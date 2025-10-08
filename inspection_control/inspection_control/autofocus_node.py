import os
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

from cv_bridge import CvBridge, CvBridgeError

import inspection_control.focus_metrics as focus_metrics

from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from std_msgs.msg import Float64, String
from viewpoint_generation_interfaces.msg import AutofocusData
from viewpoint_generation_interfaces.srv import MoveToPoseStamped

from std_srvs.srv import Trigger


class AutofocusNode(Node):

    block_callback = False

    def __init__(self):
        super().__init__('autofocus_node')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                # Focus Metric Parameters
                ('focus_metric', 'sobel'),
                ('image_topic', '/camera/image_raw/compressed'),
                ('roi_width', 100),
                ('roi_height', 100),
                ('roi_x', 0.5),
                ('roi_y', 0.5),
                # Focus Algorithm Parameters
                ('control_rate', 10.0),
                ('focus_algorithm', 'default'),
                # Save Data Parameters
                ('object', ''),
                ('save_data', False),
                ('data_path', '/tmp'),
                # Trigger
                ('autofocus_enabled', False)
            ]
        )

        # Focus Metric Parameters
        self.focus_metric = self.get_parameter(
            'focus_metric').get_parameter_value().string_value
        self.image_topic = self.get_parameter(
            'image_topic').get_parameter_value().string_value
        self.roi_width = self.get_parameter(
            'roi_width').get_parameter_value().integer_value
        self.roi_height = self.get_parameter(
            'roi_height').get_parameter_value().integer_value
        self.roi_x = self.get_parameter(
            'roi_x').get_parameter_value().double_value
        self.roi_y = self.get_parameter(
            'roi_y').get_parameter_value().double_value

        # Focus Algorithm Parameters
        self.focus_algorithm = self.get_parameter(
            'focus_algorithm').get_parameter_value().string_value
        self.control_rate = self.get_parameter(
            'control_rate').get_parameter_value().double_value

        # ROS Bag2 Writer Initialization
        self.object = self.get_parameter(
            'object').get_parameter_value().string_value
        self.count = 0
        self.save_data = self.get_parameter(
            'save_data').get_parameter_value().bool_value
        self.data_path = self.get_parameter(
            'data_path').get_parameter_value().string_value

        self.storage_options = StorageOptions(
            uri=self.data_path, storage_id='sqlite3')
        self.converter_options = ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr')
        self.writer = SequentialWriter()
        self.topic_info = TopicMetadata(
            name='autofocus_data',
            type='viewpoint_generation_interfaces/msg/AutofocusData',
            serialization_format='cdr')

        # Enable Autofocus
        self.control_step_counter = 0
        self.initial_end_effector_pose = None
        self.ehc_distance = 0.1
        self.autofocus_type = "ehc"  # EHC is default
        self.autofocus_enabled = self.get_parameter(
            'autofocus_enabled').get_parameter_value().bool_value
        self.fine_mode = False
        self.max_focus_value = 0.0
        self.max_found = False
        self.max_focus_value_distance = 0.0

        # Initialize callback groups
        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        # Allow concurrent execution
        self.subscription_callback_group = ReentrantCallbackGroup()
        self.timer_callback_group = ReentrantCallbackGroup()  # Allow concurrent execution

        # TF Listener
        self.tf_buffer = Buffer()  # store and manipulate transformations
        self.listener = TransformListener(self.tf_buffer, self)

        # Autofocus Data Message
        self.autofocus_data = AutofocusData()

        # Initialize CvBridge
        self.bridge = CvBridge()

        # EMA calculation variables with rolling window of 20 values
        self.N_ema = 20  # Window size for EMA calculation
        self.focus_values_window = []  # Rolling window of focus values
        self.ema_focus_value2 = 0  # For DEMA calculation
        self.previous_dema_focus_value = 0
        self.previous_dfv = 0
        self.kv = 0.8
        self.ddfv = 0

        # Create subscription for image topic
        self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10,
            callback_group=self.subscription_callback_group
        )

        # Move To Pose Connection
        self.pose_client = self.create_client(
            MoveToPoseStamped, 'viewpoint_traversal/move_to_pose_stamped')
        while not self.pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        # Servo Connection
        self.twist_publisher = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)
        # Admittance Control Connection
        self.admittance_publisher = self.create_publisher(
            WrenchStamped, '/admittance_node/delta_twist_cmds', 10)

        # Debug publisher for focus derivatives and ratio
        self.debug_publisher = self.create_publisher(
            String, '/autofocus_debug', 10)

        # Control loop timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop,
            callback_group=self.timer_callback_group
        )

        self.add_on_set_parameters_callback(self.parameter_callback)

    def enable_autofocus(self):
        # Implement autofocus initiation logic here
        self.get_logger().info('Beginning autofocus...')
        # Reset initial pose for fresh start
        self.initial_end_effector_pose = None
        self.autofocus_enabled = True
        # Open the bag file for writing
        if self.save_data:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            uri = f'{self.data_path}/{self.object}_{self.focus_algorithm}_{self.focus_metric}_{self.count}.bag'
            self.storage_options = StorageOptions(
                uri=uri, storage_id='sqlite3')
            self.writer.open(self.storage_options, self.converter_options)
            self.writer.create_topic(self.topic_info)

    def disable_autofocus(self):
        # Implement autofocus termination logic here
        self.get_logger().info('Ending autofocus...')
        self.autofocus_enabled = False
        # Close the bag file after writing
        self.writer.close()
        # Update parameters or state as needed
        self.count += 1

    def image_callback(self, msg: CompressedImage):
        self.autofocus_data = AutofocusData()

        # Process the incoming image message
        self.autofocus_data.header = msg.header
        self.autofocus_data.image = msg
        # Look up transform from camera frame to base frame
        try:
            transform = self.tf_buffer.lookup_transform(
                'object_frame',
                msg.header.frame_id,
                msg.header.stamp
            )
            end_effector_pose = PoseStamped()
            end_effector_pose.header = transform.header
            end_effector_pose.pose.position.x = transform.transform.translation.x
            end_effector_pose.pose.position.y = transform.transform.translation.y
            end_effector_pose.pose.position.z = transform.transform.translation.z
            end_effector_pose.pose.orientation = transform.transform.rotation

        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform {msg.header.frame_id} to object_frame: {ex}')
            return

        cv_image = self.bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding='bgr8')

        # Crop the image based on ROI parameters
        # Calculate the pixel coordinates for the ROI
        height, width, _ = cv_image.shape
        x = int(self.roi_x * width)
        y = int(self.roi_y * height)
        roi_width = self.roi_width
        roi_height = self.roi_height
        roi = cv_image[y:y+roi_height, x:x+roi_width]

        # Calculate focus metrics and update autofocus_data here
        if self.focus_metric == 'sobel':
            focus_value, image_out = focus_metrics.sobel(roi)
        elif self.focus_metric == 'squared_gradient':
            focus_value, image_out = focus_metrics.squared_gradient(roi)
        elif self.focus_metric == 'fswm':
            focus_value, image_out = focus_metrics.fswm(roi)

        self.autofocus_data.end_effector_pose = end_effector_pose
        self.autofocus_data.focus_value = focus_value
        self.autofocus_data.focus_image = self.bridge.cv2_to_compressed_imgmsg(
            image_out)

    def adaptive(self):
        # Calculate velocity
        # fine search near top. Thresholded at 0.1 to prevent false positives for noise changes
        if not self.fine_mode:
            if self.autofocus_data.smooth_ddfv < 0.1 and self.autofocus_data.dfv > 0.1:
                self.get_logger().info('Entering fine')
                self.fine_mode = True
                v = self.kv * (self.autofocus_data.ratio - 0.5)
            else:
                v = self.kv/abs(self.autofocus_data.ratio)
                if self.autofocus_data.ratio == 0.0 or self.autofocus_data.ratio == float('inf'):
                    v = 0.2  # Hardcode to keep going
                self.get_logger().info(
                    f'Coarse: ratio {self.autofocus_data.ratio:.6f}, v {v}')
        else:
            # ddfv<0 and dfv=0, here it's written as 1st instance dfv<-2 to avoid false negative, so overshooting a bit
            if self.autofocus_data.smooth_ddfv < 0.1 and self.autofocus_data.dfv < -2:
                v = 0.0
                self.max_found = True
            else:
                v = self.kv * (self.autofocus_data.ratio - 0.5)
                self.get_logger().info(
                    f'Fine mode: ratio {self.autofocus_data.ratio:.6f}, v {v}, ,dfv {self.autofocus_data.dfv}, smoothddfv {self.autofocus_data.smooth_ddfv}')
        return v

    def ehc(self):
        # Safety check to ensure we have data
        if self.autofocus_data is None:
            return 0.0

        v = 0.25
        # Initialize starting position on first call
        if self.initial_end_effector_pose is None:
            self.initial_end_effector_pose = self.autofocus_data.end_effector_pose
        if self.distance_traveled > self.ehc_distance:
            v = 0.0
            self.max_found = True
        return v

    def return_to_max(self):
        self.get_logger().info('Returning to max focus position')
        # Compare current position to maximum focus position self.distance_traveled should go down
        e = self.distance_traveled - self.max_focus_value_distance
        # Logic to return to the maximum focus position
        if abs(e) > 0.01:  # Far from target if >1 cm
            v = -0.4 if e > 0 else 0.4
        else:  # <1 cm
            v = -self.kv * e  # Proportional for precision
        # Implement the return to max logic here
        if abs(e) < 0.001:
            v = 0.0  # Stop when close enough
            # Maybe set self.max_found = False or disable autofocus
            autofocus_enabled_param = rclpy.parameter.Parameter(
                'autofocus_enabled',
                rclpy.Parameter.Type.BOOL,
                False
            )
            self.set_parameters([autofocus_enabled_param])
            # Reset for next autofocus run
            self.initial_end_effector_pose = None
            self.fine_mode = False
            self.max_found = False
        return v

    def control_loop(self):
        if not self.autofocus_data:
            return
        # Calculate EMA and ratio using rolling window of previous 20 values
        # Add current focus value to rolling window
        self.focus_values_window.append(self.autofocus_data.focus_value)

        # Keep only last N_ema values
        if len(self.focus_values_window) > self.N_ema:
            self.focus_values_window.pop(0)

        # Calculate EMA and DEMA
        if len(self.focus_values_window) == 1:
            # First value, initialize EMA values
            self.autofocus_data.ema_focus_value = self.autofocus_data.focus_value
            self.ema_focus_value2 = self.autofocus_data.focus_value
            self.autofocus_data.dema_focus_value = self.autofocus_data.focus_value
            self.previous_dema_focus_value = self.autofocus_data.dema_focus_value
        else:
            # Calculate EMA smoothing factor
            K = 2 / (len(self.focus_values_window) + 1)

            # Calculate EMA and DEMA
            self.autofocus_data.ema_focus_value = (
                K * (self.autofocus_data.focus_value - self.autofocus_data.ema_focus_value)) + self.autofocus_data.ema_focus_value
            self.ema_focus_value2 = (
                K * (self.autofocus_data.ema_focus_value - self.ema_focus_value2)) + self.ema_focus_value2
            self.autofocus_data.dema_focus_value = 2 * \
                self.autofocus_data.ema_focus_value - self.ema_focus_value2

            # Calculate ratio using current and previous DEMA values
            self.autofocus_data.ratio = self.autofocus_data.dema_focus_value / \
                self.previous_dema_focus_value

        # Compute dfv and ddfv using current and previous DEMA values
        self.autofocus_data.dfv = self.autofocus_data.dema_focus_value - \
            self.previous_dema_focus_value
        # No smoothing, will consider adding if deemed necessary
        self.ddfv = self.autofocus_data.dfv - self.previous_dfv
        # Smoothing ddFV
        K_smooth = 2 / (3 + 1)
        self.autofocus_data.smooth_ddfv = (
            K_smooth * (self.ddfv - self.autofocus_data.smooth_ddfv)) + self.autofocus_data.smooth_ddfv

        # Store current dfv for next iteration
        self.previous_dfv = self.autofocus_data.dfv
        # Store current DEMA for next iteration
        self.previous_dema_focus_value = self.autofocus_data.dema_focus_value

        # Publish debug information
        debug_msg = String()
        debug_msg.data = f"fv: {self.autofocus_data.focus_value:.6f}, dfv: {self.autofocus_data.dfv:.6f}, smooth_ddfv: {self.autofocus_data.smooth_ddfv:.6f}, ratio: {self.autofocus_data.ratio:.6f}"
        self.debug_publisher.publish(debug_msg)

        if self.autofocus_enabled:
            # Compute distance traveled, comparing current to initial
            self.distance_traveled = ((self.autofocus_data.end_effector_pose.pose.position.x - self.initial_end_effector_pose.pose.position.x)**2 +
                                      (self.autofocus_data.end_effector_pose.pose.position.y - self.initial_end_effector_pose.pose.position.y)**2 +
                                      (self.autofocus_data.end_effector_pose.pose.position.z - self.initial_end_effector_pose.pose.position.z)**2)**0.5
            # self.get_logger().info(f'EHC: distance_traveled: {self.distance_traveled:.6f}')

            # Save maximum raw focus value
            if self.autofocus_data.focus_value > self.max_focus_value:
                self.max_focus_value = self.autofocus_data.focus_value
                self.max_focus_value_distance = self.distance_traveled

            # Only run algorithm when autofocus is active
            if self.max_found:
                v = self.return_to_max()
            if self.focus_algorithm == "default":
                v = self.adaptive()
            elif self.focus_algorithm == "adaptive":
                v = self.adaptive()
            elif self.focus_algorithm == "ehc":
                v = self.ehc()

            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = self.autofocus_data.header.frame_id
            twist.twist.linear.x = 0.0
            # Twist z set by control logic. It is parallel to camera lens
            twist.twist.linear.z = float(v)
            twist.twist.linear.y = 0.0
            twist.twist.angular.x = 0.0
            twist.twist.angular.y = 0.0
            twist.twist.angular.z = 0.0

            self.control_step_counter += 1
            self.twist_publisher.publish(twist)

            # Write data if enabled
            if self.save_data:
                self.bag_autofocus_data()

    def bag_autofocus_data(self):
        self.writer.write(
            'autofocus_data',
            serialize_message(self.autofocus_data),
            self.get_clock().now().nanoseconds
        )

    def parameter_callback(self, params):
        for param in params:
            # Focus Metric Parameters
            if param.name == 'focus_metric':
                self.focus_metric = param.value
            elif param.name == 'image_topic':
                self.image_topic = param.value
                self.subscriber.destroy()
                self.subscriber = self.create_subscription(
                    CompressedImage,
                    self.image_topic,
                    self.image_callback,
                    10
                )
            elif param.name == 'roi_width':
                self.roi_width = param.value
            elif param.name == 'roi_height':
                self.roi_height = param.value
            elif param.name == 'roi_x':
                self.roi_x = param.value
            elif param.name == 'roi_y':
                self.roi_y = param.value
            # Focus Algorithm Parameters
            elif param.name == 'focus_algorithm':
                self.focus_algorithm = param.value
            # Save Data Parameters
            elif param.name == 'object':
                self.object = param.value
                self.count = 0
            elif param.name == 'save_data':
                self.save_data = param.value
            elif param.name == 'data_path':
                self.data_path = param.value
            elif param.name == 'autofocus_enabled':
                if not self.autofocus_enabled and param.value:
                    self.enable_autofocus()
                elif self.autofocus_enabled and not param.value:
                    self.disable_autofocus()

        result = SetParametersResult()
        result.successful = True

        return result


def main():
    rclpy.init()
    node = AutofocusNode()

    # Use MultiThreadedExecutor with at least 2 threads
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
