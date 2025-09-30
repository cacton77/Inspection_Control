import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions

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
                ('save_path', '/tmp/autofocus_data'),
                # Trigger
                ('autofocus_enabled', False)
            ]
        )

        # Focus Metric Parameters
        self.focus_metric = self.get_parameter('focus_metric').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.roi_width = self.get_parameter('roi_width').get_parameter_value().integer_value
        self.roi_height = self.get_parameter('roi_height').get_parameter_value().integer_value
        self.roi_x = self.get_parameter('roi_x').get_parameter_value().double_value
        self.roi_y = self.get_parameter('roi_y').get_parameter_value().double_value

        # Focus Algorithm Parameters
        self.focus_algorithm = self.get_parameter('focus_algorithm').get_parameter_value().string_value
        self.control_rate = self.get_parameter('control_rate').get_parameter_value().double_value

        # ROS Bag2 Writer Initialization
        self.object = self.get_parameter('object').get_parameter_value().string_value
        self.count = 0
        self.save_data = self.get_parameter('save_data').get_parameter_value().bool_value
        self.save_path = self.get_parameter('save_path').get_parameter_value().string_value
        
        self.storage_options = StorageOptions(uri=self.save_path, storage_id='sqlite3')
        self.converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        self.writer = SequentialWriter()

        # Enable Autofocus
        self.control_step_counter = 0
        self.autofocus_enabled = self.get_parameter('autofocus_enabled').get_parameter_value().bool_value

        # Initialize callback groups
        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        self.subscription_callback_group = MutuallyExclusiveCallbackGroup()
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()

        # TF Listener
        self.tf_buffer = Buffer()  # store and manipulate transformations
        self.listener = TransformListener(self.tf_buffer, self)

        # Autofocus Data Message
        self.autofocus_data = None

        # Initialize CvBridge
        self.bridge = CvBridge()

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
        self.autofocus_enabled = True

    def disable_autofocus(self):
        # Implement autofocus termination logic here
        self.get_logger().info('Ending autofocus...')
        self.autofocus_enabled = False

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
            self.get_logger().warn(f'Could not transform {msg.header.frame_id} to object_frame: {ex}')
            return

        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

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
        self.autofocus_data.focus_image = self.bridge.cv2_to_compressed_imgmsg(image_out)


    def control_loop(self):
        # Implement control loop logic here
        if self.autofocus_data is None:
            return

        # Calculate velocity

        # Example control logic: publish a dummy twist command
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = self.autofocus_data.header.frame_id
        twist.twist.linear.x = 0.0
        twist.twist.linear.y = 0.0
        twist.twist.linear.z = 0.0
        twist.twist.angular.x = 0.0
        twist.twist.angular.y = 0.0
        twist.twist.angular.z = 0.0

        # End condition
        if self.control_step_counter > 100:
            autofocus_enabled_param = rclpy.parameter.Parameter(
                'autofocus_enabled',
                rclpy.Parameter.Type.BOOL,
                False
            )
            self.set_parameters([autofocus_enabled_param])

            self.control_step_counter = 0
        self.control_step_counter += 1

        if self.autofocus_enabled:
            self.twist_publisher.publish(twist)

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
            elif param.name == 'save_path':
                self.save_path = param.value
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