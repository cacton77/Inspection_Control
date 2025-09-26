import rclpy
from rclpy.node import Node
from inspection_control import focus_metrics
from viewpoint_generation_interfaces.msg import FocusValue

class FocusMonitorNode(Node):
    def __init__(self):
        super().__init__('focus_monitor_node')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('focus_metric', 'laplacian'),
                ('image_topic', '/camera/image_raw/compressed'),
                ('roi', (0.5, 0.5, 100, 100)),  # (x, y, width, height)
            ]
        )
        self.focus_metric = self.get_parameter('focus_metric').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.roi = self.get_parameter('roi').get_parameter_value().double_array_value

        self.subscriber = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(FocusValue, f'{self.image_topic}/focus_value', 10)

    def image_callback(self, msg):
        # Process the incoming image message
        self.get_logger().info('Received image message')        
        focus_value = FocusValue()
        focus_value.header.stamp = self.get_clock().now().to_msg()
        focus_value.header.frame_id = "eoat_camera_link"
        focus_value.focus_metric = self.focus_metric
        self.publisher.publish(focus_value)


def __main__():
    rclpy.init()
    node = FocusMonitorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()