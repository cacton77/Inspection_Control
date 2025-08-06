import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Wrench

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')
        

        self.subscription = self.create_subscription(
            Joy,
            'joy',  # topic name
            self.joy_callback,
            10  # QoS
        )
        self.publisher = self.create_publisher(Wrench, 'teleop_force', 10)
       # self.get_logger().info('Subscribed to /joy topic')
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.force_timer_cb)
        self.get_logger().info('Joystick → Wrench mapping node started.')
    #def listener_callback(self, msg):
      #  self.get_logger().info(f'Buttons: {msg.buttons}, Axes: {msg.axes}')

    def convert_joy_to_wrench(self, msg):
        wrench = Wrench()

        # Map left stick to linear forces
        wrench.force.x = msg.axes[0] * 10.0  # Left/right → force in x
        wrench.force.y = msg.axes[1] * 10.0  # Forward/backward → force in y

        # Map right stick to torques
        wrench.torque.x = msg.axes[3] * 5.0  # Right stick L/R → torque around x
        wrench.torque.y = msg.axes[4] * 5.0  # Right stick U/D → torque around y

        # Optional: trigger button to add force in z
        #if len(msg.axes) > 2:
         #   wrench.force.z = (1.0 - msg.axes[2]) * 5.0  # Triggers often range from 1 to -1
        lt_val = msg.axes[2] if len(msg.axes) > 2 else 1.0  # 1 → -1
        rt_val = msg.axes[5] if len(msg.axes) > 5 else 1.0  # 1 → -1

        up_force = (1.0 - lt_val) / 2.0 * 5.0   # Scales to [0 → 5]
        down_force = (1.0 - rt_val) / 2.0 * 5.0 # Scales to [0 → 5]

        wrench.force.z = up_force - down_force  # Net z force: up - down
        # Optional: button 0 adds torque in z
        if len(msg.buttons) > 1:
                if msg.buttons[0]:  # A
                    wrench.torque.z = 2.0
                elif msg.buttons[1]:  # B
                    wrench.torque.z = -2.0
        return wrench

    def joy_callback(self, msg):
        self.latest_wrench = self.convert_joy_to_wrench(msg)

    def force_timer_cb(self):
        self.publisher.publish(self.latest_wrench)
        #self.get_logger().info(f"Force: {wrench.force}, Torque: {wrench.torque}")


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
