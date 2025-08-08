import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Wrench

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')
        
        self.latest_wrench = None  # in __init__()

        self.subscription = self.create_subscription(
            Joy,
            'joy',  # topic name
            self.joy_callback,
            10  # QoS
        )
        self.publisher = self.create_publisher(Wrench, 'teleop_force_threshold', 10)
       # self.get_logger().info('Subscribed to /joy topic')
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.force_timer_cb)
        self.get_logger().info('Joystick → Wrench mapping node started.')
    #def listener_callback(self, msg):
      #  self.get_logger().info(f'Buttons: {msg.buttons}, Axes: {msg.axes}')
    def apply_deadzone(self,value):
        #return value if abs(value) >= threshold else 0.0  
        if abs(value) < 0.05:
            return 0.0
        else:
            #return value
    #def apply_normalize(self,value):   
        # Scale the remaining range to 0-1
   # def apply_normalize(self, value):
             if value > 0:
               return (value - 0.05) / (1.0 - 0.05)
             else:
               return (value + 0.05) / (1.0 - 0.05)
    def convert_joy_to_wrench(self, msg):
        wrench = Wrench()

        # Map left stick to linear forces
        wrench.force.x = self.apply_deadzone(msg.axes[0] * -1.0) # Left/right → force in x

        wrench.force.y = self.apply_deadzone(msg.axes[1] * 1.0)  # Forward/backward → force in y

        # Map right stick to torques
        wrench.torque.x = self.apply_deadzone(msg.axes[3] * -1.0)  # Right stick L/R → torque around x
        wrench.torque.y = self.apply_deadzone(msg.axes[4] * 1.0)  # Right stick U/D → torque around y

        # Optional: trigger button to add force in z
        #if len(msg.axes) > 2:
         #   wrench.force.z = (1.0 - msg.axes[2]) * 5.0  # Triggers often range from 1 to -1
        lt_val = msg.axes[2] if len(msg.axes) > 2 else 1.0  # 1 → -1
        rt_val = msg.axes[5] if len(msg.axes) > 5 else 1.0  # 1 → -1

        up_force = (1.0 - rt_val) / 2.0 * 1.0   # Scales to [0 → 5]
        down_force = (1.0 - lt_val) / 2.0 * 1.0 # Scales to [0 → 5]

        wrench.force.z = self.apply_deadzone(up_force - down_force)  # Net z force: up - down
        # Optional: button 0 adds torque in z
        if len(msg.buttons) > 1:
                if msg.buttons[3]:  # A
                    wrench.torque.z = 1.0
                elif msg.buttons[0]:  # B
                    wrench.torque.z = -1.0
        return wrench

    def joy_callback(self, msg):
        self.latest_wrench = self.convert_joy_to_wrench(msg)

    def force_timer_cb(self):
        if self.latest_wrench is not None:
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
