import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist
from rclpy.time import Time
from std_srvs.srv import Trigger


class AdmittanceControlNode(Node):
    def __init__(self):
        super().__init__('admittance_control_node')
        self.subscription = self.create_subscription(
            Wrench,
            'teleop_force_threshold',  # topic name
            self.wrench_callback,
            10  # QoS
        )
        self.publisher = self.create_publisher(Twist,'/inspection_cell/servo_node/delta_twist_cmds', 10)
        
       # self.cli = self.create_client(Trigger, '/inspection_cell/servo_node/start_servo')
      #  while not self.cli.wait_for_service(timeout_sec=1.0):
        #    self.get_logger().info('service not available, waiting again...')
      #  self.req = Trigger.Request()
      #  self.cli.call_async(self.req)


        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.force_timer_callback)
        self.wrench_received = False

        # Physical parameters
        self.mass = 10.0  # kg
        self.inertia = [2.0, 2.0, 2.0]  # moment of inertia around x, y, z axes (Nm·s²)

        # Current velocities
        self.linear_vel = [0.0, 0.0, 0.0]
        self.angular_vel = [0.0, 0.0, 0.0]

          # Time tracking
        self.last_time = self.get_clock().now()

        self.get_logger().info("Physics-based Wrench-to-Twist node started.")

    def wrench_to_twist(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9  # seconds
        self.last_time = current_time

        if dt == 0:
            return

        # Update linear velocity: v = v0 + (F/m) * dt
        forces = [msg.force.x, msg.force.y, msg.force.z]
        for i in range(3):
            acceleration = forces[i] / self.mass
            self.linear_vel[i] += acceleration * dt

        # Update angular velocity: w = w0 + (τ/I) * dt
        torques = [msg.torque.x, msg.torque.y, msg.torque.z]
        for i in range(3):
            angular_acc = torques[i] / self.inertia[i]
            self.angular_vel[i] += angular_acc * dt

        #  Create Twist message
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = self.linear_vel
        twist.angular.x, twist.angular.y, twist.angular.z = self.angular_vel
        return twist

    def wrench_callback(self,msg):
        self.latest_twist = self.wrench_to_twist(msg)
        self.wrench_received = True
    
    def force_timer_callback(self): 
        if self.wrench_received:
          self.publisher.publish(self.latest_twist)

        #self.get_logger().info(
           # f"Δt: {dt:.3f}s | v: {self.linear_vel} | ω: {self.angular_vel}"
        #)

def main(args=None):
    rclpy.init(args=args)
    node = AdmittanceControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



