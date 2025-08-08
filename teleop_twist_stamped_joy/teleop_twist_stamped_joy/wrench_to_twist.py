#!/usr/bin/env python3

"""
ROS2 Wrench Twist Node

Translates geometry_msgs/Wrench messages to geometry_msgs/TwistStamped message
and publishes them at a fixed rate.

Author: Assistant
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

#from sensor_msgs.msg import Joy
from geometry_msgs.msg import WrenchStamped, TwistStamped
from std_srvs.srv import Trigger

import math

# Comment


class WrenchtoTwist(Node):
    """
    Node that converts Wrench messages to Twiststamped messages with fixed-rate publishing.
    """
    
    def __init__(self):
        super().__init__('wrench_to_twist')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 50.0),  # Hz
                ('mass', 5.0),  # Hz
                ('inertia_x', 5.0),  # Hz
                ('inertia_y', 5.0),  # Hz
                ('inertia_z', 5.0),  # Hz
                ('linear_drag_x',1.0),
                ('linear_drag_y',1.0),
                ('linear_drag_z', 1.0),
                ('rotational_drag_x',2.0),
                ('rotational_drag_y',2.0),
                ('rotational_drag_z',2.0),
              #  ('enable_button', 0),  # Safety button to enable/disable control
                ('frame_id', 'base_link'),  # Frame ID for TwistStamped
                ('wrench_topic', 'cmd_vel_stamped'),  # Topic for incoming Joy messages
                ('twist_topic', '/inspection_cell/servo_node/delta_twist_cmds')  # Topic for outgoing TwistStamped messages
            ]
        )
        
        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.mass = self.get_parameter('mass').get_parameter_value().double_value
        inertia_x = self.get_parameter('inertia_x').get_parameter_value().double_value
        inertia_y = self.get_parameter('inertia_y').get_parameter_value().double_value
        inertia_z = self.get_parameter('inertia_z').get_parameter_value().double_value
        self.D_lin = [
            self.get_parameter('linear_drag_x').get_parameter_value().double_value,
            self.get_parameter('linear_drag_y').get_parameter_value().double_value,
            self.get_parameter('linear_drag_z').get_parameter_value().double_value
        ]
        self.D_rot = [
            self.get_parameter('rotational_drag_x').get_parameter_value().double_value,
            self.get_parameter('rotational_drag_y').get_parameter_value().double_value,
            self.get_parameter('rotational_drag_z').get_parameter_value().double_value
        ]

        #self.force_scale = self.get_parameter('force_scale').get_parameter_value().double_value
        #self.torque_scale = self.get_parameter('torque_scale').get_parameter_value().double_value
       # self.x_axis = self.get_parameter('x_axis').get_parameter_value().integer_value
       # self.y_axis = self.get_parameter('y_axis').get_parameter_value().integer_value
       # self.z_axis_up = self.get_parameter('z_axis_up').get_parameter_value().integer_value
       # self.z_axis_down = self.get_parameter('z_axis_down').get_parameter_value().integer_value
       # self.roll_axis_positive = self.get_parameter('roll_axis_positive').get_parameter_value().integer_value
     #   self.roll_axis_negative = self.get_parameter('roll_axis_negative').get_parameter_value().integer_value
     #   self.pitch_axis = self.get_parameter('pitch_axis').get_parameter_value().integer_value
     # #  self.yaw_axis = self.get_parameter('yaw_axis').get_parameter_value().integer_value
      #  self.invert_x = self.get_parameter('invert_x').get_parameter_value().bool_value
     #   self.invert_y = self.get_parameter('invert_y').get_parameter_value().bool_value
      #  self.invert_z = self.get_parameter('invert_z').get_parameter_value().bool_value
       # self.enable_button = self.get_parameter('enable_button').get_parameter_value().integer_value
      #  self.deadzone = self.get_parameter('deadzone').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        wrench_topic = self.get_parameter('wrench_topic').get_parameter_value().string_value
        twist_topic = self.get_parameter('twist_topic').get_parameter_value().string_value


        
        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Publishers and subscribers
        self.twist_pub = self.create_publisher(
            TwistStamped, 
            twist_topic, 
            qos_profile
        )
        
        self.joy_sub = self.create_subscription(
            WrenchStamped,
            wrench_topic,
            self.wrench_callback,
            qos_profile
        )

       # self.cli = self.create_client(Trigger, twist_topic)
      #  while not self.cli.wait_for_service(timeout_sec=1.0):
       #     self.get_logger().info('service not available, waiting again...')
      #  self.req = Trigger.Request()
     #   self.cli.call_async(self.req)

        self.linear_vel = [0., 0., 0.]
        self.inertia = [inertia_x, inertia_y, inertia_z]
        self.angular_vel = [0., 0., 0.]


        self.wrench_received = False
        
        # Internal state
       # self.last_wrench_msg = None
        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.frame_id
        #self.last_time = self.get_clock().now()
        self.last_time= None
        # Create timer for fixed-rate publishing
        timer_period = 1.0 / self.publish_rate  # seconds
        self.timer = self.create_timer(timer_period, self.publish_twist)
        
        # Log startup info
        self.get_logger().info(f'WrenchtoTwist node started')
        self.get_logger().info(f'Publishing at {self.publish_rate} Hz')
       # self.get_logger().info(f'Enable button: {self.enable_button}')
        
    def wrench_callback(self, msg):
        """
        Process incoming Joy messages and update wrench command.
        """
        current_time = msg.header.stamp

        if not self.last_time:
            self.last_time = current_time
            return
        
        self.last_wrench_msg = msg

        dt = (current_time.sec - self.last_time.sec) + 1e-9*(current_time.nanosec - self.last_time.nanosec)

        self.last_time = current_time

        # Update linear velocity: v = v0 + (F/m) * dt
        self.forces = [msg.wrench.force.x -self.D_lin[0] * self.linear_vel[0], msg.wrench.force.y -self.D_lin[1] * self.linear_vel[1], msg.wrench.force.z -self.D_lin[2] * self.linear_vel[2]]
        for i in range(3):
            self.acceleration = self.forces[i] / self.mass
            self.linear_vel[i] += self.acceleration * dt

        # Update angular velocity: w = w0 + (τ/I) * dt
        self.torques = [msg.wrench.torque.x-self.D_rot[0] * self.angular_vel[0], msg.wrench.torque.y-self.D_rot[1] * self.angular_vel[1], msg.wrench.torque.z-self.D_rot[2] * self.angular_vel[2]]
        for i in range(3):
            self.angular_acc = self.torques[i] / self.inertia[i]
            self.angular_vel[i] += self.angular_acc * dt
    
        # Update twist message
        self.current_twist.twist.linear.x = round(self.linear_vel[0], 3)
        self.current_twist.twist.linear.y = round(self.linear_vel[1],3)
        self.current_twist.twist.linear.z = round(self.linear_vel[2],3)
        self.current_twist.twist.angular.x = round(self.angular_vel[0],3)
        self.current_twist.twist.angular.y = round(self.angular_vel[1],3)
        self.current_twist.twist.angular.z = round(self.angular_vel[2],3)
        #self.current_twist.twist.angular.z = wr if not self.invert_z else -wr
        self.wrench_received = True
        
    
    def publish_twist(self):
        """
        Publish TwistStamped message at fixed rate.
        """

        # Update timestamp
       # self.current_wrench.header.stamp = self.get_clock().now().to_msg()
        
        # Publish the message
        if self.wrench_received:
          self.current_twist.header.stamp = self.get_clock().now().to_msg() 
          self.twist_pub.publish(self.current_twist)
        
        # Optional: Log current velocities (uncomment for debugging)
        # self.get_logger().debug(
        #     f'Publishing: linear={self.current_twist.twist.linear.x:.2f}, '
        #     f'angular={self.current_twist.twist.angular.z:.2f}'
        # )


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        node = WrenchtoTwist()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()