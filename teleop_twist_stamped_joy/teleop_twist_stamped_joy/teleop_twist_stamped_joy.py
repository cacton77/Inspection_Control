#!/usr/bin/env python3

"""
ROS2 Teleop TwistStamped Joy Node

Translates sensor_msgs/Joy messages to geometry_msgs/TwistStamped messages
and publishes them at a fixed rate.

Author: Assistant
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header

import math


class TeleopTwistStampedJoy(Node):
    """
    Node that converts Joy messages to TwistStamped messages with fixed-rate publishing.
    """
    
    def __init__(self):
        super().__init__('teleop_twist_stamped_joy')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 50.0),  # Hz
                ('linear_scale', 1.0),  # Max linear velocity (m/s)
                ('angular_scale', 1.0),  # Max angular velocity (rad/s)
                ('x_axis', 0),  # Pan Camera Left/Right
                ('y_axis', 1),  # Pan Camera Up/Down
                ('z_axis', 2),  # Push/Pull Camera
                ('roll_axis', 3),  # Roll Camera About Z-Axis
                ('pitch_axis', 4),  # Pitch Camera About X-Axis
                ('yaw_axis', 5),   # Yaw Camera About Y-Axis
                ('invert_x', False),  # Invert X-axis for left/right movement
                ('invert_y', False),  # Invert Y-axis for up/down movement
                ('invert_z', False),  # Invert Z-axis for push/pull movement
                ('enable_button', 0),  # Safety button to enable/disable control
                ('deadzone', 0.05),  # Deadzone to prevent drift
                ('frame_id', 'base_link'),  # Frame ID for TwistStamped
                ('joy_topic', 'joy'),  # Topic for incoming Joy messages
                ('twist_topic', 'cmd_vel_stamped')  # Topic for outgoing TwistStamped messages
            ]
        )
        
        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.linear_scale = self.get_parameter('linear_scale').get_parameter_value().double_value
        self.angular_scale = self.get_parameter('angular_scale').get_parameter_value().double_value
        self.x_axis = self.get_parameter('x_axis').get_parameter_value().integer_value
        self.y_axis = self.get_parameter('y_axis').get_parameter_value().integer_value
        self.z_axis = self.get_parameter('z_axis').get_parameter_value().integer_value
        self.roll_axis = self.get_parameter('roll_axis').get_parameter_value().integer_value
        self.pitch_axis = self.get_parameter('pitch_axis').get_parameter_value().integer_value
        self.yaw_axis = self.get_parameter('yaw_axis').get_parameter_value().integer_value
        self.invert_x = self.get_parameter('invert_x').get_parameter_value().bool_value
        self.invert_y = self.get_parameter('invert_y').get_parameter_value().bool_value
        self.invert_z = self.get_parameter('invert_z').get_parameter_value().bool_value
        self.enable_button = self.get_parameter('enable_button').get_parameter_value().integer_value
        self.deadzone = self.get_parameter('deadzone').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        joy_topic = self.get_parameter('joy_topic').get_parameter_value().string_value
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
            Joy,
            joy_topic,
            self.joy_callback,
            qos_profile
        )
        
        # Internal state
        self.last_joy_msg = None
        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.frame_id
        
        # Create timer for fixed-rate publishing
        timer_period = 1.0 / self.publish_rate  # seconds
        self.timer = self.create_timer(timer_period, self.publish_twist)
        
        # Log startup info
        self.get_logger().info(f'TeleopTwistStampedJoy node started')
        self.get_logger().info(f'Publishing at {self.publish_rate} Hz')
        self.get_logger().info(f'Enable button: {self.enable_button}')
        
    def joy_callback(self, msg):
        """
        Process incoming Joy messages and update twist command.
        """
        self.last_joy_msg = msg
            
        # Check if enable button is pressed (safety feature) or if button is not configured
        if not msg.buttons[self.enable_button] and self.enable_button >= 0:
            # Enable button not pressed - stop the robot
            self.current_twist.twist.linear.x = 0.0
            self.current_twist.twist.linear.y = 0.0
            self.current_twist.twist.linear.z = 0.0
            self.current_twist.twist.angular.x = 0.0
            self.current_twist.twist.angular.y = 0.0
            self.current_twist.twist.angular.z = 0.0
            return
        
        # Get raw axis values
        vx_raw = msg.axes[self.x_axis]
        vy_raw = msg.axes[self.y_axis]
        vz_raw = msg.axes[self.z_axis]
        wr_raw = msg.axes[self.roll_axis]
        wp_raw = msg.axes[self.pitch_axis]
        wy_raw = msg.axes[self.yaw_axis]
        
        # Apply deadzone
        vx_filtered_deadzone = self.apply_deadzone(vx_raw)
        vx_filtered= self.apply_normalize(vx_filtered_deadzone)
        vy_filtered_deadzone = self.apply_deadzone(vy_raw)
        vy_filtered = self.apply_normalize(vy_filtered_deadzone)
        vz_filtered_deadzone = self.apply_deadzone(vz_raw)
        vz_filtered = self.apply_normalize (vz_filtered_deadzone)
        wr_filtered_deadzone = self.apply_deadzone(wr_raw)
        wr_filtered = self.apply_normalize(wr_filtered_deadzone)
        wp_filtered_deadzone = self.apply_deadzone(wp_raw)
        wp_filtered = self.apply_normalize(wp_filtered_deadzone)
        wy_filtered_deadzone = self.apply_deadzone(wy_raw)
        wy_filtered=self.apply_normalize(wy_filtered_deadzone)
                
        # Calculate scaled velocities
        vx = vx_filtered * self.linear_scale
        vy = vy_filtered * self.linear_scale
        vz = vz_filtered * self.linear_scale
        wr = wr_filtered * self.angular_scale
        wp = wp_filtered * self.angular_scale
        wy = wy_filtered * self.angular_scale
        
        # Update twist message
        self.current_twist.twist.linear.x = vx if not self.invert_x else -vx
        self.current_twist.twist.linear.y = vy if not self.invert_y else -vy
        self.current_twist.twist.linear.z = vz if not self.invert_z else -vz
        self.current_twist.twist.angular.x = wp if not self.invert_x else -wp
        self.current_twist.twist.angular.y = wy if not self.invert_y else -wy
        self.current_twist.twist.angular.z = wr if not self.invert_z else -wr
        
    def apply_deadzone(self, value):
        """
        Apply deadzone to joystick input to eliminate drift.
        """
        if abs(value) < self.deadzone:
            return 0.0
        else:
            return value
        # Scale the remaining range to 0-1
    def apply_normalize(self, value):
        if value > 0:
            return (value - self.deadzone) / (1.0 - self.deadzone)
        else:
            return (value + self.deadzone) / (1.0 - self.deadzone)
        
    
    def publish_twist(self):
        """
        Publish TwistStamped message at fixed rate.
        """
        # Update timestamp
        self.current_twist.header.stamp = self.get_clock().now().to_msg()
        
        # Publish the message
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
        node = TeleopTwistStampedJoy()
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