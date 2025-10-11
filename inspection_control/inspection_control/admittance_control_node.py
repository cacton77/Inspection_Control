#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# from sensor_msgs.msg import Joy
from geometry_msgs.msg import WrenchStamped, TwistStamped
from std_srvs.srv import Trigger

import math


class AdmittanceControlNode(Node):
    """
    Node that converts Wrench messages to Twiststamped messages with fixed-rate publishing.
    """

    def __init__(self):
        super().__init__('admittance_control')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 50.0),  # Hz
                ('mass', 5.0),  # Hz
                ('inertia_x', 5.0),  # Hz
                ('inertia_y', 5.0),  # Hz
                ('inertia_z', 5.0),  # Hz
                ('linear_drag_x', 1.0),
                ('linear_drag_y', 1.0),
                ('linear_drag_z', 1.0),
                ('rotational_drag_x', 2.0),
                ('rotational_drag_y', 2.0),
                ('rotational_drag_z', 2.0),
                #  ('enable_button', 0),  # Safety button to enable/disable control
                ('frame_id', 'eoat_camera_link'),  # Frame ID for TwistStamped
                # Topic for incoming Joy messages
                ('wrench_topic', '/teleop/wrench_cmds'),
                # Topic for outgoing TwistStamped messages
                ('twist_topic', '/servo_node/delta_twist_cmds')
            ]
        )

        # Get parameters
        self.publish_rate = self.get_parameter(
            'publish_rate').get_parameter_value().double_value
        self.mass = self.get_parameter(
            'mass').get_parameter_value().double_value
        inertia_x = self.get_parameter(
            'inertia_x').get_parameter_value().double_value
        inertia_y = self.get_parameter(
            'inertia_y').get_parameter_value().double_value
        inertia_z = self.get_parameter(
            'inertia_z').get_parameter_value().double_value
        self.D_lin = [
            self.get_parameter(
                'linear_drag_x').get_parameter_value().double_value,
            self.get_parameter(
                'linear_drag_y').get_parameter_value().double_value,
            self.get_parameter(
                'linear_drag_z').get_parameter_value().double_value
        ]
        self.D_rot = [
            self.get_parameter(
                'rotational_drag_x').get_parameter_value().double_value,
            self.get_parameter(
                'rotational_drag_y').get_parameter_value().double_value,
            self.get_parameter(
                'rotational_drag_z').get_parameter_value().double_value
        ]

        self.frame_id = self.get_parameter(
            'frame_id').get_parameter_value().string_value
        wrench_topic = self.get_parameter(
            'wrench_topic').get_parameter_value().string_value
        twist_topic = self.get_parameter(
            'twist_topic').get_parameter_value().string_value

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

        #self.joy_sub = self.create_subscription(
        #    WrenchStamped,
        #    wrench_topic,
        #    self.wrench_callback,
        #    qos_profile
       # )
        self.torque_sub = self.create_subscription(
            WrenchStamped,
            '/depth_bg_remove/wrench_cmd_in_eoat_camera_link',
            self.wrench_callback,
            qos_profile
        )
        self.linear_vel = [0., 0., 0.]
        self.inertia = [inertia_x, inertia_y, inertia_z]
        self.angular_vel = [0., 0., 0.]

        self.wrench_received = False

        # Internal state
       # self.last_wrench_msg = None
        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.frame_id
        # self.last_time = self.get_clock().now()
        self.last_time = None
        # Create timer for fixed-rate publishing
        timer_period = 1.0 / self.publish_rate  # seconds
        self.timer = self.create_timer(timer_period, self.publish_twist)

        # Log startup info
        self.get_logger().info(f'Admittance control node started')
        self.get_logger().info(f'Publishing twists at {self.publish_rate} Hz')

    def wrench_callback(self, msg):
        """
        Process incoming Joy messages and update wrench command.
        """
        current_time = msg.header.stamp

        if not self.last_time:
            self.last_time = current_time
            return

        self.last_wrench_msg = msg

        dt = (current_time.sec - self.last_time.sec) + 1e-9 * \
            (current_time.nanosec - self.last_time.nanosec)

        self.last_time = current_time

        # Update linear velocity: v = v0 + (F/m) * dt
        self.forces = [msg.wrench.force.x - self.D_lin[0] * self.linear_vel[0],
                       msg.wrench.force.y - self.D_lin[1] * self.linear_vel[1],
                       msg.wrench.force.z - self.D_lin[2] * self.linear_vel[2]]
        # print(self.forces)
        for i in range(3):
            self.acceleration = self.forces[i] / self.mass
            self.linear_vel[i] += self.acceleration * dt

        # Update angular velocity: w = w0 + (τ/I) * dt
        self.torques = [msg.wrench.torque.x - self.D_rot[0]*self.angular_vel[0], msg.wrench.torque.y -
                        self.D_rot[1]*self.angular_vel[1], msg.wrench.torque.z - self.D_rot[2]*self.angular_vel[2]]

        for i in range(3):
            self.angular_acc = self.torques[i] / self.inertia[i]
            self.angular_vel[i] += self.angular_acc * dt

        # Update twist message
        self.current_twist.twist.linear.x = round(self.linear_vel[0], 3)
        self.current_twist.twist.linear.y = round(self.linear_vel[1], 3)
        self.current_twist.twist.linear.z = round(self.linear_vel[2], 3)
        self.current_twist.twist.angular.x = round(self.angular_vel[0], 3)
        self.current_twist.twist.angular.y = round(self.angular_vel[1], 3)
        self.current_twist.twist.angular.z = round(self.angular_vel[2], 3)
       # self.current_twist.twist.angular.z = wr if not self.invert_z else -wr
        self.wrench_received = True

    def publish_twist(self):
        """
        Publish TwistStamped message at fixed rate.
        """
        # If self.current_twist is all 0, return
        # TODO: Remove this check and implement admittance-based autofocus
        if not any([
            self.current_twist.twist.linear.x,
            self.current_twist.twist.linear.y,
            self.current_twist.twist.linear.z,
            self.current_twist.twist.angular.x,
            self.current_twist.twist.angular.y,
            self.current_twist.twist.angular.z
        ]):
            return
        # Publish the message
        if self.wrench_received:
            self.current_twist.header.stamp = self.get_clock().now().to_msg()
            self.twist_pub.publish(self.current_twist)



def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    try:
        node = AdmittanceControlNode()
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
