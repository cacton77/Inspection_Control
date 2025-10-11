#!/usr/bin/env python3
# coding: utf-8

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import WrenchStamped, TwistStamped

class AdmittanceControlNode(Node):
    """
    Admittance controller that integrates commanded forces/torques into
    linear/angular velocities and publishes TwistStamped at a fixed rate.

    There are two wrench sources:
      1) teleop (e.g., user inputs)
      2) orientation controller (e.g., proportional torque from vision)
    The callbacks only STORE the latest forces/torques; all dynamics
    integration is done at a fixed rate in publish_twist().
    """

    def __init__(self):
        super().__init__('admittance_control')

        # ---------------- Parameters ----------------
        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 50.0),             # Hz
                ('mass', 5.0),                      # kg
                ('inertia_x', 5.0),                 # kg·m²
                ('inertia_y', 5.0),
                ('inertia_z', 5.0),
                ('linear_drag_x', 1.0),             # N·s/m
                ('linear_drag_y', 1.0),
                ('linear_drag_z', 1.0),
                ('rotational_drag_x', 2.0),         # N·m·s/rad
                ('rotational_drag_y', 2.0),
                ('rotational_drag_z', 2.0),
                ('frame_id', 'eoat_camera_link'),
                ('wrench_topic', '/teleop/wrench_cmds'),
                ('wrench_teleop_topic', '/teleop/wrench_cmds'),
                ('wrench_orientation_topic', '/depth_bg_remove/wrench_cmd_in_eoat_camera_link'),
                ('twist_topic', '/servo_node/delta_twist_cmds'),

                # Optional safety/comfort limits (set to 0 to disable)
                ('max_linear_speed', 0.0),          # m/s (0 = no clamp)
                ('max_angular_speed', 0.0),         # rad/s (0 = no clamp)
            ]
        )

        # Read parameters
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.mass = float(self.get_parameter('mass').value)
        self.inertia = [
            float(self.get_parameter('inertia_x').value),
            float(self.get_parameter('inertia_y').value),
            float(self.get_parameter('inertia_z').value),
        ]
        self.D_lin = [
            float(self.get_parameter('linear_drag_x').value),
            float(self.get_parameter('linear_drag_y').value),
            float(self.get_parameter('linear_drag_z').value),
        ]
        self.D_rot = [
            float(self.get_parameter('rotational_drag_x').value),
            float(self.get_parameter('rotational_drag_y').value),
            float(self.get_parameter('rotational_drag_z').value),
        ]
        self.frame_id = str(self.get_parameter('frame_id').value)
        wrench_teleop_topic = str(self.get_parameter('wrench_teleop_topic').value)
        wrench_orientation_topic = str(self.get_parameter('wrench_orientation_topic').value)
        twist_topic = str(self.get_parameter('twist_topic').value)
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)

        # ---------------- QoS ----------------
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # ---------------- Pub/Sub ----------------
        self.twist_pub = self.create_publisher(TwistStamped, twist_topic, qos_profile)

        self.teleop_sub = self.create_subscription(
            WrenchStamped, wrench_teleop_topic, self.wrench_callback_teleop, qos_profile
        )
        self.orient_sub = self.create_subscription(
            WrenchStamped, wrench_orientation_topic, self.wrench_callback_orientation, qos_profile
        )

        # ---------------- State ----------------
        # velocities (integrated state)
        self.linear_vel  = [0.0, 0.0, 0.0]   # m/s
        self.angular_vel = [0.0, 0.0, 0.0]   # rad/s

        # latest wrench commands (buffers)
        self.teleop_F = [0.0, 0.0, 0.0]      # N
        self.teleop_T = [0.0, 0.0, 0.0]      # N·m
        self.orient_F = [0.0, 0.0, 0.0]      # N
        self.orient_T = [0.0, 0.0, 0.0]      # N·m

        self.have_any_wrench = False

        # Twist message we reuse
        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.frame_id

        # timer / dt tracking
        self.last_time = None
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_twist)

        self.get_logger().info(
            f'Admittance control node started — publishing {self.publish_rate:.1f} Hz in frame "{self.frame_id}"'
        )

    # --------------- Callbacks: store only ---------------
    def wrench_callback_teleop(self, msg: WrenchStamped):
        """Store teleop wrench (forces & torques)."""
        self.teleop_F[0], self.teleop_F[1], self.teleop_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.teleop_T[0], self.teleop_T[1], self.teleop_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self.have_any_wrench = True

    def wrench_callback_orientation(self, msg: WrenchStamped):
        """Store orientation wrench (forces & torques)."""
        self.orient_F[0], self.orient_F[1], self.orient_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.orient_T[0], self.orient_T[1], self.orient_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self.have_any_wrench = True

    # --------------- Timer: integrate & publish ---------------
    def publish_twist(self):
        """Integrate admittance dynamics and publish TwistStamped at fixed rate."""
        if not self.have_any_wrench:
            return

        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return

        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.last_time = now

        # Net commands = teleop + orientation - damping
        F_cmd = [
            self.teleop_F[0] + self.orient_F[0] - self.D_lin[0] * self.linear_vel[0],
            self.teleop_F[1] + self.orient_F[1] - self.D_lin[1] * self.linear_vel[1],
            self.teleop_F[2] + self.orient_F[2] - self.D_lin[2] * self.linear_vel[2],
        ]
        T_cmd = [
            self.teleop_T[0] + self.orient_T[0] - self.D_rot[0] * self.angular_vel[0],
            self.teleop_T[1] + self.orient_T[1] - self.D_rot[1] * self.angular_vel[1],
            self.teleop_T[2] + self.orient_T[2] - self.D_rot[2] * self.angular_vel[2],
        ]

        # Integrate translational dynamics: v = v + (F/m)*dt
        inv_m = 1.0 / max(self.mass, 1e-9)
        for i in range(3):
            a = F_cmd[i] * inv_m
            self.linear_vel[i] += a * dt

        # Integrate rotational dynamics: w = w + (τ/I)*dt
        for i in range(3):
            inv_I = 1.0 / max(self.inertia[i], 1e-9)
            alpha = T_cmd[i] * inv_I
            self.angular_vel[i] += alpha * dt

        # Optional speed clamps
        if self.max_linear_speed > 0.0:
            for i in range(3):
                if self.linear_vel[i] >  self.max_linear_speed: self.linear_vel[i] =  self.max_linear_speed
                if self.linear_vel[i] < -self.max_linear_speed: self.linear_vel[i] = -self.max_linear_speed
        if self.max_angular_speed > 0.0:
            for i in range(3):
                if self.angular_vel[i] >  self.max_angular_speed: self.angular_vel[i] =  self.max_angular_speed
                if self.angular_vel[i] < -self.max_angular_speed: self.angular_vel[i] = -self.max_angular_speed

        # Fill TwistStamped
        tw = self.current_twist
        tw.header.stamp = now.to_msg()
        tw.header.frame_id = self.frame_id

        tw.twist.linear.x  = round(self.linear_vel[0], 3)
        tw.twist.linear.y  = round(self.linear_vel[1], 3)
        tw.twist.linear.z  = round(self.linear_vel[2], 3)
        tw.twist.angular.x = round(self.angular_vel[0], 3)
        tw.twist.angular.y = round(self.angular_vel[1], 3)
        tw.twist.angular.z = round(self.angular_vel[2], 3)

        # Publish (skip if everything zero)
        if not any([tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z,
                    tw.twist.angular.x, tw.twist.angular.y, tw.twist.angular.z]):
            return
        self.twist_pub.publish(tw)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AdmittanceControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f'Unhandled exception: {e}')
        else:
            print(f'Unhandled exception before node init: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
