#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult

import numpy as np
import cv2

class DepthBGRemove(Node):
    def __init__(self):
        super().__init__('depth_bg_remove')

        # ---- Parameters (override via --ros-args -p key:=val) ----
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('near_m', 0.07)   # keep >= near
        self.declare_parameter('far_m', 0.50)    # keep <= far
        self.declare_parameter('viz_enable', True)

        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.near_m = float(self.get_parameter('near_m').value)
        self.far_m  = float(self.get_parameter('far_m').value)
        self.viz_enable = bool(self.get_parameter('viz_enable').value)

        # Allow live tuning of near/far
        self.add_on_set_parameters_callback(self._on_param_update)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.bridge = CvBridge()
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, qos)
        self.pub_masked = self.create_publisher(Image, '/camera/camera/depth/foreground', 10)
        self.pub_viz = self.create_publisher(Image, '/camera/camera/depth/foreground_viz', 10) if self.viz_enable else None

        self.get_logger().info(
            f'Background remover running:\n'
            f'  depth_topic={self.depth_topic}\n'
            f'  near_m={self.near_m:.3f}, far_m={self.far_m:.3f}\n'
            f'  viz_enable={self.viz_enable}'
        )

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'near_m':
                self.near_m = float(p.value)
                self.get_logger().info(f'near_m -> {self.near_m:.3f} m')
            elif p.name == 'far_m':
                self.far_m = float(p.value)
                self.get_logger().info(f'far_m  -> {self.far_m:.3f} m')
            elif p.name == 'viz_enable':
                self.viz_enable = bool(p.value)
        return SetParametersResult(successful=True)

    def on_depth(self, msg: Image):
        # Convert ROS Image to numpy (preserve encoding)
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # Normalize to meters based on encoding
        if msg.encoding in ('16UC1', 'mono16'):
            depth_m = depth.astype(np.float32) * 1e-3   # mm -> m
        elif msg.encoding == '32FC1':
            depth_m = depth.astype(np.float32)          # already meters
        else:
            # Fallback: best effort
            depth_m = depth.astype(np.float32)

        # Build mask: valid & within [near, far]
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        mask = valid & (depth_m >= self.near_m) & (depth_m <= self.far_m)

        # Keep depth where mask True, else 0
        masked_m = np.where(mask, depth_m, 0.0).astype(np.float32)

        # Publish masked depth as 32FC1
        out = self.bridge.cv2_to_imgmsg(masked_m, encoding='32FC1')
        out.header = msg.header
        self.pub_masked.publish(out)

        # Optional visualization for quick sanity check
        if self.pub_viz is not None:
            # Scale by far limit for contrast (clip 0..far)
            denom = max(self.far_m, 1e-3)
            viz8 = np.clip((masked_m / denom) * 255.0, 0, 255).astype(np.uint8)
            viz_color = cv2.applyColorMap(viz8, cv2.COLORMAP_TURBO)
            viz_msg = self.bridge.cv2_to_imgmsg(viz_color, encoding='bgr8')
            viz_msg.header = msg.header
            self.pub_viz.publish(viz_msg)

def main():
    rclpy.init()
    node = DepthBGRemove()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
