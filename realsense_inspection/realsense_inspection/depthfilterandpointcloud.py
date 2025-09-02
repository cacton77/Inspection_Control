#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
from geometry_msgs.msg import PoseStamped

def make_pointcloud2(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """
    points_xyz: (N,3) float32 array in meters.
    """
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = 1
    msg.width = points_xyz.shape[0]
    msg.is_bigendian = False
    msg.is_dense = True  # no NaNs because we filtered them out
    msg.point_step = 12  # 3 * 4 bytes
    msg.row_step = msg.point_step * msg.width
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = points_xyz.astype(np.float32).tobytes()
    return msg


class DepthBGRemove(Node):
    def __init__(self):
        super().__init__('depth_bg_remove')

        # ---- Parameters ----
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('near_m', 0.07)
        self.declare_parameter('far_m', 0.50)
        self.declare_parameter('viz_enable', True)
        self.declare_parameter('publish_pointcloud', True)   # NEW
        self.declare_parameter('cloud_stride', 1)            # NEW (downsample for speed)

        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.near_m = float(self.get_parameter('near_m').value)
        self.far_m  = float(self.get_parameter('far_m').value)
        self.viz_enable = bool(self.get_parameter('viz_enable').value)
        self.publish_pointcloud = bool(self.get_parameter('publish_pointcloud').value)
        self.cloud_stride = int(self.get_parameter('cloud_stride').value)

        self.normal_estimate_pub = self.create_publisher(PoseStamped, 'normal_estimate', 10)

        # Live tuning
        self.add_on_set_parameters_callback(self._on_param_update)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.bridge = CvBridge()
        self.K = None
        self.depth_frame_id = None

        # Subs
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_info, qos)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, qos)

        # Pubs
        self.pub_masked = self.create_publisher(Image, '/camera/camera/depth/foreground', 10)
        self.pub_viz = self.create_publisher(Image, '/camera/camera/depth/foreground_viz', 10) if self.viz_enable else None
        self.pub_cloud = self.create_publisher(PointCloud2, '/camera/camera/depth/foreground_points', 10) if self.publish_pointcloud else None

        self.get_logger().info(
            'Background remover running:\n'
            f'  depth_topic={self.depth_topic}\n'
            f'  camera_info_topic={self.camera_info_topic}\n'
            f'  near_m={self.near_m:.3f}, far_m={self.far_m:.3f}\n'
            f'  viz_enable={self.viz_enable}, publish_pointcloud={self.publish_pointcloud}\n'
            f'  cloud_stride={self.cloud_stride}'
        )

    # Camera intrinsics
    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.depth_frame_id = msg.header.frame_id

    # Param updates
    def _on_param_update(self, params):
        for p in params:
            if p.name == 'near_m':
                self.near_m = float(p.value); self.get_logger().info(f'near_m -> {self.near_m:.3f} m')
            elif p.name == 'far_m':
                self.far_m = float(p.value); self.get_logger().info(f'far_m  -> {self.far_m:.3f} m')
            elif p.name == 'viz_enable':
                self.viz_enable = bool(p.value)
                # lazily create/destroy viz publisher
                if self.viz_enable and self.pub_viz is None:
                    self.pub_viz = self.create_publisher(Image, '/camera/depth/foreground_viz', 10)
                if not self.viz_enable and self.pub_viz is not None:
                    self.pub_viz = None
            elif p.name == 'publish_pointcloud':
                self.publish_pointcloud = bool(p.value)
                if self.publish_pointcloud and self.pub_cloud is None:
                    self.pub_cloud = self.create_publisher(PointCloud2, '/camera/depth/foreground_points', 10)
                if not self.publish_pointcloud and self.pub_cloud is not None:
                    self.pub_cloud = None
            elif p.name == 'cloud_stride':
                self.cloud_stride = max(1, int(p.value))
        return SetParametersResult(successful=True)

    def on_depth(self, msg: Image):
        # Convert ROS Image -> numpy
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # Normalize to meters
        if msg.encoding in ('16UC1', 'mono16'):
            depth_m = depth.astype(np.float32) * 1e-3
        elif msg.encoding == '32FC1':
            depth_m = depth.astype(np.float32)
        else:
            depth_m = depth.astype(np.float32)

        # Build mask
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        mask = valid & (depth_m >= self.near_m) & (depth_m <= self.far_m)

        # Masked depth
        masked_m = np.where(mask, depth_m, 0.0).astype(np.float32)

        # Publish masked depth
        out = self.bridge.cv2_to_imgmsg(masked_m, encoding='32FC1')
        out.header = msg.header
        self.pub_masked.publish(out)

        # Viz
        if self.pub_viz is not None:
            denom = max(self.far_m, 1e-3)
            viz8 = np.clip((masked_m / denom) * 255.0, 0, 255).astype(np.uint8)
            viz_color = cv2.applyColorMap(viz8, cv2.COLORMAP_TURBO)
            viz_msg = self.bridge.cv2_to_imgmsg(viz_color, encoding='bgr8')
            viz_msg.header = msg.header
            self.pub_viz.publish(viz_msg)

        # Point cloud (if enabled and we have intrinsics)
        if self.pub_cloud is not None and self.K is not None:
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

            h, w = masked_m.shape
            stride = max(1, self.cloud_stride)

            # sample valid pixels with stride for speed
            ys, xs = np.where(mask)
            if stride > 1 and xs.size > 0:
                ys = ys[::stride]; xs = xs[::stride]

            if xs.size > 0:
                z = masked_m[ys, xs]                         # (N,)
                x = (xs.astype(np.float32) - cx) * z / fx    # (N,)
                y = (ys.astype(np.float32) - cy) * z / fy    # (N,)
                pts = np.stack([x, y, z], axis=1)            # (N,3)
                avg_position = np.mean(pts, axis=0)
                normal_pose = PoseStamped()
                normal_pose.header = msg.header
                normal_pose.pose.position.x = float(avg_position[0]) # distance to object along x-axis
                normal_pose.pose.position.y = float(avg_position[1]) # 0
                normal_pose.pose.position.z = float(avg_position[2]) # 0
                normal_pose.pose.orientation.x = 0.0
                normal_pose.pose.orientation.y = 0.0
                normal_pose.pose.orientation.z = 0.0
                normal_pose.pose.orientation.w = 1.0
                # Create set of x_hat, y_hat, z_hat unit vectors to create rotation matrix
                # Cross product normal estimate with z-axis [0,0,1] to produce x_hat
                # Cross product normal estimate with x_hat to produce y_hat
                # R = [x_hat, y_hat, z_hat]
                # Convert R to quaternion and set in normal_pose.orientation

                self.normal_estimate_pub.publish(normal_pose)
                cloud = make_pointcloud2(
                    pts, frame_id=self.depth_frame_id or msg.header.frame_id, stamp=msg.header.stamp
                )
                self.pub_cloud.publish(cloud)


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
