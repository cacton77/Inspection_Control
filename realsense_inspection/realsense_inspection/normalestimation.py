#!/usr/bin/env python3
import math
import numpy as np
import numpy.linalg as LA
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header, Bool
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Quaternion


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


def _pca_plane_normal(pts_np: np.ndarray):
    """Return (centroid, unit normal) for best-fit plane to pts_np (N,3)."""
    c = pts_np.mean(axis=0)
    X = pts_np - c
    # 3x3 covariance; smallest eigenvalue's eigenvector is the plane normal
    C = (X.T @ X) / max(len(X) - 1, 1)
    w, v = LA.eigh(C)
    n = v[:, 0]
    # Make direction consistent (toward camera -Z in depth cam frame)
    if n[2] > 0:
        n = -n
    n /= (LA.norm(n) + 1e-12)
    return c, n


def _quaternion_from_z(normal: np.ndarray) -> Quaternion:
   # """Quaternion aligning +Z axis with 'normal' (xyzw)."""
#    z = normal / (LA.norm(normal) + 1e-12)
  #  tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
   # if abs(np.dot(tmp, z)) > 0.9:
   #     tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
  #  x = np.cross(tmp, z); x /= (LA.norm(x) + 1e-12)
  #  y = np.cross(z, x)
  #  R = np.stack([x, y, z], axis=1)

    z = normal / LA.norm(normal)
   # z = -normal / LA.norm(normal)
    up = np.array([0, 1, 0])
    if np.array_equal(z, up):
       x = np.array([1, 0, 0])
      #x = np.array([-1, 0, 0])
    elif np.array_equal(z, -up):
        x = np.array([-1, 0, 0])
     #  x = np.array([1, 0, 0])
    else:
         x = np.cross(up, z)
         #x = np.cross(z, up)
         x /= LA.norm(x)
    y = np.cross(z, x)
    #y = np.cross(x, z)
    R = np.stack([x, y, z], axis=1)


    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2,1] - R[1,2]) / s
        qy = (R[0,2] - R[2,0]) / s
        qz = (R[1,0] - R[0,1]) / s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / s; qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s; qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s; qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s; qz = 0.25 * s
    return Quaternion(x=qx, y=qy, z=qz, w=qw)


class DepthBGRemove(Node):
    def __init__(self):
        super().__init__('depth_bg_remove')

        # ---- Parameters ----
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('near_m', 0.07)
        self.declare_parameter('far_m', 0.50)
        self.declare_parameter('viz_enable', True)
        self.declare_parameter('publish_pointcloud', True)
        self.declare_parameter('cloud_stride', 1)
        self.declare_parameter('min_points_for_normal', 50)   # NEW: guard threshold

        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.near_m = float(self.get_parameter('near_m').value)
        self.far_m  = float(self.get_parameter('far_m').value)
        self.viz_enable = bool(self.get_parameter('viz_enable').value)
        self.publish_pointcloud = bool(self.get_parameter('publish_pointcloud').value)
        self.cloud_stride = int(self.get_parameter('cloud_stride').value)
        self.min_points_for_normal = int(self.get_parameter('min_points_for_normal').value)

        # Pubs
        self.normal_estimate_pub = self.create_publisher(PoseStamped, 'normal_estimate', 10)
        self.normal_valid_pub    = self.create_publisher(Bool, 'normal_estimate_valid', 10)
        self.pub_masked = self.create_publisher(Image, '/camera/camera/depth/foreground', 10)
        self.pub_viz = self.create_publisher(Image, '/camera/camera/depth/foreground_viz', 10) if self.viz_enable else None
        self.pub_cloud = self.create_publisher(PointCloud2, '/camera/camera/depth/foreground_points', 10) if self.publish_pointcloud else None

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

        self.get_logger().info(
            'Background remover running:\n'
            f'  depth_topic={self.depth_topic}\n'
            f'  camera_info_topic={self.camera_info_topic}\n'
            f'  near_m={self.near_m:.3f}, far_m={self.far_m:.3f}\n'
            f'  viz_enable={self.viz_enable}, publish_pointcloud={self.publish_pointcloud}\n'
            f'  cloud_stride={self.cloud_stride}, min_points_for_normal={self.min_points_for_normal}'
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
                if self.viz_enable and self.pub_viz is None:
                    self.pub_viz = self.create_publisher(Image, '/camera/camera/depth/foreground_viz', 10)
                if not self.viz_enable and self.pub_viz is not None:
                    self.pub_viz = None
            elif p.name == 'publish_pointcloud':
                self.publish_pointcloud = bool(p.value)
                if self.publish_pointcloud and self.pub_cloud is None:
                    self.pub_cloud = self.create_publisher(PointCloud2, '/camera/camera/depth/foreground_points', 10)
                if not self.publish_pointcloud and self.pub_cloud is not None:
                    self.pub_cloud = None
            elif p.name == 'cloud_stride':
                self.cloud_stride = max(1, int(p.value))
            elif p.name == 'min_points_for_normal':
                self.min_points_for_normal = max(10, int(p.value))
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

        # Guard: intrinsics available?
        if self.K is None:
            self.normal_valid_pub.publish(Bool(data=False))
            return

        # Count valid pixels
        num_valid = int(np.count_nonzero(mask))
        if num_valid < self.min_points_for_normal:
            # Not enough points for a reliable plane fit; still may publish (empty/sparse) cloud
            self.normal_valid_pub.publish(Bool(data=False))
            if self.pub_cloud is not None:
                self._publish_cloud_if_any(masked_m, mask, msg)
            return

        # Project valid points to 3D (with stride)
        pts = self._project_points(masked_m, mask)
        if pts.shape[0] == 0:
            self.normal_valid_pub.publish(Bool(data=False))
            return

        # Publish cloud (optional)
        if self.pub_cloud is not None:
            cloud = make_pointcloud2(
                pts, frame_id=self.depth_frame_id or msg.header.frame_id, stamp=msg.header.stamp
            )
            self.pub_cloud.publish(cloud)

        # Guard: enough points after stride?
        if pts.shape[0] < self.min_points_for_normal:
            self.normal_valid_pub.publish(Bool(data=False))
            return

        # ---- Dominant plane normal (PCA) ----
        centroid, normal = _pca_plane_normal(pts)

        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = self.depth_frame_id or msg.header.frame_id
        pose.pose.position.x = float(centroid[0])
        pose.pose.position.y = float(centroid[1])
        pose.pose.position.z = float(centroid[2])
        pose.pose.orientation = _quaternion_from_z(normal)

        self.normal_estimate_pub.publish(pose)
        self.normal_valid_pub.publish(Bool(data=True))

    # ---- helpers ----
    def _project_points(self, depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Project masked depth to 3D points with stride."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        ys, xs = np.where(mask)
        if xs.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        stride = max(1, self.cloud_stride)
        ys = ys[::stride]; xs = xs[::stride]
        if xs.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        z = depth_m[ys, xs].astype(np.float32)
        x = (xs.astype(np.float32) - cx) * z / fx
        y = (ys.astype(np.float32) - cy) * z / fy
        return np.stack([x, y, z], axis=1)

    def _publish_cloud_if_any(self, depth_m: np.ndarray, mask: np.ndarray, msg: Image):
        """Try to publish a sparse cloud even if there aren't enough points for normals."""
        pts = self._project_points(depth_m, mask)
        if pts.shape[0] > 0:
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
