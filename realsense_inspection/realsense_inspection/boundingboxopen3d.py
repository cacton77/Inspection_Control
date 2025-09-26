#!/usr/bin/env python3
import math
import numpy as np
import cv2
import copy
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from tf2_ros import Buffer, TransformListener, TransformException


def _quat_to_R_xyzw(x, y, z, w):
    """Return 3x3 rotation matrix from xyzw quaternion."""
    n = math.sqrt(x*x + y*y + z*z + w*w) + 1e-12
    x, y, z, w = x/n, y/n, z/n, w/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)]
    ], dtype=np.float32)


def make_pointcloud2(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """
    points_xyz: (N,3) float32 array in meters.
    """
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = 1
    msg.width = int(points_xyz.shape[0])
    msg.is_bigendian = False
    msg.is_dense = True  # we filter out invalid depths
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

        sub_cb_group = ReentrantCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()

        # ---- Parameters ----
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('bounding_box_topic', '/viewpoint_generation/bounding_box_marker')
        self.declare_parameter('near_m', 0.07)
        self.declare_parameter('far_m', 0.50)
        self.declare_parameter('viz_enable', True)
        self.declare_parameter('publish_pointcloud', True)
        self.declare_parameter('cloud_stride', 1)
        self.declare_parameter('target_frame', 'object_frame')

        # Bounding Box Parameters
        self.declare_parameter('bbox_enable', True)

        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.bounding_box_topic = self.get_parameter('bounding_box_topic').get_parameter_value().string_value
        self.near_m = float(self.get_parameter('near_m').value)
        self.far_m  = float(self.get_parameter('far_m').value)
        self.viz_enable = bool(self.get_parameter('viz_enable').value)
        self.publish_pointcloud = bool(self.get_parameter('publish_pointcloud').value)
        self.cloud_stride = int(self.get_parameter('cloud_stride').value)
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        self.bbox_enable = bool(self.get_parameter('bbox_enable').value)

        # ---- Initialize bbox fields so they're always present ----
        # Use infinities so "no box yet" behaves like "pass-through".
        self.has_bbox = True
        self.bbox_min_x = -float('inf')
        self.bbox_max_x =  float('inf')
        self.bbox_min_y = -float('inf')
        self.bbox_max_y =  float('inf')
        self.bbox_min_z = -float('inf')
        self.bbox_max_z =  float('inf')

        # Live tuning
        self.add_on_set_parameters_callback(self._on_param_update)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.bridge = CvBridge()
        self.K = None              # 3x3 intrinsics
        self.depth_frame_id = None
        self.depth_w = None
        self.depth_h = None

        # TF2 for transforms to target_frame
        self.tf_buffer = Buffer(cache_time=Duration(seconds=0.5))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subs
        self.sub_info  = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_info, qos)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, qos, callback_group=sub_cb_group)
        self.bbox_sub  = self.create_subscription(Marker, self.bounding_box_topic, self.on_bbox, qos)

        self.depth_msg = None
        self.create_timer(0.01, self.process_dmap, callback_group=timer_cb_group)

        # Pubs
        self.pub_cloud_bbox = self.create_publisher(
            PointCloud2, f'/camera/camera/depth/foreground_points_{self.target_frame}_bbox', 10
        ) if self.publish_pointcloud else None

        self.get_logger().info(
            'Background remover (Open3D) running:\n'
            f'  depth_topic={self.depth_topic}\n'
            f'  camera_info_topic={self.camera_info_topic}\n'
            f'  bounding_box_topic={self.bounding_box_topic}\n'
            f'  near_m={self.near_m:.3f}, far_m={self.far_m:.3f}\n'
            f'  viz_enable={self.viz_enable}, publish_pointcloud={self.publish_pointcloud}\n'
            f'  cloud_stride={self.cloud_stride}\n'
            f'  target_frame={self.target_frame}\n'
            f'  bbox_enable={self.bbox_enable}'
        )

    # Camera intrinsics
    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.depth_frame_id = msg.header.frame_id
        self.depth_w = msg.width
        self.depth_h = msg.height

    # Bounding box updates
    def on_bbox(self, msg: Marker):
        if msg.type != Marker.CUBE:
            self.get_logger().warn(f'Ignoring non-CUBE bounding box marker of type {msg.type}')
            return
        if msg.scale.x <= 0.0 or msg.scale.y <= 0.0 or msg.scale.z <= 0.0:
            self.get_logger().warn(f'Ignoring invalid bounding box with non-positive scale {msg.scale.x}, {msg.scale.y}, {msg.scale.z}')
            return
        # Axis-aligned box only (no rotation)
        if abs(msg.pose.orientation.x) > 1e-3 or abs(msg.pose.orientation.y) > 1e-3 or abs(msg.pose.orientation.z) > 1e-3:
            self.get_logger().warn(f'Ignoring non-axis-aligned bounding box with orientation {msg.pose.orientation}')
            return
        self.bbox_enable = True
        cx, cy, cz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        sx, sy, sz = msg.scale.x, msg.scale.y, msg.scale.z
        self.bbox_min_x = cx - 0.5 * sx
        self.bbox_max_x = cx + 0.5 * sx
        self.bbox_min_y = cy - 0.5 * sy
        self.bbox_max_y = cy + 0.5 * sy
        self.bbox_min_z = cz - 0.5 * sz
        self.bbox_max_z = cz + 0.5 * sz
        self.get_logger().info(
            f'Updated bounding box from marker: x[{self.bbox_min_x:.3f},{self.bbox_max_x:.3f}] '
            f'y[{self.bbox_min_y:.3f},{self.bbox_max_y:.3f}] '
            f'z[{self.bbox_min_z:.3f},{self.bbox_max_z:.3f}]'
        )

    # Param updates
    def _on_param_update(self, params):
        for p in params:
            if p.name == 'near_m':
                self.near_m = float(p.value); self.get_logger().info(f'near_m -> {self.near_m:.3f} m')
            elif p.name == 'far_m':
                self.far_m = float(p.value); self.get_logger().info(f'far_m  -> {self.far_m:.3f} m')
            elif p.name == 'publish_pointcloud':
                self.publish_pointcloud = bool(p.value)
                if self.publish_pointcloud and self.pub_cloud_bbox is None:
                    self.pub_cloud_bbox = self.create_publisher(
                        PointCloud2, f'/camera/camera/depth/foreground_points_{self.target_frame}_bbox', 10
                    )
                if not self.publish_pointcloud:
                    self.pub_cloud_bbox = None
            elif p.name == 'cloud_stride':
                self.cloud_stride = max(1, int(p.value))
            elif p.name == 'target_frame':
                self.target_frame = str(p.value)
                if self.publish_pointcloud:
                    self.pub_cloud_bbox = self.create_publisher(
                        PointCloud2, f'/camera/camera/depth/foreground_points_{self.target_frame}_bbox', 10
                    )
            elif p.name == 'bbox_enable':
                self.bbox_enable = bool(p.value)
        return SetParametersResult(successful=True)

    def on_depth(self, msg: Image):
        # Buffer the latest depth image
        self.depth_msg = copy.deepcopy(msg)

    def process_dmap(self):
        if self.depth_msg is None or self.K is None:
            return

        msg = self.depth_msg

        # Convert ROS Image -> numpy (depth meters)
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if msg.encoding in ('16UC1', 'mono16'):
            depth_m = depth.astype(np.float32) * 1e-3
        elif msg.encoding == '32FC1':
            depth_m = depth.astype(np.float32)
        else:
            depth_m = depth.astype(np.float32)

        # Keep only depths within [near_m, far_m]; 0 marks invalid for Open3D
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        inrange = (depth_m >= self.near_m) & (depth_m <= self.far_m)
        masked_m = np.where(valid & inrange, depth_m, 0.0).astype(np.float32)

        # Downsample by stride
        stride = max(1, self.cloud_stride)
        if stride > 1:
            masked_m = masked_m[::stride, ::stride]

        # Prepare intrinsics scaled to the decimated grid
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        fx_s, fy_s = fx / stride, fy / stride
        cx_s, cy_s = cx / stride, cy / stride
        h_s, w_s = masked_m.shape

        intr = o3d.camera.PinholeCameraIntrinsic(
            width=int(w_s), height=int(h_s),
            fx=float(fx_s), fy=float(fy_s),
            cx=float(cx_s), cy=float(cy_s)
        )

        # Open3D depth image (meters)
        o3d_depth = o3d.geometry.Image(masked_m)

        # Create point cloud from depth (no RGB)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth, intr,
            extrinsic=np.eye(4),
            depth_scale=1.0,                 # already meters
            depth_trunc=float(self.far_m),   # safety clamp
            project_valid_depth_only=True
        )

        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.size == 0:
            return

        # TF to target frame
        src_frame = msg.header.frame_id
        zero_stamp = Time(sec=0, nanosec=0)  # latest TF
        try:
            T = self.tf_buffer.lookup_transform(
                self.target_frame, src_frame, zero_stamp, timeout=Duration(seconds=0.001)
            )
            q = T.transform.rotation
            R = _quat_to_R_xyzw(q.x, q.y, q.z, q.w)
            t = T.transform.translation
            t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
            pts_tgt = (R @ pts.T).T + t_vec
        except TransformException as e:
            self.get_logger().warn(f'No TF {self.target_frame} <- {src_frame}: {e}')
            return

        # Axis-aligned bbox filter and publish
        if self.publish_pointcloud and self.bbox_enable and self.has_bbox and pts_tgt.shape[0] > 0:
            X, Y, Z = pts_tgt[:, 0], pts_tgt[:, 1], pts_tgt[:, 2]
            sel = (
                (X >= self.bbox_min_x) & (X <= self.bbox_max_x) &
                (Y >= self.bbox_min_y) & (Y <= self.bbox_max_y) &
                (Z >= self.bbox_min_z) & (Z <= self.bbox_max_z)
            )
            if np.any(sel) and self.pub_cloud_bbox is not None:
                pts_bbox = np.ascontiguousarray(pts_tgt[sel])
                cloud_bbox = make_pointcloud2(
                    pts_bbox, frame_id=self.target_frame, stamp=msg.header.stamp
                )
                self.pub_cloud_bbox.publish(cloud_bbox)


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
