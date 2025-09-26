#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Time

from tf2_ros import Buffer, TransformListener, TransformException
from sensor_msgs_py import point_cloud2  # ROS2 helper for PointCloud2

def _quat_to_R_xyzw(x, y, z, w):
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
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = 1
    msg.width = int(points_xyz.shape[0])
    msg.is_bigendian = False
    msg.is_dense = True
    msg.point_step = 12  # 3 * 4 bytes
    msg.row_step = msg.point_step * msg.width
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = points_xyz.astype(np.float32).tobytes()
    return msg

class RSPointCloudBBox(Node):
    def __init__(self):
        super().__init__('rs_pointcloud_bbox')

        # ---------- Parameters ----------
        self.declare_parameter('pointcloud_topic', '/camera/depth/color/points')
        self.declare_parameter('publish_pointcloud', True)
        self.declare_parameter('cloud_stride', 1)
        self.declare_parameter('target_frame', 'object_frame')

        # Bounding box from Marker (axis-aligned, no rotation)
        self.declare_parameter('bbox_enable', True)
        self.declare_parameter('bounding_box_topic', '/viewpoint_generation/bounding_box_marker')

        self.pointcloud_topic = self.get_parameter('pointcloud_topic').get_parameter_value().string_value
        self.publish_pointcloud = bool(self.get_parameter('publish_pointcloud').value)
        self.cloud_stride = int(self.get_parameter('cloud_stride').value)
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.bbox_enable = bool(self.get_parameter('bbox_enable').value)
        self.bounding_box_topic = self.get_parameter('bounding_box_topic').get_parameter_value().string_value

        # Initialize bbox fields (pass-through until a marker arrives)
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

        # TF2
        self.tf_buffer = Buffer(cache_time=Duration(seconds=0.5))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subs
        self.sub_cloud = self.create_subscription(PointCloud2, self.pointcloud_topic, self.on_cloud, qos)
        self.sub_bbox  = self.create_subscription(Marker, self.bounding_box_topic, self.on_bbox, qos)

        # Pub (filtered cloud in target_frame)
        self.pub_cloud_bbox = self.create_publisher(
            PointCloud2, f'/camera/camera/depth/foreground_points_{self.target_frame}_bbox', 10
        ) if self.publish_pointcloud else None

        self.get_logger().info(
            'PointCloud bbox filter running:\n'
            f'  pointcloud_topic={self.pointcloud_topic}\n'
            f'  target_frame={self.target_frame}\n'
            f'  bbox_enable={self.bbox_enable}\n'
            f'  cloud_stride={self.cloud_stride}\n'
            f'  publish_pointcloud={self.publish_pointcloud}'
        )

    # ---- Param updates ----
    def _on_param_update(self, params):
        for p in params:
            if p.name == 'publish_pointcloud':
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
            elif p.name == 'pointcloud_topic':
                # (Optional) allow retargeting subscription at runtime (not restarted here)
                self.pointcloud_topic = str(p.value)
            elif p.name == 'bounding_box_topic':
                self.bounding_box_topic = str(p.value)
        return SetParametersResult(successful=True)

    # ---- BBox from Marker (axis-aligned) ----
    def on_bbox(self, msg: Marker):
        if msg.type != Marker.CUBE:
            self.get_logger().warn(f'Ignoring non-CUBE bbox marker type {msg.type}')
            return
        if msg.scale.x <= 0.0 or msg.scale.y <= 0.0 or msg.scale.z <= 0.0:
            self.get_logger().warn('Ignoring bbox with non-positive scale')
            return
        # Require no rotation (axis-aligned box)
        if abs(msg.pose.orientation.x) > 1e-3 or abs(msg.pose.orientation.y) > 1e-3 or abs(msg.pose.orientation.z) > 1e-3:
            self.get_logger().warn('Ignoring non-axis-aligned bbox marker (orientation must be identity)')
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
            f'Updated bbox: x[{self.bbox_min_x:.3f},{self.bbox_max_x:.3f}] '
            f'y[{self.bbox_min_y:.3f},{self.bbox_max_y:.3f}] '
            f'z[{self.bbox_min_z:.3f},{self.bbox_max_z:.3f}]'
        )

    # ---- PointCloud2 handler ----
    def on_cloud(self, msg: PointCloud2):
        # Convert PointCloud2 -> Nx3 float32, skip NaNs
        # This reads only x,y,z fields (fast & memory-simple).
        pts_list = list(point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        ))
        if not pts_list:
            return
        pts = np.asarray(pts_list, dtype=np.float32)  # (N,3)

        # Optional downsample by stride
        if self.cloud_stride > 1:
            pts = pts[::self.cloud_stride]

        # TF: source frame -> target_frame
        src_frame = msg.header.frame_id
        zero_stamp = Time(sec=0, nanosec=0)  # latest available
        try:
            T = self.tf_buffer.lookup_transform(
                self.target_frame, src_frame, zero_stamp, timeout=Duration(seconds=0.001)
            )
            q = T.transform.rotation
            R = _quat_to_R_xyzw(q.x, q.y, q.z, q.w)
            t = T.transform.translation
            t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
            pts_tgt = (R @ pts.T).T + t_vec  # (N,3)
        except TransformException as e:
            self.get_logger().warn(f'No TF {self.target_frame} <- {src_frame}: {e}')
            return

        # Axis-aligned bbox filter in target_frame
        if self.bbox_enable and self.has_bbox and pts_tgt.shape[0] > 0:
            X, Y, Z = pts_tgt[:, 0], pts_tgt[:, 1], pts_tgt[:, 2]
            sel = (
                (X >= self.bbox_min_x) & (X <= self.bbox_max_x) &
                (Y >= self.bbox_min_y) & (Y <= self.bbox_max_y) &
                (Z >= self.bbox_min_z) & (Z <= self.bbox_max_z)
            )
            if np.any(sel):
                pts_bbox = np.ascontiguousarray(pts_tgt[sel])
                if self.publish_pointcloud and self.pub_cloud_bbox is not None:
                    cloud_bbox = make_pointcloud2(
                        pts_bbox, frame_id=self.target_frame, stamp=msg.header.stamp
                    )
                    self.pub_cloud_bbox.publish(cloud_bbox)

def main():
    rclpy.init()
    node = RSPointCloudBBox()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
