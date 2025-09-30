#!/usr/bin/env python3
import math
import numpy as np
import cv2
import copy

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
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
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

        sub_cb_group = ReentrantCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()

        # ---- Parameters ----
        self.declare_parameter('depth_topic', '/camera/d405_camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/d405_camera/depth/camera_info')
        self.declare_parameter('bounding_box_topic', '/viewpoint_generation/bounding_box_marker')
        self.declare_parameter('near_m', 0.07) # TODO: dmap_filter_min
        self.declare_parameter('far_m', 0.50) # TODO: dmap_filter_max
        self.declare_parameter('viz_enable', True)
        self.declare_parameter('publish_pointcloud', True)
        self.declare_parameter('cloud_stride', 1) # TODO: pcd_downsampling_stride
        self.declare_parameter('target_frame', 'object_frame')

        # Bounding Box Parameters
        self.declare_parameter('bbox_enable', True) # TODO: Remove this param
        self.declare_parameter('bbox_output_frame', 'eoat_camera_link')  # TODO: main_camera_frame

        # TODO: Change variable names according to param changes
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
        # TODO: Simplify from 6 variables to 2: self.bbox_min = (x_min, y_min, z_min), self.bbox_max = (x_max, ...)
        self.bbox_min_x = -float('inf')
        self.bbox_max_x =  float('inf')
        self.bbox_min_y = -float('inf')
        self.bbox_max_y =  float('inf')
        self.bbox_min_z = -float('inf')
        self.bbox_max_z =  float('inf')
        self.bbox_output_frame = self.get_parameter('bbox_output_frame').get_parameter_value().string_value 
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

        # TF2 for transforms to target_frame
        self.tf_buffer = Buffer(cache_time=Duration(seconds=0.5))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subs
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_info, qos)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, 1,callback_group=sub_cb_group)
        self.bbox = self.create_subscription(Marker, self.bounding_box_topic, self.on_bbox, qos)
        self.depth_msg = None
        self.create_timer(0.1, self.process_dmap, callback_group=timer_cb_group)

        # Pubs
        # TODO: Look into convention for naming point cloud topics
        self.pub_cloud_bbox_out = self.create_publisher(
            PointCloud2, f'/camera/d405_camera/depth/foreground_points_{self.bbox_output_frame}_bbox', 10
        ) if self.publish_pointcloud else None

        self.get_logger().info(
            'Background remover running:\n'
            f'  depth_topic={self.depth_topic}\n'
            f'  camera_info_topic={self.camera_info_topic}\n'
            f'  bounding_box_topic={self.bounding_box_topic}\n'
            f'  near_m={self.near_m:.3f}, far_m={self.far_m:.3f}\n'
            f'  viz_enable={self.viz_enable}, publish_pointcloud={self.publish_pointcloud}\n'
            f'  cloud_stride={self.cloud_stride}\n'
            f'  target_frame={self.target_frame}'
            f'  bbox_enable={self.bbox_enable}\n'
            f'  bbox_output_frame={self.bbox_output_frame}'
        )

    # Camera intrinsics
    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.depth_frame_id = msg.header.frame_id

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
        # TODO: Update to 2 variables
        self.bbox_min_x = cx - 0.5 * sx
        self.bbox_max_x = cx + 0.5 * sx
        self.bbox_min_y = cy - 0.5 * sy
        self.bbox_max_y = cy + 0.5 * sy
        self.bbox_min_z = cz - 0.5 * sz
        self.bbox_max_z = cz + 0.5 * sz

    def on_depth(self, msg: Image):
        # Convert ROS Image -> numpy
        self.depth_msg = msg    

    def process_dmap(self):
        if not self.depth_msg:
            return
        # TODO: Remove next line and set all msg to self.depth_msg
        msg = self.depth_msg
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

        # Point cloud (if enabled and we have intrinsics)
        if self.K:
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

            h, w = masked_m.shape
            stride = max(1, self.cloud_stride)

            ys, xs = np.where(mask)
            # TODO: We don't want to return before publishing a blank point cloud. 
            # If it is blank, we don't need to run the transforms and cropping though.
            if xs.size == 0:
             #   self._publish_bbox_marker(msg.header.stamp)
                return
            if stride > 1:
                ys = ys[::stride]; xs = xs[::stride]

            z = masked_m[ys, xs]                         # (N,)
            x = (xs.astype(np.float32) - cx) * z / fx    # (N,)
            y = (ys.astype(np.float32) - cy) * z / fy    # (N,)
            points = np.stack([x, y, z], axis=1)            # (N,3)
            # Publish cloud in source (camera) frame
            src_frame = msg.header.frame_id
            
            # TODO: If there are points alter the next line
            if True:
                try:
                    T = self.tf_buffer.lookup_transform(
                        self.target_frame, src_frame, Time(sec=0, nanosec=0), timeout=Duration(seconds=0.001))
                    q = T.transform.rotation
                    R = _quat_to_R_xyzw(q.x, q.y, q.z, q.w)
                    t = T.transform.translation
                    t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
                    pts_tgt = (R @ points.T).T + t_vec  # (N,3)

                except TransformException as e:
                    self.get_logger().warn(f'No TF {self.target_frame} <- {src_frame} at stamp: {e}')
                    return
                
                # --- Axis-aligned bounding-box filter in target_frame ---
                # TODO: Remove self.bbox_enable and use 2 variables for bounds instead of 6
                if self.bbox_enable: 
                    X, Y, Z = pts_tgt[:, 0], pts_tgt[:, 1], pts_tgt[:, 2]
                    sel = (
                        (X >= self.bbox_min_x) & (X <= self.bbox_max_x) &
                        (Y >= self.bbox_min_y) & (Y <= self.bbox_max_y) &
                        (Z >= self.bbox_min_z) & (Z <= self.bbox_max_z)
                    )

                    pts_bbox = np.ascontiguousarray(pts_tgt[sel])


                if self.bbox_output_frame: 
                    try:
                        T_out = self.tf_buffer.lookup_transform(
                            self.bbox_output_frame, self.target_frame, Time(sec=0, nanosec=0), timeout=Duration(seconds=0.001))
                        q_out = T_out.transform.rotation
                        R_out = _quat_to_R_xyzw(q_out.x, q_out.y, q_out.z, q_out.w)
                        t_out = T_out.transform.translation
                        t_vec_out = np.array([t_out.x, t_out.y, t_out.z], dtype=np.float32)
                        pts_bbox_out = (R_out @ pts_bbox.T).T + t_vec_out  # (N,3)
                        points = pts_bbox_out

                    except TransformException as e:
                        self.get_logger().warn(f'No TF {self.bbox_output_frame} <- {self.target_frame} at stamp: {e}')
                    # self._publish_bbox_marker(msg.header.stamp)
                        return              

        # TODO: Publish final point cloud here
        pcd2_msg = make_pointcloud2(
            points, frame_id=self.bbox_output_frame, stamp=msg.header.stamp
        )

        self.pointcloud_publisher.publish(pcd2_msg)


    # Param updates
    def _on_param_update(self, params):
        for p in params:
            if p.name == 'near_m':
                self.near_m = float(p.value); self.get_logger().info(f'near_m -> {self.near_m:.3f} m')
            elif p.name == 'far_m':
                self.far_m = float(p.value); self.get_logger().info(f'far_m  -> {self.far_m:.3f} m')
          #  elif p.name == 'viz_enable':
         #       self.viz_enable = bool(p.value)
          #      if self.viz_enable and self.pub_viz is None:
          #          self.pub_viz = self.create_publisher(Image, '/camera/d405_camera/depth/foreground_viz', 10)
          #      if not self.viz_enable and self.pub_viz is not None:
          #          self.pub_viz = None
            elif p.name == 'publish_pointcloud':
                self.publish_pointcloud = bool(p.value)
                if self.publish_pointcloud:
                    if self.pub_cloud_bbox_out is None:
                        self.pub_cloud_bbox_out = self.create_publisher(
                            PointCloud2, f'/camera/d405_camera/depth/foreground_points_{self.pub_cloud_bbox_out}_bbox', 10
                        )
                if not self.publish_pointcloud:
                    self.pub_cloud_bbox_out = None
            elif p.name == 'cloud_stride':
                self.cloud_stride = max(1, int(p.value))
            elif p.name == 'target_frame':
                self.target_frame = str(p.value)
            elif p.name == 'bbox_enable':
                self.bbox_enable = bool(p.value)
            elif p.name == 'bbox_output_frame':  
                new_frame = str(p.value)
                if new_frame != self.bbox_output_frame:
                    self.bbox_output_frame = new_frame
                    if self.publish_pointcloud:
                        # Recreate out publisher with new name
                        self.pub_cloud_bbox_out = self.create_publisher(
                            PointCloud2, f'/camera/d405_camera/depth/foreground_points_{self.bbox_output_frame}_bbox', 10
                        )
        return SetParametersResult(successful=True)






def main():
    rclpy.init()
    node = DepthBGRemove()
    
    # Use MultiThreadedExecutor with at least 2 threads
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
