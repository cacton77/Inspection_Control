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
   # msg.is_dense = True  # no NaNs because we filtered them out
    msg.is_dense = points_xyz.size > 0 # when empty, set is_dense False so RViz clears properly
    msg.point_step = 12  # 3 * 4 bytes
    msg.row_step = msg.point_step * msg.width
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
  #  msg.data = points_xyz.astype(np.float32).tobytes()
    msg.data = points_xyz.astype(np.float32).tobytes() if points_xyz.size else b""
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
        self.declare_parameter('near_m', 0.07)
        self.declare_parameter('far_m', 0.50)
        self.declare_parameter('viz_enable', True)
        self.declare_parameter('publish_pointcloud', True)
        self.declare_parameter('cloud_stride', 1)
        self.declare_parameter('target_frame', 'object_frame')

        # Bounding Box Parameters
        self.declare_parameter('bbox_enable', True)
       # self.declare_parameter('bbox_min_x', -0.10)
     #   self.declare_parameter('bbox_max_x',  0.10)
     #   self.declare_parameter('bbox_min_y', -0.10)
     #   self.declare_parameter('bbox_max_y',  0.10)
     #   self.declare_parameter('bbox_min_z',  0.00)
      #  self.declare_parameter('bbox_max_z',  0.40)


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
      #  self.bbox_min_x = float(self.get_parameter('bbox_min_x').value)
     #   self.bbox_max_x = float(self.get_parameter('bbox_max_x').value)
      #  self.bbox_min_y = float(self.get_parameter('bbox_min_y').value)
       # self.bbox_max_y = float(self.get_parameter('bbox_max_y').value)
      #  self.bbox_min_z = float(self.get_parameter('bbox_min_z').value)
      #  self.bbox_max_z = float(self.get_parameter('bbox_max_z').value)

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
       # self.pub_masked = self.create_publisher(Image, '/camera/d405_camera/depth/foreground', 10)
       # self.pub_viz = self.create_publisher(Image, '/camera/d405_camera/depth/foreground_viz', 10) if self.viz_enable else None
      #  self.pub_cloud = self.create_publisher(PointCloud2, '/camera/d405_camera/depth/foreground_points', 10) if self.publish_pointcloud else None
       # self.pub_cloud_target = self.create_publisher(
      #      PointCloud2, f'/camera/d405_camera/depth/foreground_points_{self.target_frame}', 10
      #  ) if self.publish_pointcloud else None
        self.pub_cloud_bbox = self.create_publisher(
            PointCloud2, f'{self.depth_topic}/foreground_points_{self.target_frame}_bbox', 10
        ) if self.publish_pointcloud else None

        # Marker for RViz to visualize the bbox
     #   self.pub_bbox_marker = self.create_publisher(Marker, f'/{self.get_name()}/bbox_marker', 1)


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
       #     f'  bbox: x[{self.bbox_min_x:.3f},{self.bbox_max_x:.3f}] '
     #       f'y[{self.bbox_min_y:.3f},{self.bbox_max_y:.3f}] '
     #       f'z[{self.bbox_min_z:.3f},{self.bbox_max_z:.3f}]'
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


    #def _publish_bbox_marker(self, stamp):
            # Center & size from min/max
      #   cx = 0.5 * (self.bbox_min_x + self.bbox_max_x)
      #   cy = 0.5 * (self.bbox_min_y + self.bbox_max_y)
      #   cz = 0.5 * (self.bbox_min_z + self.bbox_max_z)
      #   sx = max(1e-6, self.bbox_max_x - self.bbox_min_x)
     #    sy = max(1e-6, self.bbox_max_y - self.bbox_min_y)
      #   sz = max(1e-6, self.bbox_max_z - self.bbox_min_z)

       #  m = Marker()
     #    m.header.frame_id = self.target_frame
       #  m.header.stamp = stamp
      #   m.ns = 'bbox'
     #    m.id = 0
     #    m.type = Marker.CUBE
    #     m.action = Marker.ADD
    #     m.pose.position.x = cx
     #    m.pose.position.y = cy
     #    m.pose.position.z = cz
     #    m.pose.orientation.w = 1.0
     #    m.scale.x = sx
     #    m.scale.y = sy
     #    m.scale.z = sz
         # semi-transparent cyan
      #   m.color.r = 0.0
      #   m.color.g = 1.0
      #   m.color.b = 1.0
      #   m.color.a = 0.25
     #    m.lifetime = Duration(seconds=0.2).to_msg()  # short lifetime; republish each frame
     #    self.pub_bbox_marker.publish(m)

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
                  #  if self.pub_cloud is None:
                 #       self.pub_cloud = self.create_publisher(PointCloud2, '/camera/d405_camera/depth/foreground_points', 10)
                 #   if self.pub_cloud_target is None:
                #        self.pub_cloud_target = self.create_publisher(
                 #           PointCloud2, f'/camera/d405_camera/depth/foreground_points_{self.target_frame}', 10
                #        )
                    if self.pub_cloud_bbox is None:
                        self.pub_cloud_bbox = self.create_publisher(
                            PointCloud2, f'/camera/d405_camera/depth/foreground_points_{self.target_frame}_bbox', 10
                        )
                if not self.publish_pointcloud:
                    self.pub_cloud_bbox = None
               # else:
                 #   self.pub_cloud = None
                 #   self.pub_cloud_target = None
                 #   self.pub_cloud_bbox = None
            elif p.name == 'cloud_stride':
                self.cloud_stride = max(1, int(p.value))
            elif p.name == 'target_frame':
                self.target_frame = str(p.value)
                if self.publish_pointcloud:# and self.pub_cloud_target is None:
                 #   self.pub_cloud_target = self.create_publisher(
                 #       PointCloud2, f'/camera/d405_camera/depth/foreground_points_{self.target_frame}', 10
                 #   )
                    self.pub_cloud_bbox = self.create_publisher(
                        PointCloud2, f'/camera/d405_camera/depth/foreground_points_{self.target_frame}_bbox', 10
                    )
            elif p.name == 'bbox_enable':
                self.bbox_enable = bool(p.value)
           # elif p.name == 'bbox_min_x':
           #     self.bbox_min_x = float(p.value)
           # elif p.name == 'bbox_max_x':
          #      self.bbox_max_x = float(p.value)
          #  elif p.name == 'bbox_min_y':
          #      self.bbox_min_y = float(p.value)
         #   elif p.name == 'bbox_max_y':
           #     self.bbox_max_y = float(p.value)
          #  elif p.name == 'bbox_min_z':
          #      self.bbox_min_z = float(p.value)
          #  elif p.name == 'bbox_max_z':
           #     self.bbox_max_z = float(p.value)
        return SetParametersResult(successful=True)

    def on_depth(self, msg: Image):
        # Convert ROS Image -> numpy
        self.depth_msg = copy.deepcopy(msg)   


       # --- helper: publish an empty bbox cloud to clear RViz ---
    def publish_empty_bbox(self, stamp):
        if self.pub_cloud_bbox is None:
            return
        empty = np.zeros((0, 3), dtype=np.float32)
        self.pub_cloud_bbox.publish(make_pointcloud2(empty, self.target_frame, stamp))
    # Process the latest depth map 

    def process_dmap(self):
        if not self.depth_msg:
            return
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

        # Publish masked depth
      #  out = self.bridge.cv2_to_imgmsg(masked_m, encoding='32FC1')
      #  out.header = msg.header
       # self.pub_masked.publish(out)

        # Viz
        #if self.pub_viz is not None:
         #   denom = max(self.far_m, 1e-3)
         #   viz8 = np.clip((masked_m / denom) * 255.0, 0, 255).astype(np.uint8)
         #   viz_color = cv2.applyColorMap(viz8, cv2.COLORMAP_TURBO)
        #    viz_msg = self.bridge.cv2_to_imgmsg(viz_color, encoding='bgr8')
        #    viz_msg.header = msg.header
       #     self.pub_viz.publish(viz_msg)

        # Point cloud (if enabled and we have intrinsics)
        if self.publish_pointcloud and self.K is not None:
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

            h, w = masked_m.shape
            stride = max(1, self.cloud_stride)

            ys, xs = np.where(mask)
            if xs.size == 0:
                if self.bbox_enable:
                  self.publish_empty_bbox(msg.header.stamp)
             #   self._publish_bbox_marker(msg.header.stamp)
                return
            if stride > 1:
                ys = ys[::stride]; xs = xs[::stride]

            z = masked_m[ys, xs]                         # (N,)
            x = (xs.astype(np.float32) - cx) * z / fx    # (N,)
            y = (ys.astype(np.float32) - cy) * z / fy    # (N,)
            pts = np.stack([x, y, z], axis=1)            # (N,3)
            zero_stamp = Time(sec=0, nanosec=0)
            # Publish cloud in source (camera) frame
            src_frame = self.depth_frame_id or msg.header.frame_id
         #   cloud_src = make_pointcloud2(pts, frame_id=src_frame, stamp=msg.header.stamp)
          #  if self.pub_cloud is not None:
         #       self.pub_cloud.publish(cloud_src)

            # Try to transform to target_frame at the image timestamp
          #  if self.pub_cloud_target is not None:
            
            try:
                T = self.tf_buffer.lookup_transform(
                    self.target_frame, src_frame, zero_stamp, timeout=Duration(seconds=0.001))
                self.get_logger().info(f'TF {self.target_frame} <- {src_frame} at {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
                q = T.transform.rotation
                R = _quat_to_R_xyzw(q.x, q.y, q.z, q.w)
                t = T.transform.translation
                t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
                pts_tgt = (R @ pts.T).T + t_vec  # (N,3)

            #    cloud_tgt = make_pointcloud2(
            #        pts_tgt, frame_id=self.target_frame, stamp=msg.header.stamp
            #    )
             #   self.pub_cloud_target.publish(cloud_tgt)
            except TransformException as e:
                self.get_logger().warn(f'No TF {self.target_frame} <- {src_frame} at stamp: {e}')
                   # self._publish_bbox_marker(msg.header.stamp)
                if self.bbox_enable:
                    self.publish_empty_bbox(msg.header.stamp)
                return
            
            # --- Axis-aligned bounding-box filter in target_frame ---
            if self.bbox_enable: 
                if not self.has_bbox or pts_tgt.shape[0] == 0:
                    self.publish_empty_bbox(msg.header.stamp)
                    return  # and self.has_bbox and pts_tgt.shape[0] > 0:
                X, Y, Z = pts_tgt[:, 0], pts_tgt[:, 1], pts_tgt[:, 2]
                sel = (
                    (X >= self.bbox_min_x) & (X <= self.bbox_max_x) &
                    (Y >= self.bbox_min_y) & (Y <= self.bbox_max_y) &
                    (Z >= self.bbox_min_z) & (Z <= self.bbox_max_z)
                )
                if not np.any(sel):
                    self.publish_empty_bbox(msg.header.stamp)
                    return
                pts_bbox = np.ascontiguousarray(pts_tgt[sel])
                if self.pub_cloud_bbox is not None:
                    cloud_bbox = make_pointcloud2(
                        pts_bbox, frame_id=self.target_frame, stamp=msg.header.stamp
                    )
                    self.pub_cloud_bbox.publish(cloud_bbox)

            # Publish a cube marker so you can see the box in RViz
            #self._publish_bbox_marker(msg.header.stamp)

        





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
