#!/usr/bin/env python3
import math
import numpy as np
import numpy.linalg as LA
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header, Bool
from cv_bridge import CvBridge
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Quaternion
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

def _pca_plane_normal(pts_np: np.ndarray):
    """Return (centroid, unit normal) for best-fit plane to pts_np (N,3)."""
    c = pts_np.mean(axis=0)
    X = pts_np - c
    # 3x3 covariance; smallest eigenvalue's eigenvector is the plane normal
    C = (X.T @ X) / max(len(X) - 1, 1)
    w, v = LA.eigh(C)
    n = v[:, 0]
    # Make direction consistent (toward camera -Z in depth cam frame)
    if n[2] < 0:
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


        sub_cb_group = ReentrantCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()

        # ---- Parameters ----
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('near_m', 0.07)
        self.declare_parameter('far_m', 0.50)
        self.declare_parameter('viz_enable', True)
        self.declare_parameter('publish_pointcloud', True)
        self.declare_parameter('cloud_stride', 1)
        self.declare_parameter('target_frame', 'object_frame')

        # Bounding Box Parameters
        self.declare_parameter('bbox_enable', True)
        self.declare_parameter('bbox_min_x', -0.10)
        self.declare_parameter('bbox_max_x',  0.10)
        self.declare_parameter('bbox_min_y', -0.10)
        self.declare_parameter('bbox_max_y',  0.10)
        self.declare_parameter('bbox_min_z',  0.00)
        self.declare_parameter('bbox_max_z',  0.40)
        self.declare_parameter('bbox_output_frame', 'eoat_camera_link')  

        # Cropping parameters (cylindrical crop in eoat_camera_link frame)

        self.declare_parameter('crop_enable', True)
        self.declare_parameter('crop_frame', 'eoat_camera_link')  # +Z is "forward"
        self.declare_parameter('crop_radius', 0.05)               # m, radial bound about +Z
        self.declare_parameter('crop_z_min', 0.05)                # m, slab start in +Z
        self.declare_parameter('crop_z_max', 0.40)                # m, slab end in +Z
        # EOAT desired pose parameters
        self.declare_parameter('standoff_m', 0.10)                # used when mode=fixed
        self.declare_parameter('standoff_mode', 'euclidean')          # 'fixed'|'euclidean'|'along_normal'

        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.near_m = float(self.get_parameter('near_m').value)
        self.far_m  = float(self.get_parameter('far_m').value)
        self.viz_enable = bool(self.get_parameter('viz_enable').value)
        self.publish_pointcloud = bool(self.get_parameter('publish_pointcloud').value)
        self.cloud_stride = int(self.get_parameter('cloud_stride').value)
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        self.bbox_enable = bool(self.get_parameter('bbox_enable').value)
        self.bbox_min_x = float(self.get_parameter('bbox_min_x').value)
        self.bbox_max_x = float(self.get_parameter('bbox_max_x').value)
        self.bbox_min_y = float(self.get_parameter('bbox_min_y').value)
        self.bbox_max_y = float(self.get_parameter('bbox_max_y').value)
        self.bbox_min_z = float(self.get_parameter('bbox_min_z').value)
        self.bbox_max_z = float(self.get_parameter('bbox_max_z').value)
        self.bbox_output_frame = self.get_parameter('bbox_output_frame').get_parameter_value().string_value 

        # Crop reads (NEW)
        self.crop_enable   = bool(self.get_parameter('crop_enable').value)
        self.crop_frame    = self.get_parameter('crop_frame').get_parameter_value().string_value
        self.crop_radius   = float(self.get_parameter('crop_radius').value)
        self.crop_z_min    = float(self.get_parameter('crop_z_min').value)
        self.crop_z_max    = float(self.get_parameter('crop_z_max').value)
        
        self.standoff_m    = float(self.get_parameter('standoff_m').value)
        self.standoff_mode = str(self.get_parameter('standoff_mode').value).lower()
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
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, qos,callback_group=sub_cb_group)

        self.depth_msg = None
        self.create_timer(0.01, self.process_dmap, callback_group=timer_cb_group)

        # Pubs
        self.pub_masked = self.create_publisher(Image, '/camera/camera/depth/foreground', 10)
        self.pub_viz = self.create_publisher(Image, '/camera/camera/depth/foreground_viz', 10) if self.viz_enable else None
        self.pub_cloud = self.create_publisher(PointCloud2, '/camera/camera/depth/foreground_points', 10) if self.publish_pointcloud else None
        self.pub_cloud_target = self.create_publisher(
            PointCloud2, f'/camera/camera/depth/foreground_points_{self.target_frame}', 10
        ) if self.publish_pointcloud else None
        self.pub_cloud_bbox = self.create_publisher(
            PointCloud2, f'/camera/camera/depth/foreground_points_{self.target_frame}_bbox', 10
        ) if self.publish_pointcloud else None
        self.pub_cloud_bbox_out = self.create_publisher(
            PointCloud2, f'/camera/camera/depth/foreground_points_{self.bbox_output_frame}_bbox', 10
        ) if self.publish_pointcloud else None

        # NEW: cropped cloud + normal pose + validity
        self.pub_cloud_crop = self.create_publisher(
            PointCloud2, f'/{self.get_name()}/crop_cloud_{self.crop_frame}', 10
        )
        self.normal_estimate_pub = self.create_publisher(
            PoseStamped, f'/{self.get_name()}/crop_normal', 10
        )
        self.normal_valid_pub = self.create_publisher(
            Bool, f'/{self.get_name()}/crop_normal_valid', 10
        )
         # Desired EOAT pose ONLY in crop_frame
        self.pub_eoat_pose_crop = self.create_publisher(
            PoseStamped, f'/{self.get_name()}/eoat_desired_pose_in_{self.crop_frame}', 10
        )

        # Marker for RViz to visualize the bbox
        self.pub_bbox_marker = self.create_publisher(Marker, f'/{self.get_name()}/bbox_marker', 1)


        self.get_logger().info(
            'Background remover running:\n'
            f'  depth_topic={self.depth_topic}\n'
            f'  camera_info_topic={self.camera_info_topic}\n'
            f'  near_m={self.near_m:.3f}, far_m={self.far_m:.3f}\n'
            f'  viz_enable={self.viz_enable}, publish_pointcloud={self.publish_pointcloud}\n'
            f'  cloud_stride={self.cloud_stride}\n'
            f'  target_frame={self.target_frame}'
            f'  bbox_enable={self.bbox_enable}\n'
            f'  bbox: x[{self.bbox_min_x:.3f},{self.bbox_max_x:.3f}] '
            f'y[{self.bbox_min_y:.3f},{self.bbox_max_y:.3f}] '
            f'z[{self.bbox_min_z:.3f},{self.bbox_max_z:.3f}]'
            f'  bbox_output_frame={self.bbox_output_frame}' 
            f'  crop_enable={self.crop_enable}, crop_frame={self.crop_frame}, '
            f'crop_radius={self.crop_radius:.3f}, crop_z=[{self.crop_z_min:.3f},{self.crop_z_max:.3f}]'
            f'  standoff_mode={self.standoff_mode}, standoff_m={self.standoff_m:.3f}'    
        )

    # Camera intrinsics
    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.depth_frame_id = msg.header.frame_id


    def _publish_bbox_marker(self, stamp):
            # Center & size from min/max
         cx = 0.5 * (self.bbox_min_x + self.bbox_max_x)
         cy = 0.5 * (self.bbox_min_y + self.bbox_max_y)
         cz = 0.5 * (self.bbox_min_z + self.bbox_max_z)
         sx = max(1e-6, self.bbox_max_x - self.bbox_min_x)
         sy = max(1e-6, self.bbox_max_y - self.bbox_min_y)
         sz = max(1e-6, self.bbox_max_z - self.bbox_min_z)

         m = Marker()
         m.header.frame_id = self.target_frame
         m.header.stamp = stamp
         m.ns = 'bbox'
         m.id = 0
         m.type = Marker.CUBE
         m.action = Marker.ADD
         m.pose.position.x = cx
         m.pose.position.y = cy
         m.pose.position.z = cz
         m.pose.orientation.w = 1.0
         m.scale.x = sx
         m.scale.y = sy
         m.scale.z = sz
         # semi-transparent cyan
         m.color.r = 0.0
         m.color.g = 1.0
         m.color.b = 1.0
         m.color.a = 0.25
         m.lifetime = Duration(seconds=0.2).to_msg()  # short lifetime; republish each frame
         self.pub_bbox_marker.publish(m)

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
                if self.publish_pointcloud:
                    if self.pub_cloud is None:
                        self.pub_cloud = self.create_publisher(PointCloud2, '/camera/camera/depth/foreground_points', 10)
                    if self.pub_cloud_target is None:
                        self.pub_cloud_target = self.create_publisher(
                            PointCloud2, f'/camera/camera/depth/foreground_points_{self.target_frame}', 10
                        )
                    if self.pub_cloud_bbox is None:
                        self.pub_cloud_bbox = self.create_publisher(
                            PointCloud2, f'/camera/camera/depth/foreground_points_{self.target_frame}_bbox', 10
                        )
                    if self.pub_cloud_bbox_out is None:  
                        self.pub_cloud_bbox_out = self.create_publisher(
                            PointCloud2, f'/camera/camera/depth/foreground_points_{self.bbox_output_frame}_bbox', 10
                        )
                else:
                    self.pub_cloud = None
                    self.pub_cloud_target = None
                    self.pub_cloud_bbox = None
                    self.pub_cloud_bbox_out = None  
            elif p.name == 'cloud_stride':
                self.cloud_stride = max(1, int(p.value))
            elif p.name == 'target_frame':
                self.target_frame = str(p.value)
                if self.publish_pointcloud and self.pub_cloud_target is None:
                    self.pub_cloud_target = self.create_publisher(
                        PointCloud2, f'/camera/camera/depth/foreground_points_{self.target_frame}', 10
                    )
            elif p.name == 'bbox_enable':
                self.bbox_enable = bool(p.value)
            elif p.name == 'bbox_min_x':
                self.bbox_min_x = float(p.value)
            elif p.name == 'bbox_max_x':
                self.bbox_max_x = float(p.value)
            elif p.name == 'bbox_min_y':
                self.bbox_min_y = float(p.value)
            elif p.name == 'bbox_max_y':
                self.bbox_max_y = float(p.value)
            elif p.name == 'bbox_min_z':
                self.bbox_min_z = float(p.value)
            elif p.name == 'bbox_max_z':
                self.bbox_max_z = float(p.value)
            elif p.name == 'bbox_output_frame':  
                new_frame = str(p.value)
                if new_frame != self.bbox_output_frame:
                    self.bbox_output_frame = new_frame
                    if self.publish_pointcloud:
                        # Recreate out publisher with new name
                        self.pub_cloud_bbox_out = self.create_publisher(
                            PointCloud2, f'/camera/camera/depth/foreground_points_{self.bbox_output_frame}_bbox', 10
                        )
            elif p.name == 'crop_enable':
                self.crop_enable = bool(p.value)
            elif p.name == 'crop_frame':
                self.crop_frame = str(p.value)
            elif p.name == 'crop_radius':
                self.crop_radius = float(p.value)
            elif p.name == 'crop_z_min':
                self.crop_z_min = float(p.value)
            elif p.name == 'crop_z_max':
                self.crop_z_max = float(p.value)
            elif p.name == 'standoff_m':
                self.standoff_m = float(p.value)
            elif p.name == 'standoff_mode':
                self.standoff_mode = str(p.value).lower()
        
        return SetParametersResult(successful=True)

    def on_depth(self, msg: Image):
        # Convert ROS Image -> numpy
        self.depth_msg = msg
    

    def process_dmap(self):
        if not self.depth_msg:
            return
        msg = self.depth_msg
        # Convert ROS Image -> numpy
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.get_logger().info('Check one\n')
        # Normalize to meters
        if msg.encoding in ('16UC1', 'mono16'):
            depth_m = depth.astype(np.float32) * 1e-3
        elif msg.encoding == '32FC1':
            depth_m = depth.astype(np.float32)
        else:
            depth_m = depth.astype(np.float32)
        self.get_logger().info('Check two\n')
        # Build mask
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        mask = valid & (depth_m >= self.near_m) & (depth_m <= self.far_m)
        self.get_logger().info('Check three\n')
        # Masked depth
        masked_m = np.where(mask, depth_m, 0.0).astype(np.float32)
        self.get_logger().info('Check four\n')
        # Publish masked depth
        out = self.bridge.cv2_to_imgmsg(masked_m, encoding='32FC1')
        out.header = msg.header
        self.pub_masked.publish(out)
        self.get_logger().info('Check five\n')
        # Viz
        if self.pub_viz is not None:
            denom = max(self.far_m, 1e-3)
            viz8 = np.clip((masked_m / denom) * 255.0, 0, 255).astype(np.uint8)
            viz_color = cv2.applyColorMap(viz8, cv2.COLORMAP_TURBO)
            viz_msg = self.bridge.cv2_to_imgmsg(viz_color, encoding='bgr8')
            viz_msg.header = msg.header
            self.pub_viz.publish(viz_msg)
        self.get_logger().info('Check six\n')
        if self.K is None:
            self.normal_valid_pub.publish(Bool(data=False))
            return    
        self.get_logger().info('Check seven\n')
        # Point cloud (if enabled and we have intrinsics)
        if self.publish_pointcloud and self.K is not None:
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

            h, w = masked_m.shape
            stride = max(1, self.cloud_stride)

            ys, xs = np.where(mask)
            if xs.size == 0:
                self._publish_bbox_marker(msg.header.stamp)
                self.normal_valid_pub.publish(Bool(data=False))
                return
            if stride > 1:
                ys = ys[::stride]; xs = xs[::stride]
                if xs.size == 0:
                 self._publish_bbox_marker(msg.header.stamp)
                 self.normal_valid_pub.publish(Bool(data=False))
                return

            z = masked_m[ys, xs]                         # (N,)
            x = (xs.astype(np.float32) - cx) * z / fx    # (N,)
            y = (ys.astype(np.float32) - cy) * z / fy    # (N,)
            pts = np.stack([x, y, z], axis=1)            # (N,3)
            zero_stamp = Time(sec=0, nanosec=0)
            self.get_logger().info('Check eight\n')
            # Publish cloud in source (camera) frame
            src_frame = self.depth_frame_id or msg.header.frame_id
            cloud_src = make_pointcloud2(pts, frame_id=src_frame, stamp=msg.header.stamp)
            if self.pub_cloud is not None:
                self.pub_cloud.publish(cloud_src)
            self.get_logger().info('Check nine\n')
            # Try to transform to target_frame at the image timestamp
            if self.pub_cloud_target is not None:
                try:
                    T = self.tf_buffer.lookup_transform(
                        self.target_frame, src_frame, zero_stamp, timeout=Duration(seconds=0.001))
                    self.get_logger().info(f'TF {self.target_frame} <- {src_frame} at {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
                    q = T.transform.rotation
                    R = _quat_to_R_xyzw(q.x, q.y, q.z, q.w)
                    t = T.transform.translation
                    t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
                    pts_tgt = (R @ pts.T).T + t_vec  # (N,3)

                    cloud_tgt = make_pointcloud2(
                        pts_tgt, frame_id=self.target_frame, stamp=msg.header.stamp
                    )
                    self.pub_cloud_target.publish(cloud_tgt)
                except TransformException as e:
                    self.get_logger().warn(f'No TF {self.target_frame} <- {src_frame} at stamp: {e}')
                    self._publish_bbox_marker(msg.header.stamp)
                    self.normal_valid_pub.publish(Bool(data=False))
                    return
            self.get_logger().info('Check ten\n')
            # --- Axis-aligned bounding-box filter in target_frame ---
            if self.bbox_enable and pts_tgt.shape[0] > 0:
                X, Y, Z = pts_tgt[:, 0], pts_tgt[:, 1], pts_tgt[:, 2]
                sel = (
                    (X >= self.bbox_min_x) & (X <= self.bbox_max_x) &
                    (Y >= self.bbox_min_y) & (Y <= self.bbox_max_y) &
                    (Z >= self.bbox_min_z) & (Z <= self.bbox_max_z)
                )

                if np.any(sel):
                    pts_bbox = np.ascontiguousarray(pts_tgt[sel])
                    if self.pub_cloud_bbox is not None:
                        cloud_bbox = make_pointcloud2(
                            pts_bbox, frame_id=self.target_frame, stamp=msg.header.stamp
                        )
                        self.pub_cloud_bbox.publish(cloud_bbox)
            self.get_logger().info('Check eleven\n')

            if self.pub_cloud_bbox_out is not None and self.bbox_output_frame: 
                try:
                    T_out = self.tf_buffer.lookup_transform(
                        self.bbox_output_frame, self.target_frame, zero_stamp, timeout=Duration(seconds=0.001))
                    self.get_logger().info(f'TF {self.bbox_output_frame} <- {self.target_frame} at {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
                    q_out = T_out.transform.rotation
                    R_out = _quat_to_R_xyzw(q_out.x, q_out.y, q_out.z, q_out.w)
                    t_out = T_out.transform.translation
                    t_vec_out = np.array([t_out.x, t_out.y, t_out.z], dtype=np.float32)
                    pts_bbox_out = (R_out @ pts_bbox.T).T + t_vec_out  # (N,3)

                    cloud_bbox_out = make_pointcloud2(
                        pts_bbox_out, frame_id=self.bbox_output_frame, stamp=msg.header.stamp
                    )
                    self.pub_cloud_bbox_out.publish(cloud_bbox_out)
                except TransformException as e:
                    self.get_logger().warn(f'No TF {self.bbox_output_frame} <- {self.target_frame} at stamp: {e}')
                    self._publish_bbox_marker(msg.header.stamp)
                    return     
            self.get_logger().info('Check twelve\n')
            if self.crop_enable and pts_tgt.shape[0] > 0:
             try:
                T_cf = self.tf_buffer.lookup_transform(
                    self.crop_frame, self.target_frame, zero_stamp, timeout=Duration(seconds=0.001)
                )
                q_cf = T_cf.transform.rotation
                R_cf = _quat_to_R_xyzw(q_cf.x, q_cf.y, q_cf.z, q_cf.w)
                t_cf = T_cf.transform.translation
                t_vec_cf = np.array([t_cf.x, t_cf.y, t_cf.z], dtype=np.float32)

                pts_cf = (R_cf @ pts_tgt.T).T + t_vec_cf  # (N,3) in crop_frame

                # Cylindrical crop: Z slab + radial bound from +Z axis
                Xc, Yc, Zc = pts_cf[:, 0], pts_cf[:, 1], pts_cf[:, 2]
                r2 = Xc*Xc + Yc*Yc
                sel_crop = (
                    (Zc >= self.crop_z_min) & (Zc <= self.crop_z_max) &
                    (r2 <= (self.crop_radius * self.crop_radius))
                )
                if np.any(sel_crop):
                    pts_crop = np.ascontiguousarray(pts_cf[sel_crop])

                    # Publish cropped cloud (in crop_frame)
                    cloud_crop = make_pointcloud2(pts_crop, frame_id=self.crop_frame, stamp=msg.header.stamp)
                    self.pub_cloud_crop.publish(cloud_crop)

                    # Compute PCA normal (in crop_frame)
                    if pts_crop.shape[0] >= 10:
                        centroid, normal = _pca_plane_normal(pts_crop)
                        pose = PoseStamped()
                        pose.header = msg.header
                        pose.header.frame_id = self.crop_frame
                        pose.pose.position.x = float(centroid[0])
                        pose.pose.position.y = float(centroid[1])
                        pose.pose.position.z = float(centroid[2])
                        pose.pose.orientation = _quaternion_from_z(normal)
                        self.normal_estimate_pub.publish(pose)
                        self.normal_valid_pub.publish(Bool(data=True))
                         # ---- Compute standoff based on mode ----
                        if self.standoff_mode == 'euclidean':
                                d = float(LA.norm(centroid))  # ||c||
                        elif self.standoff_mode == 'along_normal':
                                # signed distance along the normal, clamp to >=0
                                d = float(max(0.0, float(np.dot(normal, centroid))))
                        else:  # 'fixed' or anything else
                                d = self.standoff_m
                        # Desired EOAT pose: back off along normal by d; +Z aligned with normal
                        p_des_cf = centroid - d*normal
                        q_des_cf = _quaternion_from_z(normal)

                        eoat_cf = PoseStamped()
                        eoat_cf.header = msg.header
                        eoat_cf.header.frame_id = self.crop_frame
                        eoat_cf.pose.position.x = float(p_des_cf[0])
                        eoat_cf.pose.position.y = float(p_des_cf[1])
                        eoat_cf.pose.position.z = float(p_des_cf[2])
                        eoat_cf.pose.orientation = q_des_cf
                        self.pub_eoat_pose_crop.publish(eoat_cf)
                    else:
                        self.normal_valid_pub.publish(Bool(data=False))
                else:
                     self.normal_valid_pub.publish(Bool(data=False))
             except TransformException as e:
                self.get_logger().warn(f'No TF {self.crop_frame} <- {self.target_frame}: {e}')
                self.normal_valid_pub.publish(Bool(data=False))
        else:
            # If crop disabled or empty, mark invalid
            self.normal_valid_pub.publish(Bool(data=False))


            # Publish a cube marker so you can see the box in RViz
            self._publish_bbox_marker(msg.header.stamp)

        self.get_logger().info('Check thirteen\n')  





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