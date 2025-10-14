#!/usr/bin/env python3
import math
import numpy as np
import cv2
import numpy.linalg as LA
import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import WrenchStamped, Wrench

from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
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
    """Return quaternion with +Z aligned with normal."""
    z = normal / LA.norm(normal)
    up = np.array([0, 1, 0])
    if np.array_equal(z, up):
       x = np.array([1, 0, 0])
      #x = np.array([-1, 0, 0])
    elif np.array_equal(z, -up):
        x = np.array([-1, 0, 0])
     #  x = np.array([1, 0, 0])
    else:
         x = np.cross(up, z)
         x /= LA.norm(x)
    y = np.cross(z, x)
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

def _z_axis_rotvec_error(z_goal: np.ndarray) -> np.ndarray:
    """Return rotation-vector ω that rotates zc=[0,0,1] onto z_goal.
    Both vectors must be expressed in the same frame."""
    zc = np.array([0.0, 0.0, 1.0], dtype=np.float32)      # camera's current Z
    zg = z_goal.astype(np.float32)
    zg /= (LA.norm(zg) + 1e-12)                           # normalize

    c = float(np.clip(np.dot(zc, zg), -1.0, 1.0))         # cos(theta)
    theta = math.acos(c)

    axis = np.cross(zc, zg)
    n = LA.norm(axis)

    if n < 1e-9:
        # zc and zg are parallel
        if c > 0.0:
            return np.zeros(3, dtype=np.float32)  # aligned
        else:
            # 180°: pick x-axis as arbitrary rotation axis
            return np.array([theta, 0.0, 0.0], dtype=np.float32)

    axis /= n
    return (theta * axis).astype(np.float32)

class DepthBGRemove(Node):
    # Node that gives desired EOAT pose based on depth image, bounding box, and cropping
    def __init__(self):
        super().__init__('depth_bg_remove')

        sub_cb_group = ReentrantCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()

        # ---- Parameters ----
        self.declare_parameter('depth_topic', '/camera/d405_camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/d405_camera/depth/camera_info')
        self.declare_parameter('bounding_box_topic', '/viewpoint_generation/bounding_box_marker')
        self.declare_parameter('dmap_filter_min', 0.07) 
        self.declare_parameter('dmap_filter_max', 0.50) 
        self.declare_parameter('viz_enable', True)
        self.declare_parameter('publish_pointcloud', True)
        self.declare_parameter('pcd_downsampling_stride', 4)
        self.declare_parameter('target_frame', 'object_frame')
        # Bounding Box Parameters
        self.declare_parameter('main_camera_frame', 'eoat_camera_link')  
        # Cropping parameters (cylindrical crop in eoat_camera_link frame)
        self.declare_parameter('crop_radius', 0.05)               # m, radial bound about +Z
        self.declare_parameter('crop_z_min', 0.05)                # m, slab start in +Z
        self.declare_parameter('crop_z_max', 0.40)        
        # EOAT desired pose parameters
        self.declare_parameter('standoff_m', 0.10)                # used when mode=fixed
        self.declare_parameter('standoff_mode', 'euclidean')          # 'fixed'|'euclidean'|'along_normal'
        # ---- Smoothing params ----
        self.declare_parameter('ema_enable', True)     # on/off
        self.declare_parameter('ema_tau', 0.25)        # seconds; try 0.2–0.5 s
        # Orientation P gains (Nm/rad) in main_camera_frame
        self.declare_parameter('K_rx', 200.0)   # torque gain about camera X
        self.declare_parameter('K_ry', 200.0)   # torque gain about camera Y
        self.declare_parameter('K_rz', 200.0)   # torque gain about camera Z (used by SO3 error)

        # Optional torque saturation (Nm)
        #self.declare_parameter('torque_limit', 3.0)
        # Safety / validity thresholds
        self.declare_parameter('min_points', 30)          # minimum points to accept target
        self.declare_parameter('stop_timeout_s', 0.20)    # optional watchdog if desired
       
        # Get parameters
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.bounding_box_topic = self.get_parameter('bounding_box_topic').get_parameter_value().string_value
        self.dmap_filter_min = float(self.get_parameter('dmap_filter_min').value)
        self.dmap_filter_max  = float(self.get_parameter('dmap_filter_max').value)
        self.viz_enable = bool(self.get_parameter('viz_enable').value)
        self.publish_pointcloud = bool(self.get_parameter('publish_pointcloud').value)
        self.pcd_downsampling_stride = int(self.get_parameter('pcd_downsampling_stride').value)
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        # Crop reads (NEW)    
        self.crop_radius   = float(self.get_parameter('crop_radius').value)
        self.crop_z_min    = float(self.get_parameter('crop_z_min').value)
        self.crop_z_max    = float(self.get_parameter('crop_z_max').value)
        self.standoff_m    = float(self.get_parameter('standoff_m').value)
        self.standoff_mode = str(self.get_parameter('standoff_mode').value).lower()
        self.ema_enable = bool(self.get_parameter('ema_enable').value)
        self.ema_tau    = float(self.get_parameter('ema_tau').value)

        # EMA state (persist across frames)
        self._ema_normal   = None      # np.ndarray (3,)
        self._ema_centroid = None      # np.ndarray (3,)
        self._ema_last_t   = None      # float seconds
        self.min_points = int(self.get_parameter('min_points').value)
        self.stop_timeout_s = float(self.get_parameter('stop_timeout_s').value)

        # Track when we last had a valid target
        self._last_target_s = None
        # ---- Initialize bbox fields so they're always present ----
        # Use infinities so "no box yet" behaves like "pass-through".
        self.bbox_min = np.array([-float('inf'), -float('inf'), -float('inf')], dtype=float)  # [xmin, ymin, zmin]
        self.bbox_max = np.array([ float('inf'),  float('inf'),  float('inf')], dtype=float)  # [xmax, ymax, zmax]
        self.main_camera_frame = self.get_parameter('main_camera_frame').get_parameter_value().string_value 

        self.K_rx = float(self.get_parameter('K_rx').value)
        self.K_ry = float(self.get_parameter('K_ry').value)
        self.K_rz = float(self.get_parameter('K_rz').value)
        #self.torque_limit = float(self.get_parameter('torque_limit').value)

        # Live tuning
        self.add_on_set_parameters_callback(self._on_param_update)
        # QoS profile 
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
        self.eoat_pointcloud_publisher = self.create_publisher(
            PointCloud2, f'/camera/d405_camera/depth/eoat_points_{self.main_camera_frame}_bbox', 10
        ) if self.publish_pointcloud else None

        self.fov_pointcloud_publisher = self.create_publisher(
            PointCloud2, f'/camera/d405_camera/depth/fov_points_{self.main_camera_frame}_bbox', 10
        ) if self.publish_pointcloud else None
        self.normal_estimate_pub = self.create_publisher(
            PoseStamped, f'/{self.get_name()}/crop_normal', 10
        )
        self.pub_eoat_pose_crop = self.create_publisher(
            PoseStamped, f'/{self.get_name()}/eoat_desired_pose_in_{self.main_camera_frame}', 10
        )
        self.pub_z_rotvec_err = self.create_publisher(
            Vector3Stamped, f'/{self.get_name()}/z_axis_rotvec_error_in_{self.main_camera_frame}', 10
        )
        self.pub_wrench_cmd = self.create_publisher(
            WrenchStamped, f'/{self.get_name()}/wrench_cmd_in_{self.main_camera_frame}', 10
        )

        self.get_logger().info(
            'Background remover running:\n'
            f'  depth_topic={self.depth_topic}\n'
            f'  camera_info_topic={self.camera_info_topic}\n'
            f'  bounding_box_topic={self.bounding_box_topic}\n'
            f'  dmap_filter_min={self.dmap_filter_min:.3f}, dmap_filter_max={self.dmap_filter_max:.3f}\n'
            f'  viz_enable={self.viz_enable}, publish_pointcloud={self.publish_pointcloud}\n'
            f'  pcd_downsampling_stride={self.pcd_downsampling_stride}\n'
            f'  target_frame={self.target_frame}'
           
            f'  main_camera_frame={self.main_camera_frame}'
            f'crop_radius={self.crop_radius:.3f}, crop_z=[{self.crop_z_min:.3f},{self.crop_z_max:.3f}]'
            f'  standoff_mode={self.standoff_mode}, standoff_m={self.standoff_m:.3f}'     
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
        
        cx, cy, cz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        sx, sy, sz = msg.scale.x, msg.scale.y, msg.scale.z
       
        # Half dimensions
        half_sizes = np.array([sx, sy, sz], dtype=float) * 0.5
        center     = np.array([cx, cy, cz], dtype=float)

        # Now update bbox as arrays 
        self.bbox_min = center - half_sizes
        self.bbox_max = center + half_sizes

    def _ema_update(self, prev, x, alpha):
        """One EMA step for vectors (broadcast-safe)."""
        if prev is None:
         return x.copy()
        return (1.0 - alpha) * prev + alpha * x

    def _now_s(self):
        t = self.get_clock().now().to_msg()
        return float(t.sec) + 1e-9 * float(t.nanosec)

    def _reset_ema(self):
        self._ema_normal = None
        self._ema_centroid = None
        self._ema_last_t = None

    def _publish_zero_wrench_and_error(self, stamp=None):
      # Zero ω (error) for visibility
        err = Vector3Stamped()
        if stamp is not None:
          err.header.stamp = stamp
        err.header.frame_id = self.main_camera_frame
        err.vector.x = err.vector.y = err.vector.z = 0.0
        self.pub_z_rotvec_err.publish(err)

      # Zero torque command
        w = WrenchStamped()
        if stamp is not None:
          w.header.stamp = stamp
        w.header.frame_id = self.main_camera_frame
        w.wrench.force.x = w.wrench.force.y = w.wrench.force.z = 0.0
        w.wrench.torque.x = w.wrench.torque.y = w.wrench.torque.z = 0.0
        self.pub_wrench_cmd.publish(w)

    def _safe_stop(self, stamp=None, reset_ema=True):
        if reset_ema:
          self._reset_ema()
       # self._last_target_s = None
        self._publish_zero_wrench_and_error(stamp)

  
 
    def on_depth(self, msg: Image):
        # Convert ROS Image -> numpy
        self.depth_msg = msg    

    def process_dmap(self):
        if not self.depth_msg:
            self._safe_stop(reset_ema=False)
            if self._last_target_s is not None and (self._now_s() - self._last_target_s) > self.ema_hold_s:
                self._reset_ema()
            return
     
       # msg = self.depth_msg
        # Convert ROS Image -> numpy
        depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding='passthrough')

        # Normalize to meters
        if self.depth_msg.encoding in ('16UC1', 'mono16'):
            depth_m = depth.astype(np.float32) * 1e-3
        elif self.depth_msg.encoding == '32FC1':
            depth_m = depth.astype(np.float32)
        else:
            depth_m = depth.astype(np.float32)

        # Build mask
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        mask = valid & (depth_m >= self.dmap_filter_min) & (depth_m <= self.dmap_filter_max)

        # Masked depth
        masked_m = np.where(mask, depth_m, 0.0).astype(np.float32)

        # Point cloud (if enabled and we have intrinsics)
        if self.K is None:
            self._safe_stop(self.depth_msg.header.stamp, reset_ema=False)
            return
        else:
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

            h, w = masked_m.shape
            stride = max(1, self.pcd_downsampling_stride)

            ys, xs = np.where(mask)
           
            if stride > 1:
                ys = ys[::stride]; xs = xs[::stride]

            z = masked_m[ys, xs]                         # (N,)
            x = (xs.astype(np.float32) - cx) * z / fx    # (N,)
            y = (ys.astype(np.float32) - cy) * z / fy    # (N,)
            points1 = np.stack([x, y, z], axis=1) 
            points2 = points1.copy()            # (N,3)
            src_frame = self.depth_msg.header.frame_id
            
            if points1.shape[0] == 0:
                # No points -> stop and try next cycle
                self._safe_stop(self.depth_msg.header.stamp, reset_ema=False)
                if self._last_target_s is not None and (self._now_s() - self._last_target_s) > self.ema_hold_s:
                  self._reset_ema()
                return
            else:
                try:
                    T = self.tf_buffer.lookup_transform(
                        self.target_frame, src_frame, Time(sec=0, nanosec=0), timeout=Duration(seconds=0.001))
                    q = T.transform.rotation
                    R = _quat_to_R_xyzw(q.x, q.y, q.z, q.w)
                    t = T.transform.translation
                    t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
                    pts_tgt = (R @ points1.T).T + t_vec  # (N,3)

                except TransformException as e:
                    self.get_logger().warn(f'No TF {self.target_frame} <- {src_frame} at stamp: {e}')
                    self._safe_stop(self.depth_msg.header.stamp, reset_ema=False)
                    if self._last_target_s is not None and (self._now_s() - self._last_target_s) > self.ema_hold_s:
                     self._reset_ema()
                    return
                
                # --- Axis-aligned bounding-box filter in target_frame ---
            
                # Check all 3 coordinates at once
                sel = np.all((pts_tgt >= self.bbox_min) & (pts_tgt <= self.bbox_max), axis=1)

              # Filter points
                pts_bbox = np.ascontiguousarray(pts_tgt[sel])


                if self.main_camera_frame: 
                    try:
                        T_out = self.tf_buffer.lookup_transform(
                            self.main_camera_frame, self.target_frame, Time(sec=0, nanosec=0), timeout=Duration(seconds=0.001))
                        q_out = T_out.transform.rotation
                        R_out = _quat_to_R_xyzw(q_out.x, q_out.y, q_out.z, q_out.w)
                        t_out = T_out.transform.translation
                        t_vec_out = np.array([t_out.x, t_out.y, t_out.z], dtype=np.float32)
                        pts_bbox_out = (R_out @ pts_bbox.T).T + t_vec_out  # (N,3)
                        
                        
                        points1= pts_bbox_out
                        
                        Xc, Yc, Zc = pts_bbox_out[:, 0], pts_bbox_out[:, 1], pts_bbox_out[:, 2]
                        r2 = Xc*Xc + Yc*Yc
                        sel_crop = (
                         (Zc >= self.crop_z_min) & (Zc <= self.crop_z_max) &
                         (r2 <= (self.crop_radius * self.crop_radius))
                        ) 
                        
                        pts_crop = np.ascontiguousarray(pts_bbox_out[sel_crop])
                        points2 = pts_crop
                  

                        # Compute PCA normal 
                        #if pts_crop.shape[0] >= 10:
                        if pts_crop.shape[0] >= self.min_points:
    ...
                            # At the end of the successful branch, remember we had a valid target:
                            self._last_target_s = now_s
                        else:
                            # Not enough points → stop
                            self._safe_stop(self.depth_msg.header.stamp)

                         centroid, normal = _pca_plane_normal(pts_crop)
                         # Timestamp in seconds for rate-independent alpha
                         stamp = self.depth_msg.header.stamp
                         now_s = float(stamp.sec) + 1e-9 * float(stamp.nanosec)

                         if not self.ema_enable:
                          cen_s = centroid
                          nrm_s = normal
                          self._ema_centroid = centroid
                          self._ema_normal   = normal
                          self._ema_last_t   = now_s
                         else:
                          if self._ema_last_t is None:
                        # First sample initializes the EMA
                            self._ema_centroid = centroid.copy()
                            self._ema_normal   = normal.copy()
                            self._ema_last_t   = now_s

                    # Keep a consistent normal hemisphere to avoid ± flips
                         #if self._ema_normal is not None and np.dot(normal, self._ema_normal) < 0.0:
                           # normal = -normal

                         dt = max(0.0, now_s - (self._ema_last_t if self._ema_last_t is not None else now_s))
                   # α from time-constant τ (handles variable frame rate)
                         alpha = 1.0 - math.exp(-dt / max(1e-3, self.ema_tau))
                         alpha = min(1.0, max(0.0, alpha))

                  # EMA updates
                         self._ema_centroid = self._ema_update(self._ema_centroid, centroid, alpha)
                         self._ema_normal   = self._ema_update(self._ema_normal,   normal,   alpha)

                  # Renormalize the smoothed normal
                         n = LA.norm(self._ema_normal) + 1e-12
                         self._ema_normal /= n

                         cen_s = self._ema_centroid
                         nrm_s = self._ema_normal
                         self._ema_last_t = now_s

                         pose = PoseStamped()
                         pose.header = self.depth_msg.header
                         pose.header.frame_id = self.main_camera_frame
                         #pose.pose.position.x = float(centroid[0])
                         pose.pose.position.x = float(cen_s[0])
                         #pose.pose.position.y = float(centroid[1])
                         pose.pose.position.y = float(cen_s[1])
                         #pose.pose.position.z = float(centroid[2])
                         pose.pose.position.z = float(cen_s[2])

                         #pose.pose.orientation = _quaternion_from_z(normal)
                         pose.pose.orientation = _quaternion_from_z(nrm_s)
                         self.normal_estimate_pub.publish(pose)
                        # ---- Compute standoff based on mode ----
                         if self.standoff_mode == 'euclidean':
                                #d = float(LA.norm(centroid))  # ||c||
                                d = float(LA.norm(cen_s))  # ||c||
                         elif self.standoff_mode == 'along_normal':
                                # signed distance along the normal, clamp to >=0
                                #d = float(max(0.0, float(np.dot(normal, centroid))))
                                d = float(max(0.0, float(np.dot(nrm_s, cen_s))))
                         else:  # 'fixed' or anything else
                                d = self.standoff_m
                        # Desired EOAT pose: back off along normal by d; +Z aligned with normal
                         #p_des_cf = centroid - d*normal
                         p_des_cf = cen_s - d*nrm_s
                         #q_des_cf = _quaternion_from_z(normal)
                         q_des_cf = _quaternion_from_z(nrm_s)

                         eoat_cf = PoseStamped()
                         eoat_cf.header = self.depth_msg.header
                         eoat_cf.header.frame_id = self.main_camera_frame
                         eoat_cf.pose.position.x = float(p_des_cf[0])
                         eoat_cf.pose.position.y = float(p_des_cf[1])
                         eoat_cf.pose.position.z = float(p_des_cf[2])
                         eoat_cf.pose.orientation = q_des_cf

                         R_goal = _quat_to_R_xyzw(q_des_cf.x, q_des_cf.y, q_des_cf.z, q_des_cf.w)
                         xg, yg, zg = R_goal[:,0], R_goal[:,1], R_goal[:,2]   # each is a length-3 unit vector

                         # --- Z-axis alignment error in main_camera_frame ---
                         omega = _z_axis_rotvec_error(zg)   # 3-vector [ωx, ωy, ωz]

                         err_msg = Vector3Stamped()
                         err_msg.header = self.depth_msg.header
                         err_msg.header.frame_id = self.main_camera_frame
                         err_msg.vector.x, err_msg.vector.y, err_msg.vector.z = map(float, omega)
                         self.pub_z_rotvec_err.publish(err_msg)

                         # Proportional torque τ = K_R * ω
                         K_R = np.diag([self.K_rx, self.K_ry, self.K_rz])
                         tau = (K_R @ omega).astype(np.float32)

                         # Saturation
                         #lim = self.torque_limit
                         #tau = np.clip(tau, -lim, lim)

                    # Publish as Wrench (torque only; set forces to 0 or add your own position control)
                         w = WrenchStamped()
                         w.header = self.depth_msg.header
                         w.header.frame_id = self.main_camera_frame
                         w.wrench.force.x = 0.0
                         w.wrench.force.y = 0.0
                         w.wrench.force.z = 0.0
                         w.wrench.torque.x = float(tau[0])
                         w.wrench.torque.y = float(tau[1])
                         w.wrench.torque.z = float(tau[2])  # will be ~0 for Z-only error
                         self.pub_wrench_cmd.publish(w)

                         self._last_target_s = self._now_s()
                         self.pub_eoat_pose_crop.publish(eoat_cf)
                       
                 
                       

                    except TransformException as e:
                        self.get_logger().warn(f'No TF {self.main_camera_frame} <- {self.target_frame} at stamp: {e}')
                        self._safe_stop(self.depth_msg.header.stamp, reset_ema=False)
                        if self._last_target_s is not None and (self._now_s() - self._last_target_s) > self.ema_hold_s:
                         self._reset_ema()
                        # self._publish_bbox_marker(self.depth_msg.header.stamp)
                        return              

       
       
        pcd2_msg = make_pointcloud2(
            points1, frame_id=self.main_camera_frame, stamp=self.depth_msg.header.stamp
        )

        self.eoat_pointcloud_publisher.publish(pcd2_msg)

        pcd2_msg_final = make_pointcloud2(
                points2, frame_id=self.main_camera_frame, stamp=self.depth_msg.header.stamp
            )

        self.fov_pointcloud_publisher.publish(pcd2_msg_final)


    # Param updates
    def _on_param_update(self, params):
        for p in params:
            if p.name == 'dmap_filter_min':
                self.dmap_filter_min = float(p.value); self.get_logger().info(f'dmap_filter_min -> {self.dmap_filter_min:.3f} m')
            elif p.name == 'dmap_filter_max':
                self.dmap_filter_max = float(p.value); self.get_logger().info(f'dmap_filter_max  -> {self.dmap_filter_max:.3f} m')
        
            elif p.name == 'publish_pointcloud':
                self.publish_pointcloud = bool(p.value)
                if self.publish_pointcloud:
                    if self.fov_pointcloud_publisher is None:
                        self.fov_pointcloud_publisher = self.create_publisher(
                            PointCloud2, f'/camera/d405_camera/depth/fov_points_{self.main_camera_frame}_bbox', 10
                        )
                if not self.publish_pointcloud:
                    self.fov_pointcloud_publisher = None
            elif p.name == 'pcd_downsampling_stride':
                self.pcd_downsampling_stride = max(1, int(p.value))
            elif p.name == 'target_frame':
                self.target_frame = str(p.value)
            elif p.name == 'main_camera_frame':  
                new_frame = str(p.value)
                if new_frame != self.main_camera_frame:
                    self.main_camera_frame = new_frame
                    if self.publish_pointcloud:
                        # Recreate out publisher with new name
                        self.fov_pointcloud_publisher = self.create_publisher(
                            PointCloud2, f'/camera/d405_camera/depth/fov_points_{self.main_camera_frame}_bbox', 10)
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
            elif p.name == 'ema_enable':
                self.ema_enable = bool(p.value)
            elif p.name == 'ema_tau':
                self.ema_tau = max(1e-3, float(p.value))
            elif p.name == 'K_rx':
                self.K_rx = float(p.value)
            elif p.name == 'K_ry':
                self.K_ry = float(p.value)
            elif p.name == 'K_rz':
                self.K_rz = float(p.value)
            elif p.name == 'min_points':
                self.min_points = max(1, int(p.value))
            #elif p.name == 'torque_limit':
                #self.torque_limit = float(p.value)

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
