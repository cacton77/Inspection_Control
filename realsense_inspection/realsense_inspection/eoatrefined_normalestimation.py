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
        self.declare_parameter('pcd_downsampling_stride', 1)
        self.declare_parameter('target_frame', 'object_frame')

        # Bounding Box Parameters
        
        self.declare_parameter('main_camera_frame', 'eoat_camera_link')  
        # Cropping parameters (cylindrical crop in eoat_camera_link frame)

        
        self.declare_parameter('crop_radius', 0.05)               # m, radial bound about +Z
        self.declare_parameter('crop_z_min', 0.05)                # m, slab start in +Z
        self.declare_parameter('crop_z_max', 0.40)
        
        
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.bounding_box_topic = self.get_parameter('bounding_box_topic').get_parameter_value().string_value
        self.dmap_filter_min = float(self.get_parameter('dmap_filter_min').value)
        self.dmap_filter_max  = float(self.get_parameter('dmap_filter_max').value)
        self.viz_enable = bool(self.get_parameter('viz_enable').value)
        self.publish_pointcloud = bool(self.get_parameter('publish_pointcloud').value)
        self.pcd_downsampling_stride = int(self.get_parameter('pcd_downsampling_stride').value)
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

       
         # ---- Initialize bbox fields so they're always present ----
        # Use infinities so "no box yet" behaves like "pass-through".
     
     
        self.bbox_min = np.array([-float('inf'), -float('inf'), -float('inf')], dtype=float)  # [xmin, ymin, zmin]
        self.bbox_max = np.array([ float('inf'),  float('inf'),  float('inf')], dtype=float)  # [xmax, ymax, zmax]
        self.main_camera_frame = self.get_parameter('main_camera_frame').get_parameter_value().string_value 
        # Crop reads (NEW)
       
      
        self.crop_radius   = float(self.get_parameter('crop_radius').value)
        self.crop_z_min    = float(self.get_parameter('crop_z_min').value)
        self.crop_z_max    = float(self.get_parameter('crop_z_max').value)
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
        
        self.fov_pointcloud_publisher = self.create_publisher(
            PointCloud2, f'/camera/d405_camera/depth/fov_points_{self.main_camera_frame}_bbox', 10
        ) if self.publish_pointcloud else None
        self.normal_estimate_pub = self.create_publisher(
            PoseStamped, f'/{self.get_name()}/crop_normal', 10
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

    def on_depth(self, msg: Image):
        # Convert ROS Image -> numpy
        self.depth_msg = msg    

    def process_dmap(self):
        if not self.depth_msg:
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
        if self.K is not None:
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
            points = np.stack([x, y, z], axis=1)            # (N,3)
            src_frame = self.depth_msg.header.frame_id
            
            if points.shape[0] > 0:
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
                        Xc, Yc, Zc = pts_bbox_out[:, 0], pts_bbox_out[:, 1], pts_bbox_out[:, 2]
                        r2 = Xc*Xc + Yc*Yc
                        sel_crop = (
                         (Zc >= self.crop_z_min) & (Zc <= self.crop_z_max) &
                         (r2 <= (self.crop_radius * self.crop_radius))
                        ) 
                        
                        pts_crop = np.ascontiguousarray(pts_bbox_out[sel_crop])
                        points = pts_crop
                  

                        # Compute PCA normal 
                        if pts_crop.shape[0] >= 10:
                         centroid, normal = _pca_plane_normal(pts_crop)
                         pose = PoseStamped()
                         pose.header = self.depth_msg.header
                         pose.header.frame_id = self.main_camera_frame
                         pose.pose.position.x = float(centroid[0])
                         pose.pose.position.y = float(centroid[1])
                         pose.pose.position.z = float(centroid[2])
                         pose.pose.orientation = _quaternion_from_z(normal)
                        self.normal_estimate_pub.publish(pose)
                        
                       
                 
                       

                    except TransformException as e:
                        self.get_logger().warn(f'No TF {self.main_camera_frame} <- {self.target_frame} at stamp: {e}')
                    # self._publish_bbox_marker(self.depth_msg.header.stamp)
                        return              

        pcd2_msg = make_pointcloud2(
            points, frame_id=self.main_camera_frame, stamp=self.depth_msg.header.stamp
        )

        self.fov_pointcloud_publisher.publish(pcd2_msg)


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
