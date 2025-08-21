# save_pcd.py (ROS 2 Jazzy)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger
import sensor_msgs_py.point_cloud2 as pc2
import struct

class Saver(Node):
    def __init__(self):
        super().__init__('pcd_saver')
        self.sub = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points',
                                            self.cb, 10)
        self.latest = None
        self.srv = self.create_service(Trigger, 'save_pcd', self.handle_save)

    def cb(self, msg):
        self.latest = msg

    def handle_save(self, req, resp):
        if self.latest is None:
            resp.success = False
            resp.message = 'No point cloud received yet.'
            return resp
        ok, path = self.write_ply(self.latest, 'capture.ply')
        resp.success = ok
        resp.message = f'Saved {path}' if ok else 'Failed to save.'
        return resp

    def write_ply(self, cloud, path):
        # read xyz (and rgb if present)
        points = []
        has_rgb = any(f.name == 'rgb' for f in cloud.fields)
        for p in pc2.read_points(cloud, skip_nans=True, field_names=('x','y','z','rgb') if has_rgb else ('x','y','z')):
            if has_rgb:
                x,y,z,rgb = p
                # rgb packed as float
                i = struct.unpack('I', struct.pack('f', rgb))[0]
                r = (i & 0x00ff0000) >> 16
                g = (i & 0x0000ff00) >> 8
                b = (i & 0x000000ff)
                points.append((x,y,z,r,g,b))
            else:
                x,y,z = p
                points.append((x,y,z))
        with open(path, 'w') as f:
            if has_rgb:
                f.write('ply\nformat ascii 1.0\nelement vertex %d\n' % len(points))
                f.write('property float x\nproperty float y\nproperty float z\n')
                f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
                for x,y,z,r,g,b in points:
                    f.write(f'{x} {y} {z} {r} {g} {b}\n')
            else:
                f.write('ply\nformat ascii 1.0\nelement vertex %d\n' % len(points))
                f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
                for x,y,z in points:
                    f.write(f'{x} {y} {z}\n')
        return True, path

def main():
    rclpy.init()
    n = Saver()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()