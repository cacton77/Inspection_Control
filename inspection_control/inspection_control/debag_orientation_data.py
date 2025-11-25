import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import csv
import rosbag2_py
from rclpy.serialization import deserialize_message
import pathlib
from viewpoint_generation_interfaces.msg import OrientationControlData
from cv_bridge import CvBridge, CvBridgeError
import matplotlib
import yaml
matplotlib.use('Agg')  # Use non-interactive backend

# Set OpenCV to not use GUI
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def generate_orientation_plots(csv_filepath):
    # Read CSV data
    df = pd.read_csv(csv_filepath)
    csv_path = pathlib.Path(csv_filepath)

    # Convert timestamps to seconds relative to start
    timestamps_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(timestamps_sec, df['error_x'], label='Error X')
    axes[0].plot(timestamps_sec, df['error_y'], label='Error Y')
    axes[0].plot(timestamps_sec, df['error_z'], label='Error Z')
    axes[0].set_title('Rotational Error Over Time')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Error (rad)')
    axes[0].legend()
    # Add torque commands
    axes[1].plot(timestamps_sec, df['torque_x'], label='Torque X')
    axes[1].plot(timestamps_sec, df['torque_y'], label='Torque Y')
    axes[1].plot(timestamps_sec, df['torque_z'], label='Torque Z')
    axes[1].set_title('Torque Commands Over Time')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Torque (Nm)')
    axes[1].legend()
    plt.tight_layout()

    plot_filename = csv_path.parent / f'{csv_path.stem}_analysis.png'
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_filename}")

def depth_to_colormap(depth_image, colormap=cv2.COLORMAP_TURBO, depth_min=None, depth_max=None):
    """
    Convert a single depth image (float) to a colormap image (BGR uint8).
    
    Args:
        depth_image: numpy array with float depth values
        colormap: OpenCV colormap constant
    
    Returns:
        BGR uint8 image suitable for video writing
    """
    # Handle invalid values (NaN, inf)
    depth_clean = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize to 0-255 range
    if not depth_min: depth_min = np.min(depth_clean[depth_clean > 0])  # Ignore zeros/invalid
    if not depth_max: depth_max = np.max(depth_clean)
    
    if depth_max > depth_min:
        normalized = ((depth_clean - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(depth_clean, dtype=np.uint8)
    
    # Where depth_image is zero, make colored image black
    mask_filtered = (depth_image == np.nan)

    # Apply colormap
    colored = cv2.applyColorMap(normalized, colormap)
    colored[mask_filtered] = [0, 0, 0]  # Black for invalid depth
    
    return colored


def debag(bag_file):
    bag_file = pathlib.Path(bag_file)
    output_dir = bag_file.with_suffix('')
    output_dir.mkdir(exist_ok=True)
    os.rename(bag_file, output_dir / bag_file.name)
    bag_file = output_dir / bag_file.name

    reader = rosbag2_py.SequentialReader()
    bridge = CvBridge()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    reader.open(rosbag2_py.StorageOptions(uri=str(bag_file)),
                rosbag2_py.ConverterOptions())

    if not reader.has_next():
        print(f"No messages in bag file {bag_file}")
        return
     # Get first message to determine video properties
    (topic, data, t) = reader.read_next()
    first_msg = deserialize_message(data, OrientationControlData)
    first_depth_image = bridge.imgmsg_to_cv2(
        first_msg.depth_image, desired_encoding='passthrough')
    first_depth_filtered_image = bridge.imgmsg_to_cv2(
        first_msg.depth_filtered_image, desired_encoding='passthrough')

    # Initialize video writer
    depth_out = cv2.VideoWriter(str(output_dir) + f"/{bag_file.stem}_depth.avi",
                          fourcc, 20.0, (first_depth_image.shape[1], first_depth_image.shape[0]))
    depth_out.write(depth_to_colormap(first_depth_image))
    depth_filtered_out = cv2.VideoWriter(str(output_dir) + f"/{bag_file.stem}_depth_filtered.avi",
                          fourcc, 20.0, (first_depth_filtered_image.shape[1], first_depth_filtered_image.shape[0]))
    depth_filtered_out.write(depth_to_colormap(first_depth_filtered_image))
    # Initialize CSV writer
    csv_file = open(bag_file.with_suffix('.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # CSV Header
    csv_writer.writerow([
        'timestamp',
        'main_camera_frame',
        'target_frame',
        'centroid_x',
        'centroid_y',
        'centroid_z',
        'normal_x',
        'normal_y',
        'normal_z',
        'goal_position_x',
        'goal_position_y',
        'goal_position_z',
        'goal_orientation_x',
        'goal_orientation_y',
        'goal_orientation_z',
        'goal_orientation_w',
        'error_x',
        'error_y',
        'error_z',
        'torque_x',
        'torque_y',
        'torque_z',
        'k_rx',
        'k_ry',
        'k_rz',
    ])

    csv_writer.writerow([
        t,
        first_msg.main_camera_frame,
        first_msg.target_frame,
        first_msg.centroid.x,
        first_msg.centroid.y,
        first_msg.centroid.z,
        first_msg.normal.x,
        first_msg.normal.y,
        first_msg.normal.z,
        first_msg.goal_pose.pose.position.x,
        first_msg.goal_pose.pose.position.y,
        first_msg.goal_pose.pose.position.z,
        first_msg.goal_pose.pose.orientation.x,
        first_msg.goal_pose.pose.orientation.y,
        first_msg.goal_pose.pose.orientation.z,
        first_msg.goal_pose.pose.orientation.w,
        first_msg.rotvec_error.x,
        first_msg.rotvec_error.y,
        first_msg.rotvec_error.z,
        first_msg.torque_cmd.x,
        first_msg.torque_cmd.y,
        first_msg.torque_cmd.z,
        first_msg.k_rx,
        first_msg.k_ry,
        first_msg.k_rz,
    ])

    while reader.has_next():
        (topic, data, t) = reader.read_next()

        # Deserialize Message
        msg = deserialize_message(data, OrientationControlData)

        try:
            cv_image = bridge.imgmsg_to_cv2(
                msg.depth_image, desired_encoding='passthrough')
            depth_out.write(depth_to_colormap(cv_image))
        except CvBridgeError as e:
            print(e)

        try:
            cv_filtered_image = bridge.imgmsg_to_cv2(
                msg.depth_filtered_image, desired_encoding='passthrough')
            depth_filtered_out.write(depth_to_colormap(cv_filtered_image, depth_min=msg.dmap_filter_min, depth_max=msg.dmap_filter_max))
        except CvBridgeError as e:
            print(e)

        # Write data to CSV
        csv_writer.writerow([
            t,
            msg.main_camera_frame,
            msg.target_frame,
            msg.centroid.x,
            msg.centroid.y,
            msg.centroid.z,
            msg.normal.x,
            msg.normal.y,
            msg.normal.z,
            msg.goal_pose.pose.position.x,
            msg.goal_pose.pose.position.y,
            msg.goal_pose.pose.position.z,
            msg.goal_pose.pose.orientation.x,
            msg.goal_pose.pose.orientation.y,
            msg.goal_pose.pose.orientation.z,
            msg.goal_pose.pose.orientation.w,
            msg.rotvec_error.x,
            msg.rotvec_error.y,
            msg.rotvec_error.z,
            msg.torque_cmd.x,
            msg.torque_cmd.y,
            msg.torque_cmd.z,
            msg.k_rx,
            msg.k_ry,
            msg.k_rz,
        ])

    # Close CSV file and video
    csv_file.close()
    # out.release()
    csv_filename = bag_file.with_suffix('.csv')

    # Generate the detailed analysis plots
    generate_orientation_plots(csv_filename)

