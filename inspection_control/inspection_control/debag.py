import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import csv
import rosbag2_py
from rclpy.serialization import deserialize_message
import pathlib
from viewpoint_generation_interfaces.msg import AutofocusData
from cv_bridge import CvBridge, CvBridgeError
import matplotlib
import yaml
matplotlib.use('Agg')  # Use non-interactive backend

# Set OpenCV to not use GUI
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def generate_focus_plots(csv_filepath):
    """
    Generate analysis plots from autofocus CSV data.

    Args:
        csv_filepath: Path to the CSV file containing autofocus data
    """
    # Read CSV data
    df = pd.read_csv(csv_filepath)
    csv_path = pathlib.Path(csv_filepath)

    # Calculate distance traveled from initial end effector pose
    initial_x = df['ee_pose_x'].iloc[0]
    initial_y = df['ee_pose_y'].iloc[0]
    initial_z = df['ee_pose_z'].iloc[0]

    distances = np.sqrt(
        (df['ee_pose_x'] - initial_x)**2 +
        (df['ee_pose_y'] - initial_y)**2 +
        (df['ee_pose_z'] - initial_z)**2
    )

    # Convert timestamps to seconds relative to start
    timestamps_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9

    # Plot 1: Focus value vs distance traveled
    plt.figure(figsize=(10, 6))
    plt.plot(distances, df['focus_value'], linewidth=1.5)
    plt.xlabel('Distance Traveled (m)', fontsize=12)
    plt.ylabel('Focus Value', fontsize=12)
    plt.title(
        f'Focus Value vs Distance Traveled - {csv_path.stem}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot1_filename = csv_path.parent / f'{csv_path.stem}_focus_vs_distance.png'
    plt.savefig(plot1_filename, dpi=150)
    plt.close()
    print(f"Plot saved to {plot1_filename}")

    # Plot 2: Multi-subplot figure
    fig, axes = plt.subplots(6, 1, figsize=(12, 16))

    # Subplot 0: distance traveled vs time
    axes[0].plot(timestamps_sec, distances, linewidth=1.5, color='tab:cyan')
    axes[0].set_xlabel('Time (s)', fontsize=11)
    axes[0].set_ylabel('Distance Traveled (m)', fontsize=11)
    axes[0].set_title('Distance Traveled vs Time', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Subplot 1: focus_value, ema_focus_value, and dema_focus_value vs time
    axes[1].plot(timestamps_sec, df['focus_value'],
                 label='focus_value', linewidth=1.5)
    axes[1].plot(timestamps_sec, df['ema_focus_value'],
                 label='ema_focus_value', linewidth=1.5)
    axes[1].plot(timestamps_sec, df['dema_focus_value'],
                 label='dema_focus_value', linewidth=1.5)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Focus Value', fontsize=11)
    axes[1].set_title('Focus Values vs Time', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Subplot 2: dfv vs time
    axes[2].plot(timestamps_sec, df['dfv'], linewidth=1.5, color='tab:orange')
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].set_ylabel('DFV', fontsize=11)
    axes[2].set_title('DFV vs Time', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    # Subplot 3: smooth_ddfv vs time
    axes[3].plot(timestamps_sec, df['smooth_ddfv'],
                 linewidth=1.5, color='tab:green')
    axes[3].set_xlabel('Time (s)', fontsize=11)
    axes[3].set_ylabel('Smooth DDFV', fontsize=11)
    axes[3].set_title('Smooth DDFV vs Time', fontsize=12)
    axes[3].grid(True, alpha=0.3)

    # Subplot 4: ratio vs time
    axes[4].plot(timestamps_sec, df['ratio'], linewidth=1.5, color='tab:red')
    axes[4].set_xlabel('Time (s)', fontsize=11)
    axes[4].set_ylabel('Ratio', fontsize=11)
    axes[4].set_title('Ratio vs Time', fontsize=12)
    axes[4].grid(True, alpha=0.3)

    # Subplot 5: velocity command vs time
    axes[5].plot(timestamps_sec, df['velocity_command'],
                 linewidth=1.5, color='tab:purple')
    axes[5].set_xlabel('Time (s)', fontsize=11)
    axes[5].set_ylabel('Velocity Command (m/s)', fontsize=11)
    axes[5].set_title('Velocity Command vs Time', fontsize=12)
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plot2_filename = csv_path.parent / f'{csv_path.stem}_analysis.png'
    plt.savefig(plot2_filename, dpi=150)
    plt.close()
    print(f"Plot saved to {plot2_filename}")


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
    first_msg = deserialize_message(data, AutofocusData)
    first_image = bridge.compressed_imgmsg_to_cv2(
        first_msg.image, desired_encoding='bgr8')

    # Initialize video writer
    out = cv2.VideoWriter(str(bag_file.with_suffix('.avi')),
                          fourcc, 20.0, (first_image.shape[1], first_image.shape[0]))
    out.write(first_image)

    # Initialize CSV writer
    csv_file = open(bag_file.with_suffix('.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # Write CSV header
    csv_writer.writerow([
        'timestamp',
        'roi_x',
        'roi_y',
        'roi_width',
        'roi_height',
        'focus_metric',
        'focus_value',
        'ema_focus_value',
        'dema_focus_value',
        'dfv',
        'smooth_ddfv',
        'ratio',
        'ee_pose_x',
        'ee_pose_y',
        'ee_pose_z',
        'ee_orient_x',
        'ee_orient_y',
        'ee_orient_z',
        'ee_orient_w',
        'velocity_command',
        'focus_mode'
        'is_focused'
    ])

    # Store ROI and focus metric info for YAML (use first message values)
    roi_x = first_msg.roi_x
    roi_y = first_msg.roi_y
    roi_width = first_msg.roi_width
    roi_height = first_msg.roi_height
    focus_metric = first_msg.focus_metric

    # Track timestamps for rate calculation
    timestamps = [t]

    # Write first message data
    csv_writer.writerow([
        t,
        first_msg.roi_x,
        first_msg.roi_y,
        first_msg.roi_width,
        first_msg.roi_height,
        first_msg.focus_metric,
        first_msg.focus_value,
        first_msg.ema_focus_value,
        first_msg.dema_focus_value,
        first_msg.dfv,
        first_msg.smooth_ddfv,
        first_msg.ratio,
        first_msg.end_effector_pose.pose.position.x,
        first_msg.end_effector_pose.pose.position.y,
        first_msg.end_effector_pose.pose.position.z,
        first_msg.end_effector_pose.pose.orientation.x,
        first_msg.end_effector_pose.pose.orientation.y,
        first_msg.end_effector_pose.pose.orientation.z,
        first_msg.end_effector_pose.pose.orientation.w,
        first_msg.velocity_command,
        first_msg.focus_mode,
        first_msg.is_focused
    ])

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        timestamps.append(t)

        # Deserialize Message
        msg = deserialize_message(data, AutofocusData)

        try:
            cv_image = bridge.compressed_imgmsg_to_cv2(
                msg.image, desired_encoding='bgr8')
            out.write(cv_image)
        except CvBridgeError as e:
            print(e)

        # Write data to CSV
        csv_writer.writerow([
            t,
            msg.roi_x,
            msg.roi_y,
            msg.roi_width,
            msg.roi_height,
            msg.focus_metric,
            msg.focus_value,
            msg.ema_focus_value,
            msg.dema_focus_value,
            msg.dfv,
            msg.smooth_ddfv,
            msg.ratio,
            msg.end_effector_pose.pose.position.x,
            msg.end_effector_pose.pose.position.y,
            msg.end_effector_pose.pose.position.z,
            msg.end_effector_pose.pose.orientation.x,
            msg.end_effector_pose.pose.orientation.y,
            msg.end_effector_pose.pose.orientation.z,
            msg.end_effector_pose.pose.orientation.w,
            msg.velocity_command,
            msg.focus_mode,
            msg.is_focused
        ])

    # Close CSV file and video
    csv_file.close()
    out.release()

    # Calculate control command rate
    # Convert timestamps from nanoseconds to seconds
    timestamps_array = np.array(timestamps) / 1e9
    time_diffs = np.diff(timestamps_array)
    avg_time_diff = np.mean(time_diffs)
    control_rate_hz = 1.0 / avg_time_diff if avg_time_diff > 0 else 0.0

    # Create YAML data
    yaml_data = {
        'roi_x': float(roi_x),
        'roi_y': float(roi_y),
        'roi_width': int(roi_width),
        'roi_height': int(roi_height),
        'focus_metric': str(focus_metric),
        'control_rate_hz': float(control_rate_hz)
    }

    # Write YAML file
    yaml_filename = bag_file.with_suffix('.yaml')
    with open(yaml_filename, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)
    print(f"YAML file saved to {yaml_filename}")

    # Read CSV and generate all plots
    csv_filename = bag_file.with_suffix('.csv')

    # Generate the detailed analysis plots
    generate_focus_plots(csv_filename)


# Find all *.bag files in and underneath current directory
bag_files = list(pathlib.Path('.').rglob('*.bag'))
to_process = []

for bag_file in bag_files:
    output_dir = bag_file.with_suffix('')
    # Split output_dir around '/'
    parts = output_dir.parts
    # If last two parts are the same, this file has already been processed. Remove it from bag_files
    if len(parts) > 1 and parts[-1] == parts[-2]:
        print(f"Skipping already processed directory: {output_dir}")
        to_process.append(bag_file)
        continue
    else:
        output_dir.mkdir(exist_ok=True)
        # Move bag_file to output_dir
        os.rename(bag_file, output_dir / bag_file.name)
        to_process.append(output_dir / bag_file.name)

print(f"Files to process: {to_process}")

for bag_file in to_process:
    debag(bag_file)
