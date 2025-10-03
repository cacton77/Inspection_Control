#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for teleop_twist_stamped_joy node."""

    # Launch arguments
    goalpose_config_file = DeclareLaunchArgument(
        'goalpose_config_file',
        default_value='goalpose.yaml',
        description='Name of controller configuration file'
    )
  
    goalpose_config = PathJoinSubstitution([
        FindPackageShare('realsense_inspection'),
        'config',
        LaunchConfiguration('goalpose_config_file')
    ])
  

    # goalpose node
    goalpose_node = Node(
        package='realsense_inspection',
        executable='eoatgoalposerefined',
        name='teleoatgoalposerefined',
        parameters=[goalpose_config],
        output='screen',
        emulate_tty=True
    )
   
    return LaunchDescription([
        goalpose_config_file,
        goalpose_node,
        
    ])
