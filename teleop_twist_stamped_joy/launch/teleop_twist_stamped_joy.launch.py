
#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for teleop_twist_stamped_joy node."""
    
    # Launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='space_mouse.yaml',
        description='Name of controller configuration file'
    )

    config = PathJoinSubstitution([
        FindPackageShare('teleop_twist_stamped_joy'), 
        'config', 
        LaunchConfiguration('config_file')
    ])
        
    # Teleop node
    teleop_node = Node(
        package='teleop_twist_stamped_joy',
        executable='teleop_twist_stamped_joy',
        name='teleop_twist_stamped_joy',
        parameters=[config],
        output='screen',
        emulate_tty=True
    )
    
    return LaunchDescription([
        config_file_arg,
        teleop_node
    ])
