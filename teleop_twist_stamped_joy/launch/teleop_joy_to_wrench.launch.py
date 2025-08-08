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
        default_value='xbox_controller.yaml',
        description='Name of controller configuration file'
    )
    config_file_arg2 = DeclareLaunchArgument(
        'config_file2',
        default_value='mass_inertia_vel.yaml',
        description='Name of controller configuration file'
    )

    config = PathJoinSubstitution([
        FindPackageShare('teleop_twist_stamped_joy'), 
        'config', 
        LaunchConfiguration('config_file')
    ])
    config2 = PathJoinSubstitution([
        FindPackageShare('teleop_twist_stamped_joy'), 
        'config', 
        LaunchConfiguration('config_file2')
    ])    

   # joy_node = Node(
     #   package='joy',
       # executable="joy_node",
      #  name='joy'
    #)

    # Teleop node
    teleop_node = Node(
        package='teleop_twist_stamped_joy',
        executable='teleop_joy_to_wrench',
        name='teleop_joy_to_wrench',
        parameters=[config],
        output='screen',
        emulate_tty=True
    )
    teleop_node2 = Node(
        package='teleop_twist_stamped_joy',
        executable='wrench_to_twist',
        name='wrench_to_twist',
        parameters=[config2],
        output='screen',
        emulate_tty=True
    )
    return LaunchDescription([
        config_file_arg,
         config_file_arg2,
        teleop_node, 
        teleop_node2,
    ])
