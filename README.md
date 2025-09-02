For the inspection control repository the following needs to be run:
1) Joy Node
2) Launch file ( admittance_control_launch.py which launches the teleop and admittance control nodes)

For the realsense_inspection repository the following needs to be run:
1)  realsense2_camera ros2 wrapper launch file ( to start the camera)
2)  depthfilterandpointcloud node to filter out depth map and create a point cloud in realtime
3)  rviz2 for visualization
