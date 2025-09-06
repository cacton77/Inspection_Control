from setuptools import find_packages, setup

package_name = 'realsense_inspection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='macs',
    maintainer_email='antara10@uw.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pcd_saver= realsense_inspection.pointcloudsave:main',
            'depth_bg_remove= realsense_inspection.depthmap_filter:main',
            'depthfilterandpointcloud= realsense_inspection.depthfilterandpointcloud:main',
            'normal_estimator= realsense_inspection.normalestimation:main',
            'pointcloudtransform = realsense_inspection.pointcloudtransform:main',
        ],
    },
)
