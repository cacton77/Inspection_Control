import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'teleop_twist_stamped_joy'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'assets'), glob('assets/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your-email@example.com',
    description='A ROS2 package that translates Joy messages to TwistStamped messages at a fixed rate',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_twist_stamped_joy = teleop_twist_stamped_joy.teleop_twist_stamped_joy:main',
            'teleop_joy_to_wrench = teleop_twist_stamped_joy.teleop_joy_to_wrench:main',
            'wrench_to_twist = teleop_twist_stamped_joy.wrench_to_twist:main',
        ],
    },
)