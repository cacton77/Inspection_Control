from setuptools import find_packages, setup

package_name = 'inspection_control'

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
    maintainer_email='macs@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'xbox_joy_to_wrench = inspection_control.teleop_node:main',
            'xbox_joy_to_wrench_threshold = inspection_control.teleop_node_threshold:main',
            'wrench_to_twist = inspection_control.admittance_control_node:main'
        ],
    },
)
