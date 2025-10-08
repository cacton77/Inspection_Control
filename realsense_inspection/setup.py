import os
from glob import glob
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
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'assets'), glob('assets/*')),
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
            'boundingbox = realsense_inspection.boundingbox:main',
            'eaot_pointcloud = realsense_inspection.pointcloudeaot:main',
            'eoat_normalestimation = realsense_inspection.eoat_normalestimation:main',
            'goal_pose = realsense_inspection.goalpose:main',
            'boundingboxnew = realsense_inspection.boundingboxnew:main',
            'boundingboxnewreducedpub = realsense_inspection.boundingboxnewreducedpub:main',
            'pointcloudeoatreducedpub = realsense_inspection.pointcloudeoatreducedpub:main',
            'boundingboxempty = realsense_inspection.boundingboxempty:main',
            'eoatempty = realsense_inspection.eoatempty:main',
            'eoatemptyrefined= realsense_inspection.eoatemptyrefined:main',
            'eoatrefined_normalestimation = realsense_inspection.eoatrefined_normalestimation:main',
            'eoatgoalposerefined= realsense_inspection.eoatgoalposerefined:main',
            'eoatgoalposerefinedsmoothed= realsense_inspection.eoatgoalposerefined_normalfilter:main'
        ],
    },
)
