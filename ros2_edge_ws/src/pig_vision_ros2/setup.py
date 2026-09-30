from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pig_vision_ros2'

setup(
    name=package_name,
    version='4.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='albertiscool',
    maintainer_email='albertiscool@users.noreply.github.com',
    description='ROS 2 Distributed Dual-Camera Edge Monitoring System for Swine Health',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dahua_dual_stream_node = pig_vision_ros2.dahua_dual_stream_node:main',
            'ocr_calibrator_node = pig_vision_ros2.ocr_calibrator_node:main',
            'swin_fusion_tracker_node = pig_vision_ros2.swin_fusion_tracker_node:main',
        ],
    },
)
