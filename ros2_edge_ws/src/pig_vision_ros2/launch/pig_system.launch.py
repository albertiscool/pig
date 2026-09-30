from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pig_vision_ros2',
            executable='dahua_dual_stream_node',
            name='dahua_dual_stream_node',
            output='screen',
            parameters=[{
                'rgb_url': 'rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0',
                'thm_url': 'rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0',
                'fps': 30.0
            }]
        ),
        Node(
            package='pig_vision_ros2',
            executable='ocr_calibrator_node',
            name='ocr_calibrator_node',
            output='screen'
        ),
        Node(
            package='pig_vision_ros2',
            executable='swin_fusion_tracker_node',
            name='swin_fusion_tracker_node',
            output='screen',
            parameters=[{
                'enable_display': True
            }]
        )
    ])
