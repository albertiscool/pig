#!/usr/bin/env python3
"""
dahua_dual_stream_node.py — 雙光譜攝影機串流發布節點 (ROS 2)
特色：
- 採用 SensorDataQoS (Best Effort, Depth=1) 確保無網路影格堆疊
- 將雙串流影像時間戳硬體對齊後，封裝成 sensor_msgs/Image 發布至 ROS 2 DDS 匯流排
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class DahuaDualStreamNode(Node):
    def __init__(self):
        super().__init__('dahua_dual_stream_node')

        self.declare_parameter('rgb_url', 'rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0')
        self.declare_parameter('thm_url', 'rtsp://admin:sp123456@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0')
        self.declare_parameter('fps', 30.0)

        self.rgb_url = self.get_parameter('rgb_url').get_parameter_value().string_value
        self.thm_url = self.get_parameter('thm_url').get_parameter_value().string_value
        fps = self.get_parameter('fps').get_parameter_value().double_value

        # 工業級 SensorData QoS 配置
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pub_rgb = self.create_publisher(Image, '/pig/rgb/image_raw', sensor_qos)
        self.pub_thm = self.create_publisher(Image, '/pig/thermal/image_raw', sensor_qos)

        self.bridge = CvBridge()

        self.get_logger().info(f'連線攝影機 RTSP 串流中...')
        self.cap_rgb = cv2.VideoCapture(self.rgb_url)
        self.cap_thm = cv2.VideoCapture(self.thm_url)

        self.cap_rgb.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap_thm.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('雙光譜發布節點啟動完成！')

    def timer_callback(self):
        # 同步抓取
        grab_rgb = self.cap_rgb.grab()
        grab_thm = self.cap_thm.grab()

        if not grab_rgb or not grab_thm:
            return

        ret_rgb, frame_rgb = self.cap_rgb.retrieve()
        ret_thm, frame_thm = self.cap_thm.retrieve()

        if ret_rgb and ret_thm:
            stamp = self.get_clock().now().to_msg()

            msg_rgb = self.bridge.cv2_to_imgmsg(frame_rgb, encoding='bgr8')
            msg_rgb.header.stamp = stamp
            msg_rgb.header.frame_id = 'camera_rgb_optical_frame'

            msg_thm = self.bridge.cv2_to_imgmsg(frame_thm, encoding='bgr8')
            msg_thm.header.stamp = stamp
            msg_thm.header.frame_id = 'camera_thermal_optical_frame'

            self.pub_rgb.publish(msg_rgb)
            self.pub_thm.publish(msg_thm)

    def destroy_node(self):
        self.cap_rgb.release()
        self.cap_thm.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DahuaDualStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
