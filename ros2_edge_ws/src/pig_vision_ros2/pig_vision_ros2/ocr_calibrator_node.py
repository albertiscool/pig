#!/usr/bin/env python3
"""
ocr_calibrator_node.py — 熱成像動態溫標解析校準節點 (ROS 2)
特色：
- 獨立節點運作，1 Hz 異步更新
- 訂閱熱成像原始影像，執行 EasyOCR 辨識畫面右上角與右下角溫標，發布至 /pig/thermal/temp_scale
"""

import re
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

class OcrCalibratorNode(Node):
    def __init__(self):
        super().__init__('ocr_calibrator_node')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_thm = self.create_subscription(
            Image,
            '/pig/thermal/image_raw',
            self.image_callback,
            sensor_qos
        )

        self.pub_scale = self.create_publisher(
            Float32MultiArray,
            '/pig/thermal/temp_scale',
            10
        )

        self.bridge = CvBridge()
        self.latest_cv_image = None

        self.get_logger().info('正在載入 EasyOCR 辨識模型...')
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.get_logger().info('EasyOCR 載入完成！')

        # 預設溫度範圍
        self.current_max_temp = 35.0
        self.current_min_temp = 20.0

        # 每 1.0 秒執行一次 OCR 解析
        self.timer = self.create_timer(1.0, self.ocr_callback)

    def image_callback(self, msg):
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            pass

    def extract_temp(self, text):
        match = re.search(r'(\d+\.\d+|\d+)', text)
        return float(match.group(1)) if match else None

    def ocr_callback(self):
        if self.latest_cv_image is None:
            return

        thm_img = self.latest_cv_image
        thm_h, thm_w = thm_img.shape[:2]

        roi_top = thm_img[0:int(thm_h * 0.25), int(thm_w * 0.8):thm_w]
        roi_bot = thm_img[int(thm_h * 0.75):thm_h, int(thm_w * 0.8):thm_w]

        try:
            top_res = self.reader.readtext(roi_top, detail=0)
            bot_res = self.reader.readtext(roi_bot, detail=0)

            updated = False
            for t in top_res:
                val = self.extract_temp(t)
                if val is not None and 10 < val < 100:
                    self.current_max_temp = val
                    updated = True
                    break

            for t in bot_res:
                val = self.extract_temp(t)
                if val is not None and -20 < val < 50:
                    self.current_min_temp = val
                    updated = True
                    break

            msg = Float32MultiArray()
            msg.data = [float(self.current_min_temp), float(self.current_max_temp)]
            self.pub_scale.publish(msg)

            if updated:
                self.get_logger().info(f'溫標校準更新: [{self.current_min_temp:.1f} °C ~ {self.current_max_temp:.1f} °C]')
        except Exception as e:
            self.get_logger().warn(f'OCR 解析異常: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = OcrCalibratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
