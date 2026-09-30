#!/usr/bin/env python3
"""
swin_fusion_tracker_node.py — 跨光譜 Swin Transformer P2P 融合與眼溫推論節點 (ROS 2)
特色：
- 採用 message_filters.ApproximateTimeSynchronizer 同步雙鏡頭影像
- 執行 YOLO 關鍵點偵測與 Swin Transformer 變形矩陣推論
- 發布眼溫數據至 /pig/detections/eye_temp 與標註影像 /pig/detections/annotated_image
"""

import os
import sys
import numpy as np
import cv2
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO

# 匯入專案模型工具函式
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from video_pipeline2 import (
    SwinTransformerP2P,
    process_yolo_detection_and_sort_points,
    get_model_warp_matrix_inference,
    get_warped_box,
    P2P_MODEL_PATH,
    YOLOV8_KEYPOINT_PATH,
    IMAGE_SIZE,
    CONF_THRESHOLD
)

class SwinFusionTrackerNode(Node):
    def __init__(self):
        super().__init__('swin_fusion_tracker_node')

        self.declare_parameter('enable_display', True)
        self.enable_display = self.get_parameter('enable_display').get_parameter_value().bool_value

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'載入推論模型中... (Device: {self.device})')

        self.detect_model = YOLO(YOLOV8_KEYPOINT_PATH)
        self.p2p_model = SwinTransformerP2P().to(self.device)
        if os.path.exists(P2P_MODEL_PATH):
            self.p2p_model.load_state_dict(torch.load(P2P_MODEL_PATH, map_location=self.device, weights_only=True))
            if self.device.type == 'cuda':
                self.p2p_model.half()
            self.p2p_model.eval()
            self.get_logger().info('Swin Transformer P2P 權重載入成功！')

        self.bridge = CvBridge()
        self.current_min_temp = 20.0
        self.current_max_temp = 35.0

        # 訂閱動態溫標
        self.sub_scale = self.create_subscription(
            Float32MultiArray,
            '/pig/thermal/temp_scale',
            self.scale_callback,
            10
        )

        # 雙光譜影像時間戳同步器
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub_rgb = message_filters.Subscriber(self, Image, '/pig/rgb/image_raw', qos_profile=sensor_qos)
        self.sub_thm = message_filters.Subscriber(self, Image, '/pig/thermal/image_raw', qos_profile=sensor_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_thm], queue_size=5, slop=0.08)
        self.sync.registerCallback(self.sync_callback)

        # 發布者
        self.pub_annotated = self.create_publisher(Image, '/pig/detections/annotated_image', 10)
        self.pub_temps = self.create_publisher(Float32MultiArray, '/pig/detections/eye_temp', 10)

        self.get_logger().info('跨光譜雙鏡頭融合節點啟動完成！')

    def scale_callback(self, msg):
        if len(msg.data) >= 2:
            self.current_min_temp = msg.data[0]
            self.current_max_temp = msg.data[1]

    def sync_callback(self, msg_rgb, msg_thm):
        try:
            frame_rgb = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding='bgr8')
            frame_thm = self.bridge.imgmsg_to_cv2(msg_thm, desired_encoding='bgr8')
        except Exception:
            return

        thm_h, thm_w = frame_thm.shape[:2]
        vis_rgb = frame_rgb.copy()
        vis_thm = frame_thm.copy()

        # 1. YOLO 關鍵點預測
        yolo_res = self.detect_model.predict(
            frame_rgb,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False,
            device=self.device,
            half=(self.device.type == 'cuda')
        )[0]
        detect_map = process_yolo_detection_and_sort_points(yolo_res)

        # 2. Swin Transformer P2P 空間變形矩陣 M
        M = get_model_warp_matrix_inference(frame_rgb, frame_thm, detect_map, self.p2p_model, self.device)
        gray_thm = cv2.cvtColor(frame_thm, cv2.COLOR_BGR2GRAY)

        left_temp, right_temp = -1.0, -1.0

        if M is not None:
            for eye_id, eye_name in [(0, 'Left Eye'), (1, 'Right Eye')]:
                if eye_id in detect_map:
                    orig_box = detect_map[eye_id][1]
                    cv2.rectangle(vis_rgb, (orig_box[0], orig_box[1]), (orig_box[2], orig_box[3]), (255, 120, 0), 2)
                    cv2.putText(vis_rgb, eye_name, (orig_box[0], orig_box[1] - 8), 0, 0.6, (255, 120, 0), 2)

                    lx1, ly1, lx2, ly2 = get_warped_box(orig_box, M)
                    lx1, ly1 = max(0, lx1), max(0, ly1)
                    lx2, ly2 = min(thm_w, lx2), min(thm_h, ly2)

                    if (lx2 - lx1) > 2 and (ly2 - ly1) > 2:
                        inner = 2
                        cx1, cy1 = max(0, lx1 + inner), max(0, ly1 + inner)
                        cx2, cy2 = min(thm_w, lx2 - inner), min(thm_h, ly2 - inner)
                        roi = gray_thm[cy1:cy2, cx1:cx2]

                        if roi.size > 0:
                            p90 = np.percentile(roi, 90)
                            pixel_val = roi[roi >= p90].mean()
                            temp = self.current_min_temp + (pixel_val / 255.0) * (self.current_max_temp - self.current_min_temp)

                            if eye_id == 0:
                                left_temp = temp
                            else:
                                right_temp = temp

                            cv2.rectangle(vis_thm, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
                            cv2.putText(vis_thm, f"{temp:.1f} C", (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 發布溫度資料
        temp_msg = Float32MultiArray()
        temp_msg.data = [float(left_temp), float(right_temp)]
        self.pub_temps.publish(temp_msg)

        # 組合標註影像發布
        combined = np.hstack((cv2.resize(vis_rgb, (800, 600)), cv2.resize(vis_thm, (800, 600))))
        out_msg = self.bridge.cv2_to_imgmsg(combined, encoding='bgr8')
        out_msg.header.stamp = msg_rgb.header.stamp
        self.pub_annotated.publish(out_msg)

        if self.enable_display:
            cv2.imshow("ROS 2 Pig Swin Tracker", cv2.resize(combined, (960, 360)))
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SwinFusionTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
