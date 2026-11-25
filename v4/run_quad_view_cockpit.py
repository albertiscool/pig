# -*- coding: utf-8 -*-
"""
run_quad_view_cockpit.py
================================================================================
🐷 智慧豬隻雙光四分割監控 & ROS 2 巡檢小車遙控整合工作台 (PyQt6 Integrated Cockpit)
================================================================================
整合功能：
1. 【左側主畫面：V4 雙光四分割即時 AI 監控看板 (1920x1080 等比縮放)】
   - 左上：可見光 RGB (0817 五官 + YOLO11-Pose 耳根 + 個體 ReID)
   - 左下：熱成像 (淚腺自動吸附 + 雙耳根溫 + 鼻鏡溫)
   - 右上：雙光 4x2 動態視差補償對齊融合
   - 右下：即時生理體溫看板 + 呼吸率帶通濾波示波器 (BPM)
   - 支援快照截圖 [S]、自動感應巡檢開關 [A]、手動記錄生理數據 [T]、重設緩衝 [R]

2. 【右側控制台：ROS 2 小車控制與狀態儀表板】
   - 模式切換服務：呼叫 /control/set_mode (rcl_interfaces/srv/SetParametersAtomically)
     提供 AUTO (自動巡邏)、MANUAL (手動控制)、STOP (停止) 三種模式，檢查服務回覆成功後才更新狀態。
   - 手控移動發布：發布 /manual/cmd_vel (geometry_msgs/msg/Twist)
     支援 7 種動作：前進、後退、原地左轉、原地右轉、前進左轉、前進右轉、停車。
     嚴格限制上限 ±0.1 m/s、±0.3 rad/s，預設 0.05 m/s、0.2 rad/s，禁止左右側移 (linear.y=0)。
   - 安全與斷線防呆：
     * 按住才移動，放開立即傳零速度 (Deadman Switch)
     * 手控期間以 20 Hz (每 50ms) 持續傳送，沒按方向時也持續傳零速度
     * 切換手控成功後立即開始 20 Hz 傳送
     * 視窗失去焦點 (Focus Lost) 時立即清除所有方向鍵並傳送零速度
     * 斷線或車端逾時 (>0.5s) 自動切回 STOP，重新連線後需再次手動切換 MANUAL
   - 車端狀態訂閱：訂閱 /control/status (std_msgs/msg/String JSON)
     即時顯示：目前模式、路線狀態、指令方向、下達指令線速度/角速度 (註明非底盤實測速度)、
     停止或拒絕切換原因、連線狀態與最後更新時間、是否需重啟巡邏 (restart_required)、
     以及獨立的底盤物理停穩確認狀態 (physical_stop_confirmed)。
================================================================================
"""

import os
import sys
import time
import json
import queue
import threading
import argparse
import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QEvent
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QGroupBox, QFrame, QLineEdit,
    QCheckBox, QSizePolicy, QStatusBar
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_quad_view_live import (
    start_live_monitor,
    DEFAULT_RGB_URL,
    DEFAULT_THM_URL,
    DEFAULT_XAVIER_IP,
    DEFAULT_UDP_PORT
)

# ==================== 【手控動作速度定義表 (依據規格書第 三 節)】 ====================
DEFAULT_LINEAR_SPEED = 0.05   # m/s (預設前進/後退速度，上限 ±0.1 m/s)
DEFAULT_ANGULAR_SPEED = 0.20  # rad/s (預設轉向角速度，上限 ±0.3 rad/s)
MAX_LINEAR_SPEED = 0.10
MAX_ANGULAR_SPEED = 0.30

MOTION_TABLE = {
    "FORWARD":       (DEFAULT_LINEAR_SPEED,  0.0,                   "前進 (FORWARD)"),
    "BACKWARD":      (-DEFAULT_LINEAR_SPEED, 0.0,                   "後退 (BACKWARD)"),
    "LEFT":          (0.0,                   DEFAULT_ANGULAR_SPEED, "原地左轉 (TURN_LEFT)"),
    "RIGHT":         (0.0,                  -DEFAULT_ANGULAR_SPEED, "原地右轉 (TURN_RIGHT)"),
    "FORWARD_LEFT":  (DEFAULT_LINEAR_SPEED,  DEFAULT_ANGULAR_SPEED, "前進左轉 (FORWARD_LEFT)"),
    "FORWARD_RIGHT": (DEFAULT_LINEAR_SPEED, -DEFAULT_ANGULAR_SPEED, "前進右轉 (FORWARD_RIGHT)"),
    "STOP":          (0.0,                   0.0,                   "停車 (STOP)"),
}


class Ros2BridgeClient(QObject):
    """
    ROS 2 通訊橋接核心：
    - 支援透過 WebSocket (rosbridge_v2 協定, 預設埠 9090) 與車端 ROS 2 雙向通訊
    - 支援「本機閉環模擬模式 (Local Loopback Simulation)」，在未連車時完整模擬車端 ROS 2 行為
      (包含 /control/set_mode 服務檢查、0.5秒無手控訊息自動跳 STOP 看門狗、/control/status 狀態回報)
    """
    sig_connection_changed = pyqtSignal(bool, str)
    sig_mode_service_reply = pyqtSignal(bool, str, str)  # (successful, target_mode, reason)
    sig_status_received = pyqtSignal(dict, float)        # (status_dict, timestamp)
    sig_tx_heartbeat = pyqtSignal(float, float, int)     # (lin_x, ang_z, hz)

    def __init__(self, host=DEFAULT_XAVIER_IP, port=9090, simulate=False):
        super().__init__()
        self.host = host
        self.port = port
        self.simulate = simulate

        self.ws = None
        self.is_connected = False
        self.running = True
        self.lock = threading.Lock()

        self._call_id_counter = 0
        self._pending_mode_requests = {}  # call_id -> target_mode

        # 模擬器內部狀態 (當 simulate=True 時模擬車端 ROS 2 控制節點)
        self._sim_mode = "STOP"
        self._sim_route_state = "MAIN"
        self._sim_reason = "INITIAL_STOP"
        self._sim_restart_required = False
        self._sim_commanded_motion = "STOP"
        self._sim_linear_x = 0.0
        self._sim_angular_z = 0.0
        self._sim_physical_stop = True
        self._sim_last_manual_cmd_ts = 0.0

        # 發送頻率統計
        self._tx_timestamps = []

        self._worker_thread = threading.Thread(target=self._network_loop, daemon=True)
        self._worker_thread.start()

        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

    def set_endpoint(self, host: str, port: int, simulate: bool):
        with self.lock:
            changed = (self.host != host or self.port != port or self.simulate != simulate)
            self.host = host
            self.port = port
            self.simulate = simulate
            if changed and self.ws is not None:
                try:
                    self.ws.close()
                except Exception:
                    pass

    def _network_loop(self):
        import websocket
        while self.running:
            if self.simulate:
                if not self.is_connected:
                    self.is_connected = True
                    self.sig_connection_changed.emit(True, "模擬車端連線 (Loopback)")
                time.sleep(0.2)
                continue

            url = f"ws://{self.host}:{self.port}"
            try:
                ws = websocket.WebSocket()
                ws.settimeout(2.5)
                ws.connect(url)
                self.ws = ws
                self.is_connected = True
                self.sig_connection_changed.emit(True, f"已連線 ROS 2 ({url})")

                # 1. 宣告發布 /manual/cmd_vel (geometry_msgs/msg/Twist)
                adv_msg = {
                    "op": "advertise",
                    "topic": "/manual/cmd_vel",
                    "type": "geometry_msgs/msg/Twist"
                }
                ws.send(json.dumps(adv_msg))

                # 2. 訂閱 /control/status (std_msgs/msg/String)
                sub_msg = {
                    "op": "subscribe",
                    "topic": "/control/status",
                    "type": "std_msgs/msg/String"
                }
                ws.send(json.dumps(sub_msg))

                ws.settimeout(0.5)
                while self.running and not self.simulate:
                    try:
                        raw = ws.recv()
                        if not raw:
                            break
                        self._handle_incoming_message(raw)
                    except websocket.WebSocketTimeoutException:
                        continue
                    except Exception:
                        break
            except Exception as e:
                pass
            finally:
                if self.ws is not None:
                    try:
                        self.ws.close()
                    except Exception:
                        pass
                    self.ws = None
                if not self.simulate:
                    if self.is_connected:
                        self.is_connected = False
                    self.sig_connection_changed.emit(False, f"等待車端連線 ({self.host}:{self.port})...")
                time.sleep(1.5)

    def _handle_incoming_message(self, raw_str: str):
        try:
            msg = json.loads(raw_str)
        except Exception:
            return

        op = msg.get("op")
        if op == "publish" and msg.get("topic") == "/control/status":
            data_str = msg.get("msg", {}).get("data", "{}")
            try:
                status_obj = json.loads(data_str) if isinstance(data_str, str) else data_str
                self.sig_status_received.emit(status_obj, time.time())
            except Exception:
                pass

        elif op == "service_response" and msg.get("service") == "/control/set_mode":
            call_id = msg.get("id", "")
            target_mode = self._pending_mode_requests.pop(call_id, "UNKNOWN")
            res_flag = bool(msg.get("result", False))
            values = msg.get("values", {}) or {}
            # rcl_interfaces/srv/SetParametersAtomically 的標準回傳結構為 values.result.successful
            inner_res = values.get("result", {})
            if isinstance(inner_res, dict) and "successful" in inner_res:
                ok = bool(res_flag and inner_res.get("successful", False))
                reason = str(inner_res.get("reason", "")) or ("OK" if ok else "SERVICE_REJECTED")
            elif "successful" in values:
                ok = bool(res_flag and values.get("successful", False))
                reason = str(values.get("reason", "")) or ("OK" if ok else "SERVICE_REJECTED")
            else:
                ok = res_flag
                reason = "OK" if ok else str(values or "SERVICE_CALL_FAILED")

            self.sig_mode_service_reply.emit(ok, target_mode, reason)

    def call_set_mode(self, target_mode: str):
        """
        二、切換模式
        呼叫 ROS 2 服務：
        服務：/control/set_mode
        型別：rcl_interfaces/srv/SetParametersAtomically
        """
        target_mode = target_mode.upper().strip()
        if target_mode not in ("AUTO", "MANUAL", "STOP"):
            return

        request_payload = {
            "parameters": [
                {
                    "name": "mode",
                    "value": {
                        "type": 4,
                        "string_value": target_mode
                    }
                }
            ]
        }

        if self.simulate:
            # 模擬車端服務驗證與狀態切換
            self._sim_mode = target_mode
            if target_mode == "MANUAL":
                self._sim_reason = "MANUAL_COMMAND"
                self._sim_restart_required = True
                self._sim_last_manual_cmd_ts = time.time()
            elif target_mode == "AUTO":
                self._sim_reason = "AUTO_PATROL_ACTIVE"
                self._sim_restart_required = False
                self._sim_commanded_motion = "AUTO_ROUTE"
                self._sim_linear_x = 0.05
                self._sim_angular_z = 0.0
                self._sim_physical_stop = False
            elif target_mode == "STOP":
                self._sim_reason = "OPERATOR_STOP"
                self._sim_commanded_motion = "STOP"
                self._sim_linear_x = 0.0
                self._sim_angular_z = 0.0
                self._sim_physical_stop = True

            self.sig_mode_service_reply.emit(True, target_mode, "MODE_SWITCH_SUCCESS")
            return

        with self.lock:
            if not self.is_connected or self.ws is None:
                self.sig_mode_service_reply.emit(False, target_mode, "尚未連線至車端 ROS 2")
                return

            self._call_id_counter += 1
            call_id = f"set_mode_{self._call_id_counter}_{int(time.time()*1000)}"
            self._pending_mode_requests[call_id] = target_mode

            srv_msg = {
                "op": "call_service",
                "id": call_id,
                "service": "/control/set_mode",
                "type": "rcl_interfaces/srv/SetParametersAtomically",
                "args": request_payload
            }
            try:
                self.ws.send(json.dumps(srv_msg))
            except Exception as e:
                self._pending_mode_requests.pop(call_id, None)
                self.sig_mode_service_reply.emit(False, target_mode, f"傳送失敗: {e}")

    def publish_manual_cmd_vel(self, linear_x: float, angular_z: float):
        """
        三、手控移動
        發布訊息：
        Topic：/manual/cmd_vel (嚴禁直接傳 /cmd_vel)
        型別：geometry_msgs/msg/Twist
        手控預設上限為 ±0.1 m/s、±0.3 rad/s，不可側移 (y=0)。
        """
        lin_x = float(np.clip(linear_x, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))
        ang_z = float(np.clip(angular_z, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

        # 避免 -0.0
        if abs(lin_x) < 1e-5:
            lin_x = 0.0
        if abs(ang_z) < 1e-5:
            ang_z = 0.0

        twist_payload = {
            "linear":  {"x": round(lin_x, 4), "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": round(ang_z, 4)}
        }

        now = time.time()
        self._tx_timestamps.append(now)
        self._tx_timestamps = [t for t in self._tx_timestamps if now - t <= 1.0]
        hz = len(self._tx_timestamps)
        self.sig_tx_heartbeat.emit(lin_x, ang_z, hz)

        if self.simulate:
            self._sim_last_manual_cmd_ts = now
            if self._sim_mode == "MANUAL":
                self._sim_linear_x = lin_x
                self._sim_angular_z = ang_z
                if lin_x > 0 and ang_z > 0:
                    self._sim_commanded_motion = "FORWARD_LEFT"
                elif lin_x > 0 and ang_z < 0:
                    self._sim_commanded_motion = "FORWARD_RIGHT"
                elif lin_x > 0:
                    self._sim_commanded_motion = "FORWARD"
                elif lin_x < 0:
                    self._sim_commanded_motion = "BACKWARD"
                elif ang_z > 0:
                    self._sim_commanded_motion = "TURN_LEFT"
                elif ang_z < 0:
                    self._sim_commanded_motion = "TURN_RIGHT"
                else:
                    self._sim_commanded_motion = "STOP"
                # 模擬物理停穩需零速持續一小段時間，體現「零速度指令 != 已確認車子停穩」
                self._sim_physical_stop = (abs(lin_x) == 0.0 and abs(ang_z) == 0.0)
            return

        with self.lock:
            if not self.is_connected or self.ws is None:
                return
            pub_msg = {
                "op": "publish",
                "topic": "/manual/cmd_vel",
                "type": "geometry_msgs/msg/Twist",
                "msg": twist_payload
            }
            try:
                self.ws.send(json.dumps(pub_msg))
            except Exception:
                pass

    def _simulation_loop(self):
        """模擬車端 /control/status 廣播與 0.5 秒超時保護"""
        while self.running:
            if self.simulate:
                now = time.time()
                # 規格第四節：車端超過 0.5 秒未收到手控訊息，會切成 STOP
                if self._sim_mode == "MANUAL" and (now - self._sim_last_manual_cmd_ts > 0.5):
                    self._sim_mode = "STOP"
                    self._sim_reason = "MANUAL_WATCHDOG_TIMEOUT_0.5S"
                    self._sim_commanded_motion = "STOP"
                    self._sim_linear_x = 0.0
                    self._sim_angular_z = 0.0
                    self._sim_physical_stop = True

                status_payload = {
                    "mode": self._sim_mode,
                    "route_state": self._sim_route_state,
                    "reason": self._sim_reason,
                    "restart_required": self._sim_restart_required,
                    "commanded_motion": self._sim_commanded_motion,
                    "linear_x": self._sim_linear_x,
                    "angular_z": self._sim_angular_z,
                    "physical_stop_confirmed": self._sim_physical_stop
                }
                self.sig_status_received.emit(status_payload, now)
            time.sleep(0.1)  # 10 Hz status broadcast

    def stop(self):
        self.running = False
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass


class QuadViewStreamWorker(QObject):
    """背景執行 V4 雙光四分割 AI 推論引擎，透過 Qt Signal 將畫布傳給主介面"""
    sig_frame_ready = pyqtSignal(np.ndarray, dict)

    def __init__(self, src_rgb, src_thm, xavier_ip, udp_port, save_video, fast_mode, ear_mode, auto_record, fallback_demo=True):
        super().__init__()
        self.src_rgb = src_rgb
        self.src_thm = src_thm
        self.xavier_ip = xavier_ip
        self.udp_port = udp_port
        self.save_video = save_video
        self.fast_mode = fast_mode
        self.ear_mode = ear_mode
        self.auto_record = auto_record
        self.fallback_demo = fallback_demo
        self.cmd_queue = queue.Queue()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send_command(self, cmd_str: str):
        self.cmd_queue.put(cmd_str)

    def _on_frame(self, canvas: np.ndarray, meta: dict):
        self.sig_frame_ready.emit(canvas.copy(), meta)

    def _run(self):
        start_live_monitor(
            src_rgb=self.src_rgb,
            src_thm=self.src_thm,
            xavier_ip=self.xavier_ip,
            udp_port=self.udp_port,
            save_video=self.save_video,
            fast_mode=self.fast_mode,
            ear_mode=self.ear_mode,
            headless=True,
            auto_record=self.auto_record,
            frame_callback=self._on_frame,
            command_queue=self.cmd_queue,
            fallback_demo=self.fallback_demo
        )


class PigQuadCockpitWindow(QMainWindow):
    """雙光四分割監控 & ROS 2 小車遙控一體化地面站主視窗"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("🐷 智慧豬隻雙光四分割 AI 監控 & ROS 2 巡檢小車遙控工作台 (Integrated Ground Control Station)")
        self.resize(1720, 960)
        self.setMinimumSize(1360, 780)

        # 控制狀態變數
        self.confirmed_mode = "STOP"       # 經服務確認或車端回報之目前模式: AUTO / MANUAL / STOP
        self.active_button_motion = None   # 當前滑鼠按住的方向鍵名稱
        self.held_keys = set()             # 當前鍵盤按住的方向鍵集合
        self.last_status_time = 0.0
        self.was_disconnected = False

        # 初始化深色科技主題樣式
        self._apply_dark_stylesheet()

        # 建立 UI 佈局
        self._init_ui()

        # 建立 ROS 2 通訊橋接器
        self.ros_client = Ros2BridgeClient(
            host=args.xavier_ip,
            port=args.ros_port,
            simulate=args.sim_ros
        )
        self.ros_client.sig_connection_changed.connect(self._on_ros_connection_changed)
        self.ros_client.sig_mode_service_reply.connect(self._on_mode_service_reply)
        self.ros_client.sig_status_received.connect(self._on_vehicle_status_received)
        self.ros_client.sig_tx_heartbeat.connect(self._on_tx_heartbeat)

        # 建立 20 Hz (50ms) 手控連續發送計時器 (規格第四節：手控期間以 20Hz 持續傳送，沒按方向也傳零速度)
        self.manual_pub_timer = QTimer(self)
        self.manual_pub_timer.setInterval(50)  # 1000ms / 20Hz = 50ms
        self.manual_pub_timer.timeout.connect(self._on_20hz_manual_tick)

        # 建立車端狀態連線逾時檢查計時器 (每 200ms 刷新一次最後更新秒數)
        self.ui_watchdog_timer = QTimer(self)
        self.ui_watchdog_timer.setInterval(200)
        self.ui_watchdog_timer.timeout.connect(self._on_ui_watchdog_tick)
        self.ui_watchdog_timer.start()

        # 監聽應用程式焦點切換事件 (規格第四節：視窗失去焦點時，清除方向按鍵並傳零速度)
        app = QApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._on_app_state_changed)

        # 啟動左側雙光四分割 AI 推論背景執行緒
        self.quad_worker = QuadViewStreamWorker(
            src_rgb=args.rgb,
            src_thm=args.thm,
            xavier_ip=args.xavier_ip,
            udp_port=args.udp_port,
            save_video=args.save,
            fast_mode=not args.full,
            ear_mode=args.ear_mode,
            auto_record=not args.no_auto,
            fallback_demo=True
        )
        self.quad_worker.sig_frame_ready.connect(self._on_quad_frame_ready)
        self.quad_worker.start()

        # 確保主視窗可接收鍵盤事件
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _apply_dark_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0f141c;
                color: #e2e8f0;
                font-family: "Microsoft JhengHei", "Segoe UI", sans-serif;
            }
            QGroupBox {
                background-color: #161f2c;
                border: 1px solid #2a3649;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 12px;
                font-size: 13px;
                font-weight: bold;
                color: #38bdf8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QPushButton {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #475569;
            }
            QPushButton:pressed {
                background-color: #0284c7;
                border-color: #38bdf8;
            }
            QLineEdit {
                background-color: #0b0f17;
                color: #38bdf8;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 8px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
            QCheckBox {
                font-size: 12px;
                color: #cbd5e1;
            }
        """)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ==================== 【左側：雙光四分割畫面區 + 影像快捷列】 ====================
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)

        # 四分割影像畫布 QLabel
        self.video_label = QLabel("🚀 正在載入 YOLO11 + Pose + 熱成像模型並連接影像源...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #05080c;
                border: 2px solid #eab308;
                border-radius: 8px;
                color: #94a3b8;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_panel.addWidget(self.video_label, stretch=1)

        # 底部影像工具列
        vision_bar = QHBoxLayout()
        vision_bar.setSpacing(8)

        self.btn_snap = QPushButton("📸 儲存四分割快照 [S]")
        self.btn_snap.clicked.connect(lambda: self._send_vision_cmd('s'))

        self.btn_auto_rec = QPushButton("🤖 自動感應巡檢錄製: 開啟 [A]")
        self.btn_auto_rec.setStyleSheet("background-color: #065f46; border-color: #10b981;")
        self.btn_auto_rec.clicked.connect(lambda: self._send_vision_cmd('a'))

        self.btn_manual_rec = QPushButton("⏺ 手動啟閉生理記錄 [T]")
        self.btn_manual_rec.clicked.connect(lambda: self._send_vision_cmd('t'))

        self.btn_reset_wave = QPushButton("🔄 重設呼吸波形 [R]")
        self.btn_reset_wave.clicked.connect(lambda: self._send_vision_cmd('r'))

        self.btn_pause_vid = QPushButton("⏯ 暫停/繼續畫面")
        self.btn_pause_vid.clicked.connect(lambda: self._send_vision_cmd('SPACE'))

        for b in (self.btn_snap, self.btn_auto_rec, self.btn_manual_rec, self.btn_reset_wave, self.btn_pause_vid):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            vision_bar.addWidget(b)

        left_panel.addLayout(vision_bar)
        main_layout.addLayout(left_panel, stretch=7)

        # ==================== 【右側：ROS 2 小車控制與車況儀表板】 ====================
        right_widget = QWidget()
        right_widget.setFixedWidth(475)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # 0. 連線設定與通訊狀態卡
        conn_group = QGroupBox("🔗 車端 ROS 2 連線設定 (rosbridge)")
        conn_layout = QVBoxLayout(conn_group)
        conn_layout.setSpacing(6)

        row_ip = QHBoxLayout()
        row_ip.addWidget(QLabel("Xavier IP:"))
        self.edit_ip = QLineEdit(self.args.xavier_ip)
        self.edit_ip.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        row_ip.addWidget(self.edit_ip, stretch=1)
        row_ip.addWidget(QLabel("Port:"))
        self.edit_port = QLineEdit(str(self.args.ros_port))
        self.edit_port.setFixedWidth(60)
        self.edit_port.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        row_ip.addWidget(self.edit_port)

        self.btn_reconnect = QPushButton("套用")
        self.btn_reconnect.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_reconnect.clicked.connect(self._on_apply_endpoint)
        row_ip.addWidget(self.btn_reconnect)
        conn_layout.addLayout(row_ip)

        row_sim = QHBoxLayout()
        self.chk_sim = QCheckBox("啟用本機 ROS 2 閉環模擬器 (無實車時可完整測試介面與 0.5s 保護)")
        self.chk_sim.setChecked(self.args.sim_ros)
        self.chk_sim.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_sim.toggled.connect(self._on_apply_endpoint)
        row_sim.addWidget(self.chk_sim)
        conn_layout.addLayout(row_sim)

        self.lbl_conn_banner = QLabel("● 正在初始化連線...")
        self.lbl_conn_banner.setStyleSheet("color: #facc15; font-weight: bold; font-size: 12px;")
        conn_layout.addWidget(self.lbl_conn_banner)
        right_layout.addWidget(conn_group)

        # 一、二、模式切換區 (/control/set_mode)
        mode_group = QGroupBox("一、二、小車操作模式切換 (/control/set_mode)")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_mode_auto = QPushButton("🤖 自動巡邏\n(AUTO)")
        self.btn_mode_manual = QPushButton("🎮 手動控制\n(MANUAL)")
        self.btn_mode_stop = QPushButton("🛑 停止車子\n(STOP)")

        for b in (self.btn_mode_auto, self.btn_mode_manual, self.btn_mode_stop):
            b.setMinimumHeight(54)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn_row.addWidget(b)

        self.btn_mode_auto.clicked.connect(lambda: self.request_mode_switch("AUTO"))
        self.btn_mode_manual.clicked.connect(lambda: self.request_mode_switch("MANUAL"))
        self.btn_mode_stop.clicked.connect(lambda: self.request_mode_switch("STOP"))
        mode_layout.addLayout(btn_row)

        self.lbl_srv_feedback = QLabel("服務狀態：等待操作指令 (切換成功後才更新模式)")
        self.lbl_srv_feedback.setWordWrap(True)
        self.lbl_srv_feedback.setStyleSheet("color: #94a3b8; font-size: 12px;")
        mode_layout.addWidget(self.lbl_srv_feedback)
        right_layout.addWidget(mode_group)

        # 三、四、手控移動面板 (/manual/cmd_vel, 20Hz, 按住才動/放開即零速)
        pad_group = QGroupBox("三、四、手控移動面板 (/manual/cmd_vel @ 20Hz)")
        pad_layout = QVBoxLayout(pad_group)
        pad_layout.setSpacing(6)

        grid = QGridLayout()
        grid.setSpacing(8)

        self.btn_fwd_left  = QPushButton("↖ 前進左轉\n[Q / W+A]\n(+0.05, +0.2)")
        self.btn_fwd       = QPushButton("▲ 前進\n[W / ↑]\n(+0.05, 0)")
        self.btn_fwd_right = QPushButton("↗ 前進右轉\n[E / W+D]\n(+0.05, -0.2)")
        self.btn_left      = QPushButton("↺ 原地左轉\n[A / ←]\n(0, +0.2)")
        self.btn_stop_now  = QPushButton("■ 停車\n[Space]\n(0, 0)")
        self.btn_right     = QPushButton("↻ 原地右轉\n[D / →]\n(0, -0.2)")
        self.btn_bwd       = QPushButton("▼ 後退\n[S / ↓]\n(-0.05, 0)")

        motion_buttons = [
            (self.btn_fwd_left,  0, 0, "FORWARD_LEFT"),
            (self.btn_fwd,       0, 1, "FORWARD"),
            (self.btn_fwd_right, 0, 2, "FORWARD_RIGHT"),
            (self.btn_left,      1, 0, "LEFT"),
            (self.btn_stop_now,  1, 1, "STOP"),
            (self.btn_right,     1, 2, "RIGHT"),
            (self.btn_bwd,       2, 1, "BACKWARD"),
        ]

        for btn, r, c, m_key in motion_buttons:
            btn.setMinimumHeight(64)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            if m_key == "STOP":
                btn.setStyleSheet("""
                    QPushButton { background-color: #7f1d1d; border: 1px solid #ef4444; color: #fef2f2; }
                    QPushButton:pressed { background-color: #dc2626; }
                """)
            grid.addWidget(btn, r, c)
            # 規格第四節：按住才移動，放開立即傳零速度
            btn.pressed.connect(lambda mk=m_key: self._on_motion_button_pressed(mk))
            btn.released.connect(self._on_motion_button_released)

        pad_layout.addLayout(grid)

        self.lbl_tx_info = QLabel("20Hz 發送器: 待命 (切換至 MANUAL 後自動以 20Hz 連續傳送)")
        self.lbl_tx_info.setStyleSheet("color: #38bdf8; font-family: Consolas, monospace; font-size: 12px;")
        pad_layout.addWidget(self.lbl_tx_info)

        lbl_safety_note = QLabel("💡 安全防呆：放開按鍵或視窗失去焦點立即送 0 速；禁左右側移 (y=0)")
        lbl_safety_note.setStyleSheet("color: #94a3b8; font-size: 11px;")
        pad_layout.addWidget(lbl_safety_note)
        right_layout.addWidget(pad_group)

        # 五、顯示車端狀態 (/control/status)
        status_group = QGroupBox("五、車端即時狀態回報 (/control/status)")
        status_layout = QGridLayout(status_group)
        status_layout.setVerticalSpacing(7)
        status_layout.setHorizontalSpacing(10)

        def _make_val_label(text="--"):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #f8fafc; font-weight: bold; font-family: Consolas, 'Microsoft JhengHei'; font-size: 13px;")
            return lbl

        self.val_mode = _make_val_label("STOP")
        self.val_route = _make_val_label("--")
        self.val_motion = _make_val_label("--")
        self.val_cmd_speeds = _make_val_label("linear.x: 0.00 m/s | angular.z: 0.00 rad/s")
        self.val_phys_stop = _make_val_label("尚未確認停穩 (False)")
        self.val_reason = _make_val_label("--")
        self.val_restart = _make_val_label("--")
        self.val_last_update = _make_val_label("尚未收到資料")

        rows = [
            ("目前模式 (mode):", self.val_mode),
            ("路線狀態 (route_state):", self.val_route),
            ("指令方向 (commanded_motion):", self.val_motion),
            ("下達速度指令 (非實測速度):", self.val_cmd_speeds),
            ("底盤物理停穩確認:", self.val_phys_stop),
            ("停止/拒絕原因 (reason):", self.val_reason),
            ("需重啟巡邏 (restart_required):", self.val_restart),
            ("連線與最後更新時間:", self.val_last_update),
        ]

        for idx, (title_text, val_widget) in enumerate(rows):
            lbl_t = QLabel(title_text)
            lbl_t.setStyleSheet("color: #94a3b8; font-size: 12px;")
            status_layout.addWidget(lbl_t, idx, 0, Qt.AlignmentFlag.AlignLeft)
            status_layout.addWidget(val_widget, idx, 1, Qt.AlignmentFlag.AlignRight)

        right_layout.addWidget(status_group)
        right_layout.addStretch(1)

        main_layout.addWidget(right_widget)
        self._update_mode_buttons_visual()

    # ==================== 【模式切換與 20Hz 手控核心邏輯】 ====================
    def request_mode_switch(self, target_mode: str):
        """發起 ROS 2 /control/set_mode 服務呼叫，等待回傳成功後才切換狀態"""
        self.lbl_srv_feedback.setText(f"⏳ 正在請求切換模式至 [{target_mode}] (/control/set_mode)...")
        self.lbl_srv_feedback.setStyleSheet("color: #facc15; font-size: 12px;")

        # 若請求非手控模式，先停止手控方向並發送一次零速度確保安全
        if target_mode != "MANUAL":
            self._clear_all_directions_and_send_zero("切換離手控模式")

        self.ros_client.call_set_mode(target_mode)
        self.setFocus()

    def _on_mode_service_reply(self, successful: bool, target_mode: str, reason: str):
        """規格第二節：檢查服務回覆是否成功，再更新介面顯示"""
        if successful:
            self.confirmed_mode = target_mode
            self._update_mode_buttons_visual()
            self.lbl_srv_feedback.setText(f"✅ 模式切換成功：目前已確認為 [{target_mode}] ({reason})")
            self.lbl_srv_feedback.setStyleSheet("color: #4ade80; font-size: 12px; font-weight: bold;")

            # 規格第四節：切換手控成功後，立即開始 20 Hz 傳送
            if target_mode == "MANUAL":
                self.held_keys.clear()
                self.active_button_motion = None
                if not self.manual_pub_timer.isActive():
                    self.manual_pub_timer.start()
                self._on_20hz_manual_tick()  # 立即送出第一筆
            else:
                if self.manual_pub_timer.isActive():
                    self.manual_pub_timer.stop()
                self.lbl_tx_info.setText(f"20Hz 發送器: 已停用 (目前模式: {target_mode})")
        else:
            self.lbl_srv_feedback.setText(f"❌ 拒絕切換至 [{target_mode}]，原因：{reason}")
            self.lbl_srv_feedback.setStyleSheet("color: #f87171; font-size: 12px; font-weight: bold;")
            self.val_reason.setText(str(reason))
            self.val_reason.setStyleSheet("color: #f87171; font-weight: bold;")

    def _update_mode_buttons_visual(self):
        styles = {
            "AUTO":   ("background-color: #065f46; border: 2px solid #10b981; color: #ffffff;", "background-color: #1e293b;"),
            "MANUAL": ("background-color: #854d0e; border: 2px solid #facc15; color: #ffffff;", "background-color: #1e293b;"),
            "STOP":   ("background-color: #991b1b; border: 2px solid #ef4444; color: #ffffff;", "background-color: #1e293b;"),
        }
        self.btn_mode_auto.setStyleSheet(styles["AUTO"][0] if self.confirmed_mode == "AUTO" else styles["AUTO"][1])
        self.btn_mode_manual.setStyleSheet(styles["MANUAL"][0] if self.confirmed_mode == "MANUAL" else styles["MANUAL"][1])
        self.btn_mode_stop.setStyleSheet(styles["STOP"][0] if self.confirmed_mode == "STOP" else styles["STOP"][1])

    def _compute_desired_velocity(self):
        """根據目前按住的按鈕或鍵盤組合計算 (linear.x, angular.z, 動作標籤)"""
        if self.confirmed_mode != "MANUAL":
            return 0.0, 0.0, "STOP"

        # 1. 優先檢查滑鼠按住的介面按鈕
        if self.active_button_motion in MOTION_TABLE:
            lx, az, label = MOTION_TABLE[self.active_button_motion]
            return lx, az, label

        # 2. 檢查鍵盤按住的方向鍵
        if "SPACE" in self.held_keys:
            return 0.0, 0.0, "停車 (STOP)"
        if "Q" in self.held_keys or ("W" in self.held_keys and "A" in self.held_keys):
            return MOTION_TABLE["FORWARD_LEFT"]
        if "E" in self.held_keys or ("W" in self.held_keys and "D" in self.held_keys):
            return MOTION_TABLE["FORWARD_RIGHT"]
        if "W" in self.held_keys:
            return MOTION_TABLE["FORWARD"]
        if "S" in self.held_keys:
            return MOTION_TABLE["BACKWARD"]
        if "A" in self.held_keys:
            return MOTION_TABLE["LEFT"]
        if "D" in self.held_keys:
            return MOTION_TABLE["RIGHT"]

        return 0.0, 0.0, "零速度待命 (ZERO_HOLD)"

    def _on_20hz_manual_tick(self):
        """規格第四節：手控模式期間，以 20 Hz 持續傳送；沒有按方向時也傳零速度"""
        if self.confirmed_mode != "MANUAL":
            return
        lin_x, ang_z, _ = self._compute_desired_velocity()
        self.ros_client.publish_manual_cmd_vel(lin_x, ang_z)

    def _on_motion_button_pressed(self, motion_key: str):
        if self.confirmed_mode != "MANUAL":
            self.lbl_srv_feedback.setText("⚠️ 請先切換至「手控 (MANUAL)」模式才能控制移動！")
            self.lbl_srv_feedback.setStyleSheet("color: #facc15; font-size: 12px; font-weight: bold;")
            return
        self.active_button_motion = motion_key
        lin_x, ang_z, _ = self._compute_desired_velocity()
        self.ros_client.publish_manual_cmd_vel(lin_x, ang_z)

    def _on_motion_button_released(self):
        """規格第四節：放開立即傳零速度"""
        self.active_button_motion = None
        if self.confirmed_mode == "MANUAL":
            self.ros_client.publish_manual_cmd_vel(0.0, 0.0)
        self.setFocus()

    def _clear_all_directions_and_send_zero(self, trigger_reason=""):
        """清除所有方向鍵並立即傳送零速度"""
        had_motion = bool(self.held_keys or self.active_button_motion)
        self.held_keys.clear()
        self.active_button_motion = None
        if self.confirmed_mode == "MANUAL" or had_motion:
            self.ros_client.publish_manual_cmd_vel(0.0, 0.0)
        if trigger_reason and self.confirmed_mode == "MANUAL":
            self.lbl_tx_info.setText(f"🛑 已強制歸零 ({trigger_reason}) | lin.x=0.00, ang.z=0.00")

    # ==================== 【鍵盤與視窗失焦保護事件】 ====================
    def keyPressEvent(self, event: QKeyEvent):
        if event.isAutoRepeat():
            return
        key = event.key()
        key_map = {
            Qt.Key.Key_W: "W", Qt.Key.Key_Up: "W",
            Qt.Key.Key_S: "S", Qt.Key.Key_Down: "S",
            Qt.Key.Key_A: "A", Qt.Key.Key_Left: "A",
            Qt.Key.Key_D: "D", Qt.Key.Key_Right: "D",
            Qt.Key.Key_Q: "Q",
            Qt.Key.Key_E: "E",
            Qt.Key.Key_Space: "SPACE",
        }
        if key in key_map:
            if self.confirmed_mode == "MANUAL":
                self.held_keys.add(key_map[key])
                lin_x, ang_z, _ = self._compute_desired_velocity()
                self.ros_client.publish_manual_cmd_vel(lin_x, ang_z)
            else:
                self.lbl_srv_feedback.setText("⚠️ 目前非手控模式，方向鍵已鎖定！請先點擊 [手控 MANUAL]。")
                self.lbl_srv_feedback.setStyleSheet("color: #facc15; font-size: 12px;")
            return

        # 視覺系統快捷鍵
        if key == Qt.Key.Key_R:
            self._send_vision_cmd('r')
        elif key == Qt.Key.Key_T or key == Qt.Key.Key_V:
            self._send_vision_cmd('t')
        elif key == Qt.Key.Key_P:
            self._send_vision_cmd('s')

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.isAutoRepeat():
            return
        key = event.key()
        key_map = {
            Qt.Key.Key_W: "W", Qt.Key.Key_Up: "W",
            Qt.Key.Key_S: "S", Qt.Key.Key_Down: "S",
            Qt.Key.Key_A: "A", Qt.Key.Key_Left: "A",
            Qt.Key.Key_D: "D", Qt.Key.Key_Right: "D",
            Qt.Key.Key_Q: "Q",
            Qt.Key.Key_E: "E",
            Qt.Key.Key_Space: "SPACE",
        }
        if key in key_map:
            self.held_keys.discard(key_map[key])
            if self.confirmed_mode == "MANUAL":
                lin_x, ang_z, _ = self._compute_desired_velocity()
                self.ros_client.publish_manual_cmd_vel(lin_x, ang_z)

    def focusOutEvent(self, event):
        """規格第四節：視窗失去焦點時，清除方向按鍵並傳零速度"""
        self._clear_all_directions_and_send_zero("視窗失去焦點")
        super().focusOutEvent(event)

    def _on_app_state_changed(self, state):
        if state != Qt.ApplicationState.ApplicationActive:
            self._clear_all_directions_and_send_zero("切換至其他視窗")

    # ==================== 【車端狀態訂閱更新與斷線處理】 ====================
    def _on_ros_connection_changed(self, connected: bool, info_str: str):
        if connected:
            self.lbl_conn_banner.setText(f"🟢 {info_str}")
            self.lbl_conn_banner.setStyleSheet("color: #4ade80; font-weight: bold; font-size: 12px;")
            if self.was_disconnected:
                # 規格第四節：車端超過 0.5 秒未收到手控訊息會切成 STOP；重新連線後需再次切換手控
                self.confirmed_mode = "STOP"
                if self.manual_pub_timer.isActive():
                    self.manual_pub_timer.stop()
                self._update_mode_buttons_visual()
                self.lbl_srv_feedback.setText("⚠️ 已重新連線！因安全機制車端已切為 STOP，如需手控請再次點擊 [MANUAL]。")
                self.lbl_srv_feedback.setStyleSheet("color: #facc15; font-size: 12px; font-weight: bold;")
                self.was_disconnected = False
        else:
            self.was_disconnected = True
            self.confirmed_mode = "STOP"
            if self.manual_pub_timer.isActive():
                self.manual_pub_timer.stop()
            self.held_keys.clear()
            self.active_button_motion = None
            self._update_mode_buttons_visual()
            self.lbl_conn_banner.setText(f"🔴 {info_str}")
            self.lbl_conn_banner.setStyleSheet("color: #f87171; font-weight: bold; font-size: 12px;")

    def _on_vehicle_status_received(self, data: dict, ts: float):
        """
        規格第五節：解析 /control/status JSON 並更新介面
        {
          "mode": "MANUAL",
          "route_state": "MAIN",
          "reason": "MANUAL_COMMAND",
          "restart_required": true,
          "commanded_motion": "FORWARD",
          "linear_x": 0.05,
          "angular_z": 0.0,
          "physical_stop_confirmed": false
        }
        """
        self.last_status_time = ts

        car_mode = str(data.get("mode", "UNKNOWN")).upper()
        route_state = str(data.get("route_state", "--"))
        reason = str(data.get("reason", "--"))
        restart_req = bool(data.get("restart_required", False))
        cmd_motion = str(data.get("commanded_motion", "--"))
        lin_x = float(data.get("linear_x", 0.0))
        ang_z = float(data.get("angular_z", 0.0))
        phys_stop = bool(data.get("physical_stop_confirmed", False))

        # 若車端安全層或 0.5 秒逾時自動切回 STOP，同步介面狀態並停止 20Hz 發送，要求重新切換手控
        if self.confirmed_mode == "MANUAL" and car_mode == "STOP":
            self.confirmed_mode = "STOP"
            if self.manual_pub_timer.isActive():
                self.manual_pub_timer.stop()
            self.held_keys.clear()
            self.active_button_motion = None
            self._update_mode_buttons_visual()
            self.lbl_srv_feedback.setText(f"⚠️ 車端已自動切換為 STOP (原因: {reason})，請重新點擊 [MANUAL] 啟動手控！")
            self.lbl_srv_feedback.setStyleSheet("color: #f87171; font-size: 12px; font-weight: bold;")
        elif car_mode in ("AUTO", "MANUAL", "STOP") and car_mode != self.confirmed_mode:
            self.confirmed_mode = car_mode
            self._update_mode_buttons_visual()

        # 1. 目前模式與路線狀態
        mode_colors = {"AUTO": "#4ade80", "MANUAL": "#facc15", "STOP": "#f87171"}
        self.val_mode.setText(car_mode)
        self.val_mode.setStyleSheet(f"color: {mode_colors.get(car_mode, '#f8fafc')}; font-weight: bold; font-size: 14px;")
        self.val_route.setText(route_state)

        # 2. 指令方向、線速度、角速度 (明確標示為下達的指令值，非底盤實測速度)
        self.val_motion.setText(cmd_motion)
        self.val_cmd_speeds.setText(f"{lin_x:+.2f} m/s | {ang_z:+.2f} rad/s (指令值)")

        # 3. 底盤物理停穩確認 (嚴格遵守：不能把零速度指令顯示成「已確認車子停穩」)
        if phys_stop:
            self.val_phys_stop.setText("✅ 已確認車子物理停穩 (True)")
            self.val_phys_stop.setStyleSheet("color: #4ade80; font-weight: bold;")
        else:
            if abs(lin_x) < 1e-4 and abs(ang_z) < 1e-4:
                self.val_phys_stop.setText("⚠️ 指令為零，但尚未確認車子停穩 (False)")
            else:
                self.val_phys_stop.setText("🏃 車體移動/運動中，未停穩 (False)")
            self.val_phys_stop.setStyleSheet("color: #fb923c; font-weight: bold;")

        # 4. 停止或拒絕切換的原因
        self.val_reason.setText(reason)
        if any(w in reason.upper() for w in ["STOP", "OBSTACLE", "ERROR", "TIMEOUT", "REJECT", "FAIL", "LIDAR", "RADAR"]):
            self.val_reason.setStyleSheet("color: #f87171; font-weight: bold;")
        else:
            self.val_reason.setStyleSheet("color: #38bdf8; font-weight: bold;")

        # 5. 是否需要重新開始巡邏：restart_required
        if restart_req:
            self.val_restart.setText("⚠️ 是 (True - 需重新開始巡邏)")
            self.val_restart.setStyleSheet("color: #facc15; font-weight: bold;")
        else:
            self.val_restart.setText("否 (False)")
            self.val_restart.setStyleSheet("color: #94a3b8; font-weight: bold;")

    def _on_tx_heartbeat(self, lin_x: float, ang_z: float, hz: int):
        self.lbl_tx_info.setText(
            f"📡 /manual/cmd_vel [{hz:2d} Hz] -> lin.x={lin_x:+.2f} m/s, ang.z={ang_z:+.2f} rad/s"
        )

    def _on_ui_watchdog_tick(self):
        if self.last_status_time <= 0:
            self.val_last_update.setText("離線 / 尚未收到狀態")
            self.val_last_update.setStyleSheet("color: #f87171; font-weight: bold;")
            return
        age_ms = int((time.time() - self.last_status_time) * 1000)
        ts_str = time.strftime("%H:%M:%S", time.localtime(self.last_status_time))
        if age_ms < 600:
            self.val_last_update.setText(f"🟢 連線正常 | {ts_str} ({age_ms} ms 前)")
            self.val_last_update.setStyleSheet("color: #4ade80; font-weight: bold;")
        else:
            self.val_last_update.setText(f"🔴 訊號延遲/中斷 | {ts_str} ({age_ms/1000:.1f}s 前)")
            self.val_last_update.setStyleSheet("color: #f87171; font-weight: bold;")

    def _on_apply_endpoint(self):
        ip = self.edit_ip.text().strip() or DEFAULT_XAVIER_IP
        try:
            port = int(self.edit_port.text().strip())
        except ValueError:
            port = 9090
        sim = self.chk_sim.isChecked()
        self.ros_client.set_endpoint(ip, port, sim)
        self.setFocus()

    # ==================== 【左側雙光四分割畫面更新】 ====================
    def _on_quad_frame_ready(self, canvas_bgr: np.ndarray, meta: dict):
        rgb_img = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        scaled_pix = pix.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pix)

        # 同步更新下方按鈕狀態文字
        auto_on = meta.get("auto_mode", True)
        is_rec = meta.get("is_recording", False)
        self.btn_auto_rec.setText(f"🤖 自動感應巡檢錄製: {'開啟 [A]' if auto_on else '關閉 [A]'}")
        self.btn_auto_rec.setStyleSheet(
            "background-color: #065f46; border-color: #10b981;" if auto_on else "background-color: #1e293b;"
        )
        self.btn_manual_rec.setText("⏹ 停止並儲存生理記錄 [T]" if is_rec else "⏺ 手動啟動生理記錄 [T]")
        self.btn_manual_rec.setStyleSheet(
            "background-color: #991b1b; border-color: #ef4444;" if is_rec else "background-color: #1e293b;"
        )

    def _send_vision_cmd(self, cmd: str):
        self.quad_worker.send_command(cmd)
        self.setFocus()

    def closeEvent(self, event):
        self._clear_all_directions_and_send_zero("關閉程式")
        if self.manual_pub_timer.isActive():
            self.manual_pub_timer.stop()
        self.quad_worker.send_command("QUIT")
        self.ros_client.stop()
        time.sleep(0.2)
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="智慧豬隻雙光四分割 AI 監控 & ROS 2 小車遙控整合工作台")
    parser.add_argument("--rgb", type=str, default=DEFAULT_RGB_URL, help="可見光 RTSP 或影片路徑")
    parser.add_argument("--thm", type=str, default=DEFAULT_THM_URL, help="熱成像 RTSP 或影片路徑")
    parser.add_argument("--xavier_ip", type=str, default=DEFAULT_XAVIER_IP, help="Xavier IP 位址")
    parser.add_argument("--ros_port", type=int, default=9090, help="ROS 2 rosbridge WebSocket Port (預設: 9090)")
    parser.add_argument("--udp_port", type=int, default=DEFAULT_UDP_PORT, help="豬臉追蹤 UDP Port (預設: 5007)")
    parser.add_argument("--ear_mode", type=str, default="pose", choices=["pose", "yolo_bbox"], help="耳根偵測模式")
    parser.add_argument("--save", action="store_true", help="同時將四分割畫面錄製為 MP4")
    parser.add_argument("--full", action="store_true", help="啟用全幀推論")
    parser.add_argument("--no_auto", action="store_true", help="預設關閉自動感應開錄")
    parser.add_argument("--demo", action="store_true", help="使用本地預錄雙光影片展示四分割畫面")
    parser.add_argument("--sim_ros", action="store_true", help="啟用本機 ROS 2 閉環模擬模式 (免連車即可測試所有按鈕)")
    args = parser.parse_args()

    if args.demo:
        demo_rgb = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "484-6_11410181546_可見光_正面(30秒呼吸34次).mp4")
        demo_thm = os.path.join(PROJECT_ROOT, "v3", "data", "test_vid", "484-6_11410181546_熱成像_正面(30秒呼吸34次).mp4")
        if os.path.exists(demo_rgb) and os.path.exists(demo_thm):
            args.rgb = demo_rgb
            args.thm = demo_thm

    app = QApplication(sys.argv)
    win = PigQuadCockpitWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
