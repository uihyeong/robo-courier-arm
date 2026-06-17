"""
엘리베이터 버튼 노드 (arm_elevator.py).

Phase 1 (UPDOWN):  YOLOv8 → UP/DOWN 버튼 감지 → 해석적 IK → 누르기
Phase 2 (NUMBER):  YOLO-seg + EasyOCR → 숫자 버튼 감지 → 해석적 IK → 누르기

상태 전이:
  IDLE → UPDOWN_READY → UPDOWN_PRESS → WAIT → NUMBER_READY → NUMBER_PRESS → DONE

토픽:
  구독: /target_floor   (Int32) — 목표 층수
  구독: /elevator_ready (Bool)  — Scout 탑승 완료 → 숫자 버튼 Phase 시작
  발행: /robot_status   (String) — MOVING/UPDOWN_PRESSED/ELEVATOR_ARRIVED/NUMBER_PRESSED/FAILED/NEED_REPOSITION
"""

import math
import os
import datetime
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.time
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, Int32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros
import tf2_geometry_msgs

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

try:
    import easyocr
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

# ─── 모델 경로 ────────────────────────────────────────────────────────────────

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPDOWN_MODEL_PATH = os.path.join(_REPO_ROOT, 'yolo', 'weights', 'best.pt')
NUM_MODEL_PATH    = os.path.join(_REPO_ROOT, 'yolo', 'weights', 'best_num.pt')

# ─── OpenMANIPULATOR-X 링크 파라미터 ─────────────────────────────────────────

L1    = 0.0595
L2    = math.sqrt(0.024**2 + 0.128**2)
ALPHA = math.atan2(0.128, 0.024)
L3    = 0.124
L4    = 0.126

JOINT_LIMITS = [
    (-math.pi, math.pi + 0.65),  # +0.65 rad 여유: ±π 경계 왼쪽 버튼 지원
    (-2.0,     1.5),
    (-1.5,     1.4),
    (-1.7,     1.97),
]

JOINT_NAMES        = ['joint1', 'joint2', 'joint3', 'joint4']
HOME_JOINTS        = [+3.1400, -1.9190, 1.2701,  0.7240]
NUMBER_HOME_JOINTS = [+3.1400, -1.9190, 1.2701,  0.7240]
MOVE_SPEED   = 0.5
MIN_DURATION = 2.0
# 단일 관절 1회 이동 안전 상한(rad). 이보다 크면 위험 동작(한 바퀴 회전 등)으로 보고 차단.
# 엘리베이터 팔은 홈(+pi)↔버튼(≈0) 약 180°(π) 이동이 정상이므로 그보다 크게 둔다.
MAX_JOINT_STEP = 4.5

# ─── 인식 파라미터 ────────────────────────────────────────────────────────────

UPDOWN_CONF_MIN   = 0.7
NUM_CONF_MIN      = 0.5
NUM_PRESS_CONF    = 0.7
OCR_INTERVAL      = 5
BUTTON_OFFSET_X   = 0.075
MAX_FAIL          = 3
LIT_GREEN_RATIO   = 0.30
LIT_BRIGHT_RATIO  = 0.60

# ─── 상태 상수 ────────────────────────────────────────────────────────────────

IDLE          = 'IDLE'
UPDOWN_READY  = 'UPDOWN_READY'
UPDOWN_PRESS  = 'UPDOWN_PRESS'
WAIT          = 'WAIT'
NUMBER_READY  = 'NUMBER_READY'
NUMBER_PRESS  = 'NUMBER_PRESS'
NUMBER_WAIT   = 'NUMBER_WAIT'
DONE          = 'DONE'


# ─── 해석적 IK ────────────────────────────────────────────────────────────────

def solve_ik(X: float, Y: float, Z: float):
    j1_raw = math.atan2(Y, X)

    # 홈(+π)에서 3사분면(X<0,Y<0) 버튼: j1_raw ≈ -π+ε → j1_raw+2π ≈ π+ε 로도 시도
    j1_list = [j1_raw]
    if j1_raw < -math.pi / 2:
        j1_list.append(j1_raw + 2 * math.pi)

    r  = math.sqrt(X**2 + Y**2)
    wr = r - L4
    wz = Z
    dr = wr
    dz = wz - L1
    D  = math.sqrt(dr**2 + dz**2)

    if D > (L2 + L3) * 0.999:
        return None
    if D < abs(L2 - L3) * 1.001:
        return None

    c_psi = (D**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    c_psi = max(-1.0, min(1.0, c_psi))

    for j1 in j1_list:
        for psi in (-math.acos(c_psi), math.acos(c_psi)):
            s_psi  = math.sin(psi)
            gamma  = math.atan2(L3 * s_psi, L2 + L3 * c_psi)
            alpha1 = math.atan2(dz, dr) - gamma
            j2     = ALPHA - alpha1
            j3     = -psi - ALPHA
            j4     = -(j2 + j3)

            angles = [j1, j2, j3, j4]
            if all(lo <= a <= hi for a, (lo, hi) in zip(angles, JOINT_LIMITS)):
                return angles

    return None


def _shortest_path(target: float, current: float) -> float:
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff


def make_trajectory(target_joints: list, current_joints: list, speed: float = MOVE_SPEED):
    target_joints = [_shortest_path(t, c) for t, c in zip(target_joints, current_joints)]
    # 관절 한계로 클램프 (_shortest_path 는 한계를 무시하므로 필수)
    target_joints = [max(lo, min(hi, t))
                     for t, (lo, hi) in zip(target_joints, JOINT_LIMITS)]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
    # 과대 이동(한 바퀴 회전 등)이면 None 반환 → 호출부에서 중단
    if max_disp > MAX_JOINT_STEP:
        return None, max_disp
    duration = max(max_disp / speed, MIN_DURATION)

    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES

    pt = JointTrajectoryPoint()
    pt.positions = target_joints
    pt.velocities = [0.0] * 4
    secs  = int(duration)
    nsecs = int((duration - secs) * 1e9)
    pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
    traj.points.append(pt)

    return traj, duration


# ─── 노드 ────────────────────────────────────────────────────────────────────

class ArmElevatorNode(Node):
    def __init__(self):
        super().__init__('arm_elevator')

        self.lock   = threading.Lock()
        self.bridge = CvBridge()

        self.state = IDLE

        self.target_floor   = None
        self.current_floor  = -1
        self.target_button  = None

        self.current_joints = None
        self.moving         = False

        self.depth_image = None
        self.fx, self.fy = 1380.0, 1380.0
        self.cx, self.cy = 960.0, 540.0

        self.ocr_cache   = {}
        self.frame_count = 0

        self._fail_updown = 0

        self.latest_frame      = None
        self._frame_lock       = threading.Lock()
        self._last_updown_bbox = None
        self._last_number_bbox = None
        self._writer           = None
        _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._record_path = os.path.expanduser(f'~/recordings/elevator_{_ts}.mp4')

        self._elevator_ready_event = threading.Event()

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        self.create_subscription(Int32,        '/target_floor',    self._cb_target_floor,    10)
        self.create_subscription(JointState,   '/joint_states',    self._cb_joint_state,     10)
        self.create_subscription(PointStamped, '/target_point',    self._cb_target_point,    10)
        self.create_subscription(Bool,         '/elevator_ready',  self._cb_elevator_ready,  10)

        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self._cb_camera_info, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw',
                                 self._cb_depth, 10)

        self.updown_model = None
        self.num_model    = None
        self.ocr          = None

        if _YOLO_AVAILABLE:
            try:
                self.updown_model = YOLO(UPDOWN_MODEL_PATH)
                self.get_logger().info('UP/DOWN YOLO 모델 로드 완료')
            except Exception as e:
                self.get_logger().warn(f'UP/DOWN YOLO 로드 실패: {e}')

            try:
                self.num_model = YOLO(NUM_MODEL_PATH)
                self.get_logger().info('숫자 YOLO 모델 로드 완료')
            except Exception as e:
                self.get_logger().warn(f'숫자 YOLO 로드 실패: {e}')

        if _OCR_AVAILABLE and self.num_model is not None:
            self.get_logger().info('EasyOCR 초기화 중...')
            self.ocr = easyocr.Reader(['en'], gpu=False)
            self.get_logger().info('EasyOCR 초기화 완료')

        if self.updown_model is not None or self.num_model is not None:
            self.create_subscription(Image, '/camera/camera/color/image_raw',
                                     self._cb_image, 10)
            # 표시/녹화 + YOLO/OCR 추론은 별도 루프 스레드에서 수행한다.
            # (카메라 콜백은 프레임 저장만 → executor 가 OCR 로 막히지 않아 끊김/정지 방지)
            threading.Thread(target=self._loop, daemon=True).start()

        self.get_logger().info('arm_elevator 노드 시작. 2초 후 홈 이동...')
        self._home_timer = self.create_timer(2.0, self._move_to_home_once)

    # ─── 콜백 ───────────────────────────────────────────────────────────────

    def _cb_elevator_ready(self, msg: Bool):
        if self.state == WAIT:
            self.get_logger().info('/elevator_ready 수신 → 숫자 버튼 Phase 시작')
            self._elevator_ready_event.set()
        else:
            self.get_logger().warn(f'/elevator_ready 무시 (state={self.state})')

    def _cb_joint_state(self, msg: JointState):
        with self.lock:
            self.current_joints = msg

    def _cb_target_floor(self, msg: Int32):
        floor = msg.data
        if self.state not in (IDLE, DONE):
            self.get_logger().warn(f'작업 중 ({self.state}). /target_floor 무시.')
            return
        self.target_floor  = floor
        self.target_button = 'up_button' if floor > self.current_floor else 'down_button'
        self.ocr_cache.clear()
        self.frame_count = 0
        self.get_logger().info(f'목표 층: {floor}F | {self.target_button} 누르기 대기')
        self.state = UPDOWN_READY

    def _cb_camera_info(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx = msg.k[2]; self.cy = msg.k[5]

    def _cb_depth(self, msg: Image):
        with self.lock:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.depth_image = raw.astype(np.float32) / 1000.0

    def _cb_target_point(self, msg: PointStamped):
        if self.moving:
            self.get_logger().warn('이동 중. /target_point 무시.')
            return
        X, Y, Z = msg.point.x, msg.point.y, msg.point.z
        self.get_logger().info(f'/target_point 수신: ({X:.3f}, {Y:.3f}, {Z:.3f})')
        threading.Thread(target=self._press_button, args=(X, Y, Z, '수동'), daemon=True).start()

    # ─── 녹화 ────────────────────────────────────────────────────────────────

    def _write_frame(self, frame):
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(self._record_path, fourcc, 20, (w, h))
        self._writer.write(frame)

    # ─── 이미지 처리 ─────────────────────────────────────────────────────────

    def _cb_image(self, msg: Image):
        # 콜백은 프레임 저장만 (가볍게) — 무거운 추론/표시는 _loop 스레드가 담당
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self._frame_lock:
            self.latest_frame = frame

    def _loop(self):
        while rclpy.ok():
            with self._frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            if frame is None:
                time.sleep(0.02)
                continue
            with self.lock:
                depth = self.depth_image.copy() if self.depth_image is not None else None

            state = self.state

            # 팔 이동 중에는 무거운 YOLO/OCR 추론은 건너뛰고 화면 표시/녹화만 유지
            # (이동 중에도 매 루프 최신 프레임을 녹화하므로 끊김/정지화면 없음)
            if self.moving:
                cv2.putText(frame, f'MOVING | State: {state}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            elif state == UPDOWN_READY and self.updown_model is not None:
                self._process_updown(frame, depth)
            elif state == NUMBER_READY and self.num_model is not None:
                self._process_number(frame, depth)
            else:
                label = f'State: {state} | Target: {self.target_floor}F'
                cv2.putText(frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                if state == WAIT and self._last_updown_bbox is not None:
                    x1, y1, x2, y2 = self._last_updown_bbox
                    ratio = self._get_lit_ratio()
                    if self.target_button == 'down_button':
                        color = (255, 255, 255)
                        ratio_label = f'brightness={ratio:.3f} (>{LIT_BRIGHT_RATIO})' if ratio is not None else 'brightness=N/A'
                    else:
                        color = (0, 255, 0)
                        ratio_label = f'green={ratio:.3f} (>{LIT_GREEN_RATIO})' if ratio is not None else 'green=N/A'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, ratio_label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                if state == NUMBER_WAIT and self._last_number_bbox is not None:
                    x1, y1, x2, y2 = self._last_number_bbox
                    ratio = self._get_lit_ratio(self._last_number_bbox)
                    color = (0, 255, 0)
                    ratio_label = f'green={ratio:.3f} (>{LIT_GREEN_RATIO})' if ratio is not None else 'green=N/A'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, ratio_label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            self._write_frame(frame)
            cv2.imshow('ElevatorArm', frame)
            cv2.waitKey(1)

    # ─── Phase 1: UP/DOWN 인식 ───────────────────────────────────────────────

    def _process_updown(self, frame, depth):
        results = self.updown_model(frame, conf=0.5, verbose=False)
        colors  = {'up_button': (0, 255, 0), 'down_button': (0, 0, 255)}

        for result in results:
            for box in result.boxes:
                cls  = result.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx_box = (x1 + x2) // 2
                cy_box = (y1 + y2) // 2
                color  = colors.get(cls, (255, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{cls} {conf:.2f}',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if cls != self.target_button or conf < UPDOWN_CONF_MIN or depth is None:
                    continue

                region = depth[cy_box-2:cy_box+3, cx_box-2:cx_box+3]
                valid  = region[(region > 0) & ~np.isnan(region)]
                if len(valid) == 0:
                    continue
                d = float(np.median(valid))

                X_cam = (cx_box - self.cx) / self.fx * d
                Y_cam = (cy_box - self.cy) / self.fy * d

                pt_cam = PointStamped()
                pt_cam.header.frame_id = 'camera_color_optical_frame'
                pt_cam.header.stamp    = rclpy.time.Time().to_msg()
                pt_cam.point.x = X_cam
                pt_cam.point.y = Y_cam
                pt_cam.point.z = d

                try:
                    pt_w = self.tf_buffer.transform(pt_cam, 'world')
                    X, Y, Z = pt_w.point.x, pt_w.point.y, pt_w.point.z
                    cv2.putText(frame, f'({X:.2f},{Y:.2f},{Z:.2f})',
                                (x1, y1 - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                except Exception as e:
                    self.get_logger().warn(f'TF 변환 실패: {e}')
                    continue

                self._last_updown_bbox = (x1, y1, x2, y2)
                self.state = UPDOWN_PRESS
                self.get_logger().info(f'{cls} 감지! IK 시작')
                cv2.putText(frame, f'UPDOWN | Target: {self.target_button}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                threading.Thread(
                    target=self._press_button,
                    args=(X, Y, Z - 0.031, cls),
                    daemon=True,
                ).start()
                return

        cv2.putText(frame, f'UPDOWN | Target: {self.target_button}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ─── Phase 2: 숫자 인식 ─────────────────────────────────────────────────

    def _process_number(self, frame, depth):
        self.frame_count += 1
        run_ocr = (self.frame_count % OCR_INTERVAL == 0)

        results = self.num_model(frame, conf=NUM_CONF_MIN, verbose=False)

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                pad  = 5
                cx1  = max(0, x1 - pad);  cy1 = max(0, y1 - pad)
                cx2  = min(frame.shape[1], x2 + pad)
                cy2  = min(frame.shape[0], y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]

                key = f'{x1//20}_{y1//20}_{x2//20}_{y2//20}'
                if run_ocr or key not in self.ocr_cache:
                    self.ocr_cache[key] = self._read_number(crop)
                number = self.ocr_cache.get(key)

                matched = (number == self.target_floor)
                color   = (0, 255, 0) if matched else (180, 180, 180)
                label   = str(number) if number is not None else '?'

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if matched and conf > NUM_PRESS_CONF and depth is not None:
                    self.state = NUMBER_PRESS
                    cv2.putText(frame, f'NUMBER | Target: {self.target_floor}F',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    self._trigger_number_press(depth, x1, y1, x2, y2)
                    return

        cv2.putText(frame, f'NUMBER | Target: {self.target_floor}F',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def _read_number(self, crop):
        if self.ocr is None:
            return None
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None

        scale   = max(64 / max(h, w), 1.0)
        resized = cv2.resize(crop, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        ocr_results = self.ocr.readtext(enhanced, allowlist='0123456789Bb', detail=1)
        best_conf, best_num = 0.0, None
        for (_, text, conf) in ocr_results:
            text = text.strip().upper()
            if text.isdigit() and conf > best_conf:
                best_conf = conf;  best_num = int(text)
            elif text.startswith('B') and text[1:].isdigit() and conf > best_conf:
                best_conf = conf;  best_num = -int(text[1:])
        return best_num

    def _trigger_number_press(self, depth, x1, y1, x2, y2):
        self._last_number_bbox = (x1, y1, x2, y2)
        cx_box = (x1 + x2) // 2
        cy_box = (y1 + y2) // 2

        region = depth[cy_box-2:cy_box+3, cx_box-2:cx_box+3]
        valid  = region[(region > 0.1) & ~np.isnan(region)]
        if len(valid) == 0:
            self.get_logger().warn('깊이값 없음. 재시도...')
            self.state = NUMBER_READY
            return
        d = float(np.median(valid))

        X_cam = (cx_box - self.cx) / self.fx * d
        Y_cam = (cy_box - self.cy) / self.fy * d

        pt_cam = PointStamped()
        pt_cam.header.frame_id = 'camera_color_optical_frame'
        pt_cam.header.stamp    = rclpy.time.Time().to_msg()
        pt_cam.point.x = X_cam
        pt_cam.point.y = Y_cam
        pt_cam.point.z = d

        try:
            pt_w = self.tf_buffer.transform(pt_cam, 'world')
            X, Y, Z = pt_w.point.x, pt_w.point.y, pt_w.point.z
        except Exception as e:
            self.get_logger().warn(f'TF 변환 실패: {e}')
            self.state = NUMBER_READY
            return

        self.get_logger().info(
            f'{self.target_floor}층 버튼 감지! 위치: ({X:.3f}, {Y:.3f}, {Z:.3f})')
        threading.Thread(
            target=self._press_button,
            args=(X, Y, Z - 0.031, f'{self.target_floor}층'),
            daemon=True,
        ).start()

    # ─── 공통 IK + 이동 ──────────────────────────────────────────────────────

    def _on_press_fail(self, phase_updown: bool):
        if phase_updown:
            self._fail_updown += 1
            count = self._fail_updown
            self.get_logger().warn(f'UP/DOWN 실패 {count}/{MAX_FAIL}회')
            if count >= MAX_FAIL:
                self.get_logger().error(f'연속 {MAX_FAIL}회 실패 → NEED_REPOSITION 발행')
                self.status_pub.publish(String(data='NEED_REPOSITION'))
                self._fail_updown = 0
                self.state = IDLE
            else:
                self.state = UPDOWN_READY
        else:
            self.get_logger().warn('숫자 버튼 실패 → 재시도')
            self.state = NUMBER_READY

    def _get_lit_ratio(self, bbox=None) -> float | None:
        frame = self.latest_frame
        if bbox is None:
            bbox = self._last_updown_bbox
        if frame is None or bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if self.target_button == 'down_button':
            return float(hsv[:, :, 2].mean()) / 255.0
        else:
            mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([95, 255, 255]))
            return np.count_nonzero(mask) / (roi.shape[0] * roi.shape[1])

    def _check_button_lit(self) -> bool:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            ratio = self._get_lit_ratio()
            if ratio is None:
                time.sleep(0.2)
                continue
            if self.target_button == 'down_button':
                threshold = LIT_BRIGHT_RATIO
                self.get_logger().info(f'DOWN 점등: brightness={ratio:.3f} (기준 {threshold})')
            else:
                threshold = LIT_GREEN_RATIO
                self.get_logger().info(f'UP 점등: green_ratio={ratio:.3f} (기준 {threshold})')
            return ratio > threshold
        self.get_logger().warn('점등 확인 불가 (5초 대기 후에도 bbox 없음) → 실패로 간주')
        return False

    def _wait_for_button_unlit(self):
        TIMEOUT = 60.0
        self.get_logger().info(f'버튼 소등 대기 중 (최대 {TIMEOUT:.0f}초)...')
        deadline  = time.time() + TIMEOUT
        last_log  = time.time()
        while time.time() < deadline:
            ratio = self._get_lit_ratio()
            if ratio is None:
                time.sleep(0.5)
                continue
            if time.time() - last_log >= 5.0:
                self.get_logger().info(f'소등 대기 중... green_ratio={ratio:.3f}')
                last_log = time.time()
            if ratio <= LIT_GREEN_RATIO:
                self.get_logger().info(f'✅ 버튼 소등! (green_ratio={ratio:.3f}) 엘리베이터 도착')
                self.status_pub.publish(String(data='ELEVATOR_ARRIVED'))
                self._last_updown_bbox = None
                return
            time.sleep(0.5)
        self.get_logger().warn(f'소등 타임아웃 ({TIMEOUT:.0f}초) → 강제 진행')
        self.status_pub.publish(String(data='ELEVATOR_ARRIVED'))
        self._last_updown_bbox = None

    def _return_home_then_wait_number(self):
        ok = self._send_trajectory(HOME_JOINTS)
        if ok:
            self.get_logger().info('홈 복귀 완료. 자세 안정화 대기 중...')
        time.sleep(1.0)

        LIT_TIMEOUT = 10.0
        deadline_lit = time.time() + LIT_TIMEOUT
        lit_confirmed = False
        while time.time() < deadline_lit:
            ratio = self._get_lit_ratio(self._last_number_bbox)
            if ratio is None:
                time.sleep(0.2)
                continue
            self.get_logger().info(f'점등 대기 중... green_ratio={ratio:.3f} (기준 >{LIT_GREEN_RATIO})')
            if ratio > LIT_GREEN_RATIO:
                self.get_logger().info(f'✅ 숫자 버튼 점등 확인! (green_ratio={ratio:.3f})')
                lit_confirmed = True
                break
            time.sleep(0.3)
        if not lit_confirmed:
            self.get_logger().warn(f'점등 타임아웃 ({LIT_TIMEOUT:.0f}초) → 강제 진행')

        self.get_logger().info('소등 대기 중... (엘리베이터 도착 확인)')
        TIMEOUT = 60.0
        deadline = time.time() + TIMEOUT
        last_log = time.time()
        while time.time() < deadline:
            ratio = self._get_lit_ratio(self._last_number_bbox)
            if ratio is None:
                time.sleep(0.5)
                continue
            if time.time() - last_log >= 5.0:
                self.get_logger().info(f'소등 대기 중... green_ratio={ratio:.3f}')
                last_log = time.time()
            if ratio <= LIT_GREEN_RATIO:
                self.get_logger().info(f'✅ 숫자 버튼 소등! (green_ratio={ratio:.3f})')
                break
            time.sleep(0.5)
        else:
            self.get_logger().warn(f'소등 타임아웃 ({TIMEOUT:.0f}초) → 강제 진행')

        self.status_pub.publish(String(data='NUMBER_PRESSED'))
        self.get_logger().info('NUMBER_PRESSED 발행 → Scout Mini 이동 시작')

        self._last_number_bbox = None
        self.state = DONE
        self.get_logger().info('✅ 전체 시퀀스 완료! 3초 후 홈 복귀')
        threading.Timer(3.0, self._move_to_home).start()

    def _press_button(self, X: float, Y: float, Z: float, label: str = ''):
        phase_updown = (self.state == UPDOWN_PRESS)
        sign = math.copysign(1.0, X)

        approach_x = X - BUTTON_OFFSET_X * sign
        joints_approach = solve_ik(approach_x, Y, Z)
        if joints_approach is None:
            self.get_logger().error(
                f'IK 해 없음 [{label}]: ({approach_x:.3f},{Y:.3f},{Z:.3f})')
            self.status_pub.publish(String(data='FAILED'))
            self._on_press_fail(phase_updown)
            return

        self.get_logger().info(f'[{label}] → ({approach_x:.3f},{Y:.3f},{Z:.3f})')
        if not self._send_trajectory(joints_approach):
            self.get_logger().error(f'❌ [{label}] 이동 실패')
            self.status_pub.publish(String(data='FAILED'))
            self._on_press_fail(phase_updown)
            return

        self.get_logger().info(f'✅ [{label}] 버튼 누르기 완료')
        if phase_updown:
            self.current_floor = self.target_floor
            self.state = WAIT
            self.get_logger().info('UP/DOWN 완료. 홈 복귀 후 점등 확인...')
            threading.Thread(target=self._return_home_then_wait, daemon=True).start()
        else:
            self.state = NUMBER_WAIT
            self.get_logger().info('숫자 버튼 완료. 홈 복귀 후 점등→소등 확인...')
            threading.Thread(target=self._return_home_then_wait_number, daemon=True).start()

    def _return_home_then_wait(self):
        ok = self._send_trajectory(HOME_JOINTS)
        if ok:
            self.get_logger().info('홈 복귀 완료. 자세 안정화 대기 중...')
        else:
            self.get_logger().error('홈 복귀 실패. 점등 확인 진행')

        time.sleep(1.0)

        if not self._check_button_lit():
            self.get_logger().warn('버튼 점등 미확인 → 재시도')
            self._on_press_fail(phase_updown=True)
            return

        self._fail_updown = 0
        self.get_logger().info('✅ UP/DOWN 버튼 점등 확인! 엘리베이터 도착 대기(소등)...')
        self.status_pub.publish(String(data='UPDOWN_PRESSED'))

        self._wait_for_button_unlit()

        self.get_logger().info('숫자 패널 자세로 이동 중...')
        self._send_trajectory(NUMBER_HOME_JOINTS)

        self.get_logger().info('Scout Mini /elevator_ready 대기 중...')
        self._elevator_ready_event.clear()
        self._elevator_ready_event.wait()
        self._start_number_phase()

    def _start_number_phase(self):
        self.get_logger().info('숫자 버튼 인식 Phase 시작!')
        self.ocr_cache.clear()
        self.frame_count = 0
        self.state = NUMBER_READY

    # ─── 액션 전송 ───────────────────────────────────────────────────────────

    def _send_trajectory(self, target_joints: list, blocking: bool = True) -> bool:
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('arm_controller 액션 서버 없음!')
            return False

        with self.lock:
            js = self.current_joints
        # 안전장치: /joint_states 가 없으면 현재값을 0 으로 가정하지 않는다.
        # (0 으로 가정하면 +pi/-pi 경계에서 한 바퀴 회전하는 위험 동작이 발생)
        if js is None:
            self.get_logger().error('/joint_states 미수신 → 안전을 위해 이동 중단')
            return False
        current = [None] * 4
        for i, name in enumerate(JOINT_NAMES):
            if name in js.name:
                current[i] = js.position[js.name.index(name)]
        if any(c is None for c in current):
            self.get_logger().error(f'joint_states 관절 누락 {current} → 이동 중단')
            return False

        traj, duration = make_trajectory(target_joints, current)
        if traj is None:
            self.get_logger().error(
                f'단일 관절 이동량 {duration:.2f}rad 과대(>{MAX_JOINT_STEP}) '
                f'→ 위험 동작 차단, 이동 중단 (current={[round(c,3) for c in current]}, '
                f'target={[round(t,3) for t in target_joints]})')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.moving = True
        self.status_pub.publish(String(data='MOVING'))

        future   = self._arm_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error('액션 수락 타임아웃')
                self.moving = False
                return False
            time.sleep(0.05)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('액션 거부됨')
            self.moving = False
            return False

        if not blocking:
            self.moving = False
            return True

        result_future = goal_handle.get_result_async()
        deadline = time.time() + duration + 5.0
        while not result_future.done():
            if time.time() > deadline:
                self.get_logger().error('이동 실행 타임아웃')
                self.moving = False
                return False
            time.sleep(0.1)

        self.moving = False
        code = result_future.result().result.error_code
        return code == FollowJointTrajectory.Result.SUCCESSFUL

    # ─── Home ────────────────────────────────────────────────────────────────

    def _move_to_home_once(self):
        self._home_timer.cancel()
        threading.Thread(target=self._move_to_home, daemon=True).start()

    def _move_to_home(self):
        self.get_logger().info('홈 포지션으로 이동 중...')
        ok = self._send_trajectory(HOME_JOINTS)
        if ok:
            self.get_logger().info('✅ 홈 도착!')
        else:
            self.get_logger().error('❌ 홈 이동 실패')

        if self.state == DONE:
            self.state = IDLE
            self.target_floor = None
            self.get_logger().info('✅ 작업 완료. /target_floor 대기 중...')
        elif self.state == IDLE:
            self.get_logger().info('초기 홈 완료. /target_floor 대기 중...')


# ─── 엔트리포인트 ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ArmElevatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._writer is not None:
            node._writer.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
