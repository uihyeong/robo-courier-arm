"""
프레시백 회수 노드 (arm_recover.py).

회수 시퀀스 (8스텝):
  1. 엘리베이터 홈 (배달 마지막 자세, 그리퍼 닫힌 상태)
  2. 오른쪽 확인 (TABLE_LOOK)
  3. YOLO 박스 인식 대기
  4. 고리에 끼우기 (HOOK_JOINTS — 조인트값 직접 입력)
  5. joint4 올리기 (고개 들어올려 백 걸기)
  6. 홈 복귀, joint4 올린 상태 유지
  7. joint4만 내리기
  8. 엘리베이터 홈 복귀 → 주행 재개

상태 전이:
  IDLE → /start_recover → RECOVER → /recover_done → IDLE

실행:
  ros2 launch open_manipulator_x_bringup hardware.launch.py
  ros2 launch realsense2_camera rs_launch.py
  python3 nodes/real_robot/arm_recover.py
  ros2 topic pub --once /start_recover std_msgs/Bool "{data: true}"
"""

import datetime
import math
import os
import re
import threading
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    import cv2
    import numpy as np
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image as ImageMsg
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False

try:
    from ultralytics import YOLO as UltralyticsYOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

try:
    import easyocr
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

# ─── 모델 경로 ────────────────────────────────────────────────────────────────

_REPO_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOM_MODEL_PATH   = os.path.join(_REPO_ROOT, 'yolo', 'weights', 'best_room.pt')
HANDLE_MODEL_PATH = os.path.join(_REPO_ROOT, 'yolo', 'weights', 'best_handle.pt')
ROOM_CONF         = 0.6
HANDLE_CONF       = 0.50
OCR_INTERVAL      = 5


JOINT_LIMITS = [
    (-math.pi, math.pi),
    (-2.0,     1.5),
    (-1.5,     1.4),
    (-1.7,     1.97),
]

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']

# ─── 관절 상수 ────────────────────────────────────────────────────────────────

HOME_JOINTS          = [3.141,  -1.3963,  1.2217,  0.5236]
TABLE_LOOK_JOINTS    = [1.571,  -1.3963,  1.2217,  0.5236]
# joint1 을 +3.140(=+pi 쪽)으로 둔다. -3.140 과 물리적으로 동일한 자세지만,
# HOME 계열(+3.141)과 같은 부호 쪽에 둬서 마지막 스텝의 +pi/-pi wrap(한 바퀴 회전)을 제거.
ELEVATOR_HOME_JOINTS = [3.1400, -1.9190,  1.2701,  0.7240]

ROOM_SIGN_JOINTS = [ 1.571, -2.0203,  1.5002, -0.044]
HOOK_JOINTS      = [1.571, 0.468, -0.331, -0.206]

# joint4만 올린 상태 (step 5): HOOK 위치에서 joint4만 변경
HOOK_LIFT_JOINTS  = [HOOK_JOINTS[0], HOOK_JOINTS[1], HOOK_JOINTS[2], -0.600]
HOOK_HOVER_JOINTS = [1.571, 0.405, -0.204, -0.600]

# joint4 올린 채로 홈 이동 (step 7): HOME[0:3] + 올린 joint4
HOME_HOOK_JOINTS  = [HOME_JOINTS[0], HOME_JOINTS[1], HOME_JOINTS[2], -0.600]

# joint4만 내리기 (step 8): HOME[0:3] + 내린 joint4
HOME_DOWN_JOINTS  = [HOME_JOINTS[0], HOME_JOINTS[1], HOME_JOINTS[2],  0.300]

MOVE_SPEED   = 0.4
MIN_DURATION = 2.0
STEP_DELAY   = 1.5
# 단일 관절 1회 이동 안전 상한(rad). 이보다 크면 위험 동작으로 보고 차단(한 바퀴 회전 방지).
# 180°(π) 정상 이동은 허용하고 360°(2π) 회전만 막도록 4.5 로 통일(3 노드 공통).
MAX_JOINT_STEP = 4.5

# ─── 시퀀스 정의 ─────────────────────────────────────────────────────────────

RECOVER_STEPS = [
    ('호수 확인',                         ROOM_SIGN_JOINTS),
    ('오른쪽 확인',                       TABLE_LOOK_JOINTS),
    ('고리에 끼우기',                     HOOK_JOINTS),
    ('joint4 올리기 (백 들어올리기)',      HOOK_LIFT_JOINTS),
    ('살짝 호버',                         HOOK_HOVER_JOINTS),
    ('홈 복귀 (joint4 올린 상태 유지)',   HOME_HOOK_JOINTS),
    ('joint4 내리기',                     HOME_DOWN_JOINTS),
    ('엘리베이터 홈 복귀',                ELEVATOR_HOME_JOINTS),
]

# ─── 상태 상수 ────────────────────────────────────────────────────────────────

IDLE    = 'IDLE'
RECOVER = 'RECOVER'
DONE    = 'DONE'

def _shortest_path(target, current):
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff


def make_trajectory(target_joints, current_joints):
    target_joints = [_shortest_path(t, c) for t, c in zip(target_joints, current_joints)]
    # 관절 한계로 클램프 (_shortest_path 는 한계를 무시하므로 필수)
    target_joints = [max(lo, min(hi, t))
                     for t, (lo, hi) in zip(target_joints, JOINT_LIMITS)]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
    # 과대 이동(한 바퀴 회전 등)이면 None 반환 → 호출부에서 중단
    if max_disp > MAX_JOINT_STEP:
        return None, max_disp
    duration = max(max_disp / MOVE_SPEED, MIN_DURATION)
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

class ArmRecoverNode(Node):

    def __init__(self):
        super().__init__('arm_recover')
        self.lock           = threading.Lock()
        self.current_joints = None
        self.state          = IDLE

        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self.status_pub  = self.create_publisher(String, '/robot_status',  10)
        self.recover_pub = self.create_publisher(Bool,   '/recover_done',  10)

        self.create_subscription(JointState, '/joint_states',    self._cb_joints,         10)
        self.create_subscription(Bool,       '/start_recover',   self._cb_start_recover,  10)

        self._current_step_en  = 'Waiting'

        # 카메라 / 호수 인식
        self.bridge            = None
        self._latest_frame     = None
        self._frame_lock       = threading.Lock()
        self._frame_count      = 0
        self._ocr_active       = False
        self._handle_active    = False
        self._latest_room_text = None
        self._latest_room_bbox = None
        self._room_model       = None
        self._handle_model     = None
        self._latest_handle_bbox = None
        self._latest_handle_xyz  = None
        self._ocr              = None
        self.depth_image       = None
        self.fx, self.fy       = 1380.0, 1380.0
        self.cx, self.cy       = 960.0,  540.0

        self._writer = None
        _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._record_path = os.path.expanduser(f'~/recordings/recover_{_ts}.mp4')

        if _CV_AVAILABLE:
            self.bridge = CvBridge()
            self.create_subscription(
                ImageMsg,   '/camera/camera/color/image_raw',                    self._cb_image,       10)
            self.create_subscription(
                ImageMsg,   '/camera/camera/aligned_depth_to_color/image_raw',   self._cb_depth,       10)
            self.create_subscription(
                CameraInfo, '/camera/camera/color/camera_info',                  self._cb_camera_info, 10)

        if _YOLO_AVAILABLE and _CV_AVAILABLE:
            try:
                self._room_model = UltralyticsYOLO(ROOM_MODEL_PATH)
                self.get_logger().info('호수 YOLO 모델 로드 완료')
            except Exception as e:
                self.get_logger().warn(f'호수 YOLO 로드 실패: {e}')
            try:
                self._handle_model = UltralyticsYOLO(HANDLE_MODEL_PATH)
                self.get_logger().info('핸들 YOLO 모델 로드 완료')
            except Exception as e:
                self.get_logger().warn(f'핸들 YOLO 로드 실패: {e}')

        if _OCR_AVAILABLE and self._room_model is not None:
            self.get_logger().info('EasyOCR 초기화 중...')
            self._ocr = easyocr.Reader(['en'], gpu=False)
            self.get_logger().info('EasyOCR 초기화 완료')

        if _CV_AVAILABLE and self._room_model is not None:
            threading.Thread(target=self._display_loop, daemon=True).start()

        self.get_logger().info('arm_recover 노드 시작. /start_recover 대기 중...')
        self._home_timer = self.create_timer(2.0, self._init_home)

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def _cb_joints(self, msg):
        with self.lock:
            self.current_joints = msg

    def _cb_image(self, msg: ImageMsg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self._frame_lock:
            self._latest_frame = frame
        if self._ocr_active:
            self._process_room_sign(frame)
        if self._handle_active:
            self._detect_handle(frame)

    def _cb_depth(self, msg: ImageMsg):
        import numpy as _np
        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        with self.lock:
            self.depth_image = raw.astype(_np.float32) / 1000.0

    def _cb_camera_info(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx = msg.k[2]; self.cy = msg.k[5]

    def _process_room_sign(self, frame):
        if self._room_model is None:
            return
        self._frame_count += 1
        if self._frame_count % OCR_INTERVAL != 0:
            return
        results = self._room_model(frame, conf=ROOM_CONF, verbose=False)
        for box in results[0].boxes:
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = frame.shape[:2]
            # bbox 위쪽 40%만 크롭 — 숫자는 상단, 영어는 하단
            y_cut = y1 + int((y2 - y1) * 0.4)
            roi = frame[max(0, y1):min(h, y_cut), max(0, x1):min(w, x2)]
            if roi.size == 0:
                continue
            room_text = self._run_ocr(roi)
            if room_text:
                self.get_logger().info(f'호수 인식: {room_text} (conf={conf:.2f})')
                self._latest_room_text = room_text
                self._latest_room_bbox = (x1, y1, x2, y2)

    def _detect_handle(self, frame):
        if self._handle_model is None:
            return
        results = self._handle_model(frame, conf=HANDLE_CONF, verbose=False)[0]
        best_box, best_conf = None, 0.0
        for box in results.boxes:
            conf = float(box.conf)
            if conf > best_conf:
                best_conf = conf
                best_box  = box
        if best_box is None:
            return
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        self._latest_handle_bbox = (x1, y1, x2, y2)

        cx_px = (x1 + x2) // 2
        cy_px = (y1 + y2) // 2
        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None
        if depth is None:
            return
        h, w = depth.shape
        region = depth[max(0, cy_px-2):min(h, cy_px+3),
                       max(0, cx_px-2):min(w, cx_px+3)]
        valid = region[(region > 0.1) & ~np.isnan(region)]
        if len(valid) == 0:
            return
        d = float(np.median(valid))
        X = (cx_px - self.cx) / self.fx * d
        Y = (cy_px - self.cy) / self.fy * d
        Z = d
        self._latest_handle_xyz = (X, Y, Z)
        self.get_logger().info(
            f'handle_hole xyz=({X:.3f}, {Y:.3f}, {Z:.3f}) m (conf={best_conf:.2f})')

    def _run_ocr(self, roi) -> str | None:
        if self._ocr is None:
            return None
        results = self._ocr.readtext(roi, allowlist='0123456789', detail=0)
        text = ''.join(results).strip()
        m = re.match(r'\d+', text)
        if not m:
            return None
        digits = m.group()
        return digits if len(digits) >= 3 else None

    def _write_frame(self, frame):
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(self._record_path, fourcc, 20, (w, h))
        self._writer.write(frame)

    def _display_loop(self):
        while rclpy.ok():
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            vis = frame.copy()
            cv2.putText(vis, f'Step: {self._current_step_en}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if self._latest_room_bbox is not None:
                rx1, ry1, rx2, ry2 = self._latest_room_bbox
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
                cv2.putText(vis, f'Room: {self._latest_room_text}',
                            (rx1, max(ry1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            if self._latest_handle_bbox is not None:
                hx1, hy1, hx2, hy2 = self._latest_handle_bbox
                cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
                label = 'handle_hole'
                if self._latest_handle_xyz is not None:
                    X, Y, Z = self._latest_handle_xyz
                    label = f'handle_hole ({X:.3f}, {Y:.3f}, {Z:.3f})'
                cv2.putText(vis, label,
                            (hx1, max(hy1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self._write_frame(vis)
            cv2.imshow('Recover', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)

    def _cb_start_recover(self, msg: Bool):
        if not msg.data:
            return
        if self.state != IDLE:
            self.get_logger().warn(f'작업 중 ({self.state}). /start_recover 무시.')
            return
        self.get_logger().info('/start_recover 수신 → 회수 시작')
        self.state = RECOVER
        threading.Thread(target=self._run_recover_flow, daemon=True).start()

    # ─── 회수 흐름 ───────────────────────────────────────────────────────────

    def _run_recover_flow(self):
        self.get_logger().info('회수 시퀀스 시작')
        ok = self._run_sequence(RECOVER_STEPS, '회수')
        if not ok:
            self.get_logger().error('회수 실패')
            self.status_pub.publish(String(data='FAILED'))
            self.state = IDLE
            return

        self.get_logger().info('✅ 회수 완료')
        self.status_pub.publish(String(data='RECOVER_DONE'))
        self.recover_pub.publish(Bool(data=True))
        self.state = IDLE
        self.get_logger().info('✅ /start_recover 대기 중...')

    # ─── 시퀀스 실행 ─────────────────────────────────────────────────────────

    def _run_sequence(self, steps, name) -> bool:
        self.get_logger().info(f'{name} 시퀀스 시작 ({len(steps)}스텝)')
        for i, (label, joints) in enumerate(steps):
            self.get_logger().info(f'[{i+1}/{len(steps)}] {label}')
            self._current_step_en = f'[{i+1}/{len(steps)}] {label}'
            self._handle_active = (label == '오른쪽 확인')
            # OCR은 ROOM_SIGN_JOINTS 스텝에서만 활성
            if joints is ROOM_SIGN_JOINTS:
                self._frame_count = 0
            self._ocr_active = (joints is ROOM_SIGN_JOINTS)
            # 검출이 꺼진 스텝에서는 이전 bbox가 화면에 남지 않도록 클리어
            if not self._handle_active:
                self._latest_handle_bbox = None
                self._latest_handle_xyz  = None
            if not self._ocr_active:
                self._latest_room_bbox  = None
                self._latest_room_text  = None
            time.sleep(STEP_DELAY)

            if joints is not None:
                if not self.move_to_joints(joints, label):
                    self.get_logger().error(f'{label} 실패')
                    self._handle_active = False
                    return False

        self._handle_active = False
        self._ocr_active = False
        self._latest_room_bbox = None
        self._current_step_en = f'{name} Done'
        self.get_logger().info(f'{name} 시퀀스 완료')
        return True

    # ─── 팔 이동 ─────────────────────────────────────────────────────────────

    def move_to_joints(self, joints, label='') -> bool:
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('arm_controller 서버 없음!')
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

        traj, duration = make_trajectory(joints, current)
        if traj is None:
            self.get_logger().error(
                f'{label}: 단일 관절 이동량 {duration:.2f}rad 과대(>{MAX_JOINT_STEP}) '
                f'→ 위험 동작 차단, 이동 중단 (current={[round(c,3) for c in current]}, '
                f'target={[round(t,3) for t in joints]})')
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self.status_pub.publish(String(data='MOVING'))

        future = self._arm_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error('액션 수락 타임아웃')
                return False
            time.sleep(0.05)

        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('액션 거부됨')
            return False

        rf = gh.get_result_async()
        deadline = time.time() + duration + 5.0
        while not rf.done():
            if time.time() > deadline:
                self.get_logger().error('실행 타임아웃')
                return False
            time.sleep(0.1)

        ok = (rf.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL)
        if not ok:
            self.get_logger().error(f'{label} error_code={rf.result().result.error_code}')
        return ok

    # ─── 초기 홈 ─────────────────────────────────────────────────────────────

    def _init_home(self):
        self._home_timer.cancel()
        threading.Thread(target=self.move_to_joints,
                         args=(ELEVATOR_HOME_JOINTS, 'init_home'), daemon=True).start()


# ─── 엔트리포인트 ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ArmRecoverNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if node._writer is not None:
            node._writer.release()
        if _CV_AVAILABLE:
            import cv2 as _cv2
            _cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
