"""
홈 자세 → 룸사인 자세로 이동 후 YOLO+OCR로 호수 번호판을 실시간 인식.

실행:
  ros2 run elevator_button_robot detect_room_sign
  python3 nodes/real_robot/detect_room_sign.py
"""

import math
import os
import re
import threading
import time

import cv2
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

try:
    import easyocr
    _OCR_OK = True
except ImportError:
    _OCR_OK = False

# ─── 경로 / 파라미터 ──────────────────────────────────────────────────────────

_REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH     = os.path.join(_REPO_ROOT, 'yolo', 'weights', 'best_room.pt')
ROOM_CONF      = 0.6
OCR_INTERVAL   = 5

JOINT_NAMES      = ['joint1', 'joint2', 'joint3', 'joint4']
MOVE_SPEED       = 0.5
MIN_DURATION     = 2.0
MAX_JOINT_STEP   = 4.5

HOME_JOINTS      = [3.141,  -1.3963,  1.2217,  0.5236]
ROOM_SIGN_JOINTS = [1.571,  -2.0203,  1.5002, -0.044]


# ─── 헬퍼 ────────────────────────────────────────────────────────────────────

def _shortest_path(target: float, current: float) -> float:
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff


def make_trajectory(target_joints: list, current_joints: list):
    target_joints = [_shortest_path(t, c) for t, c in zip(target_joints, current_joints)]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
    if max_disp > MAX_JOINT_STEP:
        return None, max_disp
    duration = max(max_disp / MOVE_SPEED, MIN_DURATION)

    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES
    pt = JointTrajectoryPoint()
    pt.positions   = target_joints
    pt.velocities  = [0.0] * 4
    secs  = int(duration)
    nsecs = int((duration - secs) * 1e9)
    pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
    traj.points.append(pt)
    return traj, duration


# ─── 노드 ────────────────────────────────────────────────────────────────────

class RoomSignDetector(Node):
    def __init__(self):
        super().__init__('room_sign_detector')
        self.bridge         = CvBridge()
        self.latest_frame   = None
        self.current_joints = None
        self.frame_count      = 0
        self._latest_room_text = None  # 마지막 인식 결과 캐시
        self._lock            = threading.Lock()

        if _YOLO_OK:
            self.model = YOLO(MODEL_PATH)
            self.get_logger().info('YOLO 모델 로드 완료')
        else:
            self.model = None
            self.get_logger().warn('ultralytics 없음 — YOLO 비활성')

        if _OCR_OK:
            self.get_logger().info('EasyOCR 초기화 중...')
            self.ocr = easyocr.Reader(['en'], gpu=False)
            self.get_logger().info('EasyOCR 초기화 완료')
        else:
            self.ocr = None
            self.get_logger().warn('easyocr 없음 — OCR 비활성')

        self._arm_client  = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.status_pub   = self.create_publisher(String, '/robot_status', 10)
        self.room_num_pub = self.create_publisher(String, '/room_number',  10)

        self.create_subscription(Image,      '/camera/camera/color/image_raw', self._cb_image,  10)
        self.create_subscription(JointState, '/joint_states',                  self._cb_joints, 10)

        threading.Thread(target=self._move_sequence, daemon=True).start()

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def _cb_image(self, msg):
        with self._lock:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _cb_joints(self, msg):
        with self._lock:
            self.current_joints = msg

    # ─── 팔 이동 ─────────────────────────────────────────────────────────────

    def _move_to(self, joints, label) -> bool:
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('arm_controller 서버 없음')
            return False

        with self._lock:
            js = self.current_joints
        if js is None:
            self.get_logger().error('/joint_states 미수신')
            return False

        current = [None] * 4
        for i, name in enumerate(JOINT_NAMES):
            if name in js.name:
                current[i] = js.position[js.name.index(name)]
        if any(c is None for c in current):
            self.get_logger().error('관절 정보 누락')
            return False

        traj, dur = make_trajectory(joints, current)
        if traj is None:
            self.get_logger().error(f'{label}: 이동량 {dur:.2f}rad 과대 → 차단')
            return False

        self.status_pub.publish(String(data='MOVING'))
        future = self._arm_client.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj))
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
        deadline = time.time() + dur + 5.0
        while not rf.done():
            if time.time() > deadline:
                self.get_logger().error('이동 타임아웃')
                return False
            time.sleep(0.1)

        ok = (rf.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL)
        if ok:
            self.get_logger().info(f'✅ {label} 완료')
        else:
            self.get_logger().error(f'{label} error_code={rf.result().result.error_code}')
        return ok

    def _move_sequence(self):
        time.sleep(1.0)  # joint_states 수신 대기
        if not self._move_to(HOME_JOINTS, '홈'):
            return
        if not self._move_to(ROOM_SIGN_JOINTS, '룸사인 자세'):
            return
        self.get_logger().info('룸사인 인식 시작 (q: 종료)')

    # ─── OCR ─────────────────────────────────────────────────────────────────

    def _run_ocr(self, roi) -> str | None:
        if self.ocr is None or roi.size == 0:
            return None
        results = self.ocr.readtext(roi, allowlist='0123456789', detail=0)
        text = ''.join(results).strip()
        # 앞에서 연속으로 이어지는 숫자만 추출 (ex. "531xx9" → "531")
        m = re.match(r'\d+', text)
        if not m:
            return None
        digits = m.group()
        # 3자리 미만이면 노이즈로 판단
        return digits if len(digits) >= 3 else None

    # ─── 메인 루프 ────────────────────────────────────────────────────────────

    def run(self):
        while rclpy.ok():
            with self._lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                time.sleep(0.05)
                continue

            if self.model is None:
                cv2.imshow('Room Sign Detector', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.05)
                continue

            self.frame_count += 1
            results  = self.model(frame, conf=ROOM_CONF, verbose=False)
            vis      = frame.copy()

            for box in results[0].boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if self.frame_count % OCR_INTERVAL == 0:
                    h, w = frame.shape[:2]
                    # bbox 위쪽 40%만 크롭 — 숫자는 상단, 영어는 하단
                    y_cut = y1 + int((y2 - y1) * 0.4)
                    roi = frame[max(0, y1):min(h, y_cut), max(0, x1):min(w, x2)]
                    result = self._run_ocr(roi)
                    if result:
                        self._latest_room_text = result
                        self.get_logger().info(f'호수 인식: {result} (conf={conf:.2f})')
                        self.room_num_pub.publish(String(data=result))

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if self._latest_room_text:
                    cv2.putText(vis, self._latest_room_text, (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Room Sign Detector', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)

        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = RoomSignDetector()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    node.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
