import math
import os
import threading
import time

import cv2
import easyocr
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ultralytics import YOLO


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(_REPO_ROOT, 'yolo', 'weights', 'best_room.pt')
CONFIDENCE   = 0.83
OCR_INTERVAL = 5  # 매 N프레임마다 OCR 실행

JOINT_NAMES      = ['joint1', 'joint2', 'joint3', 'joint4']
MOVE_SPEED       = 0.5
MIN_DURATION     = 2.0
ROOM_SIGN_JOINTS = [-3.141, -2.0203, 1.5002, 0.502]


def _shortest_path(target: float, current: float) -> float:
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff


def make_trajectory(target_joints: list, current_joints: list):
    target_joints = [_shortest_path(t, c) for t, c in zip(target_joints, current_joints)]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
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


class RoomSignDetector(Node):
    def __init__(self):
        super().__init__('room_sign_detector')
        self.bridge = CvBridge()
        self.latest_frame   = None
        self.current_joints = None
        self.lock = threading.Lock()
        self.frame_count = 0

        self.model = YOLO(MODEL_PATH)
        self.get_logger().info('YOLO 모델 로드 완료')

        self.get_logger().info('EasyOCR 초기화 중...')
        self.ocr = easyocr.Reader(['en'], gpu=False)
        self.get_logger().info('EasyOCR 초기화 완료')

        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')
        self.status_pub   = self.create_publisher(String, '/robot_status',   10)
        self.room_num_pub = self.create_publisher(String, '/room_number',    10)

        self.create_subscription(Image,      '/camera/camera/color/image_raw', self._cb_image,  10)
        self.create_subscription(JointState, '/joint_states',                  self._cb_joints, 10)

        self._move_timer = self.create_timer(2.0, self._move_once)
        self.get_logger().info('실시간 감지 시작! q: 종료')

    def _cb_image(self, msg):
        with self.lock:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _cb_joints(self, msg):
        with self.lock:
            self.current_joints = msg

    def _move_once(self):
        self._move_timer.cancel()
        threading.Thread(target=self._move_to_room_sign, daemon=True).start()

    def _move_to_room_sign(self):
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('arm_controller 서버 없음!')
            return

        with self.lock:
            js = self.current_joints
        current = [0.0] * 4
        if js is not None:
            for i, name in enumerate(JOINT_NAMES):
                if name in js.name:
                    current[i] = js.position[js.name.index(name)]

        traj, duration = make_trajectory(ROOM_SIGN_JOINTS, current)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.status_pub.publish(String(data='MOVING'))
        future = self._arm_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error('액션 수락 타임아웃')
                return
            time.sleep(0.05)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('액션 거부됨')
            return

        result_future = goal_handle.get_result_async()
        deadline = time.time() + duration + 5.0
        while not result_future.done():
            if time.time() > deadline:
                self.get_logger().error('이동 타임아웃')
                return
            time.sleep(0.1)

        self.get_logger().info('✅ ROOM_SIGN_JOINTS 이동 완료')

    def _run_ocr(self, frame, x1, y1, x2, y2) -> str | None:
        """bbox ROI에서 EasyOCR로 호수 읽기."""
        h, w = frame.shape[:2]
        roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if roi.size == 0:
            return None
        results = self.ocr.readtext(roi, allowlist='0123456789BbGg', detail=0)
        text = ''.join(results).strip()
        return text if text else None

    def run(self):
        while rclpy.ok():
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                continue

            self.frame_count += 1
            results  = self.model(frame, conf=CONFIDENCE, device=0, verbose=False)
            annotated = results[0].plot()

            for box in results[0].boxes:
                conf = float(box.conf)
                cls  = int(box.cls)
                name = self.model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                room_text = None
                if self.frame_count % OCR_INTERVAL == 0:
                    room_text = self._run_ocr(frame, x1, y1, x2, y2)
                    if room_text:
                        self.get_logger().info(f'호수 인식: {room_text} (YOLO: {name} {conf:.2f})')
                        self.room_num_pub.publish(String(data=room_text))
                    else:
                        self.get_logger().info(f'감지: {name} ({conf:.2f}) — OCR 실패')

                label = f'{name} {conf:.2f}'
                if room_text:
                    label += f' → {room_text}호'
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Room Sign Detector', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = RoomSignDetector()

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    node.run()


if __name__ == '__main__':
    main()
