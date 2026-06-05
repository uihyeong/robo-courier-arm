"""
배달 노드 (arm_delivery.py).

픽업 → 호수 번호판 인식 → 배달 시퀀스를 토픽 트리거로 자동 실행.

상태 전이:
  IDLE
    → /start_pickup 수신
  PICKUP  — 픽업 시퀀스 (9스텝)
    → /pickup_done 발행
  ROOM_SIGN — ROOM_SIGN_JOINTS 이동 + YOLO+OCR 루프
    → /room_number 발행 (인식마다)
  WAITING_ALIGN — /aligned_ready 수신 대기
    → /aligned_ready 수신
  DELIVER — 배달 시퀀스 (12스텝)
    → /delivery_done 발행
  DONE → IDLE

토픽:
  구독: /start_pickup  (Bool)   — 픽업 시작 트리거
  구독: /aligned_ready (Bool)   — Scout 목적지 정렬 완료 → 배달 시작
  발행: /pickup_done   (Bool)   — 픽업 완료
  발행: /room_number   (String) — 인식된 호수 (예: "529")
  발행: /delivery_done (Bool)   — 배달 완료
  발행: /robot_status  (String) — MOVING/PICKUP_DONE/DELIVERY_DONE/FAILED

실행:
  ros2 launch open_manipulator_x_bringup hardware.launch.py
  ros2 launch realsense2_camera rs_launch.py
  ros2 run courier_arm arm_delivery
  ros2 topic pub --once /start_pickup std_msgs/Bool "{data: true}"
"""

import math
import os
import threading
import time

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
import tf2_geometry_msgs
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, JointState
from std_msgs.msg import Bool, String

from courier_arm.ik import solve_ik, make_trajectory, JOINT_NAMES

try:
    import cv2
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

_REPO_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOM_MODEL_PATH  = os.path.join(_REPO_ROOT, 'yolo', 'weights', 'best_room.pt')
ROOM_CONF        = 0.83
OCR_INTERVAL     = 5

YOLO_MODEL_PATH   = 'yolov8n.pt'
YOLO_CONF         = 0.15
GRAB_HOVER_OFFSET = 0.04
DETECT_Z_OFFSET   = -0.02
DETECT_Y_OFFSET   = -0.05

HIGHLIGHT_CLASSES = {
    'suitcase', 'backpack', 'handbag',
    'book', 'bottle', 'cup', 'bowl', 'box',
    'refrigerator',
}

# ─── 관절 상수 ────────────────────────────────────────────────────────────────

HOME_JOINTS         = [3.141,  -1.3963,  1.2217,  0.5236]
TABLE_LOOK_JOINTS   = [1.571,  -1.3963,  1.2217,  0.5236]
BASKET_LOOK_JOINTS  = [-3.116, -0.387,   0.755,   1.164]
BASKET_PLACE_JOINTS = [3.1032,  0.00767, 1.41126, -1.41433]
BASKET_GRIP_JOINTS  = [3.122,   0.457,   0.831,   0.305]
ROOM_SIGN_JOINTS    = [-3.141, -2.0203,  1.5002,  -0.044]

GRIPPER_OPEN  = [0.020]
GRIPPER_CLOSE = [0.000]

MOVE_SPEED   = 0.4
MIN_DURATION = 2.0
STEP_DELAY   = 1.5

# ─── 웨이포인트 ───────────────────────────────────────────────────────────────

TABLE_HOVER  = ( 0.013,  0.298,  0.100)
TABLE_GRIP   = ( 0.013,  0.298,  0.040)
BASKET_HOVER = (-0.165,  0.009,  0.123)
DEST_HOVER   = ( 0.013,  0.298,  0.100)
DEST_PLACE   = ( 0.013,  0.298,  0.040)

# ─── AUTO_GRAB sentinel ───────────────────────────────────────────────────────

class _AutoGrab:
    """YOLO 감지 → IK → 잡기 자동화. joints 필드에 넣어 사용."""
    def __init__(self, fallback_xyz):
        self.fallback_xyz = fallback_xyz

AUTO_GRAB_TABLE  = _AutoGrab(TABLE_GRIP)
AUTO_GRAB_BASKET = _AutoGrab(BASKET_HOVER)

# ─── 시퀀스 정의 ─────────────────────────────────────────────────────────────

PICKUP_STEPS = [
    ('홈',                             HOME_JOINTS,         None,         None),
    ('책상 방향 확인 (joint1 오른쪽)',   TABLE_LOOK_JOINTS,   None,         None),
    ('그리퍼 열기 (접근 전)',            None,                None,         GRIPPER_OPEN),
    ('박스 위 호버',                     None,                TABLE_HOVER,  None),
    ('박스 잡기 위치',                   None,                TABLE_GRIP,   None),
    ('그리퍼 닫기 (잡기)',               None,                None,         GRIPPER_CLOSE),
    ('바구니에 내려놓기',                BASKET_PLACE_JOINTS, None,         None),
    ('그리퍼 열기 (박스 놓기)',          None,                None,         GRIPPER_OPEN),
    ('홈 복귀',                          HOME_JOINTS,         None,         None),
]

DELIVER_STEPS = [
    ('홈',                              HOME_JOINTS,         None,         None),
    ('바구니 확인 (joint4 틸트)',         BASKET_LOOK_JOINTS,  None,         None),
    ('그리퍼 열기 (접근 전)',             None,                None,         GRIPPER_OPEN),
    ('바구니 박스 잡기',                  BASKET_GRIP_JOINTS,  None,         None),
    ('그리퍼 닫기 (잡기)',                None,                None,         GRIPPER_CLOSE),
    ('박스 들어올리기',                   None,                BASKET_HOVER, None),
    ('목적지 방향 확인 (joint1 오른쪽)',  TABLE_LOOK_JOINTS,   None,         None),
    ('목적지 책상 위 호버',               None,                DEST_HOVER,   None),
    ('목적지에 내려놓기',                 None,                DEST_PLACE,   None),
    ('그리퍼 열기 (박스 놓기)',           None,                None,         GRIPPER_OPEN),
    ('위로 호버',                         None,                DEST_HOVER,   None),
    ('홈 복귀',                           HOME_JOINTS,         None,         None),
]

# ─── 상태 상수 ────────────────────────────────────────────────────────────────

IDLE           = 'IDLE'
PICKUP         = 'PICKUP'
ROOM_SIGN      = 'ROOM_SIGN'
WAITING_ALIGN  = 'WAITING_ALIGN'
DELIVER        = 'DELIVER'
DONE           = 'DONE'


# ─── 노드 ────────────────────────────────────────────────────────────────────

class ArmDeliveryNode(Node):

    def __init__(self):
        super().__init__('arm_delivery')
        self.lock           = threading.Lock()
        self.current_joints = None
        self.state          = IDLE

        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')
        self._gripper_client = ActionClient(
            self, GripperCommand,
            '/gripper_controller/gripper_cmd')

        self.status_pub   = self.create_publisher(String, '/robot_status',  10)
        self.pickup_pub   = self.create_publisher(Bool,   '/pickup_done',   10)
        self.room_pub     = self.create_publisher(String, '/room_number',   10)
        self.delivery_pub = self.create_publisher(Bool,   '/delivery_done', 10)

        self.create_subscription(JointState, '/joint_states',   self._cb_joints,        10)
        self.create_subscription(Bool,       '/start_pickup',   self._cb_start_pickup,  10)
        self.create_subscription(Bool,       '/aligned_ready',  self._cb_aligned_ready, 10)

        self._aligned_ready_event = threading.Event()

        self.bridge         = None
        self._latest_frame  = None
        self._frame_lock    = threading.Lock()
        self._frame_count   = 0
        self.depth_image    = None
        self.fx, self.fy    = 615.0, 615.0
        self.cx, self.cy    = 320.0, 240.0

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._grab_model      = None
        self._current_step_en = 'Waiting'

        self._room_model    = None
        self._ocr           = None
        self._ocr_active    = False

        if _CV_AVAILABLE:
            self.bridge = CvBridge()

        if _YOLO_AVAILABLE and _CV_AVAILABLE:
            try:
                self._grab_model = UltralyticsYOLO(YOLO_MODEL_PATH)
                self.get_logger().info(f'박스 YOLO 모델 로드 완료 ({YOLO_MODEL_PATH})')
            except Exception as e:
                self.get_logger().warn(f'박스 YOLO 로드 실패: {e}')

            try:
                self._room_model = UltralyticsYOLO(ROOM_MODEL_PATH)
                self.get_logger().info('호수 YOLO 모델 로드 완료')
            except Exception as e:
                self.get_logger().warn(f'호수 YOLO 로드 실패: {e}')

        if _OCR_AVAILABLE and self._room_model is not None:
            self.get_logger().info('EasyOCR 초기화 중...')
            self._ocr = easyocr.Reader(['en'], gpu=False)
            self.get_logger().info('EasyOCR 초기화 완료')

        if _CV_AVAILABLE:
            self.create_subscription(
                ImageMsg, '/camera/camera/color/image_raw', self._cb_image, 10)
            self.create_subscription(
                ImageMsg, '/camera/camera/aligned_depth_to_color/image_raw', self._cb_depth, 10)
            self.create_subscription(
                CameraInfo, '/camera/camera/color/camera_info', self._cb_camera_info, 10)
            if self._grab_model is not None:
                threading.Thread(target=self._yolo_display_loop, daemon=True).start()

        self.get_logger().info('arm_delivery 노드 시작. /start_pickup 대기 중...')
        self._home_timer = self.create_timer(2.0, self._init_home)

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def _cb_joints(self, msg):
        with self.lock:
            self.current_joints = msg

    def _cb_start_pickup(self, msg: Bool):
        if not msg.data:
            return
        if self.state != IDLE:
            self.get_logger().warn(f'작업 중 ({self.state}). /start_pickup 무시.')
            return
        self.get_logger().info('/start_pickup 수신 → 픽업 시작')
        self.state = PICKUP
        threading.Thread(target=self._run_pickup_flow, daemon=True).start()

    def _cb_aligned_ready(self, msg: Bool):
        if not msg.data:
            return
        if self.state == WAITING_ALIGN:
            self.get_logger().info('/aligned_ready 수신 → 배달 시작')
            self._aligned_ready_event.set()
        else:
            self.get_logger().warn(f'/aligned_ready 무시 (state={self.state})')

    def _cb_image(self, msg: ImageMsg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self._frame_lock:
            self._latest_frame = frame

        if self._ocr_active:
            self._process_room_sign(frame)

    def _cb_depth(self, msg: ImageMsg):
        with self.lock:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.depth_image = raw.astype(np.float32) / 1000.0

    def _cb_camera_info(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx = msg.k[2]; self.cy = msg.k[5]

    # ─── YOLO 시각화 루프 ────────────────────────────────────────────────────

    def _yolo_display_loop(self):
        while rclpy.ok():
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.05)
                continue

            results = self._grab_model(frame, conf=YOLO_CONF, verbose=False)[0]
            vis = frame.copy()
            for box in results.boxes:
                cls_name = results.names[int(box.cls)]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color         = (0, 255, 0) if cls_name in HIGHLIGHT_CLASSES else (160, 160, 160)
                display_label = 'box' if cls_name in HIGHLIGHT_CLASSES else cls_name
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, display_label,
                            (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(vis, f'Step: {self._current_step_en}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('Delivery YOLO', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)

    # ─── AUTO_GRAB: YOLO 감지 → 3D 변환 → IK → 잡기 ─────────────────────────

    def _detect_and_grab(self, fallback_xyz) -> bool:
        if self._grab_model is None:
            return self._fallback_grab(fallback_xyz)

        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            self.get_logger().warn('프레임 없음 → 폴백')
            return self._fallback_grab(fallback_xyz)

        results = self._grab_model(frame, conf=YOLO_CONF, verbose=False)[0]
        best_box, best_conf = None, 0.0
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            conf     = float(box.conf)
            if cls_name in HIGHLIGHT_CLASSES and conf > best_conf:
                best_conf = conf
                best_box  = box

        if best_box is None:
            self.get_logger().warn('박스 감지 실패 → 폴백')
            return self._fallback_grab(fallback_xyz)

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cx_px = (x1 + x2) // 2
        cy_px = y1 + int((y2 - y1) * 0.75)
        cls_name = results.names[int(best_box.cls)]
        self.get_logger().info(f'감지: {cls_name} {best_conf:.2f}  pixel=({cx_px},{cy_px})')

        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None
        if depth is None:
            self.get_logger().warn('뎁스 없음 → 폴백')
            return self._fallback_grab(fallback_xyz)

        h, w = depth.shape
        region = depth[max(0, cy_px-2):min(h, cy_px+3),
                       max(0, cx_px-2):min(w, cx_px+3)]
        valid = region[(region > 0.1) & ~np.isnan(region)]
        if len(valid) == 0:
            self.get_logger().warn('뎁스값 없음 → 폴백')
            return self._fallback_grab(fallback_xyz)
        d = float(np.median(valid))

        X_cam = (cx_px - self.cx) / self.fx * d
        Y_cam = (cy_px - self.cy) / self.fy * d

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
            self.get_logger().warn(f'TF 변환 실패: {e} → 폴백')
            return self._fallback_grab(fallback_xyz)

        Z += DETECT_Z_OFFSET
        Y += DETECT_Y_OFFSET
        self.get_logger().info(f'world=({X:.3f},{Y:.3f},{Z:.3f})  depth={d:.3f}m')

        ok = self.move_to_xyz(X, Y, Z + GRAB_HOVER_OFFSET, label='감지 호버')
        if not ok:
            self.get_logger().warn('호버 IK 실패 → 폴백')
            return self._fallback_grab(fallback_xyz)

        self.send_gripper(GRIPPER_CLOSE)
        time.sleep(0.3)
        return True

    def _fallback_grab(self, xyz) -> bool:
        self.get_logger().info(f'폴백 좌표: {xyz}')
        ok = self.move_to_xyz(*xyz, label='폴백')
        if ok:
            self.send_gripper(GRIPPER_CLOSE)
            time.sleep(0.3)
        return ok

    # ─── 호수 번호판 인식 ─────────────────────────────────────────────────────

    def _process_room_sign(self, frame):
        if self._room_model is None:
            return

        self._frame_count += 1
        run_ocr = (self._frame_count % OCR_INTERVAL == 0)
        if not run_ocr:
            return

        results = self._room_model(frame, conf=ROOM_CONF, verbose=False)
        for box in results[0].boxes:
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = frame.shape[:2]
            roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if roi.size == 0:
                continue

            room_text = self._run_ocr(roi)
            if room_text:
                self.get_logger().info(f'호수 인식: {room_text} (conf={conf:.2f})')
                self.room_pub.publish(String(data=room_text))

    def _run_ocr(self, roi) -> str | None:
        if self._ocr is None:
            return None
        results = self._ocr.readtext(roi, allowlist='0123456789BbGg', detail=0)
        text = ''.join(results).strip()
        return text if text else None

    # ─── 전체 흐름 ────────────────────────────────────────────────────────────

    def _run_pickup_flow(self):
        self.get_logger().info('픽업 시퀀스 시작')
        ok = self._run_sequence(PICKUP_STEPS, '픽업')
        if not ok:
            self.get_logger().error('픽업 실패')
            self.status_pub.publish(String(data='FAILED'))
            self.state = IDLE
            return

        self.get_logger().info('✅ 픽업 완료')
        self.status_pub.publish(String(data='PICKUP_DONE'))
        self.pickup_pub.publish(Bool(data=True))

        self.state = ROOM_SIGN
        self.get_logger().info('호수 번호판 인식 시작 → ROOM_SIGN_JOINTS 이동')
        self.move_to_joints(ROOM_SIGN_JOINTS, 'room_sign')

        self._frame_count = 0
        self._ocr_active  = True

        self.state = WAITING_ALIGN
        self.get_logger().info('OCR 실행 중. /aligned_ready 대기...')
        self._aligned_ready_event.clear()
        self._aligned_ready_event.wait()
        self._ocr_active = False

        self.state = DELIVER
        self.get_logger().info('배달 시퀀스 시작')
        ok = self._run_sequence(DELIVER_STEPS, '배달')
        if not ok:
            self.get_logger().error('배달 실패')
            self.status_pub.publish(String(data='FAILED'))
            self.state = IDLE
            return

        self.get_logger().info('✅ 배달 완료')
        self.status_pub.publish(String(data='DELIVERY_DONE'))
        self.delivery_pub.publish(Bool(data=True))

        self.state = IDLE
        self.get_logger().info('✅ 전체 배달 시퀀스 완료. /start_pickup 대기 중...')

    # ─── 시퀀스 실행 ─────────────────────────────────────────────────────────

    def _run_sequence(self, steps, name) -> bool:
        self.get_logger().info(f'{name} 시퀀스 시작 ({len(steps)}스텝)')
        for i, (label, joints, xyz, gripper) in enumerate(steps):
            self.get_logger().info(f'[{i+1}/{len(steps)}] {label}')
            self._current_step_en = f'[{i+1}/{len(steps)}] {label}'
            time.sleep(STEP_DELAY)

            if gripper is not None:
                self.send_gripper(gripper)

            if isinstance(joints, _AutoGrab):
                if not self._detect_and_grab(joints.fallback_xyz):
                    self.get_logger().error(f'{label} 실패')
                    return False
            elif joints is not None:
                if not self.move_to_joints(joints, label):
                    self.get_logger().error(f'{label} 실패')
                    return False
            elif xyz is not None:
                if solve_ik(*xyz) is None:
                    self.get_logger().error(f'IK 불가 {xyz} → 스킵')
                else:
                    if not self.move_to_xyz(*xyz, label=label):
                        self.get_logger().error(f'{label} 실패')
                        return False

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
        current = [0.0] * 4
        if js:
            for i, name in enumerate(JOINT_NAMES):
                if name in js.name:
                    current[i] = js.position[js.name.index(name)]

        traj, duration = make_trajectory(joints, current, MOVE_SPEED)
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

    def move_to_xyz(self, X, Y, Z, label='') -> bool:
        joints = solve_ik(X, Y, Z)
        if joints is None:
            self.get_logger().error(f'IK 해 없음: ({X:.3f}, {Y:.3f}, {Z:.3f})')
            return False
        return self.move_to_joints(joints, label)

    # ─── 그리퍼 ──────────────────────────────────────────────────────────────

    def send_gripper(self, position: list) -> bool:
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('gripper_controller 서버 없음!')
            return False

        goal = GripperCommand.Goal()
        goal.command.position   = position[0]
        goal.command.max_effort = 50.0 if position == GRIPPER_OPEN else 10.0

        future = self._gripper_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error('그리퍼 수락 타임아웃')
                return False
            time.sleep(0.05)

        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('그리퍼 액션 거부됨')
            return False

        rf = gh.get_result_async()
        deadline = time.time() + 5.0
        while not rf.done():
            if time.time() > deadline:
                self.get_logger().error('그리퍼 실행 타임아웃')
                return False
            time.sleep(0.05)

        return True

    # ─── 초기 홈 ─────────────────────────────────────────────────────────────

    def _init_home(self):
        self._home_timer.cancel()
        threading.Thread(target=self.move_to_joints,
                         args=(HOME_JOINTS, 'init_home'), daemon=True).start()


# ─── 엔트리포인트 ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ArmDeliveryNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if _CV_AVAILABLE:
            import cv2 as _cv2
            _cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
