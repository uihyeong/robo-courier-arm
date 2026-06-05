"""
배달 모션 데모 테스트.

시퀀스:
  [픽업]
    1. 홈
    2. 그리퍼 열기
    3. 책상 방향 확인 (TABLE_LOOK_JOINTS — joint1 오른쪽 90°)
    4. [자동] YOLO 박스 감지 → IK → 하강 → 그리퍼 닫기
    5. 박스 들어올리기 (TABLE_LOOK_JOINTS — 오른쪽 유지)
    6. 바구니에 내려놓기
    7. 그리퍼 열기
    8. 바구니 위 후퇴
    9. 홈 복귀

  [Scout Mini 이동]  ← 외부 단계

  [배달]
    1. 홈
    2. 그리퍼 열기
    3. 바구니 위 이동
    4. 바구니 확인 (BASKET_LOOK_JOINTS — joint4 틸트)
    5. [자동] YOLO 박스 감지 → IK → 하강 → 그리퍼 닫기
    6. 박스 들어올리기
    7. 목적지 책상 위 호버
    8. 목적지에 내려놓기
    9. 그리퍼 열기
   10. 목적지 위 후퇴
   11. 홈 복귀

실행:
  ros2 launch open_manipulator_x_bringup hardware.launch.py
  ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
  ros2 run tf2_ros static_transform_publisher --x 0.12 --y 0.01 --z 0.062 \\
      --roll 0 --pitch 0 --yaw 0 --frame-id link5 --child-frame-id camera_link
  python3 nodes/real_robot/test_delivery_motion.py

조작:
  Enter → 다음 단계
  q     → 즉시 종료 (홈 복귀 후)
  r     → 처음부터 다시
"""

import math
import threading
import time

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
import tf2_geometry_msgs
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ─── YOLO / 카메라 (선택적 임포트) ───────────────────────────────────────────

try:
    from ultralytics import YOLO as UltralyticsYOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

try:
    import cv2
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image as ImageMsg
    _CAMERA_AVAILABLE = True
except ImportError:
    _CAMERA_AVAILABLE = False

YOLO_MODEL_PATH = 'yolov8n.pt'
YOLO_CONF         = 0.15   # 낮을수록 더 잘 감지 (오탐 증가)
GRAB_HOVER_OFFSET = 0.04   # 감지 지점 위 4cm 호버 후 하강
DETECT_Z_OFFSET   = -0.02  # Z 보정: 팔이 위 → 음수↑, 아래(바닥 끌림) → 양수↑
DETECT_Y_OFFSET   = -0.05  # Y 보정: 너무 멀리 감 → 음수로 줄임

# 감지 대상 클래스 (초록 강조)
HIGHLIGHT_CLASSES = {
    'suitcase', 'backpack', 'handbag',
    'book', 'bottle', 'cup', 'bowl', 'box',
    'refrigerator',  # COCO에 box 없음 → 직육면체 박스가 refrigerator로 오인식됨
}

# ─── 링크 파라미터 ────────────────────────────────────────────────────────────

L1    = 0.0595
L2    = math.sqrt(0.024**2 + 0.128**2)
ALPHA = math.atan2(0.128, 0.024)
L3    = 0.124
L4    = 0.126

JOINT_LIMITS = [
    (-math.pi, math.pi),
    (-1.5,     1.5),
    (-1.5,     1.4),
    (-1.7,     1.97),
]

JOINT_NAMES   = ['joint1', 'joint2', 'joint3', 'joint4']
HOME_JOINTS         = [3.141,  -1.3963,  1.2217,  0.5236]
BASKET_LOOK_JOINTS  = [-3.116, -0.387,   0.755,   1.164]   # 바구니 확인용 — 실측 joint_states 2026-05-26
TABLE_LOOK_JOINTS   = [1.571,  -1.3963,  1.2217,  0.5236]  # 책상 확인용 (joint1 오른쪽 90°)
BASKET_PLACE_JOINTS = [3.1032,  0.00767, 1.41126, -1.41433]  # 픽업 바구니 내려놓기 — 실측 2026-05-26
BASKET_GRIP_JOINTS  = [3.122,   0.457,   0.831,   0.305]    # 배달 바구니에서 잡기 — 실측 2026-05-26
GRIPPER_OPEN  = [0.020]   # 실측 최대 0.040 rad
GRIPPER_CLOSE = [0.000]   # 살살 잡기 — 박스 찌그러짐 방지 (0.010 너무 헐렁, -0.010 너무 셈)
GRIPPER_REST  = [-0.005]  # 홈 자세용 닫힘 (스프링 방향, 튕김 없음)
MOVE_SPEED    = 0.4
MIN_DURATION  = 2.0

# ─── 웨이포인트 ───────────────────────────────────────────────────────────────

TABLE_HOVER  = ( 0.013,  0.298,  0.100)  # 실측 2026-05-26
TABLE_GRIP   = ( 0.013,  0.298,  0.040)  # 실측 2026-05-26 (AUTO_GRAB 폴백)

BASKET_HOVER = (-0.165,  0.009,  0.123)  # 실측 2026-05-26
BASKET_PLACE = (-0.165,  0.009,  0.063)  # 실측 2026-05-26 (AUTO_GRAB 폴백)

DEST_HOVER   = ( 0.013,  0.298,  0.100)  # 픽업 위치와 동일 (실측 2026-05-26)
DEST_PLACE   = ( 0.013,  0.298,  0.040)  # 픽업 위치와 동일 (실측 2026-05-26)

# ─── IK ──────────────────────────────────────────────────────────────────────

def solve_ik(X, Y, Z):
    j1 = math.atan2(Y, X)
    r  = math.sqrt(X**2 + Y**2)
    wr = r - L4
    dr = wr
    dz = Z - L1
    D  = math.sqrt(dr**2 + dz**2)
    if D > (L2 + L3) * 0.999 or D < abs(L2 - L3) * 1.001:
        return None
    c_psi = max(-1.0, min(1.0, (D**2 - L2**2 - L3**2) / (2.0 * L2 * L3)))
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

def _shortest_path(target, current):
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff

def make_trajectory(target_joints, current_joints):
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

# ─── AUTO_GRAB sentinel ───────────────────────────────────────────────────────

class _AutoGrab:
    """YOLO 감지 → IK → 잡기 자동화. joints 필드에 넣어 사용."""
    def __init__(self, fallback_xyz):
        self.fallback_xyz = fallback_xyz  # 감지 실패 시 폴백 좌표

# 픽업용 (책상 위 박스) / 배달용 (바구니 안 박스)
AUTO_GRAB_TABLE  = _AutoGrab(TABLE_GRIP)
AUTO_GRAB_BASKET = _AutoGrab(BASKET_PLACE)

# ─── 시퀀스 정의 ─────────────────────────────────────────────────────────────

# 스텝 형식: (한글 라벨, joints, xyz, gripper, 영어 라벨)
PICKUP_STEPS = [
    ('홈',                             HOME_JOINTS,        None,         None,         'Home'),
    ('책상 방향 확인 (joint1 오른쪽)',   TABLE_LOOK_JOINTS,  None,         None,         'Look at Table'),
    ('그리퍼 열기 (접근 전)',            None,               None,         GRIPPER_OPEN, 'Gripper Open'),
    ('박스 위 호버',                     None,               TABLE_HOVER,  None,         'Hover over Box'),
    ('박스 잡기 위치',                   None,               TABLE_GRIP,   None,         'Move to Grip'),
    ('그리퍼 닫기 (잡기)',               None,               None,         GRIPPER_CLOSE,'Grip Box'),
    ('바구니에 내려놓기',                BASKET_PLACE_JOINTS, None,        None,         'Place in Basket'),
    ('그리퍼 열기 (박스 놓기)',          None,               None,         GRIPPER_OPEN, 'Gripper Open'),
    ('홈 복귀',                          HOME_JOINTS,        None,         None,         'Home'),
]

DELIVER_STEPS = [
    ('홈',                              HOME_JOINTS,        None,         None,         'Home'),
    ('바구니 확인 (joint4 틸트)',         BASKET_LOOK_JOINTS, None,         None,         'Look into Basket'),
    ('그리퍼 열기 (접근 전)',             None,               None,         GRIPPER_OPEN, 'Gripper Open'),
    ('바구니 박스 잡기',                  BASKET_GRIP_JOINTS,  None,        None,         'Grip Box Position'),
    ('그리퍼 닫기 (잡기)',               None,               None,         GRIPPER_CLOSE,'Grip Box'),
    ('박스 들어올리기',                  None,               BASKET_HOVER, None,         'Lift Box'),
    ('목적지 방향 확인 (joint1 오른쪽)', TABLE_LOOK_JOINTS,  None,         None,         'Look at Dest'),
    ('목적지 책상 위 호버',              None,               DEST_HOVER,   None,         'Hover over Dest'),
    ('목적지에 내려놓기',                None,               DEST_PLACE,   None,         'Place at Dest'),
    ('그리퍼 열기 (박스 놓기)',          None,               None,         GRIPPER_OPEN, 'Gripper Open'),
    ('위로 호버',                        None,               DEST_HOVER,   None,         'Hover Up'),
    ('홈 복귀',                          HOME_JOINTS,        None,         None,         'Home'),
]

# ─── 테스트 노드 ──────────────────────────────────────────────────────────────

class DeliveryTestNode(Node):

    def __init__(self):
        super().__init__('test_delivery_motion')
        self.lock           = threading.Lock()
        self.current_joints = None

        # 액션 클라이언트
        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')
        self._gripper_client = ActionClient(
            self, GripperCommand,
            '/gripper_controller/gripper_cmd')

        # 관절 상태
        self.create_subscription(JointState, '/joint_states', self._cb_joints, 10)

        # 카메라 파라미터
        self.depth_image = None
        self.fx, self.fy = 615.0, 615.0
        self.cx, self.cy = 320.0, 240.0

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # YOLO 시각화 + 감지
        self._yolo_model   = None
        self._latest_frame = None
        self._frame_lock   = threading.Lock()
        self.bridge        = None
        self._current_step    = 'Waiting'
        self._current_step_en = 'Waiting'

        if _YOLO_AVAILABLE and _CAMERA_AVAILABLE:
            self.get_logger().info(f'YOLO 모델 로드 중: {YOLO_MODEL_PATH}')
            self._yolo_model = UltralyticsYOLO(YOLO_MODEL_PATH)
            self.bridge = CvBridge()
            self.create_subscription(
                ImageMsg, '/camera/camera/color/image_raw', self._cb_image, 10)
            self.create_subscription(
                # aligned_depth_to_color: 컬러와 동일 해상도(640×480)로 픽셀 1:1 대응
                # realsense 실행 시 align_depth.enable:=true 필요
                ImageMsg, '/camera/camera/aligned_depth_to_color/image_raw', self._cb_depth, 10)
            self.create_subscription(
                CameraInfo, '/camera/camera/color/camera_info', self._cb_camera_info, 10)
            threading.Thread(target=self._yolo_display_loop, daemon=True).start()
            self.get_logger().info('✅ YOLO 시각화 + AUTO_GRAB 활성화')
        else:
            self.get_logger().warn(
                'YOLO 비활성화 — AUTO_GRAB은 폴백 좌표로 동작')

        self.get_logger().info('테스트 노드 시작. 2초 후 홈 이동...')

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def _cb_joints(self, msg):
        with self.lock:
            self.current_joints = msg

    def _cb_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self._frame_lock:
            self._latest_frame = frame

    def _cb_depth(self, msg):
        with self.lock:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.depth_image = raw.astype(np.float32) / 1000.0

    def _cb_camera_info(self, msg):
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

            results = self._yolo_model(frame, conf=YOLO_CONF, verbose=False)[0]
            vis = frame.copy()
            for box in results.boxes:
                cls_name = results.names[int(box.cls)]
                conf     = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color        = (0, 255, 0) if cls_name in HIGHLIGHT_CLASSES else (160, 160, 160)
                display_label = 'box' if cls_name in HIGHLIGHT_CLASSES else cls_name
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, display_label,
                            (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(vis, f'Step: {self._current_step_en}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(vis, 'YOLOv8n  Delivery Demo',
                        (10, vis.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow('Delivery YOLO', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)

    # ─── AUTO_GRAB: YOLO 감지 → 3D 변환 → IK → 잡기 ─────────────────────────

    def _detect_and_grab(self, fallback_xyz) -> bool:
        """YOLO로 박스 감지 후 IK로 잡기. 실패 시 fallback_xyz로 폴백."""

        # YOLO / 카메라 미활성화 → 폴백
        if self._yolo_model is None:
            print('  ⚠️  YOLO 비활성화 → 폴백 좌표로 이동')
            return self._fallback_grab(fallback_xyz)

        # 최신 프레임 확보
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            print('  ⚠️  카메라 프레임 없음 → 폴백')
            return self._fallback_grab(fallback_xyz)

        # YOLO 추론
        results = self._yolo_model(frame, conf=YOLO_CONF, verbose=False)[0]
        best_box, best_conf = None, 0.0
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            conf     = float(box.conf)
            if cls_name in HIGHLIGHT_CLASSES and conf > best_conf:
                best_conf = conf
                best_box  = box

        if best_box is None:
            print('  ⚠️  박스 감지 실패 → 폴백')
            return self._fallback_grab(fallback_xyz)

        # 픽셀: X는 중심, Y는 바운딩박스 하단 75% 지점 (박스 아랫부분 높이 기준으로 잡기)
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cx_px = (x1 + x2) // 2
        cy_px = y1 + int((y2 - y1) * 0.75)   # 상단에서 75% 아래 = 박스 하단부
        cls_name = results.names[int(best_box.cls)]
        print(f'  감지: {cls_name} {best_conf:.2f}  pixel=({cx_px},{cy_px})  bbox=({x1},{y1},{x2},{y2})')

        # 뎁스
        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None
        if depth is None:
            print('  ⚠️  뎁스 없음 → 폴백')
            return self._fallback_grab(fallback_xyz)

        h, w = depth.shape
        region = depth[max(0, cy_px-2):min(h, cy_px+3),
                       max(0, cx_px-2):min(w, cx_px+3)]
        valid = region[(region > 0.1) & ~np.isnan(region)]
        if len(valid) == 0:
            print('  ⚠️  뎁스값 없음 → 폴백')
            return self._fallback_grab(fallback_xyz)
        d = float(np.median(valid))

        # 카메라 3D 좌표
        X_cam = (cx_px - self.cx) / self.fx * d
        Y_cam = (cy_px - self.cy) / self.fy * d

        # TF 변환 (camera → world)
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
            print(f'  ⚠️  TF 변환 실패: {e} → 폴백')
            return self._fallback_grab(fallback_xyz)

        Z += DETECT_Z_OFFSET   # 카메라 TF 오차 보정
        Y += DETECT_Y_OFFSET   # Y 거리 오차 보정
        print(f'  world=({X:.3f},{Y:.3f},{Z:.3f})  depth={d:.3f}m  (Z{DETECT_Z_OFFSET:+.3f} Y{DETECT_Y_OFFSET:+.3f})')

        # 호버 위치로 이동 → 바로 그리퍼 닫기 (하강 없음)
        ok = self.move_to_xyz(X, Y, Z + GRAB_HOVER_OFFSET, label='감지 호버')
        if not ok:
            print('  ⚠️  호버 IK 실패 → 폴백')
            return self._fallback_grab(fallback_xyz)

        print('  그리퍼 닫기 (박스 잡기)')
        self.send_gripper(GRIPPER_CLOSE)
        time.sleep(0.3)
        return True

    def _fallback_grab(self, xyz) -> bool:
        """폴백: 하드코딩 좌표로 이동 후 잡기."""
        print(f'  폴백 좌표: {xyz}')
        ok = self.move_to_xyz(*xyz, label='폴백 하강')
        if ok:
            self.send_gripper(GRIPPER_CLOSE)
            time.sleep(0.3)
        return ok

    # ─── 팔 이동 ─────────────────────────────────────────────────────────────

    def move_to_joints(self, joints, label=''):
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            print('  ❌ arm_controller 서버 없음!')
            return False

        with self.lock:
            js = self.current_joints
        current = [0.0] * 4
        if js:
            for i, name in enumerate(JOINT_NAMES):
                if name in js.name:
                    current[i] = js.position[js.name.index(name)]

        traj, duration = make_trajectory(joints, current)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self._arm_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                print('  ❌ 수락 타임아웃')
                return False
            time.sleep(0.05)

        gh = future.result()
        if not gh.accepted:
            print('  ❌ 액션 거부됨')
            return False

        rf = gh.get_result_async()
        deadline = time.time() + duration + 5.0
        while not rf.done():
            if time.time() > deadline:
                print('  ❌ 실행 타임아웃')
                return False
            time.sleep(0.1)

        ok = (rf.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL)
        print('  ✅ 완료' if ok else f'  ❌ 실패 (error_code={rf.result().result.error_code})')
        return ok

    def move_to_xyz(self, X, Y, Z, label=''):
        joints = solve_ik(X, Y, Z)
        if joints is None:
            print(f'  ❌ IK 해 없음: ({X:.3f}, {Y:.3f}, {Z:.3f})')
            return False
        print(f'  IK: j={[f"{j:.3f}" for j in joints]}')
        return self.move_to_joints(joints, label)

    # ─── 그리퍼 ──────────────────────────────────────────────────────────────

    def send_gripper(self, position: list) -> bool:
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            print('  ❌ gripper_controller 서버 없음!')
            return False

        goal = GripperCommand.Goal()
        goal.command.position   = position[0]
        # 열기(0.038): 50.0 / 닫기(0.026): 10.0 (박스 stall 감지)
        goal.command.max_effort = 50.0 if position == GRIPPER_OPEN else 10.0

        future = self._gripper_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                print('  ❌ 그리퍼 수락 타임아웃')
                return False
            time.sleep(0.05)

        gh = future.result()
        if not gh.accepted:
            print('  ❌ 그리퍼 액션 거부됨')
            return False

        rf = gh.get_result_async()
        deadline = time.time() + 5.0
        while not rf.done():
            if time.time() > deadline:
                print('  ❌ 그리퍼 실행 타임아웃')
                return False
            time.sleep(0.05)

        print('  ✅ 완료')
        return True

    # ─── 시퀀스 실행 ─────────────────────────────────────────────────────────

    def run_sequence(self, steps, name):
        print(f'\n{"="*50}')
        print(f' {name} 시퀀스 시작')
        print(f'{"="*50}')

        for i, step in enumerate(steps):
            label, joints, xyz, gripper = step[0], step[1], step[2], step[3]
            label_en = step[4] if len(step) > 4 else label
            print(f'\n[{i+1}/{len(steps)}] {label}')
            self._current_step    = f'[{i+1}/{len(steps)}] {label}'
            self._current_step_en = f'[{i+1}/{len(steps)}] {label_en}'

            key = input('  Enter: 실행 / q: 종료 / r: 처음부터 > ').strip().lower()
            if key == 'q':
                print('  종료 → 홈 복귀')
                self._current_step_en = 'Going Home...'
                self.move_to_joints(HOME_JOINTS, 'home')
                return 'quit'
            if key == 'r':
                return 'restart'

            # 그리퍼
            if gripper is not None:
                label_g = '열기' if gripper == GRIPPER_OPEN else '닫기'
                print(f'  그리퍼 {label_g} ({gripper[0]} rad)')
                self.send_gripper(gripper)
                time.sleep(0.3)

            # AUTO_GRAB sentinel
            if isinstance(joints, _AutoGrab):
                self._detect_and_grab(joints.fallback_xyz)
            # 관절 직접 지정
            elif joints is not None:
                self.move_to_joints(joints, label)
                pass
            # XYZ → IK
            elif xyz is not None:
                if solve_ik(*xyz) is None:
                    print(f'  ⚠️  IK 불가 {xyz} → 스킵')
                else:
                    self.move_to_xyz(*xyz, label=label)

        self._current_step    = f'{name} 완료'
        self._current_step_en = f'{name} Done'
        print(f'\n✅ {name} 시퀀스 완료!\n')
        return 'done'


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = DeliveryTestNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    time.sleep(2.0)
    print('\n홈 포지션으로 이동...')
    node.move_to_joints(HOME_JOINTS, 'home')

    print('\n─── 웨이포인트 IK 확인 ───')
    all_ok = True
    for name, xyz in [
        ('TABLE_HOVER',  TABLE_HOVER),
        ('TABLE_GRIP',   TABLE_GRIP),
        ('BASKET_HOVER', BASKET_HOVER),
        ('BASKET_PLACE', BASKET_PLACE),
        ('DEST_HOVER',   DEST_HOVER),
        ('DEST_PLACE',   DEST_PLACE),
    ]:
        j = solve_ik(*xyz)
        status = '✅' if j else '❌ IK 불가'
        print(f'  {name:14s} {str(xyz):35s} {status}')
        if not j:
            all_ok = False
    print('\n모든 웨이포인트 IK 가능 ✅' if all_ok
          else '\n⚠️  IK 불가 웨이포인트 있음. 해당 스텝은 건너뜁니다.')

    while True:
        print('\n─── 메뉴 ───')
        print('  1. 픽업  (PICKUP)  — 책상 박스 → 바구니')
        print('  2. 배달  (DELIVER) — 바구니 → 목적지 책상')
        print('  h. 홈 복귀')
        print('  q. 종료')
        choice = input('선택 > ').strip().lower()

        if choice == '1':
            result = node.run_sequence(PICKUP_STEPS, '픽업')
            if result == 'quit':
                break
        elif choice == '2':
            result = node.run_sequence(DELIVER_STEPS, '배달')
            if result == 'quit':
                break
        elif choice == 'h':
            node.move_to_joints(HOME_JOINTS, 'home')
        elif choice == 'q':
            node.move_to_joints(HOME_JOINTS, 'home')
            break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
