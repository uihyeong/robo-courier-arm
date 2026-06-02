"""
정지 중 접촉 감지 노드.

팔이 정지해 있을 때 joint effort를 모니터링하다가
사람이 팔을 건드리면 joint3·4를 접어 움츠러든 뒤 홈으로 복귀합니다.

동작 흐름:
  1. 시작 시 홈 포지션 effort를 N샘플 평균으로 baseline 측정
  2. 팔이 정지 상태 (velocity ≈ 0) + 로봇이 이동 중이 아닐 때만 모니터링
  3. baseline 대비 THRESHOLD 초과 시 접촉 판정
  4. /contact_detected 발행 → SHRINK 자세 (joint3·4 접기) → 2초 후 홈 복귀

실행:
  python3 nodes/real_robot/contact_detector.py

real_robot_unified.py와 병렬 실행 가능.
/robot_status가 MOVING이면 자동으로 모니터링 중단.
"""

import math
import threading
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ─── 파라미터 ─────────────────────────────────────────────────────────────────

MONITOR_JOINTS    = ['joint3']  # 실측상 joint3 effort만 탭 시 유의미한 변화 (~18 units)
COLLISION_THRESHOLD = 15     # baseline 대비 이 값 초과 시 접촉 판정 (Dynamixel 전류 단위)
                              # 1 unit ≈ 2.69 mA → 15 units ≈ 40 mA
VELOCITY_STILL    = 0.01     # 이 값 이하면 정지로 판정 (rad/s)
CALIBRATE_SAMPLES = 30       # baseline 측정에 사용할 샘플 수
COOLDOWN_SEC      = 3.0      # 접촉 감지 후 재감지까지 대기 시간 (초)

JOINT_NAMES   = ['joint1', 'joint2', 'joint3', 'joint4']
HOME_JOINTS   = [-3.141, -0.9948,  0.6981,  0.2967]
# joint3·4를 최대로 접어서 link3과 link4가 포개지는 자세
SHRINK_JOINTS = [-3.141, -0.9948,  1.3000, -1.5700]
SHRINK_HOLD_SEC  = 2.0   # 움츠린 자세 유지 후 홈 복귀까지 대기 (초)
MOVE_SPEED       = 0.5   # 홈 복귀 속도 (rad/s)
MIN_DURATION     = 2.0   # 홈 복귀 최소 이동 시간 (초)
SHRINK_SPEED     = 2.0   # 움츠리기 속도 (rad/s) — 빠르게 반응
SHRINK_MIN_DUR   = 0.5   # 움츠리기 최소 이동 시간 (초)


def make_trajectory(target_joints, current_joints,
                    speed: float = MOVE_SPEED, min_dur: float = MIN_DURATION):
    def shortest(t, c):
        diff = (t - c + math.pi) % (2 * math.pi) - math.pi
        return c + diff

    target_joints = [shortest(t, c) for t, c in zip(target_joints, current_joints)]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
    duration = max(max_disp / speed, min_dur)

    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES

    pt = JointTrajectoryPoint()
    pt.positions  = target_joints
    pt.velocities = [0.0] * 4
    secs  = int(duration)
    nsecs = int((duration - secs) * 1e9)
    pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
    traj.points.append(pt)

    return traj, duration


class ContactDetectorNode(Node):
    def __init__(self):
        super().__init__('contact_detector')

        self.lock = threading.Lock()

        self.current_joints = None   # 최신 JointState
        self.robot_moving   = False  # unified 노드가 이동 중인지

        # baseline: 홈에서 측정한 고정 기준값
        self.baseline        = {}    # {joint_name: float}
        self.calibrated      = False
        self._calib_buffer   = {j: [] for j in MONITOR_JOINTS}

        # 접촉 감지 쿨다운
        self._last_contact_t = 0.0

        # arm_controller 액션
        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self.contact_pub = self.create_publisher(Bool,   '/contact_detected', 10)
        self.status_pub  = self.create_publisher(String, '/contact_status',   10)

        self.create_subscription(JointState, '/joint_states',  self._cb_joint_state,  10)
        self.create_subscription(String,     '/robot_status',  self._cb_robot_status, 10)

        self.get_logger().info(
            f'접촉 감지 노드 시작. 홈 포지션에서 {CALIBRATE_SAMPLES}샘플 baseline 측정 중...')

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def _cb_robot_status(self, msg: String):
        self.robot_moving = (msg.data == 'MOVING')

    def _cb_joint_state(self, msg: JointState):
        with self.lock:
            self.current_joints = msg

        if not self.calibrated:
            self._collect_calibration(msg)
            return

        self._check_contact(msg)

    # ─── Baseline 측정 ───────────────────────────────────────────────────────

    def _collect_calibration(self, msg: JointState):
        """홈 포지션 effort를 샘플링해서 baseline 계산."""
        # 팔이 정지 상태일 때만 샘플 수집
        if not self._is_still(msg):
            return

        for j in MONITOR_JOINTS:
            if j in msg.name:
                idx = msg.name.index(j)
                effort = msg.effort[idx]
                if effort != 0.0:   # 0은 미보고 값
                    self._calib_buffer[j].append(effort)

        # 모든 관절이 충분한 샘플 모이면 완료
        if all(len(v) >= CALIBRATE_SAMPLES for v in self._calib_buffer.values()):
            self.baseline = {
                j: sum(v) / len(v) for j, v in self._calib_buffer.items()
            }
            self.calibrated = True
            self.get_logger().info('✅ Baseline 측정 완료:')
            for j, val in self.baseline.items():
                self.get_logger().info(f'   {j}: {val:.1f}')
            self.get_logger().info(f'접촉 감지 시작! (threshold: ±{COLLISION_THRESHOLD})')

    # ─── 접촉 감지 ───────────────────────────────────────────────────────────

    def _is_still(self, msg: JointState) -> bool:
        """모든 관절 속도가 VELOCITY_STILL 이하면 정지로 판정."""
        for j in MONITOR_JOINTS:
            if j in msg.name:
                idx = msg.name.index(j)
                if abs(msg.velocity[idx]) > VELOCITY_STILL:
                    return False
        return True

    def _check_contact(self, msg: JointState):
        # 이동 중이거나 팔이 움직이는 중이면 스킵
        if self.robot_moving or not self._is_still(msg):
            return

        # 쿨다운 체크
        if time.time() - self._last_contact_t < COOLDOWN_SEC:
            return

        # 각 관절 effort vs baseline 비교
        for j in MONITOR_JOINTS:
            if j not in msg.name:
                continue
            idx    = msg.name.index(j)
            effort = msg.effort[idx]
            if effort == 0.0:
                continue

            deviation = abs(effort - self.baseline[j])
            if deviation > COLLISION_THRESHOLD:
                self.get_logger().warn(
                    f'⚠️  접촉 감지! [{j}] baseline={self.baseline[j]:.1f} '
                    f'현재={effort:.1f} 편차={deviation:.1f}')
                self._on_contact()
                return

    def _on_contact(self):
        self._last_contact_t = time.time()
        self.contact_pub.publish(Bool(data=True))
        self.status_pub.publish(String(data='CONTACT_DETECTED'))
        threading.Thread(target=self._shrink_then_home, daemon=True).start()

    # ─── 움츠리기 → 홈 복귀 ─────────────────────────────────────────────────

    def _shrink_then_home(self):
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('arm_controller 없음!')
            return

        # 1단계: joint3·4 빠르게 접기 (움츠리기)
        self.get_logger().info('⚠️  접촉! joint3·4 접는 중...')
        ok = self._send_joints(SHRINK_JOINTS, speed=SHRINK_SPEED, min_dur=SHRINK_MIN_DUR)
        if not ok:
            self.get_logger().error('움츠리기 실패')

        # 2초 유지
        time.sleep(SHRINK_HOLD_SEC)

        # 2단계: 홈 복귀 (일반 속도)
        self.get_logger().info('홈으로 복귀 중...')
        self._send_joints(HOME_JOINTS)
        self.get_logger().info('✅ 홈 복귀 완료. 모니터링 재개.')
        self.status_pub.publish(String(data='CONTACT_RESOLVED'))

    def _send_joints(self, target_joints: list,
                     speed: float = MOVE_SPEED, min_dur: float = MIN_DURATION) -> bool:
        with self.lock:
            js = self.current_joints
        current = [0.0] * 4
        if js is not None:
            for i, name in enumerate(JOINT_NAMES):
                if name in js.name:
                    current[i] = js.position[js.name.index(name)]

        traj, duration = make_trajectory(target_joints, current, speed=speed, min_dur=min_dur)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future   = self._arm_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                return False
            time.sleep(0.05)

        goal_handle = future.result()
        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        deadline = time.time() + duration + 5.0
        while not result_future.done():
            if time.time() > deadline:
                return False
            time.sleep(0.1)

        return result_future.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL


# ─── 엔트리포인트 ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ContactDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
