"""
Scout Mini 통합 노드 뼈대 (scout.py).

로봇팔 노드(arm_elevator.py, arm_delivery.py)와 토픽으로 통신.
실제 이동 로직(Nav2 웨이포인트, Pure Pursuit 등)은 팀원이 채워야 할 부분.

─── 전체 시나리오 흐름 ───────────────────────────────────────────────────────
1. 앱에서 층수 입력 → /target_floor 발행
2. Scout가 엘리베이터 홀로 이동 (TODO: 팀원 구현)
3. arm_elevator가 UP/DOWN 버튼 누르기
4. arm_elevator가 /robot_status = UPDOWN_PRESSED 발행
5. Scout가 엘리베이터 탑승 후 /elevator_ready 발행
6. arm_elevator가 층 버튼 누르기 후 /robot_status = NUMBER_PRESSED 발행
7. Scout가 목적지 층으로 이동 후 픽업 지점 앞 정지 (TODO)
8. /start_pickup 발행 → arm_delivery가 픽업 실행
9. arm_delivery가 /pickup_done 발행
10. arm_delivery가 호수 인식 후 /room_number 발행
11. Scout가 /room_number를 rooms.yaml과 비교 → 목적지 좌표 찾아 이동 (TODO)
12. 목적지 문 앞 정렬 완료 → /aligned_ready 발행
13. arm_delivery가 배달 실행 후 /delivery_done 발행
14. Scout가 복귀 (TODO)

─── 토픽 인터페이스 ──────────────────────────────────────────────────────────
발행:
  /target_floor   (Int32)  — 목표 층수
  /elevator_ready (Bool)   — 엘리베이터 탑승 완료 (버튼 앞 정지)
  /start_pickup   (Bool)   — 픽업 시작 트리거
  /aligned_ready  (Bool)   — 목적지 문 앞 정렬 완료 → 배달 시작

구독:
  /robot_status   (String) — arm 상태 (UPDOWN_PRESSED/NUMBER_PRESSED/PICKUP_DONE/DELIVERY_DONE/NEED_REPOSITION)
  /room_number    (String) — 인식된 호수
  /pickup_done    (Bool)   — 픽업 완료
  /delivery_done  (Bool)   — 배달 완료

─── 팀원 확인 필요 사항 ──────────────────────────────────────────────────────
- 웨이포인트 파일 경로 (WAYPOINTS_FILE)
- /target_floor 발행 시점 (앱 직접? Scout 노드에서?)
- Pure Pursuit vs Nav2 최종 확인
- /elevator_ready 발행 조건 (버튼 앞 몇 cm 정지?)
"""

import os
import threading
import time

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, String

# ─── 웨이포인트 / 맵 설정 ─────────────────────────────────────────────────────
# TODO: 팀원에게 파일 경로 확인 후 수정

WAYPOINTS_FILE = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'rooms.yaml')  # 호수별 좌표

# ─── 상태 상수 ────────────────────────────────────────────────────────────────

IDLE              = 'IDLE'
GOING_ELEVATOR    = 'GOING_ELEVATOR'   # 엘리베이터 홀로 이동 중
WAITING_UPDOWN    = 'WAITING_UPDOWN'   # UP/DOWN 버튼 누르기 대기
BOARDING          = 'BOARDING'         # 엘리베이터 탑승 중
WAITING_NUMBER    = 'WAITING_NUMBER'   # 층 버튼 누르기 대기
GOING_PICKUP      = 'GOING_PICKUP'     # 픽업 지점으로 이동 중
WAITING_PICKUP    = 'WAITING_PICKUP'   # 픽업 완료 대기
ROOM_SCANNING     = 'ROOM_SCANNING'    # 호수 번호판 스캔 중
GOING_DELIVERY    = 'GOING_DELIVERY'   # 배달 목적지로 이동 중
WAITING_DELIVERY  = 'WAITING_DELIVERY' # 배달 완료 대기
RETURNING         = 'RETURNING'        # 복귀 중
DONE              = 'DONE'


class ScoutNode(Node):

    def __init__(self):
        super().__init__('scout')
        self.state        = IDLE
        self.target_floor = None
        self.room_number  = None
        self.room_coords  = {}  # 호수 → (x, y) 좌표

        # 발행
        self.floor_pub     = self.create_publisher(Int32,  '/target_floor',   10)
        self.elev_ready_pub = self.create_publisher(Bool,  '/elevator_ready', 10)
        self.pickup_pub    = self.create_publisher(Bool,   '/start_pickup',   10)
        self.aligned_pub   = self.create_publisher(Bool,   '/aligned_ready',  10)

        # 구독
        self.create_subscription(String, '/robot_status',  self._cb_robot_status,  10)
        self.create_subscription(String, '/room_number',   self._cb_room_number,   10)
        self.create_subscription(Bool,   '/pickup_done',   self._cb_pickup_done,   10)
        self.create_subscription(Bool,   '/delivery_done', self._cb_delivery_done, 10)

        self._load_rooms()
        self.get_logger().info('scout 노드 시작. 층수 입력을 기다리는 중...')

    # ─── rooms.yaml 로드 ──────────────────────────────────────────────────────

    def _load_rooms(self):
        if not os.path.exists(WAYPOINTS_FILE):
            self.get_logger().warn(f'rooms.yaml 없음: {WAYPOINTS_FILE}')
            return
        with open(WAYPOINTS_FILE) as f:
            data = yaml.safe_load(f)
        for room_id, coords in data.items():
            self.room_coords[str(room_id)] = (coords['x'], coords['y'])
        self.get_logger().info(f'호수 {len(self.room_coords)}개 로드: {list(self.room_coords.keys())}')

    # ─── arm 상태 수신 ────────────────────────────────────────────────────────

    def _cb_robot_status(self, msg: String):
        status = msg.data
        self.get_logger().info(f'/robot_status: {status}')

        if status == 'UPDOWN_PRESSED' and self.state == WAITING_UPDOWN:
            # UP/DOWN 버튼 눌림 + 점등 확인 완료 → 탑승 시작
            self.state = BOARDING
            threading.Thread(target=self._board_elevator, daemon=True).start()

        elif status == 'NUMBER_PRESSED' and self.state == WAITING_NUMBER:
            # 층 버튼 눌림 → 엘리베이터 탑승 후 이동 시작
            self.state = GOING_PICKUP
            threading.Thread(target=self._go_to_pickup, daemon=True).start()

        elif status == 'NEED_REPOSITION':
            self.get_logger().warn('NEED_REPOSITION 수신 → 재정렬 필요 (TODO)')
            # TODO: Scout가 버튼 앞에서 미세 재정렬 후 /target_floor 재발행

        elif status == 'FAILED':
            self.get_logger().error('arm FAILED 수신 → 비상 정지')
            self.state = IDLE

    def _cb_room_number(self, msg: String):
        room = msg.data.strip()
        if room == self.room_number:
            return
        self.room_number = room
        self.get_logger().info(f'호수 인식: {room}')

        if room in self.room_coords:
            x, y = self.room_coords[room]
            self.get_logger().info(f'  → 목적지 좌표: ({x:.2f}, {y:.2f})')
        else:
            self.get_logger().warn(f'  → rooms.yaml에 {room}호 없음')

    def _cb_pickup_done(self, msg: Bool):
        if not msg.data:
            return
        self.get_logger().info('/pickup_done 수신 → 호수 인식 대기 중')
        self.state = ROOM_SCANNING

    def _cb_delivery_done(self, msg: Bool):
        if not msg.data:
            return
        self.get_logger().info('/delivery_done 수신 → 복귀')
        self.state = RETURNING
        threading.Thread(target=self._return_home, daemon=True).start()

    # ─── 이동 함수 (TODO: 팀원 구현) ─────────────────────────────────────────

    def start_mission(self, floor: int):
        """외부에서 층수 입력 시 호출. 앱 연동 또는 수동 토픽으로 대체 가능."""
        if self.state != IDLE:
            self.get_logger().warn(f'작업 중 ({self.state}). 무시.')
            return
        self.target_floor = floor
        self.get_logger().info(f'미션 시작: {floor}층')

        # 로봇팔에 목표 층수 전달
        self.floor_pub.publish(Int32(data=floor))

        self.state = GOING_ELEVATOR
        threading.Thread(target=self._go_to_elevator, daemon=True).start()

    def _go_to_elevator(self):
        self.get_logger().info('엘리베이터 홀로 이동 중... (TODO: Nav2 / Pure Pursuit 구현)')
        # TODO: 팀원 구현
        # nav2_client.send_goal(ELEVATOR_WAYPOINT)
        # nav2_client.wait_for_result()

        self.get_logger().info('엘리베이터 홀 도착 (TODO: 실제 이동 구현)')
        self.state = WAITING_UPDOWN
        # arm_elevator.py가 /target_floor를 이미 받았으므로 자동으로 버튼 누르기 시작

    def _board_elevator(self):
        self.get_logger().info('엘리베이터 탑승 중... (TODO: 진입 이동 구현)')
        # TODO: 엘리베이터 문 열림 감지 + 진입 이동
        # nav2_client.send_goal(ELEVATOR_INSIDE_WAYPOINT)

        self.get_logger().info('탑승 완료 → /elevator_ready 발행')
        self.elev_ready_pub.publish(Bool(data=True))
        self.state = WAITING_NUMBER

    def _go_to_pickup(self):
        self.get_logger().info('픽업 지점으로 이동 중... (TODO: 구현)')
        # TODO: 픽업 웨이포인트로 이동

        self.get_logger().info('픽업 지점 도착 → /start_pickup 발행')
        self.pickup_pub.publish(Bool(data=True))
        self.state = WAITING_PICKUP

    def _navigate_to_room(self, room: str):
        """인식된 호수 좌표로 이동."""
        if room not in self.room_coords:
            self.get_logger().error(f'rooms.yaml에 {room}호 없음 → 배달 불가')
            return False
        x, y = self.room_coords[room]
        self.get_logger().info(f'{room}호 ({x:.2f}, {y:.2f})로 이동 중... (TODO: Nav2 구현)')
        # TODO: Nav2로 (x, y) 이동
        # nav2_client.send_goal_xy(x, y)
        return True

    def _align_at_door(self):
        """문 앞 정렬 완료 → /aligned_ready 발행."""
        self.get_logger().info('문 앞 정렬 완료 → /aligned_ready 발행 (TODO: 실제 정렬 구현)')
        # TODO: 문 방향으로 회전 정렬
        self.aligned_pub.publish(Bool(data=True))
        self.state = WAITING_DELIVERY

    def _go_to_delivery(self):
        """픽업 완료 후 배달 목적지로 이동."""
        # room_number가 아직 없으면 ROOM_SCANNING에서 대기
        timeout = time.time() + 30.0
        while not self.room_number and time.time() < timeout:
            self.get_logger().info('호수 인식 대기 중...')
            time.sleep(2.0)

        if not self.room_number:
            self.get_logger().error('호수 인식 실패 (30초 타임아웃)')
            self.state = IDLE
            return

        self.state = GOING_DELIVERY
        ok = self._navigate_to_room(self.room_number)
        if not ok:
            self.state = IDLE
            return

        self._align_at_door()

    def _return_home(self):
        self.get_logger().info('복귀 중... (TODO: 출발 지점 복귀 구현)')
        # TODO: 출발 지점 또는 충전 스테이션으로 이동
        self.state = IDLE
        self.room_number = None
        self.get_logger().info('복귀 완료. 다음 미션 대기 중.')


# ─── 엔트리포인트 ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ScoutNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # 수동 테스트: 터미널에서 층수 입력
    # 실제 배포 시 앱 연동으로 대체하거나 /target_floor 토픽 수신으로 변경
    try:
        while rclpy.ok():
            print('\n층수 입력 (q: 종료): ', end='', flush=True)
            inp = input().strip()
            if inp.lower() == 'q':
                break
            try:
                floor = int(inp)
                node.start_mission(floor)
            except ValueError:
                print('숫자를 입력하세요.')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
