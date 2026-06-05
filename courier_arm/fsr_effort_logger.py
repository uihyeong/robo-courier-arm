"""
FSR406 + joint effort 동시 로깅 스크립트.

FSR로 관절을 탭했을 때의 effort 값을 확인해
contact_detector.py의 COLLISION_THRESHOLD(현재 80) 튜닝에 사용.

사용법:
  python3 nodes/real_robot/fsr_effort_logger.py --port /dev/ttyACM0

출력 CSV 컬럼:
  time_ms,
  q1 q2 q3 q4,        (관절 위치 rad)
  v1 v2 v3 v4,        (관절 속도 rad/s)
  e1 e2 e3 e4,        (관절 effort)
  fsr_raw,
  label               (0=정상 / 1=FSR 탭 감지)

단축키:
  Ctrl+C  종료 후 fsr_effort_log.csv 저장
"""

import argparse
import csv
import os
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial

JOINT_NAMES    = ['joint1', 'joint2', 'joint3', 'joint4']
FSR_THRESHOLD  = 10
SERIAL_BAUD    = 9600
OUTPUT_FILE    = os.path.join(os.path.dirname(__file__), 'fsr_effort_log.csv')


class FsrEffortLogger(Node):
    def __init__(self):
        super().__init__('fsr_effort_logger')
        self._lock = threading.Lock()
        self._pos  = [0.0] * 4
        self._vel  = [0.0] * 4
        self._eff  = [0.0] * 4
        self.create_subscription(JointState, '/joint_states', self._cb_joint, 10)

    def _cb_joint(self, msg: JointState):
        with self._lock:
            for i, name in enumerate(JOINT_NAMES):
                if name in msg.name:
                    idx = msg.name.index(name)
                    if idx < len(msg.position): self._pos[i] = msg.position[idx]
                    if idx < len(msg.velocity): self._vel[i] = msg.velocity[idx]
                    if idx < len(msg.effort):   self._eff[i] = msg.effort[idx]

    def snapshot(self):
        with self._lock:
            return list(self._pos), list(self._vel), list(self._eff)


def serial_ros_loop(port: str):
    rclpy.init()
    node = FsrEffortLogger()

    rows      = []
    start_ms  = int(time.time() * 1000)
    stop_flag = threading.Event()

    print(f'포트 {port} 연결 중...')
    try:
        ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
    except Exception as e:
        print(f'시리얼 연결 실패: {e}')
        return

    def serial_reader():
        try:
            time.sleep(2.0)
            ser.readline()  # 헤더 버리기
            print('로깅 시작! FSR로 관절을 탭하세요. Ctrl+C로 종료.\n', flush=True)
            print(f"{'time_ms':>8}  {'fsr':>5}  {'label':>5}  "
                  f"{'e1':>7}  {'e2':>7}  {'e3':>7}  {'e4':>7}", flush=True)
            print('-' * 60, flush=True)

            while not stop_flag.is_set():
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) != 3:
                    continue
                try:
                    fsr_val = int(parts[1])
                    label   = 1 if fsr_val > FSR_THRESHOLD else 0  # Python 측 판단 (아두이노 무시)
                except ValueError:
                    continue

                pos, vel, eff = node.snapshot()
                t_ms = int(time.time() * 1000) - start_ms

                row = [t_ms,
                       round(pos[0], 4), round(pos[1], 4), round(pos[2], 4), round(pos[3], 4),
                       round(vel[0], 4), round(vel[1], 4), round(vel[2], 4), round(vel[3], 4),
                       round(eff[0], 2), round(eff[1], 2), round(eff[2], 2), round(eff[3], 2),
                       fsr_val, label]
                rows.append(row)

                marker = '  ◀ TAP' if label else ''
                print(f"{t_ms:>8}  {fsr_val:>5}  {label:>5}  "
                      f"{eff[0]:>7.1f}  {eff[1]:>7.1f}  {eff[2]:>7.1f}  {eff[3]:>7.1f}{marker}",
                      flush=True)

        except Exception as e:
            print(f'[serial_reader 오류] {e}')

    # 시리얼은 별도 스레드, ROS2 spin은 메인 스레드
    threading.Thread(target=serial_reader, daemon=True).start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        ser.close()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'time_ms',
                'q1', 'q2', 'q3', 'q4',
                'v1', 'v2', 'v3', 'v4',
                'e1', 'e2', 'e3', 'e4',
                'fsr_raw',
                'label',
            ])
            writer.writerows(rows)

        print(f'\n✅ {len(rows)}행 저장: {OUTPUT_FILE}')
        _print_summary(rows)


def _print_summary(rows):
    tap_rows = [r for r in rows if r[14] == 1]  # label==1
    if not tap_rows:
        print('탭 감지 없음.')
        return
    for i, name in enumerate(JOINT_NAMES, start=9):  # e1~e4는 index 9~12
        vals = [abs(r[i]) for r in tap_rows]
        print(f'{name}  탭 시 effort  max={max(vals):.1f}  avg={sum(vals)/len(vals):.1f}')
    print(f'\n현재 COLLISION_THRESHOLD = 15  —  위 max 값 참고해서 조정하세요.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/ttyACM0')
    args = parser.parse_args()
    serial_ros_loop(args.port)
