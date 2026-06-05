"""
버튼 점등/소등 HSV 튜닝 도구 (ROS2 토픽 버전).

unified.py 와 동일하게 /camera/camera/color/image_raw 토픽을 사용.

실행:
    ros2 launch realsense2_camera rs_launch.py          # 카메라 드라이버 먼저
    python3 nodes/real_robot/test_button_lit.py

단축키:
    드래그      ROI 직접 선택
    R           ROI 초기화 (화면 중앙 100×100)
    L           현재 값을 "켜짐" 샘플로 기록
    U           현재 값을 "꺼짐" 샘플로 기록
    P           기록된 샘플 요약 및 권장 LIT_GREEN_RATIO 출력
    S           현재 프레임을 파일로 저장
    Q / ESC     종료
"""

import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

HSV_LOW  = np.array([35,  50,  50])
HSV_HIGH = np.array([85, 255, 255])

WIN = "test_button_lit  |  드래그:ROI  L:켜짐  U:꺼짐  P:요약  S:저장  Q:종료"

lit_samples   = []
unlit_samples = []
roi   = None
p1    = None
dragging = False
save_idx = 0


def mouse_cb(event, x, y, flags, param):
    global roi, p1, dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = (x, y); dragging = True
    elif event == cv2.EVENT_LBUTTONUP and dragging:
        x1, y1 = min(p1[0], x), min(p1[1], y)
        x2, y2 = max(p1[0], x), max(p1[1], y)
        if x2 - x1 > 5 and y2 - y1 > 5:
            roi = (x1, y1, x2, y2)
        dragging = False


def compute(frame, r):
    x1, y1, x2, y2 = r
    h, w = frame.shape[:2]
    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if crop.size == 0:
        return 0.0, np.zeros(3)
    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    ratio = np.count_nonzero(mask) / (crop.shape[0] * crop.shape[1])
    hmean = hsv.mean(axis=(0, 1))
    return ratio, hmean


def print_summary():
    print("\n" + "=" * 50)
    if lit_samples:
        print(f"  켜짐  {len(lit_samples):2d}개  "
              f"min={min(lit_samples):.4f}  max={max(lit_samples):.4f}  mean={np.mean(lit_samples):.4f}")
    else:
        print("  켜짐  샘플 없음 (L 키로 기록)")
    if unlit_samples:
        print(f"  꺼짐  {len(unlit_samples):2d}개  "
              f"min={min(unlit_samples):.4f}  max={max(unlit_samples):.4f}  mean={np.mean(unlit_samples):.4f}")
    else:
        print("  꺼짐  샘플 없음 (U 키로 기록)")
    if lit_samples and unlit_samples:
        gap = min(lit_samples) - max(unlit_samples)
        mid = (min(lit_samples) + max(unlit_samples)) / 2
        if gap > 0:
            print(f"\n  ✅ 권장 LIT_GREEN_RATIO = {mid:.3f}")
            print(f"     real_robot_unified.py 90번째 줄을 이 값으로 수정")
        else:
            print(f"\n  ⚠️  분포 겹침 (gap={gap:.4f}) → HSV 범위 조정 필요")
    print("=" * 50 + "\n")


class ButtonLitTester(Node):
    def __init__(self):
        super().__init__('button_lit_tester')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.lock = threading.Lock()
        self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self._image_cb, 10)
        self.get_logger().info("카메라 토픽 대기 중...")

    def _image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.lock:
            self.latest_frame = frame

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None


def main():
    global roi, save_idx

    rclpy.init()
    node = ButtonLitTester()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, mouse_cb)
    print("\n[시작] 카메라 토픽 수신 대기... (드라이버가 켜져 있어야 합니다)")
    print("       ros2 launch realsense2_camera rs_launch.py")

    while True:
        frame = node.get_frame()
        if frame is None:
            key = cv2.waitKey(100) & 0xFF
            if key in (ord('q'), 27):
                break
            continue

        h, w = frame.shape[:2]
        if roi is None:
            roi = (w//2 - 50, h//2 - 50, w//2 + 50, h//2 + 50)

        ratio, hmean = compute(frame, roi)
        x1, y1, x2, y2 = roi

        vis = frame.copy()
        color = (0, 220, 0) if ratio > 0.10 else (60, 60, 220)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"green_ratio={ratio:.4f}", (x1, max(y1-8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        cv2.putText(vis, f"H={hmean[0]:.0f}  S={hmean[1]:.0f}  V={hmean[2]:.0f}",
                    (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(vis, f"lit={len(lit_samples)}  unlit={len(unlit_samples)}",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 1)
        cv2.imshow(WIN, vis)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            roi = (w//2 - 50, h//2 - 50, w//2 + 50, h//2 + 50)
            print("[R] ROI 중앙 리셋")
        elif key == ord('l'):
            lit_samples.append(ratio)
            print(f"[켜짐] green_ratio={ratio:.4f}  (누적 {len(lit_samples)}개)")
        elif key == ord('u'):
            unlit_samples.append(ratio)
            print(f"[꺼짐] green_ratio={ratio:.4f}  (누적 {len(unlit_samples)}개)")
        elif key == ord('p'):
            print_summary()
        elif key == ord('s'):
            fname = f"button_capture_{save_idx:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[S] 저장: {fname}")
            save_idx += 1

    print_summary()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
