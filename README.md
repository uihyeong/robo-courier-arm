<h1 align="center">🤖 courier_arm</h1>

<p align="center">
  <b>자율주행 택배 로봇의 로봇팔 ROS2 패키지</b><br/>
  엘리베이터 버튼을 스스로 누르고, 박스를 픽업·배달하며, 보냉백을 회수합니다.<br/>
  OpenMANIPULATOR-X + RealSense D435 · Scout Mini 모바일 베이스 탑재용
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/build-ament__python-blue"/>
  <img src="https://img.shields.io/badge/YOLOv8-mAP50%2098.7%25-00BFFF?logo=yolo&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green"/>
</p>

<p align="center">
  <img src="https://github.com/uihyeong/elevator-button-robot/raw/main/media/demo1.gif" width="44%"/>
  <img src="https://github.com/uihyeong/elevator-button-robot/raw/main/media/demo_delivery.gif" width="44%"/>
</p>
<p align="center"><em>좌: 엘리베이터 버튼 누르기 &nbsp;|&nbsp; 우: 바구니 → 목적지 배달</em></p>

---

## 📋 Overview

`courier_arm`은 캡스톤 자율주행 택배 로봇의 **로봇팔 파트**를 담당하는 배포용 ROS2 패키지입니다.
카메라로 버튼·호수·박스를 인식하고 **해석적 IK**(MoveIt2 불필요)로 모션을 수행하며, 각 기능은
ROS2 토픽으로 느슨하게 결합돼 Scout Mini 주행 파트와 독립적으로 동작합니다.

> 전체 개발 히스토리·시뮬레이션(Isaac Sim/Lab)·학습 코드는 개발 레포
> [`elevator-button-robot`](https://github.com/uihyeong/elevator-button-robot) 참고.

---

## ✨ Features

- **🛗 엘리베이터 버튼 조작** — YOLOv8로 UP/DOWN·층수 버튼을 인식하고 해석적 IK로 누름. 버튼 점등(HSV)·소등을 감지해 엘리베이터 도착까지 자동 판단
- **📦 픽업 & 배달** — 책상 위 박스를 바구니로 픽업 → 호수 인식 → 목적지 책상으로 배달 (Joint 지령 + XYZ→IK 혼용)
- **🔢 호수 인식** — YOLO + EasyOCR로 호실 번호판을 읽어 `/room_number` 발행
- **🧊 프레시백 회수** — 배달 후 보냉백을 팔 고리에 다시 거는 회수 모션
- **🛡 접촉 감지** — 정지 중 외부 접촉을 joint effort로 감지해 자동 후퇴 (병렬 노드)

---

## 🧩 Nodes

`ros2 run courier_arm <executable>` 로 실행합니다.

| Executable | 설명 | 트리거 / 입력 |
|------------|------|---------------|
| `arm_elevator` | UP/DOWN + 층수 버튼 누르기 | `/target_floor`, `/elevator_ready` |
| `arm_delivery` | 픽업 → 호수 인식 → 배달 | `/start_pickup`, `/aligned_ready` |
| `arm_recover` | 보냉백(프레시백) 회수 | `/start_recover` |
| `detect_room_sign` | 호수 번호판 OCR → `/room_number` | (자동) |
| `contact_detector` | 충돌 감지 후 자동 후퇴 | (병렬 실행) |
| `scout` | Scout Mini 연동 뼈대 | — |

---

## 🛠 Tech Stack

| 분야 | 내용 |
|------|------|
| 미들웨어 | ROS2 Humble (`ament_python`) |
| 로봇팔 | OpenMANIPULATOR-X (4-DOF) + U2D2 |
| 카메라 | Intel RealSense D435 (RGB-D, 1080p + aligned depth) |
| 인식 | YOLOv8 · YOLO-seg (mAP50 98.7%) + EasyOCR |
| 역기구학 | 해석적 IK (수식 직접 유도, MoveIt2 불필요) |
| 언어 | Python 3.10 |

---

## ⚙️ Installation

### 1. 사전 요구

- Ubuntu 22.04 + **ROS2 Humble**
- 실제 로봇 구동 시: OpenMANIPULATOR-X + U2D2, RealSense D435
- 워크스페이스에 아래 패키지가 빌드돼 있어야 합니다.
  - [`open_manipulator`](https://github.com/ROBOTIS-GIT/open_manipulator) (`open_manipulator_x_bringup`)
  - `realsense2_camera` (`sudo apt install ros-humble-realsense2-camera`)

### 2. 패키지 클론 & 빌드

```bash
# colcon 워크스페이스에 클론 (폴더명 = 패키지명)
cd ~/colcon_ws/src
git clone https://github.com/uihyeong/robo-courier-arm.git courier_arm

# Python 의존성
pip install -r courier_arm/requirements.txt

# 빌드
cd ~/colcon_ws
rosdep install --from-paths src --ignore-src -r -y   # ROS 의존성 자동 설치
colcon build --packages-select courier_arm --symlink-install
source install/setup.bash
```

> **주의**: `numpy < 2.0.0` 필요 (cv_bridge가 NumPy 1.x 기준 빌드). OpenCV는 cv_bridge/시스템 ROS가 제공하므로 `pip install opencv-python` 금지(Qt 백엔드 충돌).

---

## ▶️ Usage

### 공통 준비 (3개 터미널)

```bash
# 터미널 1 — 하드웨어 컨트롤러 (U2D2 연결 후)
ros2 launch open_manipulator_x_bringup hardware.launch.py

# 터미널 2 — D435 카메라 (aligned depth + 1080p, 노드 요구사항)
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  rgb_camera.color_profile:=1920,1080,30

# 터미널 3 — 카메라 TF (link5 기준, 유지)
ros2 run tf2_ros static_transform_publisher \
  --x 0.12 --y 0.01 --z 0.062 --roll 0.0 --pitch 0.0 --yaw 0.0 \
  --frame-id link5 --child-frame-id camera_link
```

### 🛗 엘리베이터

```bash
ros2 launch courier_arm elevator.launch.py        # arm_elevator + contact_detector 동시 실행

ros2 topic pub --once /target_floor   std_msgs/Int32 "{data: 3}"     # 목표 층 (음수=지하)
ros2 topic pub --once /elevator_ready std_msgs/Bool  "{data: true}"  # 탑승 완료 → 층수 버튼 단계
```

### 📦 픽업 / 배달

```bash
ros2 run courier_arm arm_delivery

ros2 topic pub --once /start_pickup  std_msgs/Bool "{data: true}"   # 박스 픽업 → 바구니
ros2 topic pub --once /aligned_ready std_msgs/Bool "{data: true}"   # 목적지 정렬 완료 → 배달
```

### 🔢 호수 인식 · 🧊 회수

```bash
ros2 run courier_arm detect_room_sign                              # /room_number 발행
ros2 run courier_arm arm_recover
ros2 topic pub --once /start_recover std_msgs/Bool "{data: true}"  # 보냉백 회수
```

---

## 🔌 Topic Interface

| Topic | Type | 방향 | 노드 | 설명 |
|-------|------|------|------|------|
| `/target_floor` | `Int32` | IN | elevator | 목표 층수 (음수=지하, 예: -1=B1) |
| `/target_point` | `PointStamped` | IN | elevator | 수동 월드 좌표 직접 지정 |
| `/elevator_ready` | `Bool` | IN | elevator | 탑승 완료 → 층수 버튼 단계 시작 |
| `/start_pickup` | `Bool` | IN | delivery | 픽업 시퀀스 트리거 |
| `/aligned_ready` | `Bool` | IN | delivery | 목적지 정렬 완료 → 배달 시작 |
| `/start_delivery` | `Bool` | IN | delivery | 배달 시퀀스 직접 트리거 |
| `/start_recover` | `Bool` | IN | recover | 보냉백 회수 트리거 |
| `/robot_status` | `String` | OUT | all | 상태 문자열 (아래 참고) |
| `/pickup_done` · `/delivery_done` · `/recover_done` | `Bool` | OUT | delivery / recover | 각 단계 완료 |
| `/room_number` | `String` | OUT | delivery / detect_room_sign | 인식된 호수 (예: `"529"`) |
| `/contact_detected` | `Bool` | OUT | contact_detector | 접촉 감지 |
| `/contact_status` | `String` | OUT | contact_detector | `CONTACT_DETECTED` / `CONTACT_RESOLVED` |

**`/robot_status` 값:**
- **elevator** — `MOVING` / `UPDOWN_PRESSED` / `ELEVATOR_ARRIVED` / `NUMBER_PRESSED` / `NEED_REPOSITION` / `FAILED`
- **delivery** — `MOVING` / `PICKUP` / `PICKUP_DONE` / `ROOM_SIGN` / `WAITING_ALIGN` / `DELIVER` / `DELIVERY_DONE` / `FAILED`
- **recover** — `MOVING` / `RECOVER` / `RECOVER_DONE` / `FAILED`

---

## 🔁 State Machines

```
elevator :  IDLE → UPDOWN_READY → UPDOWN_PRESS → WAIT → NUMBER_READY → NUMBER_PRESS → NUMBER_WAIT → DONE
delivery :  IDLE → PICKUP → ROOM_SIGN → WAITING_ALIGN → DELIVER → DONE
recover  :  IDLE → RECOVER → DONE
```

---

## 🧠 Models

학습된 YOLO 가중치가 `yolo/weights/` 에 포함되어 **추가 학습 없이 바로 실행**됩니다.

| 가중치 | 용도 | 사용 노드 |
|--------|------|-----------|
| `best.pt` | UP/DOWN 버튼 (mAP50 98.7%) | arm_elevator |
| `best_num.pt` | 층수 버튼 영역 분할 | arm_elevator |
| `best_room.pt` | 호실 번호판 | arm_delivery · detect_room_sign · arm_recover |
| `best_box.pt` | 배달 박스 | arm_delivery |
| `best_handle.pt` | 보냉백 손잡이 | arm_recover |

---

## 🗂 Package Structure

```
courier_arm/                       # ament_python 패키지
├── courier_arm/
│   ├── arm_elevator.py            # 🛗 엘리베이터 버튼 누르기
│   ├── arm_delivery.py            # 📦 픽업 → 호수 인식 → 배달
│   ├── arm_recover.py             # 🧊 보냉백 회수
│   ├── detect_room_sign.py        # 🔢 호수 번호판 OCR → /room_number
│   ├── contact_detector.py        # 🛡 접촉 감지 후 자동 후퇴
│   ├── scout.py                   # Scout Mini 연동 뼈대
│   ├── test_button_lit.py         # 버튼 점등 HSV 튜닝 도구
│   ├── test_delivery_motion.py    # 픽업/배달 수동 데모 (스텝 진행)
│   └── fsr_effort_logger.py       # FSR + effort 로깅 유틸
├── launch/
│   └── elevator.launch.py         # arm_elevator + contact_detector
├── yolo/weights/                  # 학습된 YOLO 가중치 (best*.pt)
├── rooms.yaml                     # 호수 → 내비 웨이포인트 매핑
├── package.xml · setup.py         # ROS2 패키지 매니페스트
└── requirements.txt               # Python 의존성
```

---

## 📄 License

[MIT License](LICENSE)
