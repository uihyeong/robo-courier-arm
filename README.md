# robo-courier-arm

ROS2 robot arm package for autonomous elevator button pressing and package delivery.  
Mounted on a Scout Mini mobile robot as part of an autonomous delivery system.

---

## Features

- **Elevator button pressing** — Detects UP/DOWN and floor buttons using YOLOv8 + EasyOCR, then presses them via analytical IK
- **Pick-and-place delivery** — Picks up a package from a table, reads the room number sign, and delivers to the destination
- **Contact detection** — Monitors joint effort to detect unexpected collisions and retract automatically

## Hardware

| Component | Model |
|-----------|-------|
| Robot arm | OpenMANIPULATOR-X |
| Depth camera | Intel RealSense D435 |
| Mobile base | Scout Mini (teammate's) |

## Tech Stack

- ROS2 Humble
- YOLOv8 (UP/DOWN button detection, room sign detection)
- EasyOCR (floor number & room number recognition)
- Analytical IK (no MoveIt2 required)

---

## Package Structure

```
robo-courier-arm/
├── elevator_robot/
│   ├── ik.py                  # Shared analytical IK (OpenMANIPULATOR-X)
│   ├── arm_elevator.py        # Elevator button pressing node
│   ├── arm_delivery.py        # Pick-and-place delivery node
│   ├── contact_detector.py    # Collision detection node
│   ├── detect_room_sign.py    # Room number sign recognition node
│   └── scout.py               # Scout Mini integration skeleton
├── launch/
│   └── elevator.launch.py     # Launches arm_elevator + contact_detector
├── yolo/weights/
│   ├── best.pt                # UP/DOWN button detection model
│   ├── best_num.pt            # Floor number detection model
│   └── best_room.pt           # Room sign detection model
└── rooms.yaml                 # Room number → navigation waypoint mapping
```

---

## Installation

```bash
# Clone into your colcon workspace
cd ~/colcon_ws/src
git clone https://github.com/uihyeong/robo-courier-arm.git elevator_robot

# Build
cd ~/colcon_ws
colcon build --packages-select elevator_robot --symlink-install
source install/setup.bash
```

**Dependencies** (must be present in colcon_ws):
- `open_manipulator_x_bringup`
- `realsense2_camera`

---

## Usage

### Elevator Mode

```bash
# 1. Hardware controller
ros2 launch open_manipulator_x_bringup hardware.launch.py

# 2. Camera
ros2 launch realsense2_camera rs_launch.py

# 3. Camera TF
ros2 run tf2_ros static_transform_publisher \
  --x 0.12 --y 0.01 --z 0.062 \
  --roll 0.0 --pitch 0.0 --yaw 0.0 \
  --frame-id link5 --child-frame-id camera_link

# 4. Run node (+ optional contact detector)
ros2 launch elevator_robot elevator.launch.py

# 5. Send target floor
ros2 topic pub --once /target_floor std_msgs/Int32 "{data: 3}"
```

### Delivery Mode

```bash
ros2 run elevator_robot arm_delivery

# Trigger pickup
ros2 topic pub --once /start_pickup std_msgs/Bool "{data: true}"

# Signal arrival at destination (after Scout Mini aligns)
ros2 topic pub --once /aligned_ready std_msgs/Bool "{data: true}"
```

---

## Topic Interface

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/target_floor` | `Int32` | IN | Target floor (negative = basement) |
| `/elevator_ready` | `Bool` | IN | Scout Mini boarded → start floor button phase |
| `/start_pickup` | `Bool` | IN | Trigger pickup sequence |
| `/aligned_ready` | `Bool` | IN | Scout Mini aligned at destination |
| `/robot_status` | `String` | OUT | `MOVING` / `UPDOWN_PRESSED` / `PICKUP_DONE` / `DELIVERY_DONE` / `FAILED` |
| `/pickup_done` | `Bool` | OUT | Pickup complete |
| `/room_number` | `String` | OUT | Recognized room number (e.g. `"529"`) |
| `/delivery_done` | `Bool` | OUT | Delivery complete |
| `/contact_detected` | `Bool` | OUT | Collision detected |

---

## State Machine

### Elevator Node
```
IDLE → UPDOWN_READY → UPDOWN_PRESS → WAIT → NUMBER_READY → NUMBER_PRESS → DONE
```

### Delivery Node
```
IDLE → PICKUP → ROOM_SIGN → WAITING_ALIGN → DELIVER → DONE
```
