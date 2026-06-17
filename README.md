# robo-courier-arm

ROS2 robot arm package for autonomous elevator button pressing and package delivery.  
Mounted on a Scout Mini mobile robot as part of an autonomous delivery system.

---

## Features

- **Elevator button pressing** — Detects UP/DOWN and floor buttons using YOLOv8 + EasyOCR, then presses them via analytical IK
- **Pick-and-place delivery** — Picks up a package from a table, reads the room number sign, and delivers to the destination
- **Package recovery** — Re-acquires the FreshBag (insulated delivery bag) onto its hook using YOLO detection
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
├── courier_arm/
│   ├── arm_elevator.py        # Elevator button pressing node
│   ├── arm_delivery.py        # Pick-and-place delivery node
│   ├── arm_recover.py         # FreshBag recovery node (re-hook the delivery bag)
│   ├── contact_detector.py    # Collision detection node
│   ├── detect_room_sign.py    # Room number sign recognition node
│   └── scout.py               # Scout Mini integration skeleton
├── launch/
│   └── elevator.launch.py     # Launches arm_elevator + contact_detector
├── yolo/weights/
│   ├── best.pt                # UP/DOWN button detection model
│   ├── best_num.pt            # Floor number detection model
│   ├── best_room.pt           # Room sign detection model
│   └── best_handle.pt         # Bag handle detection model
└── rooms.yaml                 # Room number → navigation waypoint mapping
```

---

## Installation

```bash
# Clone into your colcon workspace
cd ~/colcon_ws/src
git clone https://github.com/uihyeong/robo-courier-arm.git courier_arm

# Build
cd ~/colcon_ws
colcon build --packages-select courier_arm --symlink-install
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

# 2. Camera (aligned depth + 1080p — required by the nodes)
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  rgb_camera.color_profile:=1920,1080,30

# 3. Camera TF
ros2 run tf2_ros static_transform_publisher \
  --x 0.12 --y 0.01 --z 0.062 \
  --roll 0.0 --pitch 0.0 --yaw 0.0 \
  --frame-id link5 --child-frame-id camera_link

# 4. Run node (+ optional contact detector)
ros2 launch courier_arm elevator.launch.py

# 5. Send target floor
ros2 topic pub --once /target_floor std_msgs/Int32 "{data: 3}"
```

### Delivery Mode

```bash
ros2 run courier_arm arm_delivery

# Trigger pickup (package on table → Scout Mini basket)
ros2 topic pub --once /start_pickup std_msgs/Bool "{data: true}"

# Signal arrival at destination (after Scout Mini aligns)
ros2 topic pub --once /aligned_ready std_msgs/Bool "{data: true}"

# (optional) Trigger the delivery sequence directly
ros2 topic pub --once /start_delivery std_msgs/Bool "{data: true}"
```

### Recovery Mode

```bash
ros2 run courier_arm arm_recover

# Trigger FreshBag recovery (re-hook the bag onto the arm)
ros2 topic pub --once /start_recover std_msgs/Bool "{data: true}"
```

---

## Topic Interface

| Topic | Type | Direction | Node | Description |
|-------|------|-----------|------|-------------|
| `/target_floor` | `Int32` | IN | elevator | Target floor (negative = basement) |
| `/target_point` | `PointStamped` | IN | elevator | Manual world coordinate override |
| `/elevator_ready` | `Bool` | IN | elevator | Scout Mini boarded → start floor button phase |
| `/start_pickup` | `Bool` | IN | delivery | Trigger pickup sequence |
| `/aligned_ready` | `Bool` | IN | delivery | Scout Mini aligned at destination |
| `/start_delivery` | `Bool` | IN | delivery | Trigger delivery sequence directly |
| `/start_recover` | `Bool` | IN | recover | Trigger FreshBag recovery |
| `/robot_status` | `String` | OUT | all | Status string (see below) |
| `/pickup_done` | `Bool` | OUT | delivery | Pickup complete |
| `/delivery_done` | `Bool` | OUT | delivery | Delivery complete |
| `/recover_done` | `Bool` | OUT | recover | Recovery complete |
| `/room_number` | `String` | OUT | delivery / detect_room_sign | Recognized room number (e.g. `"529"`) |
| `/contact_detected` | `Bool` | OUT | contact_detector | Collision detected |
| `/contact_status` | `String` | OUT | contact_detector | `CONTACT_DETECTED` / `CONTACT_RESOLVED` |

**`/robot_status` values by node:**
- **elevator** — `MOVING` / `UPDOWN_PRESSED` / `ELEVATOR_ARRIVED` / `NUMBER_PRESSED` / `NEED_REPOSITION` / `FAILED`
- **delivery** — `MOVING` / `PICKUP` / `PICKUP_DONE` / `ROOM_SIGN` / `WAITING_ALIGN` / `DELIVER` / `DELIVERY_DONE` / `FAILED`
- **recover** — `MOVING` / `RECOVER` / `RECOVER_DONE` / `FAILED`

---

## State Machine

### Elevator Node
```
IDLE → UPDOWN_READY → UPDOWN_PRESS → WAIT → NUMBER_READY → NUMBER_PRESS → NUMBER_WAIT → DONE
```

### Delivery Node
```
IDLE → PICKUP → ROOM_SIGN → WAITING_ALIGN → DELIVER → DONE
```

### Recovery Node
```
IDLE → RECOVER → DONE
```
