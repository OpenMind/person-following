# Person Following System

A real-time person following and greeting system using visual tracking, re-identification, geofencing, and OM1 robot integration.

## Overview

This project enables a robot to track and follow a specific person using a camera and optional LiDAR. It supports two operation modes:

- **Greeting Mode** — Robot proactively approaches new people, greets them (via OM1), remembers greeted persons to avoid re-greeting, and switches to the next ungreeted person. Includes geofencing to keep the robot within a defined area.
- **Following Mode** — Robot continuously tracks and follows an enrolled target without history or greeting logic.

### Features

**Two-Stage Appearance Matching**
- Stage 1: Lab color histogram matching (fast, lighting-robust)
- Stage 2: OpenCLIP embedding verification (semantic, cross-view robust)

**Distance-Bucketed Feature Storage**
- Features stored at 0.5m intervals with direction awareness (approaching / leaving)

**Greeting System with History**
- Remembers greeted persons across sessions (persisted via pickle file)
- Fast-accept with background history verification to avoid re-greeting
- Zenoh-based handshake with OM1 (`om/person_greeting` topic)

**Geofencing (Greeting Mode Only)**
- Hard boundary: blocks forward movement beyond radius
- Soft boundary: scales down speed in transition zone
- Auto-return to center after exhausting search rotations
- Obstacle-aware return navigation via `/om/paths`

**Multi-Robot Support**
- Unitree Go2 (front camera + RPLidar)
- Unitree G1 (Insta360 camera + RPLidar)
- Tron robot (USB camera + RPLidar)
- Intel RealSense D435i (depth camera)

**ROS 2 Integration** — Publishes tracking status and position for robot control
**HTTP Control API** — Remote control via REST endpoints (two servers: vision and follower)

---

## System Architecture

### Greeting Mode (Full System)

```
Camera + LiDAR/Depth ──► tracked_person_publisher_ros.py
                                      │
                         ┌────────────▼────────────┐
                         │    PersonFollowingSystem  │
                         │  YOLO11n Detection        │
                         │  BoTSORT / ByteTrack      │
                         │  Two-stage Matching       │
                         │  History (pickle file)    │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                   ▼
             /tracked_person    /tracked_person    /tracked_person
               /status           /position          /detection_image
                    │                 │
                    └────────┬────────┘
                             ▼
               person_follow_greet.py
              ┌──────────────────────────────┐
              │  State Machine               │
              │  IDLE → SWITCHING →          │
              │  APPROACHING →               │
              │  GREETING_IN_PROGRESS        │
              │  SEARCHING                   │
              │  RETURNING_TO_CENTER         │
              ├──────────────────────────────┤
              │  GeofenceManager             │
              │  MotionController (PD)       │
              │  SearchBehavior              │
              │  Zenoh ↔ OM1                 │
              └─────────────┬────────────────┘
                            │
                     /cmd_vel (Twist)
```

### Following Mode (Simple)

```
Camera + LiDAR  ──┐
                  ├──► tracked_person_publisher_ros.py ──► /tracked_person/position
RealSense Depth ──┘           (--mode following)                    │
                                                         person_follower.py
                                                         PD Control ──► /cmd_vel
```

---

## Core Project Structure

| File | Description |
|------|-------------|
| `person_following/nodes/tracked_person_publisher_ros.py` | Main vision node: YOLO detection, tracking, feature extraction, ROS publisher, HTTP command server (port 2001) |
| `person_following/nodes/person_following_system.py` | Core tracking logic: detection, BoTSORT/ByteTrack, two-stage re-id, history, switch/greeting state |
| `person_following/nodes/person_follow_greet.py` | Full greeting node: state machine, geofencing, Zenoh OM1 integration, obstacle avoidance, HTTP server (port 2000) |
| `person_following/nodes/person_follower.py` | Simple following node: PD control, safety timeout (for following mode only) |
| `person_following/managers/geofence_manager.py` | Geofence boundary enforcement, soft/hard radius, return target generation |
| `person_following/managers/model_manager.py` | TensorRT engine auto-compilation from ONNX models |
| `person_following/controllers/motion_controller.py` | PD motion controller with obstacle-aware path selection |
| `person_following/utils/clothing_matcher_lab_openclip.py` | Lab histogram + OpenCLIP feature extraction and matching |
| `person_following/utils/yolo_detector.py` | TensorRT YOLO person detection |
| `person_following/utils/target_state.py` | Target state: distance buckets, direction, feature storage |
| `person_following/utils/state_machine.py` | Follower state machine: `FollowerState`, `SearchBehavior`, `ReturnToCenter` |
| `person_following/utils/switch_state.py` | Switch candidate queue management |
| `person_following/utils/http_server.py` | HTTP server for follower node mode/geofence control |
| `person_following/utils/zenoh_msgs.py` | Zenoh message types for OM1 `om/person_greeting` topic |
| `person_following/controllers/person_following_command.py` | HTTP command server for vision node (enroll, switch, greeting_ack, etc.) |
| `launch/greeting_launch.py` | Launch file for full greeting system (multi-robot) |
| `launch/person_following_launch.py` | Launch file for simple person following (RealSense) |
| `config/<robot>_params.yaml` | Per-robot ROS parameters (go2, g1, tron) |
| `extrinsics-files/` | LiDAR-camera extrinsics per robot |
| `intrinsics-files/` | Camera intrinsics per robot |

---

## Operation Modes

### Greeting Mode
The robot autonomously greets new persons in sequence:

1. **IDLE** — Waiting for `SWITCH` command from OM1 via Zenoh
2. **SWITCHING** — Vision system selects a candidate (skipping previously greeted persons). Robot waits for tracking result. Geofence check: rejects candidates outside boundary.
3. **APPROACHING** — Robot moves toward the person using PD control with obstacle avoidance. Geofence constraints applied.
4. **GREETING_IN_PROGRESS** — Person has approached within threshold. Robot signals OM1 (`APPROACHED`). Saves person features to history.
5. **SEARCHING** — No person found: robot rotates in steps, pauses, calls switch. After max rotations, returns to center.
6. **RETURNING_TO_CENTER** — Robot navigates back inside geofence using obstacle-aware path planning.

### Following Mode
Simple continuous tracking — works with any supported camera input (RealSense depth or camera + LiDAR):
- Enroll a target (nearest person or via HTTP)
- Robot follows using PD control, stops if tracking lost for > timeout seconds
- No history, no switch, no geofencing, no Zenoh

---

## Tracking System States (Vision)

| State | Description |
|-------|-------------|
| `INACTIVE` | No target enrolled. Detection and tracking running, awaiting command. |
| `TRACKING_ACTIVE` | Target locked by track ID. Features extracted at 0.5m distance buckets. Background history verification in progress (greeting mode). |
| `SEARCHING` | Target lost. Re-identification via two-stage matching at ~3 Hz. Greeting mode times out after `--searching-timeout` seconds (default 5s). Following mode searches indefinitely. |
| `SWITCHING` | Finding next ungreeted candidate. Fast-accept with background verification. Skips persons already in history. |

---

## Greeting System — How It Works

### Person Identification & History
- **Enrollment at approach**: When a person is accepted as target, their appearance features (Lab histograms + CLIP embeddings) are saved at distance buckets.
- **History**: After successful greeting (`greeting_ack`), the person's features are saved to a pickle file (persists across restarts).
- **History check during switch**: New candidates are immediately accepted (fast path), then verified against history in the background at ~3 Hz for up to 3 checks. If a match is found, the target is rejected and the system goes inactive.

### Zenoh Handshake with OM1
```
OM1 ──► om/person_greeting (status=SWITCH=2) ──► Start/continue person search
Robot ──► om/person_greeting (status=APPROACHED=1) ──► Person greeted, OM1 can now interact
```

### Switch Logic
1. OM1 sends `SWITCH` signal
2. Vision system scans visible persons, sorts by distance
3. Fast-accepts first candidate (nearest, non-excluded, inside geofence)
4. Runs 3 background verification checks against history
5. If in history → clear target, go INACTIVE
6. If not in history → continue TRACKING_ACTIVE

---

## Geofencing

The geofence center is set automatically from the first odometry reading.

| Zone | Behavior |
|------|----------|
| Inside soft radius | Normal movement |
| Soft zone (soft_radius → hard_radius) | Forward speed scaled down linearly |
| At hard boundary | Forward movement blocked |
| Person outside geofence | Candidate rejected, search continues |
| Max rotations exceeded | Return to center |

**Default Parameters (Go2):**
- Hard radius: 30m
- Soft radius: 28m
- Return speed: 0.4 m/s
- Max search rotations: 48

---

## Obstacle Avoidance

The system uses the `/om/paths` topic (published by the OM1 stack) to detect blocked directions and react accordingly. Path data is considered stale after 1 second of no updates, at which point the path is treated as blocked.

### Path Directions

OM1 provides 10 discrete path directions indexed 0–9:

```
Index:  0     1     2     3     4     5     6     7     8     9
Angle: 60°   45°   30°   15°   0°  -15°  -30°  -45°  -60°  180°
       (left)                (fwd)                  (right) (back)
```

The `Paths` message contains:
- `paths` — list of available (unblocked) path indices
- `blocked_by_obstacle_idx` — paths blocked by obstacles
- `blocked_by_hazard_idx` — paths blocked by hazards

### During APPROACHING State

When the robot is moving toward a person, the `MotionController` checks only whether the **forward path** (index 4, 0°) is in the safe paths list:

1. **Within `target_distance + 0.7m` of person** → obstacle check is skipped entirely, move freely
2. **Forward path safe** → move normally toward person
3. **Forward path blocked** → stop, call `/switch` on vision system, transition to `SEARCHING`

No alternative angles are attempted during approach — if the forward path is blocked, the robot immediately abandons the current target and searches for another person.

### During RETURNING_TO_CENTER State

When returning to center after exceeding the geofence, the `ReturnToCenter` module checks path safety on the route back:

| Duration blocked | Behavior |
|-----------------|----------|
| 0 – 5s | Stop and wait for path to clear |
| 5 – 30s | Try alternative angles (±15°, ±30°, ±45°) |
| > 30s | Give up return, transition to `SEARCHING` from current position |

---

## ROS Topics

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/tracked_person/status` | `std_msgs/String` (JSON) | Tracking state, position, approached flag, operation mode, num persons |
| `/person_following_robot/tracked_person/position` | `geometry_msgs/PoseStamped` | Person position relative to robot (x=lateral, z=forward) |
| `/tracked_person/detection_image` | `sensor_msgs/Image` | Visualization image with bounding boxes and overlays |
| `/cmd_vel` | `geometry_msgs/Twist` | Robot velocity commands |
| `/person_follower/state` | `std_msgs/String` | Current follower state machine state |
| `/api/sport/request` | `unitree_api/Request` | Unitree sport mode control (classical walk enable/disable) |
| `om/person_greeting` (Zenoh) | `PersonGreetingStatus` | Greeting handshake status (APPROACHED=1) |

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/.../color/image_raw` | `sensor_msgs/Image` | Color camera feed (RealSense mode) |
| `/camera/.../depth/image_rect_raw` | `sensor_msgs/Image` | Aligned depth feed (RealSense mode) |
| `/camera/go2/image_raw/best_effort` | `sensor_msgs/Image` | Go2 front camera |
| `/camera/insta360/image_raw` | `sensor_msgs/Image` | G1 Insta360 camera |
| `/image_raw` | `sensor_msgs/Image` | Tron USB camera |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR scan (Go2/G1/Tron mode) |
| `/odom` | `nav_msgs/Odometry` | Robot odometry (for geofencing) |
| `/om/paths` | `om_api/Paths` | Safe/blocked path info (obstacle avoidance) |
| `/sportmodestate` | `unitree_go/SportModeState` | Unitree sport mode state |
| `om/person_greeting` (Zenoh) | `PersonGreetingStatus` | Greeting commands from OM1 (SWITCH=2) |

---

## HTTP Control APIs

### Vision System — Port 2001

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | System status (tracking state, position, history, etc.) |
| `/enroll` | POST | Enroll nearest person as target |
| `/clear` | POST | Clear current target (no history save) |
| `/switch` | POST | Switch to next ungreeted person (greeting mode only) |
| `/greeting_ack` | POST | Acknowledge greeting: save features to history, clear target |
| `/clear_history` | POST | Clear history from memory |
| `/delete_history` | POST | Clear history and delete file |
| `/save_history` | POST | Save history to file |
| `/load_history` | POST | Load history from file |
| `/set_max_history` | POST | Set max history size `{"size": N}` |
| `/command` | POST | Generic command `{"cmd": "set_mode", "mode": "greeting"\|"following"}` |
| `/quit` | POST | Shutdown vision node |

### Follower Node — Port 2000

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Follower state and operation mode |
| `/healthz` | GET | Health check |
| `/get_mode` | GET | Get current operation mode |
| `/geofence` | GET | Geofence status (center, radius, distance, boundaries) |
| `/set_mode` | POST | Set mode `{"mode": "greeting"\|"following"}` |
| `/command` | POST | Generic command (set_mode) |
| `/geofence/reset_center` | POST | Reset geofence center to current position |
| `/geofence/enable` | POST | Enable geofencing |
| `/geofence/disable` | POST | Disable geofencing |

---

## Installation & Running

### Option 1: Docker (Easiest)

```bash
# Set robot type (go2, g1, or tron)
export ROBOT_TYPE=go2

# Pull and run
docker pull openmindagi/person_following:latest
docker compose up
```

Or build your own image:

```bash
git clone <repo>
cd person-following
docker build -t person-following:latest .

docker run \
  --rm \
  --runtime=nvidia \
  --privileged \
  --network=host \
  -e ROBOT_TYPE=go2 \
  -v /dev:/dev \
  person-following:latest \
  ros2 launch /opt/person_following/launch/greeting_launch.py
```

### Option 2: Local Installation

#### Prerequisites

```bash
# System dependencies
sudo apt update
sudo apt install -y \
  git cmake build-essential pkg-config ninja-build \
  python3-pip python3-venv python3-dev \
  curl ca-certificates gnupg lsb-release

# Install ROS 2 Jazzy (Ubuntu 24.04)
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y \
  ros-jazzy-ros-base \
  python3-rosdep \
  python3-colcon-common-extensions

sudo rosdep init || true
rosdep update
source /opt/ros/jazzy/setup.bash
```

#### Python Environment

```bash
git clone <repo>
cd person-following

# Using uv (recommended)
uv venv --python /usr/bin/python3 --system-site-packages
uv sync --all-extras
source .venv/bin/activate

# Or using pip
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install pycuda open-clip-torch boxmot onnx onnxruntime ultralytics
pip install "numpy==1.26.4"  # must be <2.0
```

#### Build ROS Message Packages

The greeting system requires `om_api` and `unitree_api` message packages:

```bash
cd person-following
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

#### Running — Greeting Mode (recommended)

```bash
export ROBOT_TYPE=go2   # or g1, tron
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source .venv/bin/activate

ros2 launch launch/greeting_launch.py
```

#### Running — Simple Following Mode

Following mode works with any supported camera input — RealSense depth camera or Go2/G1/Tron camera + LiDAR. The `--mode following` flag is independent of camera mode.

**With RealSense (depth camera):**
```bash
# Launch RealSense camera node first
source /opt/ros/jazzy/setup.bash
source ~/realsense_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true \
  enable_gyro:=false \
  enable_accel:=false

# Then launch person following
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
ros2 launch launch/person_following_launch.py \
  yolo_det:=./engine/yolo11n.engine \
  yolo_seg:=./engine/yolo11s-seg.engine
```

**With Go2/G1/Tron (camera + LiDAR):**
```bash
export ROBOT_TYPE=go2   # or g1, tron
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source .venv/bin/activate

python3 person_following/nodes/tracked_person_publisher_ros.py \
  --mode following \
  --cmd-port 2001

# In another terminal, start the follower node
python3 person_following/nodes/person_follower.py
```

#### Running — Vision Node Only (standalone)

```bash
python3 person_following/nodes/tracked_person_publisher_ros.py \
  --mode greeting \
  --cmd-port 2001 \
  --scan-topic /scan \
  --display
```

---

## Configuration

### Per-Robot Parameters

Robot-specific parameters are in `config/<ROBOT_TYPE>_params.yaml`. Set `ROBOT_TYPE` environment variable to select:

| Variable | Default (Go2) | Description |
|----------|--------------|-------------|
| `target_distance` | 0.8m | Desired following distance |
| `max_linear_speed` | 0.7 m/s | Max forward speed |
| `max_angular_speed` | 1.6 rad/s | Max rotation speed |
| `geofence_enabled` | true | Enable geofencing |
| `geofence_radius` | 30.0m | Hard boundary radius |
| `geofence_soft_radius` | 28.0m | Soft boundary (speed scaling starts here) |
| `geofence_return_speed` | 0.4 m/s | Return-to-center speed |
| `geofence_max_search_rotations` | 48 | Rotations before returning to center |
| `search_rotation_angle` | 15.0° | Degrees per search rotation step |
| `search_wait_time` | 1.0s | Pause between search rotations |
| `cmd_port` | 2001 | Vision system HTTP port |
| `http_port` | 2000 | Follower HTTP port |

### Calibration Files

Each robot requires calibration files in:
- `intrinsics-files/camera_intrinsics_<robot>.yaml` — Camera matrix and image size
- `extrinsics-files/lidar_camera_extrinsics_<robot>.yaml` — LiDAR-to-camera transform (translation + rotation_euler)

### Key Vision Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `greeting` | Operation mode: `greeting` or `following` |
| `--clothing-threshold` | 0.8 | Lab color histogram similarity threshold |
| `--clip-threshold` | 0.8 | OpenCLIP cosine similarity threshold |
| `--min-mask-coverage` | 35% | Minimum segmentation mask coverage for feature extraction |
| `--approach-distance` | 1.0m | Distance threshold to trigger `approached` flag |
| `--searching-timeout` | 5.0s | Time before giving up search (greeting mode) |
| `--max-history-size` | 1 | Max persons to remember in history |
| `--switch-interval` | 0.3s | Feature check interval during switch (~3 Hz) |
| `--switch-timeout` | 1.0s | Max time per candidate during switch |
| `--tracker` | `botsort` | Tracker: `botsort` or `bytetrack` |

---

## Controls

### Keyboard (when `--display` is enabled)

| Key | Action | Mode |
|-----|--------|------|
| `e` | Enroll nearest person as target | Both |
| `c` | Clear current target | Both |
| `m` | Toggle between greeting / following mode | Both |
| `s` | Print status | Both |
| `q` | Quit | Both |
| `w` | Switch to next ungreeted person | Greeting only |
| `g` | Greeting acknowledge (save to history) | Greeting only |
| `h` | Clear history (memory only) | Greeting only |

### Vision System HTTP (port 2001)

```bash
# --- Status / Info ---
curl http://127.0.0.1:2001/healthz
curl http://127.0.0.1:2001/status
curl http://127.0.0.1:2001/get_mode

# --- Target Control ---
curl -X POST http://127.0.0.1:2001/enroll            # Enroll nearest person as target
curl -X POST http://127.0.0.1:2001/clear             # Clear current target (no history save)

# --- Mode ---
curl -X POST http://127.0.0.1:2001/command \
  -H 'Content-Type: application/json' \
  -d '{"cmd":"set_mode","mode":"greeting"}'
curl -X POST http://127.0.0.1:2001/command \
  -H 'Content-Type: application/json' \
  -d '{"cmd":"set_mode","mode":"following"}'

# --- Greeting mode only ---
curl -X POST http://127.0.0.1:2001/switch            # Switch to next ungreeted person
curl -X POST http://127.0.0.1:2001/greeting_ack      # Acknowledge greeting, save to history
curl -X POST http://127.0.0.1:2001/clear_history     # Clear history from memory
curl -X POST http://127.0.0.1:2001/delete_history    # Clear memory + delete history file
curl -X POST http://127.0.0.1:2001/save_history      # Force save history to file
curl -X POST http://127.0.0.1:2001/load_history      # Force load history from file
curl -X POST http://127.0.0.1:2001/command \
  -H 'Content-Type: application/json' \
  -d '{"cmd":"set_max_history","size":5}'            # Set max history size

# --- Lifecycle ---
curl -X POST http://127.0.0.1:2001/quit              # Shutdown vision node
```

### Follower HTTP (port 2000)

```bash
# --- Status / Info ---
curl http://127.0.0.1:2000/healthz
curl http://127.0.0.1:2000/status                    # State + operation mode
curl http://127.0.0.1:2000/get_mode

# --- Mode ---
curl -X POST http://127.0.0.1:2000/set_mode \
  -H 'Content-Type: application/json' \
  -d '{"mode":"following"}'
curl -X POST http://127.0.0.1:2000/set_mode \
  -H 'Content-Type: application/json' \
  -d '{"mode":"greeting"}'

# --- Geofence ---
curl http://127.0.0.1:2000/geofence                  # Geofence status (center, radius, distance)
curl -X POST http://127.0.0.1:2000/geofence/enable   # Enable geofencing
curl -X POST http://127.0.0.1:2000/geofence/disable  # Disable geofencing
curl -X POST http://127.0.0.1:2000/geofence/reset_center  # Reset center to current position
```

---

## Feature Matching Details

### Distance Buckets
- Features stored at **0.5m intervals** from enrollment distance
- Separate buckets for **approaching** and **leaving** directions (handles front/back view differences)
- Quality threshold: ≥35% segmentation mask coverage to save, ≥30% to match (≥25% minimum to attempt extraction during search)
- Frame margin: 20px from left/right edges (partial persons excluded)

### Two-Stage Matching

**Stage 1: Lab Color Histogram Filter**
- Extracts Lab color histograms from segmentation mask
- Fast comparison (< 1ms)
- Default threshold: 0.8 cosine similarity

**Stage 2: OpenCLIP Semantic Verification**
- ViT-B-16 model with laion2b_s34b_b88k weights
- Computes cosine similarity of embeddings
- Default threshold: 0.8

**Selection**: Highest CLIP similarity among Stage 1 + Stage 2 passed candidates.

### History Verification (Greeting Mode)
- After fast-accepting a switch candidate, runs 3 background checks at ~3 Hz
- Uses both clothing + CLIP — both must pass for a history match
- If match found: clears target (person already greeted), goes INACTIVE
- If 3 checks pass without match: confirms new person, continues tracking

---

## Optional: Download OpenCLIP Model Locally

```bash
python - <<'PY'
import open_clip

open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="laion2b_s34b_b88k",
    device="cpu",
    cache_dir="./model",
)
print("Downloaded into: ./model")
PY
```

---

## Supported Robots

| Robot | Camera | Distance | Config |
|-------|--------|----------|--------|
| Unitree Go2 | Front camera (BGR) + RPLidar | LiDAR clusters | `config/go2_params.yaml` |
| Unitree G1 | Insta360 + RPLidar | LiDAR clusters | `config/g1_params.yaml` |
| Tron | USB camera + RPLidar | LiDAR clusters | `config/tron_params.yaml` |
| Any robot | Intel RealSense D435i | Depth (aligned) | Via `--camera-mode realsense` |

---

## Acknowledgments

- [YOLO](https://github.com/ultralytics/ultralytics) — Object detection
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) — Multi-object tracking (BoTSORT / ByteTrack)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — Vision-language embeddings
- [Intel RealSense](https://github.com/IntelRealSense/librealsense) — Depth camera SDK
- [ROS 2](https://docs.ros.org/en/jazzy/) — Robot Operating System
- [Zenoh](https://zenoh.io/) — OM1 messaging integration
