# Person Following System

A real-time person following system using visual tracking and re-identification for robotic applications.

## Overview

This project enables a robot to track and follow a specific person using a RealSense depth camera. The system enrolls a target person and continuously tracks them, with the ability to re-identify the target if they are temporarily lost.


### Features
Two-Stage Matching

Stage 1: Lab color histogram matching (fast, lighting-robust)
Stage 2: OpenCLIP embedding verification (semantic, cross-view robust)


1. Distance-Bucketed Feature Storage - Features stored at 0.5m intervals with direction awareness
2. Robust Re-identification - Handles occlusions, viewpoint changes, and temporary loss
3. ROS 2 Integration - Publishes tracking status for robot control
4. HTTP Control API - Remote enrollment and management via REST endpoints


### Experiment Setup

Thor Device w/ Jetpack 7 + CUDA

### Core Project Structure

| File | Description |
|------|-------------|
| `main.py` | Main entry point for running the person following system with visualization |
| `person_following_system.py` | Core system that coordinates detection, tracking, and re-identification |
| `target_state.py` | Manages the tracked target's state and stored appearance features |
| `clothing_matcher_lab_openclip.py` | Extracts and matches clothing appearance using color and semantic visual features |
| `yolo_detector.py` | Detects people in camera frames using YOLO |
| `tracked_person_publisher_ros.py` | Realsense-ROS node that publishes tracked person position for robot navigation |
| `person_following_command.py` | Manage and set up HTTP endpoint to send (enroll/clear/quit/status) command to control the system|

### System Architecture
```bash
RealSense D435i → ROS 2 Camera Node → Person Following System
                                            ↓
                                    YOLO11 Detection
                                            ↓
                                    BoTSORT Tracking
                                            ↓
                        ┌───────────────────┴────────────────────┐
                        ↓                                        ↓
              TRACKING_ACTIVE                              SEARCHING
                        ↓                                        ↓
            Feature Extraction                      Re-identification
         (Lab + OpenCLIP @ 0.5m buckets)        (Two-stage matching)
                        ↓                                        ↓
                    ROS Topic                               ROS Topic
              /tracked_person/status                /tracked_person/status
```

### Installation & Running

#### Option 1: Docker (EASIEST)
##### Pull Pre-built Image with Docker Compose
1. **Pull the image**
```bash
docker pull openmindagi/person-following:v0.1.0
```

3. **Start with Docker Compose**
```bash
docker compose up
```

##### OR Build your own docker image
```bash
git clone <repo>
cd person-following-system
docker build -t person-following:latest .

docker run \
  --rm \
  --runtime=nvidia \
  --privileged \
  --network=host \
  -v $(pwd)/engine:/opt/person_following/engine \ (optional)
  -v /dev:/dev \
  person-following:latest \
  bash /opt/person_following/src/start_person_following.sh
```

### Option 2: Local Installation

##### Build librealsense from Source
```bash
# System prerequisites
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update

# Install core build + Python tools
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
  python3-colcon-common-extensions \
  python3-vcstool

# Initialize rosdep
sudo rosdep init || true
rosdep update

# Source ROS
source /opt/ros/jazzy/setup.bash

# Build librealsense from source (recommended on Ubuntu 24.04)
# Install librealsense build dependencies
sudo apt install -y \
  libssl-dev libusb-1.0-0-dev libudev-dev \
  libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev

# Clone + udev rules
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.57.5
sudo ./scripts/setup_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger

# (Optional but helpful)
sudo usermod -aG plugdev $USER
sudo usermod -aG video $USER

# Build
# If you do NOT need pyrealsense2 (ROS topics only), use BUILD_PYTHON_BINDINGS=OFF:
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_WITH_CUDA=ON \
  -DBUILD_PYTHON_BINDINGS=OFF \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF

cmake --build . -j"$(nproc)"
sudo cmake --install .
sudo ldconfig

# Verify installed
pkg-config --modversion realsense2

# Verify device visibility
rs-enumerate-devices
realsense-viewer # on GUI supported env

# Build realsense-ros (ROS 2 wrapper) from source
# Intel’s ROS wrapper instructions recommend building from source.
mkdir -p ~/realsense_ws/src
cd ~/realsense_ws/src
git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master

# Install dependencies via rosdep
cd ~/realsense_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y \
  --rosdistro jazzy \
  --skip-keys=librealsense2

# Build
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

# Source workspace
echo "source ~/realsense_ws/install/setup.bash" >> ~/.bashrc # Optional
source ~/realsense_ws/install/setup.bash

# Launch RealSense ROS node and confirm topics
source /opt/ros/jazzy/setup.bash
source ~/realsense_ws/install/setup.bash

ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true \
  enable_gyro:=false \
  enable_accel:=false

# In another terminal
source /opt/ros/jazzy/setup.bash
ros2 topic list | grep camera
```


##### Install and run person-following
```bash
git clone <repo>
cd person-following-system
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

source /opt/ros/jazzy/setup.bash
source ~/realsense_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true \
  enable_gyro:=false \
  enable_accel:=false

cd /path/to/person-following-sys
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install pycuda open-clip-torch boxmot onnx onnxruntime ultralytics
pip install "numpy==1.26.4" # need to be <2 to make it work


source /opt/ros/jazzy/setup.bash
source ~/realsense_ws/install/setup.bash
source .venv/bin/activate

python3 src/tracked_person_publisher_ros.py \
  --yolo-det ./engine/yolo11n.engine \
  --yolo-seg ./engine/yolo11s-seg.engine \
  --cmd-host 127.0.0.1 \
  --cmd-port 8080 \
  --display (optional)

# Echo status
ros2 topic echo /tracked_person/status

```


## Controls
##### Endpoint HTTP Control
```bash
  curl -X POST http://127.0.0.1:8080/enroll
  curl -X POST http://127.0.0.1:8080/command -H 'Content-Type: application/json' -d '{"cmd":"enroll"}'
  curl -X POST http://127.0.0.1:8080/clear
  curl http://127.0.0.1:8080/status
  curl -X POST http://127.0.0.1:8080/quit
```


##### Keyboard on camera preview
| Key | Action |
|-----|--------|
| `e` | Enroll the closest person as target (Nearest will auto enroll otherwise set --no-auto-enroll)|
| `c` | Clear current target |
| `s` | Print system status |
| `q` | Quit |


## System States

##### INACTIVE
- No target enrolled
- Detection and tracking active, but no person following
- Waiting for enrollment command (You can also set with auto enroll with nearest person with --auto-enroll)

##### TRACKING_ACTIVE
- Target locked and tracked by ID
- Distance-based feature storage (0.5m buckets)
- Movement direction detection (approaching/leaving)
- Automatic feature extraction at key distances

##### SEARCHING
- Target lost (occluded, left frame, or track dropped)
- Re-identification via two-stage matching:
  1. Lab color histogram matching (fast filter)
  2. OpenCLIP embedding verification (semantic confirmation)
- Throttled feature extraction (~3 fps by default)
- Searches all visible persons for best match

##### Feature Storage Strategy

##### Distance Buckets
- Features stored at **0.5m intervals** relative to enrollment distance
- Example: Enrolled at 3.0m → buckets at 2.5m, 3.0m, 3.5m, 4.0m, etc.

##### Direction Awareness
- **Approaching**: Target moving closer to camera
- **Leaving**: Target moving away from camera
- Separate features for each direction (handles different views)

##### Quality Thresholds
- **Saving**: Minimum 35% mask coverage
- **Matching**: Minimum 30% mask coverage
- Frame margin: 20px from left/right edges

##### Matching Process

##### Active Tracking
1. Locate target by tracker ID
2. Detect movement direction (speed > 0.4 m/s)
3. Check if new bucket should be saved
4. Extract features if conditions met:
   - New bucket not yet populated
   - Within frame margins
   - Time since last save > 0.3s
5. Continue tracking

##### Re-identification (Searching)
1. **Stage 1: Clothing Filter**
   - Extract Lab histograms from all candidates
   - Compute similarity vs. stored features
   - Keep candidates with similarity ≥ threshold (default 0.8)

2. **Stage 2: CLIP Verification**
   - Extract OpenCLIP embeddings
   - Compute cosine similarity
   - Verify candidates with similarity ≥ threshold (default 0.8)

3. **Selection**
   - Choose candidate with highest CLIP similarity
   - Resume tracking with new track ID




## Acknowledgments

- [YOLO](https://github.com/ultralytics/ultralytics) - Object detection
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) - Multi-object tracking
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Vision-language embeddings
- [Intel RealSense](https://github.com/IntelRealSense/librealsense) - Depth camera SDK
- [ROS 2](https://docs.ros.org/en/jazzy/) - Robot Operating System
