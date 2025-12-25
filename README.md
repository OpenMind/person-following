# Person Following System

A real-time person following system using visual tracking and re-identification for robotic applications.

## Overview

This project enables a robot to track and follow a specific person using a RealSense depth camera. The system enrolls a target person and continuously tracks them, with the ability to re-identify the target if they are temporarily lost.

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Main entry point for running the person following system with visualization |
| `person_following_system.py` | Core system that coordinates detection, tracking, and re-identification |
| `target_state.py` | Manages the tracked target's state and stored appearance features |
| `clothing_matcher_lab_openclip.py` | Extracts and matches clothing appearance using color and visual features |
| `yolo_detector.py` | Detects people in camera frames using YOLO |
| `tracked_person_publisher.py` | ROS 2 node that publishes tracked person position for robot navigation |

## Usage

### Basic Usage

### ROS 2 Integration

```bash
python tracked_person_publisher.py --yolo-det yolo11n.engine --yolo-seg yolo11s-seg.engine
```
In another terminal run
```bash
ros2 topic echo /tracked_person/pose
```

If ROS Jazzy not installed

```bash
python main.py --yolo-det yolo11n.engine --yolo-seg yolo11s-seg.engine
```

### Run Without Display

```bash
python main.py --yolo-det yolo11n.engine --yolo-seg yolo11s-seg.engine --no-display
```


## Controls

| Key | Action |
|-----|--------|
| `e` | Enroll the closest person as target (Nearest will auto enroll otherwise set --no-auto-enroll)|
| `c` | Clear current target |
| `s` | Print system status |
| `q` | Quit |

## Installation

```bash
pip install boxmot
pip install open-clip-torch
pip install pyrealsense2
pip install opencv-python numpy
```

For ROS 2 integration:
```bash
pip install rclpy
```

## Requirements

- Intel RealSense camera (D400 series)
- TensorRT engine files for YOLO detection and segmentation
- CUDA-capable GPU

## Install librealsense + pyrealsense2
1) Install build dependencies
```bash
sudo apt update
sudo apt install -y \
  git cmake build-essential pkg-config \
  libusb-1.0-0-dev libudev-dev libssl-dev \
  python3-dev python3-pip \
  libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
```

2) Clone librealsense and install udev rules (permissions)
```bash
cd ~
git clone https://github.com/realsenseai/librealsense.git
cd librealsense
sudo ./scripts/setup_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```
add your user to common device groups:
```bash
sudo usermod -aG plugdev $USER
sudo usermod -aG video $USER
```

3) Build and install librealsense + pyrealsense2
```bash
cd ~/librealsense
git fetch --all --tags
# Choose a stable tag (example):
git checkout v2.57.4

mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_GRAPHICAL_EXAMPLES=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE=$(which python3)

make -j$(nproc)
sudo make install
sudo ldconfig
```

4) Verify librealsense tools
You should now have tools in /usr/local/bin.
```bash
/usr/local/bin/rs-enumerate-devices
```
5) Verify pyrealsense2 import
```bash
python3 -c "import pyrealsense2 as rs; print('pyrealsense2 OK:', rs.__file__)"
```

6) Using pyrealsense2 inside a Python venv
```bash
cd ~/your-project
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

### Install Torch and Tensorrt
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### If .engine not working build again
1) YOLO detection
```bash
pip install ultralytics onnx
yolo export model=yolo11n.pt format=onnx device=cpu
yolo export model=yolo11n.pt format=onnx device=0
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolo11n.onnx \
  --saveEngine=yolo11n_fp16.engine \
  --fp16
```

2) YOLO Segmentation
```bash
yolo export model=yolo11s-seg.pt format=onnx imgsz=640
yolo export model=yolo11s-seg.pt format=engine device=0 half=True imgsz=640
```

3) Convert Ultralytics .engine → raw TensorRT engine (*.raw.engine)
```bash

Ultralytics writes extra metadata into the beginning of the .engine file (a small header + JSON). Native TensorRT loaders (tensorrt.Runtime.deserialize_cuda_engine) expect the file to start with a TensorRT plan header, so they fail with magicTag / serialization errors. Converting to *.raw.engine strips the Ultralytics metadata, leaving a standard TensorRT engine that TensorRT can deserialize normally.

python - <<'PY'
import json

src="yolo11s-seg.engine"
dst="yolo11s-seg.raw.engine"

data=open(src,"rb").read()
if len(data) < 4:
    raise SystemExit("Engine file too small")

meta_len = int.from_bytes(data[:4], "little", signed=True)

# Ultralytics format: [int32 meta_len][meta_json_bytes][tensorrt_engine_bytes]
if 0 < meta_len < 1_000_000 and 4 + meta_len < len(data):
    meta_blob = data[4:4+meta_len]
    # validate it really is JSON metadata
    json.loads(meta_blob.decode("utf-8"))
    engine_bytes = data[4+meta_len:]
    open(dst, "wb").write(engine_bytes)
    print(f"OK: wrote {dst}, bytes={len(engine_bytes)} (stripped Ultralytics metadata {meta_len} bytes)")
else:
    raise SystemExit("Did not detect Ultralytics metadata header. Nothing written.")
PY

```