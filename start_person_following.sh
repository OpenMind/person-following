#!/usr/bin/env bash
# Auto-start script for person following system (ROS Jazzy + realsense-ros + HTTP control)
# Strong cleanup guarantee: kills ros2 launch + realsense2_camera_node via process-group kill.
# Shows tracker output in the terminal.

set -euo pipefail

echo "=========================================="
echo "  Person Following System - Auto Start"
echo "=========================================="
echo ""


# Config
PROJECT_ROOT="${PROJECT_ROOT:-/opt/person_following}"
ENGINE_DIR="${ENGINE_DIR:-${PROJECT_ROOT}/engine}"

YOLO_DET="${YOLO_DET:-${ENGINE_DIR}/yolo11n.engine}"
YOLO_SEG="${YOLO_SEG:-${ENGINE_DIR}/yolo11s-seg.engine}"

COLOR_TOPIC="${COLOR_TOPIC:-/camera/camera/color/image_raw}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/camera/aligned_depth_to_color/image_raw}"
INFO_TOPIC="${INFO_TOPIC:-/camera/camera/color/camera_info}"

CMD_HOST="${CMD_HOST:-0.0.0.0}"
CMD_PORT="${CMD_PORT:-8080}"

REALSENSE_LOG="${REALSENSE_LOG:-/tmp/realsense.log}"
TRACKER_LOG="${TRACKER_LOG:-/tmp/tracker.log}"

# Optional: set to 1 ONLY if you want last-resort pkill by name (may kill other realsense nodes!)
AGGRESSIVE_CLEANUP="${AGGRESSIVE_CLEANUP:-0}"


REALSENSE_PID=""
TRACKER_PID=""
CLEANED_UP=0

# Helpers
kill_tree_by_pgid() {
  # Kill entire process group of a PID (works best when started via `setsid`).
  local pid="${1:-}"
  if [[ -z "${pid}" ]]; then return 0; fi
  if ! kill -0 "${pid}" 2>/dev/null; then return 0; fi

  local pgid
  pgid="$(ps -o pgid= "${pid}" 2>/dev/null | tr -d ' ' || true)"
  if [[ -z "${pgid}" ]]; then
    kill "${pid}" 2>/dev/null || true
    return 0
  fi

  # Graceful stop
  kill -TERM "--" "-${pgid}" 2>/dev/null || true

  # Wait a bit
  for _ in {1..30}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done

  # Force kill
  kill -KILL "--" "-${pgid}" 2>/dev/null || true
}

wait_for_topic() {
  local topic="$1"
  local timeout="${2:-25}"
  local start
  start="$(date +%s)"

  echo "[start] Waiting for topic: ${topic} (timeout=${timeout}s)"
  while true; do
    if ros2 topic list 2>/dev/null | grep -qx "${topic}"; then
      echo "[start] Topic ready: ${topic}"
      return 0
    fi
    local now
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      echo "[start] ERROR: Timeout waiting for topic: ${topic}"
      return 1
    fi
    sleep 0.5
  done
}

cleanup() {
  # Run once
  if [[ "${CLEANED_UP}" == "1" ]]; then
    return 0
  fi
  CLEANED_UP=1

  echo ""
  echo "[start] Cleanup..."

  # Try graceful quit for tracker first (best effort)
  curl -s -X POST "http://${CMD_HOST}:${CMD_PORT}/quit" >/dev/null 2>&1 || true
  sleep 0.3

  if [[ -n "${TRACKER_PID}" ]]; then
    echo "[start] Stopping tracker group (pid=${TRACKER_PID})..."
    kill_tree_by_pgid "${TRACKER_PID}" || true
  fi

  if [[ -n "${REALSENSE_PID}" ]]; then
    echo "[start] Stopping realsense launch group (pid=${REALSENSE_PID})..."
    kill_tree_by_pgid "${REALSENSE_PID}" || true
  fi

  # Last resort (optional, may kill unrelated processes!)
  if [[ "${AGGRESSIVE_CLEANUP}" == "1" ]]; then
    echo "[start] Aggressive cleanup enabled: pkill realsense2_camera_node"
    pkill -f realsense2_camera_node 2>/dev/null || true
    pkill -f "ros2.*realsense2_camera" 2>/dev/null || true
  fi

  echo "[start] Cleanup done."
}

# Ensure cleanup happens on:
# - Normal exit (tracker quit/crash/script completion)
# - Ctrl+C (SIGINT)
# - docker stop (SIGTERM)
trap cleanup EXIT SIGINT SIGTERM

# Source ROS safely under set -u
set +u
source /opt/ros/jazzy/setup.bash
source /opt/realsense_ws/install/setup.bash
set -u

echo "[start] Python: $(python3 -c 'import sys; print(sys.executable)' 2>/dev/null || echo 'unknown')"
echo ""

# Check engines
if [[ ! -f "${YOLO_DET}" ]]; then
  echo "[start] ERROR: yolo det engine not found: ${YOLO_DET}"
  exit 1
fi
if [[ ! -f "${YOLO_SEG}" ]]; then
  echo "[start] ERROR: yolo seg engine not found: ${YOLO_SEG}"
  exit 1
fi

echo "[start] Engine files found:"
echo "  - ${YOLO_DET}"
echo "  - ${YOLO_SEG}"
echo ""

# Start RealSense camera node (new session => new process group)
# (keeps logs in file; enable tee if you want it in terminal too)
echo "[start] Starting RealSense camera node..."
setsid ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true \
  enable_gyro:=false \
  enable_accel:=false \
  > "${REALSENSE_LOG}" 2>&1 &

REALSENSE_PID=$!
echo "[start] RealSense launch PID: ${REALSENSE_PID}"
echo "[start] Logs: ${REALSENSE_LOG}"

# Wait for topics instead of fixed sleep
if ! wait_for_topic "${COLOR_TOPIC}" 25; then
  echo "[start] ERROR: RealSense failed to publish color topic."
  tail -n 120 "${REALSENSE_LOG}" || true
  exit 1
fi

if ! wait_for_topic "${DEPTH_TOPIC}" 25; then
  echo "[start] ERROR: RealSense failed to publish depth topic."
  tail -n 120 "${REALSENSE_LOG}" || true
  exit 1
fi

echo ""
echo "[start] RealSense camera ready"
echo ""

# Start tracker (new session => new process group)
# Show tracker output in terminal AND save to /tmp/tracker.log
echo "[start] Starting person following tracker..."
echo "[start] Default waits for ENROLL command (HTTP)."
echo ""

export PYTHONUNBUFFERED=1

setsid bash -lc \
  "python3 -u '${PROJECT_ROOT}/person_following/tracked_person_publisher_ros.py' \
    --yolo-det '${YOLO_DET}' \
    --yolo-seg '${YOLO_SEG}' \
    --color-topic '${COLOR_TOPIC}' \
    --depth-topic '${DEPTH_TOPIC}' \
    --camera-info-topic '${INFO_TOPIC}' \
    --cmd-host '${CMD_HOST}' \
    --cmd-port '${CMD_PORT}'" \
  &

TRACKER_PID=$!
echo "[start] Tracker launch PID: ${TRACKER_PID}"

sleep 1
if ! kill -0 "${TRACKER_PID}" 2>/dev/null; then
  echo "[start] ERROR: tracker failed to start."
  exit 1
fi

echo ""
echo "=========================================="
echo "  System Ready!"
echo "=========================================="
echo ""
echo "ROS Topics:"
echo "  - /tracked_person/status   (std_msgs/String JSON)"
echo "  - ${COLOR_TOPIC}"
echo "  - ${DEPTH_TOPIC}"
echo ""
echo "HTTP Control:"
echo "  Base:   http://${CMD_HOST}:${CMD_PORT}"
echo "  enroll: curl -X POST http://${CMD_HOST}:${CMD_PORT}/command -H 'Content-Type: application/json' -d '{\"cmd\":\"enroll\"}'"
echo "  clear:  curl -X POST http://${CMD_HOST}:${CMD_PORT}/clear"
echo "  status: curl http://${CMD_HOST}:${CMD_PORT}/status"
echo "  quit:   curl -X POST http://${CMD_HOST}:${CMD_PORT}/quit"
echo ""
echo "To view topics:  ros2 topic list"
echo "To echo status:  ros2 topic echo /tracked_person/status"
echo ""
echo "[start] Running... (Ctrl+C to stop)"
echo ""

# Keep script running until tracker exits. EXIT trap will cleanup realsense too.
wait "${TRACKER_PID}"
