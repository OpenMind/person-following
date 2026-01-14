#!/usr/bin/env python3
"""
ROS 2 Tracked Person Publisher for Robot Dog Following.

Subscribes to realsense2_camera_node ROS topics for camera input, allowing the camera
to be shared with other ROS nodes.

Publishes
---------
/tracked_person/status : std_msgs/String (JSON)
  - is_tracked : bool
  - x          : float  (lateral offset in meters; +right / -left)
  - z          : float  (distance forward in meters)

/person_following_robot/tracked_person/position : geometry_msgs/PoseStamped
  - Position in camera_color_optical_frame

/tracked_person/detection_image : sensor_msgs/Image
  - BGR8 annotated image with detection visualization
  - Includes bounding boxes, tracking IDs, status overlay
  - Use with rqt_image_view or rviz for headless debugging

Control API
-----------------
This script now exposes a small HTTP control API (no extra deps) that lets a host
machine send commands into the running process:

Commands: enroll | clear | status | quit

Examples (when using `--network host`):
  curl -X POST http://127.0.0.1:8080/command -H 'Content-Type: application/json' -d '{"cmd":"enroll"}'
  curl -X POST http://127.0.0.1:8080/enroll
  curl -X POST http://127.0.0.1:8080/clear
  curl http://127.0.0.1:8080/status
  curl -X POST http://127.0.0.1:8080/quit


Default is to WAIT for an explicit enroll command (keyboard 'e' or HTTP 'enroll').
If you still want the previous behavior, start with `--auto-enroll`.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import signal
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from person_following_command import Command, CommandServer, SharedStatus

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tracked_person_publisher_ros")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Person Following with ROS 2")

    # Model paths
    p.add_argument(
        "--yolo-det",
        type=str,
        default="/opt/person_following/engine/yolo11n.engine",
        help="Path to YOLO detection TensorRT engine",
    )
    p.add_argument(
        "--yolo-seg",
        type=str,
        default="/opt/person_following/engine/yolo11s-seg.engine",
        help="Path to YOLO segmentation TensorRT engine",
    )

    # OpenCLIP settings
    p.add_argument(
        "--clip-model", type=str, default="ViT-B-16", help="OpenCLIP model name"
    )
    p.add_argument(
        "--clip-pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="OpenCLIP pretrained weights",
    )

    # Thresholds
    p.add_argument(
        "--clothing-threshold",
        type=float,
        default=0.8,
        help="Lab clothing similarity threshold",
    )
    p.add_argument(
        "--clip-threshold",
        type=float,
        default=0.8,
        help="OpenCLIP similarity threshold",
    )
    p.add_argument(
        "--min-mask-coverage",
        type=float,
        default=35.0,
        help="Minimum mask coverage percentage",
    )
    p.add_argument(
        "--search-interval",
        type=float,
        default=0.33,
        help="Search mode feature extraction interval (seconds)",
    )

    # Tracker
    p.add_argument(
        "--tracker",
        type=str,
        default="botsort",
        choices=["botsort", "bytetrack"],
        help="Tracker type",
    )

    # ROS camera topic configuration
    p.add_argument(
        "--color-topic",
        type=str,
        default="/camera/camera/color/image_raw",
        help="ROS color image topic",
    )
    p.add_argument(
        "--depth-topic",
        type=str,
        default="/camera/camera/aligned_depth_to_color/image_raw",
        help="ROS depth image topic (aligned depth)",
    )
    p.add_argument(
        "--camera-info-topic",
        type=str,
        default="/camera/camera/color/camera_info",
        help="ROS camera info topic",
    )
    p.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="Depth scale (uint16 to meters), default 0.001",
    )

    # Display and output
    p.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Enable visualization window",
    )
    p.add_argument(
        "--save-video", type=str, default=None, help="Save output to video file"
    )
    p.add_argument(
        "--publish-hz", type=float, default=15.0, help="ROS 2 publish rate (Hz)"
    )

    # Enrollment behavior (default OFF; enroll via endpoint or keypress)
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--auto-enroll",
        action="store_true",
        default=False,
        help="Enable auto-enrollment (previous default)",
    )
    g.add_argument(
        "--no-auto-enroll",
        action="store_true",
        help="Disable auto-enrollment (deprecated; default already OFF)",
    )

    # Control API
    p.add_argument(
        "--cmd-host",
        type=str,
        default="127.0.0.1",
        help="Command API bind host (default: 127.0.0.1)",
    )
    p.add_argument(
        "--cmd-port", type=int, default=8080, help="Command API port (default: 8080)"
    )
    p.add_argument(
        "--no-command-server",
        action="store_true",
        help="Disable the HTTP command server",
    )

    return p.parse_args()


# ROS 2 Camera Subscriber
class RealSenseROSCamera:
    """
    ROS 2 camera subscriber for RealSense via realsense2_camera_node.

    Instead of opening the camera device directly (which blocks other nodes),
    this class subscribes to ROS topics published by realsense2_camera_node.
    """

    def __init__(
        self,
        node: Node,
        color_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        depth_scale: float = 0.001,
    ):
        self.node = node
        self.bridge = CvBridge()
        self.depth_scale = depth_scale

        # Thread-safe frame storage
        self._lock = threading.Lock()
        self._color_frame: Optional[np.ndarray] = None
        self._depth_frame: Optional[np.ndarray] = None

        # Camera intrinsics (populated from camera_info)
        self.fx: float = 0.0
        self.fy: float = 0.0
        self.cx: float = 0.0
        self.cy: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self._intrinsics_received = False

        # QoS for camera topics (BEST_EFFORT matches camera node default)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribe to camera info for intrinsics
        self._info_sub = node.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_callback, qos
        )

        # Synchronized subscribers for color and depth
        self._color_sub = message_filters.Subscriber(
            node, Image, color_topic, qos_profile=qos
        )
        self._depth_sub = message_filters.Subscriber(
            node, Image, depth_topic, qos_profile=qos
        )

        # ApproximateTimeSynchronizer matches frames by timestamp
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub],
            queue_size=10,
            slop=0.1,
        )
        self._sync.registerCallback(self._sync_callback)

        self._running = True

        logger.info("RealSenseROSCamera subscribing to:")
        logger.info(f"  Color: {color_topic}")
        logger.info(f"  Depth: {depth_topic}")
        logger.info(f"  Info:  {camera_info_topic}")

        self._wait_for_frames(timeout=10.0)

    def _wait_for_frames(self, timeout: float):
        """Wait for camera to start publishing frames."""
        logger.info(f"Waiting for camera frames (timeout={timeout}s)...")
        start_time = time.time()

        while self._color_frame is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        if self._color_frame is None:
            logger.error("Timeout! No frames received.")
            logger.error("Check that realsense2_camera_node is running:")
            logger.error("  ros2 topic list | grep camera")
            logger.error("  ros2 topic hz /camera/color/image_raw")
        else:
            logger.info(f"Camera ready: {self.width}x{self.height}")
            if self._intrinsics_received:
                logger.info(
                    f"  Intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}"
                )

    def _camera_info_callback(self, msg: CameraInfo):
        """Extract camera intrinsics from CameraInfo message."""
        if not self._intrinsics_received:
            self.width = msg.width
            self.height = msg.height
            # K matrix is [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self._intrinsics_received = True

    def _sync_callback(self, color_msg: Image, depth_msg: Image):
        """Process synchronized color and depth frames."""
        if not self._running:
            return

        try:
            # Convert color image to BGR (OpenCV format)
            if color_msg.encoding == "rgb8":
                color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            elif color_msg.encoding == "bgr8":
                color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            else:
                color = self.bridge.imgmsg_to_cv2(color_msg, "passthrough")
                if len(color.shape) == 3 and color.shape[2] == 3:
                    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # Convert depth to meters
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

            if depth_raw.dtype == np.uint16:
                depth = depth_raw.astype(np.float64) * self.depth_scale
            elif depth_raw.dtype == np.float32:
                depth = depth_raw.astype(np.float64)
            else:
                depth = depth_raw.astype(np.float64) * self.depth_scale

            with self._lock:
                self._color_frame = color
                self._depth_frame = depth
                if self.width == 0:
                    self.height, self.width = color.shape[:2]

        except Exception as e:
            logger.error(f"Frame processing error: {e}")

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the latest synchronized color and depth frames."""
        with self._lock:
            if self._color_frame is None or self._depth_frame is None:
                return None, None
            return self._color_frame.copy(), self._depth_frame.copy()

    def stop(self):
        """Stop the camera subscriber."""
        self._running = False
        logger.info("RealSenseROSCamera stopped")


# ROS 2 Publisher Node
class TrackedPersonPublisher(Node):
    """ROS 2 node for publishing tracked person status."""

    def __init__(self):
        super().__init__("tracked_person_publisher")

        # JSON status publisher
        self.publisher = self.create_publisher(String, "/tracked_person/status", 10)
        # PoseStamped publisher
        self.pose_publisher = self.create_publisher(
            PoseStamped, "/person_following_robot/tracked_person/position", 10
        )
        # Detection visualization image publisher
        self.image_publisher = self.create_publisher(
            Image, "/tracked_person/detection_image", 10
        )
        self.bridge = CvBridge()
        self.publish_count = 0

    def publish_status(self, is_tracked: bool, x: float, z: float):
        """Publish tracking status as JSON and PoseStamped."""
        # JSON format
        msg = String()
        msg.data = json.dumps(
            {"is_tracked": is_tracked, "x": round(x, 3), "z": round(z, 3)}
        )
        self.publisher.publish(msg)

        if is_tracked:
            now = self.get_clock().now()

            # PoseStamped in camera frame for person_follower.py
            # Camera optical frame: x=right, y=down, z=forward
            pose = PoseStamped()
            pose.header.stamp = now.to_msg()
            pose.header.frame_id = "camera_color_optical_frame"
            pose.pose.position.x = x  # lateral offset in meters
            pose.pose.position.y = 0.0  # not used
            pose.pose.position.z = z  # distance forward in meters
            pose.pose.orientation.w = 1.0  # identity quaternion
            self.pose_publisher.publish(pose)

        self.publish_count += 1

    def publish_detection_image(self, image: np.ndarray):
        """Publish detection visualization image."""
        try:
            msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_color_optical_frame"
            self.image_publisher.publish(msg)
        except Exception as e:
            self.get_logger().warning(f"Failed to publish detection image: {e}")


# Utility
def compute_lateral_offset(bbox, distance: float, fx: float, cx: float) -> float:
    """Compute lateral offset from camera center using pinhole model."""
    x1, y1, x2, y2 = bbox
    bbox_cx = (x1 + x2) / 2.0
    pixel_offset = bbox_cx - cx
    return (pixel_offset * distance) / fx


def draw_visualization(
    frame, result, system, is_tracked, x_offset, distance, publish_count, cmd_url: str
):
    """Draw tracking visualization on frame."""
    display = frame.copy()
    H, W = display.shape[:2]

    # Draw all tracks
    for track in result.get("all_tracks", []):
        x1, y1, x2, y2 = track["bbox"]
        track_id = track["track_id"]
        target_id = getattr(getattr(system, "target", None), "track_id", None)

        if target_id is not None and track_id == target_id:
            color = (0, 255, 0)
            thickness = 2
        else:
            color = (128, 128, 128)
            thickness = 1

        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            display,
            f"ID:{track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    # Draw target with distance
    if result.get("target_found") and "bbox" in result:
        x1, y1, x2, y2 = result["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

        label = f"TARGET @{distance:.2f}m (x:{x_offset:+.2f}m)"
        cv2.putText(
            display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(display, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
        cv2.line(display, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)

    # Draw candidates
    for cand in system.get_candidates_info():
        x1, y1, x2, y2 = cand["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            display,
            f"L:{cand['clothing_sim']:.2f} C:{cand['clip_sim']:.2f}",
            (x1, y2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
        )

    status = result.get("status", "UNKNOWN")
    fps = result.get("fps", 0)

    status_color = {
        "TRACKING_ACTIVE": (0, 255, 0),
        "SEARCHING": (0, 165, 255),
        "INACTIVE": (128, 128, 128),
    }.get(status, (255, 255, 255))
    tracked_color = (0, 255, 0) if is_tracked else (0, 0, 255)

    cv2.rectangle(display, (0, 0), (W, 35), (0, 0, 0), -1)
    cv2.putText(
        display,
        f"Status: {status}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2,
    )
    cv2.putText(
        display,
        f"Tracked: {'YES' if is_tracked else 'NO'}",
        (220, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        tracked_color,
        1,
    )
    cv2.putText(
        display,
        f"FPS:{fps:.0f} Pub:{publish_count}",
        (W - 150, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    cv2.rectangle(display, (0, H - 50), (W, H), (0, 0, 0), -1)
    cv2.putText(
        display,
        "'e'=enroll | 'c'=clear | 's'=status | 'q'=quit",
        (10, H - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        display,
        f"HTTP: {cmd_url}  (POST /command)",
        (10, H - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    if status == "SEARCHING":
        time_lost = result.get("time_lost", 0)
        cv2.putText(
            display,
            f"SEARCHING... ({time_lost:.1f}s)",
            (W // 2 - 80, H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

    return display


def main() -> None:
    """Main entry point for the ROS 2 tracked person publisher."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("PERSON FOLLOWING SYSTEM - ROS 2")
    logger.info("=" * 60)

    rclpy.init()
    ros_node = TrackedPersonPublisher()
    logger.info("ROS 2 node initialized")
    logger.info("Publishing to: /tracked_person/status")
    logger.info("Publishing to: /person_following_robot/tracked_person/position")
    logger.info("Publishing to: /tracked_person/detection_image")

    logger.info("Initializing camera via ROS topics...")
    camera = RealSenseROSCamera(
        node=ros_node,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
        depth_scale=args.depth_scale,
    )

    if camera.width == 0:
        logger.error("Camera not available. Exiting.")
        ros_node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    logger.info("Initializing person following system...")
    from person_following_system import PersonFollowingSystem

    system = PersonFollowingSystem(
        yolo_detection_engine=args.yolo_det,
        yolo_seg_engine=args.yolo_seg,
        device="cuda",
        tracker_type=args.tracker,
        use_clip=True,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clothing_threshold=args.clothing_threshold,
        clip_threshold=args.clip_threshold,
        min_mask_coverage=args.min_mask_coverage,
        search_interval=args.search_interval,
    )

    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.save_video, fourcc, 15, (camera.width, camera.height)
        )
        logger.info(f"Saving video to: {args.save_video}")

    auto_enroll = bool(args.auto_enroll) and not bool(args.no_auto_enroll)
    publish_period = 1.0 / max(args.publish_hz, 1.0)
    last_publish_time = 0.0

    # Command server + queue
    cmd_queue: "queue.Queue[Command]" = queue.Queue(maxsize=32)
    shared_status = SharedStatus()
    cmd_server: Optional[CommandServer] = None
    cmd_url = "(disabled)"

    if not args.no_command_server:
        cmd_server = CommandServer(
            args.cmd_host, args.cmd_port, cmd_queue, shared_status
        )
        cmd_server.start()
        cmd_url = cmd_server.url
        logger.info(f"Command API listening at: {cmd_url}")
    else:
        logger.info("Command API disabled (--no-command-server)")

    stop_event = threading.Event()

    def _sigterm_handler(signum, frame):  # noqa: ARG001
        stop_event.set()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    pending_enroll = False
    last_action = {"name": None, "ok": None, "detail": None, "ts": None}

    logger.info("=" * 60)
    logger.info(
        f"Auto-enroll: {'ON' if auto_enroll else 'OFF'} (use --auto-enroll to enable)"
    )
    logger.info(f"Publish rate: {args.publish_hz} Hz")
    logger.info("Controls: 'e'=enroll, 'c'=clear, 's'=status, 'q'=quit")
    logger.info(f"HTTP commands: {cmd_url}")
    logger.info("=" * 60)

    def _drain_commands(color_frame: np.ndarray, depth_frame: np.ndarray) -> None:
        nonlocal pending_enroll, last_action
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except queue.Empty:
                return

            if cmd.name == "quit":
                last_action = {
                    "name": "quit",
                    "ok": True,
                    "detail": "queued",
                    "ts": cmd.ts,
                }
                stop_event.set()
                return

            if cmd.name == "clear":
                try:
                    system.clear_target()
                    last_action = {
                        "name": "clear",
                        "ok": True,
                        "detail": None,
                        "ts": cmd.ts,
                    }
                    logger.info("Target cleared (HTTP)")
                except Exception as e:
                    last_action = {
                        "name": "clear",
                        "ok": False,
                        "detail": str(e),
                        "ts": cmd.ts,
                    }
                    logger.warning(f"Clear failed (HTTP): {e}")
                continue

            if cmd.name == "enroll":
                pending_enroll = True
                last_action = {
                    "name": "enroll",
                    "ok": None,
                    "detail": "pending",
                    "ts": cmd.ts,
                }
                logger.info("Enroll requested (HTTP)")
                continue

    try:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(ros_node, timeout_sec=0.001)

            color_frame, depth_frame = camera.get_frames()
            if color_frame is None or depth_frame is None:
                continue

            _drain_commands(color_frame, depth_frame)

            if pending_enroll:
                pending_enroll = False
                try:
                    ok = bool(system.enroll_target(color_frame, depth_frame))
                    last_action = {
                        "name": "enroll",
                        "ok": ok,
                        "detail": None,
                        "ts": time.time(),
                    }
                    if ok:
                        logger.info("Enrolled target (HTTP/keyboard)")
                    else:
                        logger.warning("Enrollment failed (HTTP/keyboard)")
                except Exception as e:
                    last_action = {
                        "name": "enroll",
                        "ok": False,
                        "detail": str(e),
                        "ts": time.time(),
                    }
                    logger.warning(f"Enrollment failed: {e}")

            result = system.process_frame(color_frame, depth_frame)
            status = result.get("status", "UNKNOWN")

            is_tracked = bool(
                result.get("target_found", False)
                and result.get("bbox") is not None
                and result.get("distance") is not None
            )

            x_offset, distance = 0.0, 0.0
            if is_tracked:
                distance = float(result["distance"])
                if np.isfinite(distance) and distance > 0.1:
                    x_offset = float(
                        compute_lateral_offset(
                            result["bbox"], distance, camera.fx, camera.cx
                        )
                    )
                else:
                    is_tracked = False

            if auto_enroll and status == "INACTIVE":
                try:
                    if system.enroll_target(color_frame, depth_frame):
                        logger.info("Auto-enrolled nearest person")
                except Exception:
                    pass

            current_time = time.time()
            if current_time - last_publish_time >= publish_period:
                ros_node.publish_status(is_tracked, x_offset, distance)
                last_publish_time = current_time

                if ros_node.publish_count % 30 == 0 and is_tracked:
                    logger.info(f"Target: x={x_offset:+.2f}m z={distance:.2f}m")

            target_id = getattr(getattr(system, "target", None), "track_id", None)
            shared_status.set(
                {
                    "ok": True,
                    "ts": current_time,
                    "status": status,
                    "is_tracked": bool(is_tracked),
                    "x": float(round(x_offset, 3)),
                    "z": float(round(distance, 3)),
                    "fps": float(result.get("fps", 0) or 0),
                    "target_track_id": target_id,
                    "auto_enroll": bool(auto_enroll),
                    "publish_count": int(ros_node.publish_count),
                    "last_action": dict(last_action),
                }
            )

            # Always draw and publish visualization to ROS topic
            display = draw_visualization(
                color_frame,
                result,
                system,
                is_tracked,
                x_offset,
                distance,
                ros_node.publish_count,
                cmd_url,
            )
            ros_node.publish_detection_image(display)

            if video_writer:
                video_writer.write(display)

            if args.display:
                cv2.imshow("Person Following - ROS 2", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                    last_action = {
                        "name": "quit",
                        "ok": True,
                        "detail": "keyboard",
                        "ts": time.time(),
                    }
                elif key == ord("e"):
                    pending_enroll = True
                elif key == ord("c"):
                    system.clear_target()
                    last_action = {
                        "name": "clear",
                        "ok": True,
                        "detail": "keyboard",
                        "ts": time.time(),
                    }
                elif key == ord("s"):
                    status_dict = system.get_status()
                    logger.info("=" * 40)
                    for k, v in status_dict.items():
                        logger.info(f"  {k}: {v}")
                    logger.info(f"  ROS publish count: {ros_node.publish_count}")
                    logger.info(f"  HTTP: {cmd_url}")
                    logger.info("=" * 40)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        try:
            camera.stop()
        except Exception:
            pass

        if video_writer:
            video_writer.release()
        if args.display:
            cv2.destroyAllWindows()

        if cmd_server is not None:
            try:
                cmd_server.stop()
            except Exception:
                pass

        logger.info(f"Published {ros_node.publish_count} messages")
        ros_node.destroy_node()
        rclpy.shutdown()
        logger.info("Done.")


if __name__ == "__main__":
    main()
