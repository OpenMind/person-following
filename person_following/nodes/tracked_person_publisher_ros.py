"""
ROS 2 Tracked Person Publisher for Robot Dog Following.

When realsense rgbd depth camera is used, it subscribes to realsense2_camera_node ROS topics for camera input
When built in unitree go2 camera + RPLidar is used, it subscribe ros2 /scan and /camera/image_raw for data input

Publishes
---------
/tracked_person/status : std_msgs/String (JSON)
  - is_tracked    : bool   (whether target is being tracked)
  - x             : float  (lateral offset in meters; +right / -left)
  - z             : float  (distance forward in meters)
  - mode          : str    (INACTIVE, TRACKING_ACTIVE, SEARCHING, SWITCHING, APPROACHED)
  - mode_duration : float  (seconds in current mode)
  - approached    : bool   (whether target has approached within threshold)
  - operation_mode: str    (greeting or following)
  - num_persons   : int    (number of persons currently detected)

Control API
-----------------
Commands: enroll | clear | switch | clear_history | delete_history |
          save_history | load_history | set_max_history | status | quit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String

from person_following.controllers.person_following_command import (
    Command,
    CommandServer,
    SharedStatus,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tracked_person_publisher_ros")

OperationMode = Literal["greeting", "following"]

MODEL_DIR = str(Path(__file__).resolve().parents[2] / "model")
ENGINE_DIR = str(Path(__file__).resolve().parents[2] / "engine")
EXTRINSICS_CACHE_DIR = str(Path(__file__).resolve().parents[2] / "extrinsics-files")
INTRINSICS_CACHE_DIR = str(Path(__file__).resolve().parents[2] / "intrinsics-files")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Person Following with ROS 2")

    # Pick calibration defaults based on docker-compose env ROBOT_TYPE
    robot_type = os.getenv("ROBOT_TYPE", "go2").strip().lower()
    calib_defaults = {
        "go2": (
            f"{INTRINSICS_CACHE_DIR}/camera_intrinsics_go2.yaml",
            f"{EXTRINSICS_CACHE_DIR}/lidar_camera_extrinsics_go2.yaml",
            "/camera/go2/image_raw/best_effort",
            "go2",
        ),
        "tron": (
            f"{INTRINSICS_CACHE_DIR}/camera_intrinsics_tron_usb.yaml",
            f"{EXTRINSICS_CACHE_DIR}/lidar_camera_extrinsics_tron_usb.yaml",
            "/image_raw",
            "go2",
        ),
        "g1": (
            f"{INTRINSICS_CACHE_DIR}/camera_intrinsics_g1.yaml",
            f"{EXTRINSICS_CACHE_DIR}/lidar_camera_extrinsics_g1.yaml",
            "rtsp://localhost:8554/top_camera_raw",  # default RTSP URL
            "rtsp",
        ),
    }

    if robot_type not in calib_defaults:
        logging.getLogger("tracked_person_publisher_ros").warning(
            f"Unknown ROBOT_TYPE={robot_type!r}; defaulting to 'go2'"
        )
        robot_type = "go2"
    intr_default, extr_default, camera_topic, camera_mode = calib_defaults[robot_type]

    # Operation mode
    p.add_argument(
        "--mode",
        type=str,
        default="greeting",
        choices=["greeting", "following"],
        help="Operation mode: 'greeting' (full features with history/switch) or 'following' (simple tracking)",
    )

    # Model paths
    p.add_argument(
        "--yolo-det",
        type=str,
        default="/opt/person_following/engine/yolo11used for realsensen.engine",
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
        default="/camera/realsense2_camera_node/color/image_raw",
        help="ROS color image topic (only used for realsense depth camera)",
    )
    p.add_argument(
        "--depth-topic",
        type=str,
        default="/camera/realsense2_camera_node/depth/image_rect_raw",
        help="ROS depth image topic (aligned depth (only used for realsense depth camera))",
    )
    p.add_argument(
        "--camera-info-topic",
        type=str,
        default="/camera/realsense2_camera_node/color/camera_info",
        help="ROS camera info topic (only used for realsense depth camera)",
    )
    p.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="Depth scale (uint16 to meters), default 0.001 (only used for realsense depth camera)",
    )

    # Camera mode
    p.add_argument(
        "--camera-mode",
        type=str,
        default=camera_mode,
        choices=["realsense", "go2", "rtsp"],
        help="Camera mode. 'realsense' uses color+depth topics. "
        "'go2' uses ROS image topic + /scan. "
        "'rtsp' uses RTSP stream + /scan.",
    )

    # Go2 front camera + LiDAR topics
    p.add_argument(
        "--image-topic",
        type=str,
        default=camera_topic,
        help="Go2 front camera image topic (BGR/RGB Image)",
    )
    p.add_argument(
        "--scan-topic",
        type=str,
        default="/scan",
        help="LiDAR LaserScan topic",
    )
    p.add_argument(
        "--rtsp-url",
        type=str,
        default=(
            camera_topic
            if camera_mode == "rtsp"
            else "rtsp://localhost:8554/top_camera_raw"
        ),
        help="RTSP stream URL for --camera-mode rtsp (e.g. rtsp://192.168.123.164:8554/live)",
    )
    p.add_argument(
        "--rtsp-fps",
        type=int,
        default=30,
        help="Target FPS for RTSP capture (used in --camera-mode rtsp)",
    )
    p.add_argument(
        "--rtsp-decode-format",
        type=str,
        default="H264",
        help="RTSP decode fourcc format (used in --camera-mode rtsp)",
    )
    p.add_argument(
        "--rtsp-reconnect-delay",
        type=float,
        default=1.0,
        help="Seconds to wait before RTSP reconnection attempt (default: 1.0)",
    )
    p.add_argument(
        "--intrinsics-yaml",
        type=str,
        default=intr_default,
        help="Path to camera intrinsics YAML. Used in --camera-mode go2.",
    )

    # Go2 extrinsics
    p.add_argument(
        "--extrinsics-yaml",
        type=str,
        default=extr_default,
        help="Path to camera extrinsics YAML. Used in --camera-mode go2.",
    )

    # LiDAR clustering params
    p.add_argument(
        "--lidar-range-jump",
        type=float,
        default=0.35,
        help="Range jump threshold (m) to split LiDAR clusters",
    )
    p.add_argument(
        "--lidar-min-cluster-size",
        type=int,
        default=4,
        help="Minimum number of LiDAR points for a valid cluster",
    )

    # Display and output
    p.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Enable OpenCV visualization window",
    )
    p.add_argument(
        "--save-video", type=str, default=None, help="Save output to video file"
    )
    p.add_argument(
        "--publish-hz", type=float, default=15.0, help="ROS 2 publish rate (Hz)"
    )
    p.add_argument(
        "--no-publish-detection-image",
        action="store_true",
        default=False,
        help="Disable publishing /tracked_person/detection_image",
    )

    # Enrollment behavior
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--auto-enroll",
        action="store_true",
        default=False,
        help="Enable auto-enrollment",
    )
    g.add_argument(
        "--no-auto-enroll",
        action="store_true",
        help="Disable auto-enrollment (default)",
    )

    # Control API
    p.add_argument(
        "--cmd-host",
        type=str,
        default="127.0.0.1",
        help="Command API bind host (default: 127.0.0.1)",
    )
    p.add_argument(
        "--cmd-port", type=int, default=2001, help="Command API port (default: 2001)"
    )
    p.add_argument(
        "--no-command-server",
        action="store_true",
        help="Disable the HTTP command server",
    )

    # History settings
    p.add_argument(
        "--max-history-size",
        type=int,
        default=1,
        help="Maximum number of greeted persons to remember (default: 1)",
    )
    p.add_argument(
        "--history-file",
        type=str,
        default=None,
        help="Path to history file (default: ~/.person_following_history.pkl)",
    )
    p.add_argument(
        "--no-auto-load-history",
        action="store_true",
        help="Don't auto-load history on startup",
    )
    p.add_argument(
        "--no-auto-save-history",
        action="store_true",
        help="Don't auto-save history after switch",
    )

    # Approach and searching timeout settings
    p.add_argument(
        "--approach-distance",
        type=float,
        default=1.0,
        help="Distance threshold for 'approached' status (default: 1.0m)",
    )
    p.add_argument(
        "--searching-timeout",
        type=float,
        default=5.0,
        help="Searching mode timeout before auto-save and inactive (default: 5.0s)",
    )

    return p.parse_args()


# ROS 2 Camera Subscribers
class RealSenseROSCamera:
    """ROS 2 camera subscriber for RealSense via realsense2_camera_node."""

    def __init__(
        self,
        node: Node,
        color_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        depth_scale: float = 0.001,
    ):
        """
        Initialize RealSense camera interface via ROS topics.

        Parameters
        ----------
        node : Node
            ROS 2 node for creating subscriptions.
        timeout : float, optional
            Timeout waiting for first frames in seconds, by default 5.0.
        color_topic : str, optional
            Topic for color images, by default '/camera/color/image_raw'.
        depth_topic : str, optional
            Topic for depth images, by default '/camera/depth/image_rect_raw'.
        camera_info_topic : str, optional
            Topic for camera intrinsics, by default '/camera/color/camera_info'.
        """
        self.node = node
        self.bridge = CvBridge()
        self.depth_scale = depth_scale

        self._lock = threading.Lock()
        self._color_frame: Optional[np.ndarray] = None
        self._depth_frame: Optional[np.ndarray] = None

        self.fx: float = 0.0
        self.fy: float = 0.0
        self.cx: float = 0.0
        self.cy: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self._intrinsics_received = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._info_sub = node.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_callback, qos
        )

        self._color_sub = message_filters.Subscriber(
            node, Image, color_topic, qos_profile=qos
        )
        self._depth_sub = message_filters.Subscriber(
            node, Image, depth_topic, qos_profile=qos
        )

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
        """
        Wait for initial frames from camera topics.

        Parameters
        ----------
        timeout : float
            Maximum time to wait in seconds.
        """
        logger.info(f"Waiting for camera frames (timeout={timeout}s)...")
        start_time = time.time()
        while self._color_frame is None and (time.time() - start_time) < timeout:
            time.sleep(0.05)
        if self._color_frame is None:
            logger.error("Timeout! No frames received.")
        else:
            logger.info(f"Camera ready: {self.width}x{self.height}")

    def _camera_info_callback(self, msg: CameraInfo):
        """
        Handle camera info messages.

        Extracts and stores camera intrinsics (fx, fy, cx, cy).

        Parameters
        ----------
        msg : CameraInfo
            ROS CameraInfo message.
        """
        if not self._intrinsics_received:
            self.width = msg.width
            self.height = msg.height
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self._intrinsics_received = True

    def _sync_callback(self, color_msg: Image, depth_msg: Image):
        """
        Handle synchronized color and depth image messages.

        Parameters
        ----------
        color_msg : Image
            ROS Image message (BGR8 or RGB8).
        depth_msg : Image
            ROS Image message (16UC1 depth in mm).
        """
        if not self._running:
            return
        try:
            if color_msg.encoding == "rgb8":
                color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            elif color_msg.encoding == "bgr8":
                color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            else:
                color = self.bridge.imgmsg_to_cv2(color_msg, "passthrough")
                if len(color.shape) == 3 and color.shape[2] == 3:
                    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

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

    def get_frames(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
        """
        Get latest camera frames.

        Returns
        -------
        color : np.ndarray or None
            BGR color image.
        depth : np.ndarray or None
            Depth image in millimeters (uint16).
        aux : dict or None
            Auxiliary data with camera intrinsics ('fx', 'fy', 'cx', 'cy').
        """
        with self._lock:
            if self._color_frame is None or self._depth_frame is None:
                return None, None, None
            return self._color_frame.copy(), self._depth_frame.copy(), None

    def stop(self):
        """
        Stop camera and release resources.

        Destroys ROS subscriptions.
        """
        self._running = False
        logger.info("RealSenseROSCamera stopped")


def load_intrinsics_yaml(path: str) -> Tuple[float, float, float, float, int, int]:
    """
    Load camera intrinsics from YAML file.

    Parameters
    ----------
    path : str
        Path to YAML file with camera_matrix and image dimensions.

    Returns
    -------
    fx : float
        Focal length x.
    fy : float
        Focal length y.
    cx : float
        Principal point x.
    cy : float
        Principal point y.
    width : int
        Image width.
    height : int
        Image height.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    w, h = data.get("image_size", [640, 480])
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return fx, fy, cx, cy, int(w), int(h)


def load_extrinsics_yaml(path: str) -> Tuple[float, float, float, float, float, float]:
    """
    Load camera extrinsics from YAML file.

    Parameters
    ----------
    path : str
        Path to YAML file with translation and rotation_euler.

    Returns
    -------
    tx : float
        Translation x in meters.
    ty : float
        Translation y in meters.
    tz : float
        Translation z in meters.
    rx : float
        Rotation roll in radians.
    ry : float
        Rotation pitch in radians.
    rz : float
        Rotation yaw in radians.

    Raises
    ------
    FileNotFoundError
        If the extrinsics YAML file does not exist.
    ValueError
        If the YAML file is missing required fields.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Extrinsics file not found: '{path}'\n"
            f"Please create the extrinsics YAML file with translation and rotation_euler fields."
        )

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Extrinsics file '{path}' is empty or invalid YAML.")

    if "translation" not in data:
        raise ValueError(f"Extrinsics file '{path}' missing 'translation' field.")

    if "rotation_euler" not in data:
        raise ValueError(f"Extrinsics file '{path}' missing 'rotation_euler' field.")

    translation = data.get("translation", {})
    rotation = data.get("rotation_euler", {})

    tx = float(translation.get("x", 0.0))
    ty = float(translation.get("y", 0.0))
    tz = float(translation.get("z", 0.0))
    rx = float(rotation.get("roll", 0.0))
    ry = float(rotation.get("pitch", 0.0))
    rz = float(rotation.get("yaw", 0.0))

    return tx, ty, tz, rx, ry, rz


def euler_xyz_to_R(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Convert Euler angles (XYZ order) to rotation matrix.

    Parameters
    ----------
    rx : float
        Rotation around X axis in radians.
    ry : float
        Rotation around Y axis in radians.
    rz : float
        Rotation around Z axis in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


class Go2ROSCameraLidar:
    """Go2 front camera subscriber + LiDAR subscriber with projection."""

    def __init__(
        self,
        node: Node,
        image_topic: str,
        scan_topic: str,
        intrinsics_yaml: str,
        tx: float,
        ty: float,
        tz: float,
        rx: float,
        ry: float,
        rz: float,
    ):
        """
        Initialize Go2 camera with LiDAR interface via ROS topics.

        Parameters
        ----------
        node : Node
            ROS 2 node for creating subscriptions.
        timeout : float, optional
            Timeout waiting for first frames in seconds, by default 5.0.
        image_topic : str, optional
            Topic for camera images, by default '/go2_camera/image_raw'.
        scan_topic : str, optional
            Topic for LiDAR scans, by default '/scan'.
        intrinsics_yaml : str, optional
            Path to camera intrinsics YAML, by default None.
        """
        self.node = node
        self.bridge = CvBridge()

        self._lock = threading.Lock()
        self._color_frame: Optional[np.ndarray] = None
        self._latest_scan: Optional[LaserScan] = None
        self._frame_seq: int = 0
        self._scan_seq: int = 0
        self._cached_aux: Optional[dict] = None
        self._cached_scan_seq: int = -1

        fx, fy, cx, cy, w, h = load_intrinsics_yaml(intrinsics_yaml)
        self._yaml_width = w
        self._yaml_height = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = w
        self.height = h

        self.R_CL = euler_xyz_to_R(rx, ry, rz)
        self.t_CL = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._img_sub = node.create_subscription(
            Image, image_topic, self._image_callback, qos
        )
        self._scan_sub = node.create_subscription(
            LaserScan, scan_topic, self._scan_callback, qos
        )

        self._running = True

        logger.info("Go2ROSCameraLidar subscribing to:")
        logger.info(f"  Image: {image_topic}")
        logger.info(f"  Scan:  {scan_topic}")

        self._wait_for_frames(timeout=10.0)

    def _wait_for_frames(self, timeout: float):
        """
        Wait for initial frames from camera and LiDAR topics.

        Parameters
        ----------
        timeout : float
            Maximum time to wait in seconds.
        """
        logger.info(f"Waiting for Go2 camera frames (timeout={timeout}s)...")
        start_time = time.time()
        while self._color_frame is None and (time.time() - start_time) < timeout:
            time.sleep(0.05)
        if self._color_frame is None:
            logger.error("Timeout! No Go2 camera frames received.")
        else:
            logger.info(f"Go2 camera ready: {self.width}x{self.height}")

    def _image_callback(self, msg: Image):
        """
        Handle camera image messages.

        Parameters
        ----------
        msg : Image
            ROS Image message.
        """
        if not self._running:
            return
        try:
            if msg.encoding == "rgb8":
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            H, W = frame.shape[:2]
            if (W, H) != (self.width, self.height):
                sx = W / float(self.width) if self.width > 0 else 1.0
                sy = H / float(self.height) if self.height > 0 else 1.0
                self.fx *= sx
                self.fy *= sy
                self.cx *= sx
                self.cy *= sy
                self.width = W
                self.height = H
            with self._lock:
                self._color_frame = frame  # no copy — callback owns this frame
                self._frame_seq += 1
        except Exception as e:
            logger.error(f"Go2 image callback error: {e}")

    def _scan_callback(self, msg: LaserScan):
        """
        Handle LiDAR scan messages.

        Parameters
        ----------
        msg : LaserScan
            ROS LaserScan message.
        """
        if not self._running:
            return
        with self._lock:
            self._latest_scan = msg
            self._scan_seq += 1

    def _project_scan(
        self, scan: LaserScan, frame_shape: Tuple[int, int]
    ) -> Optional[dict]:
        """
        Project LiDAR scan points onto camera image plane.

        Parameters
        ----------
        scan : LaserScan
            LiDAR scan message.
        frame_shape : tuple of int
            Image dimensions (height, width).

        Returns
        -------
        dict or None
            Projection data with 'lidar_uv', 'lidar_ranges', 'lidar_scan_idx',
            'uv', 'idx', 'r', 'xy', 'scan_xy', or None if projection fails.
        """
        H, W = frame_shape
        ranges_full = np.array(scan.ranges, dtype=np.float64)
        n = len(ranges_full)
        if n == 0:
            return None
        angles = scan.angle_min + np.arange(n, dtype=np.float64) * scan.angle_increment
        valid = (
            np.isfinite(ranges_full)
            & (ranges_full > float(scan.range_min))
            & (ranges_full < float(scan.range_max))
        )
        if not np.any(valid):
            return None
        idx = np.where(valid)[0].astype(np.int32)
        r = ranges_full[idx]
        a = angles[idx]
        x_lidar = r * np.cos(a)
        y_lidar = r * np.sin(a)
        z_lidar = np.zeros_like(x_lidar)
        P_L = np.vstack([x_lidar, y_lidar, z_lidar])
        P_C = self.R_CL.T @ (P_L - self.t_CL)
        pts_cv = np.vstack([-P_C[1, :], -P_C[2, :], P_C[0, :]])
        Z = pts_cv[2, :]
        ok = Z > 0.01
        if not np.any(ok):
            return None
        pts = pts_cv[:, ok]
        idx_ok = idx[ok]
        r_ok = r[ok]
        x_ok = x_lidar[ok]
        y_ok = y_lidar[ok]
        K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64
        )
        uvw = K @ pts
        uv = (uvw[:2, :] / uvw[2:3, :]).T
        uv_int = np.round(uv).astype(np.int32)
        in_img = (
            (uv_int[:, 0] >= 0)
            & (uv_int[:, 0] < W)
            & (uv_int[:, 1] >= 0)
            & (uv_int[:, 1] < H)
        )
        if not np.any(in_img):
            return None
        uv_final = uv_int[in_img]
        idx_final = idx_ok[in_img]
        r_final = r_ok[in_img]
        xy_final = np.column_stack([x_ok[in_img], y_ok[in_img]])
        # scan_xy = ALL valid XY points (not just in-image), for KF tracker
        all_xy = np.column_stack([x_lidar, y_lidar])
        return {
            # Original keys (used by draw_visualization, _get_distance_from_lidar)
            "lidar_uv": uv_final,
            "lidar_scan_idx": idx_final,
            "lidar_ranges": r_final,
            # Keys expected by lidar_tracker.py cluster_in_bbox()
            "uv": uv_final,
            "idx": idx_final,
            "r": r_final,
            "xy": xy_final,
            # All LiDAR XY for KF tracker (not just in-image)
            "scan_xy": all_xy,
        }

    def get_frames(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
        """
        Get latest camera frame and LiDAR projection.

        Returns
        -------
        color : np.ndarray or None
            BGR color image.
        depth : None
            Always None (no depth camera on Go2).
        aux : dict or None
            Auxiliary data with LiDAR projection and camera intrinsics.
        """
        with self._lock:
            if self._color_frame is None:
                return None, None, None
            frame = self._color_frame.copy()
            scan = self._latest_scan
            scan_seq = self._scan_seq

        # Cache scan projection — only recompute when scan changes
        if scan is not None:
            if scan_seq != self._cached_scan_seq:
                self._cached_aux = self._project_scan(scan, frame.shape[:2])
                self._cached_scan_seq = scan_seq
            aux = self._cached_aux
        else:
            aux = None
        return frame, None, aux

    def stop(self):
        """
        Stop camera/LiDAR and release resources.

        Destroys ROS subscriptions.
        """
        self._running = False
        logger.info("Go2ROSCameraLidar stopped")


class Go2RTSPCameraLidar:
    """Go2 camera via RTSP stream + LiDAR via ROS2 /scan with projection.

    Replaces the previous ROS2 image topic subscription with a direct RTSP
    capture thread (based on the Insta360 Stream bridge pattern). LiDAR
    data is still received via ROS2 LaserScan subscription.
    """

    def __init__(
        self,
        node: Node,
        rtsp_url: str,
        rtsp_fps: int,
        rtsp_decode_format: str,
        rtsp_reconnect_delay: float,
        scan_topic: str,
        intrinsics_yaml: str,
        tx: float,
        ty: float,
        tz: float,
        rx: float,
        ry: float,
        rz: float,
    ):
        """
        Initialize Go2 camera via RTSP with LiDAR interface.

        Parameters
        ----------
        node : Node
            ROS 2 node for creating LiDAR subscription.
        rtsp_url : str
            RTSP stream URL for the camera.
        rtsp_fps : int
            Target FPS for RTSP capture.
        rtsp_decode_format : str
            Fourcc decode format string (e.g. 'H264').
        rtsp_reconnect_delay : float
            Seconds to wait before reconnection attempts.
        scan_topic : str
            ROS2 topic for LiDAR LaserScan.
        intrinsics_yaml : str
            Path to camera intrinsics YAML.
        tx, ty, tz : float
            LiDAR-to-camera translation in meters.
        rx, ry, rz : float
            LiDAR-to-camera rotation (roll, pitch, yaw) in radians.
        """
        self.node = node
        self.rtsp_url = rtsp_url
        self.rtsp_fps = rtsp_fps
        self.rtsp_decode_format = rtsp_decode_format
        self.rtsp_reconnect_delay = rtsp_reconnect_delay

        self._lock = threading.Lock()
        self._color_frame: Optional[np.ndarray] = None
        self._latest_scan: Optional[LaserScan] = None
        self._scan_seq: int = 0
        self._cached_aux: Optional[dict] = None
        self._cached_scan_seq: int = -1

        # Load camera intrinsics
        fx, fy, cx, cy, w, h = load_intrinsics_yaml(intrinsics_yaml)
        self._yaml_width = w
        self._yaml_height = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = w
        self.height = h

        # LiDAR-to-camera extrinsics
        self.R_CL = euler_xyz_to_R(rx, ry, rz)
        self.t_CL = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)

        # --- RTSP capture setup ---
        self._cap: Optional[cv2.VideoCapture] = None
        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._running = True
        self._intrinsics_scaled = False

        self._initialize_capture()

        # Start RTSP capture thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="rtsp-capture"
        )
        self._capture_thread.start()

        # Start frame consumer thread (moves frames from queue to _color_frame)
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop, daemon=True, name="rtsp-consumer"
        )
        self._consumer_thread.start()

        # FPS tracking for RTSP stream
        self._frame_count = 0
        self._last_fps_time = time.time()

        # --- LiDAR subscription via ROS2 ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._scan_sub = node.create_subscription(
            LaserScan, scan_topic, self._scan_callback, qos
        )

        logger.info("Go2RTSPCameraLidar initialized:")
        logger.info(f"  RTSP URL: {rtsp_url}")
        logger.info(f"  RTSP FPS: {rtsp_fps}")
        logger.info(f"  RTSP Decode: {rtsp_decode_format}")
        logger.info(f"  LiDAR Scan: {scan_topic}")

        self._wait_for_frames(timeout=10.0)

    def _initialize_capture(self) -> bool:
        """Initialize OpenCV VideoCapture for the RTSP stream."""
        try:
            self._cap = cv2.VideoCapture(self.rtsp_url)
            if not self._cap.isOpened():
                logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
                return False

            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FPS, self.rtsp_fps)
            self._cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*self.rtsp_decode_format),
            )
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 100)

            logger.info("Successfully connected to RTSP stream")
            return True
        except Exception as e:
            logger.error(f"Error initializing RTSP capture: {e}")
            return False

    def _capture_loop(self):
        """Dedicated thread for reading frames from RTSP stream."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                logger.warning("Attempting to reconnect to RTSP stream...")
                self._initialize_capture()
                time.sleep(self.rtsp_reconnect_delay)
                continue

            try:
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning(
                        "Failed to capture RTSP frame, attempting reconnection..."
                    )
                    self._cap.release()
                    self._initialize_capture()
                    time.sleep(self.rtsp_reconnect_delay)
                    continue

                # Drop old frames if queue is full (keep latest only)
                try:
                    self._capture_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._capture_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._capture_queue.put_nowait(frame)
                    except queue.Full:
                        pass

                self._frame_count += 1

            except Exception as e:
                logger.error(f"RTSP capture error: {e}")
                time.sleep(0.1)

    def _consumer_loop(self):
        """Consume frames from the capture queue and update _color_frame."""
        while self._running:
            try:
                frame = self._capture_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                H, W = frame.shape[:2]

                # Scale intrinsics once if actual frame size differs from YAML
                if not self._intrinsics_scaled and (W, H) != (
                    self._yaml_width,
                    self._yaml_height,
                ):
                    sx = W / float(self._yaml_width) if self._yaml_width > 0 else 1.0
                    sy = H / float(self._yaml_height) if self._yaml_height > 0 else 1.0
                    self.fx *= sx
                    self.fy *= sy
                    self.cx *= sx
                    self.cy *= sy
                    self.width = W
                    self.height = H
                    self._intrinsics_scaled = True
                    logger.info(
                        f"Intrinsics scaled for RTSP frame size: {W}x{H} "
                        f"(YAML was {self._yaml_width}x{self._yaml_height})"
                    )
                elif not self._intrinsics_scaled:
                    self.width = W
                    self.height = H
                    self._intrinsics_scaled = True

                with self._lock:
                    self._color_frame = frame.copy()
            except Exception as e:
                logger.error(f"RTSP consumer error: {e}")

    def _wait_for_frames(self, timeout: float):
        """Wait for initial frames from RTSP stream."""
        logger.info(f"Waiting for RTSP camera frames (timeout={timeout}s)...")
        start_time = time.time()
        while self._color_frame is None and (time.time() - start_time) < timeout:
            time.sleep(0.05)
        if self._color_frame is None:
            logger.error("Timeout! No RTSP camera frames received.")
        else:
            logger.info(f"RTSP camera ready: {self.width}x{self.height}")

    def _scan_callback(self, msg: LaserScan):
        """Handle LiDAR scan messages."""
        if not self._running:
            return
        with self._lock:
            self._latest_scan = msg
            self._scan_seq += 1

    def _project_scan(
        self, scan: LaserScan, frame_shape: Tuple[int, int]
    ) -> Optional[dict]:
        """
        Project LiDAR scan points onto camera image plane.

        Returns
        -------
        dict or None
            Projection data with 'lidar_uv', 'lidar_ranges', 'lidar_scan_idx',
            'uv', 'idx', 'r', 'xy', 'scan_xy', or None if projection fails.
        """
        H, W = frame_shape
        ranges_full = np.array(scan.ranges, dtype=np.float64)
        n = len(ranges_full)
        if n == 0:
            return None
        angles = scan.angle_min + np.arange(n, dtype=np.float64) * scan.angle_increment
        valid = (
            np.isfinite(ranges_full)
            & (ranges_full > float(scan.range_min))
            & (ranges_full < float(scan.range_max))
        )
        if not np.any(valid):
            return None
        idx = np.where(valid)[0].astype(np.int32)
        r = ranges_full[idx]
        a = angles[idx]
        x_lidar = r * np.cos(a)
        y_lidar = r * np.sin(a)
        z_lidar = np.zeros_like(x_lidar)
        P_L = np.vstack([x_lidar, y_lidar, z_lidar])
        P_C = self.R_CL.T @ (P_L - self.t_CL)
        pts_cv = np.vstack([-P_C[1, :], -P_C[2, :], P_C[0, :]])
        Z = pts_cv[2, :]
        ok = Z > 0.01
        if not np.any(ok):
            return None
        pts = pts_cv[:, ok]
        idx_ok = idx[ok]
        r_ok = r[ok]
        x_ok = x_lidar[ok]
        y_ok = y_lidar[ok]
        K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64
        )
        uvw = K @ pts
        uv = (uvw[:2, :] / uvw[2:3, :]).T
        uv_int = np.round(uv).astype(np.int32)
        in_img = (
            (uv_int[:, 0] >= 0)
            & (uv_int[:, 0] < W)
            & (uv_int[:, 1] >= 0)
            & (uv_int[:, 1] < H)
        )
        if not np.any(in_img):
            return None
        uv_final = uv_int[in_img]
        idx_final = idx_ok[in_img]
        r_final = r_ok[in_img]
        xy_final = np.column_stack([x_ok[in_img], y_ok[in_img]])
        # scan_xy = ALL valid XY points (not just in-image), for KF tracker
        all_xy = np.column_stack([x_lidar, y_lidar])
        return {
            # Original keys (used by draw_visualization, _get_distance_from_lidar)
            "lidar_uv": uv_final,
            "lidar_scan_idx": idx_final,
            "lidar_ranges": r_final,
            # Keys expected by lidar_tracker.py cluster_in_bbox()
            "uv": uv_final,
            "idx": idx_final,
            "r": r_final,
            "xy": xy_final,
            # All LiDAR XY for KF tracker (not just in-image)
            "scan_xy": all_xy,
        }

    def get_frames(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
        """
        Get latest camera frame and LiDAR projection.

        Returns
        -------
        color : np.ndarray or None
            BGR color image from RTSP stream.
        depth : None
            Always None (no depth camera).
        aux : dict or None
            Auxiliary data with LiDAR projection and scan_xy for KF tracker.
        """
        with self._lock:
            if self._color_frame is None:
                return None, None, None
            frame = self._color_frame.copy()
            scan = self._latest_scan
            scan_seq = self._scan_seq

        # Cache scan projection — only recompute when scan changes
        if scan is not None:
            if scan_seq != self._cached_scan_seq:
                self._cached_aux = self._project_scan(scan, frame.shape[:2])
                self._cached_scan_seq = scan_seq
            aux = self._cached_aux
        else:
            aux = None
        return frame, None, aux

    def get_rtsp_fps(self) -> float:
        """Get the current RTSP capture FPS."""
        current_time = time.time()
        elapsed = current_time - self._last_fps_time
        if elapsed > 0:
            fps = self._frame_count / elapsed
        else:
            fps = 0.0
        return fps

    def reset_fps_counter(self):
        """Reset the FPS counter."""
        self._frame_count = 0
        self._last_fps_time = time.time()

    def stop(self):
        """Stop RTSP capture and LiDAR, release resources."""
        self._running = False
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        logger.info("Go2RTSPCameraLidar stopped")


# ROS 2 Publisher Node
class TrackedPersonPublisher(Node):
    """ROS 2 node for publishing tracked person status and detection image."""

    def __init__(self):
        """
        Initialize ROS 2 node for publishing tracked person status.

        Creates publishers for:
        - /tracked_person/status (String/JSON)
        - /tracked_person/pose (PoseStamped)
        - /tracked_person/detection_image (Image)
        """
        super().__init__("tracked_person_publisher")

        self.publisher = self.create_publisher(String, "/tracked_person/status", 10)

        self.pose_publisher = self.create_publisher(
            PoseStamped, "/person_following_robot/tracked_person/position", 10
        )

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_publisher = self.create_publisher(
            Image, "/tracked_person/detection_image", image_qos
        )

        self.bridge = CvBridge()
        self.publish_count = 0

        self.get_logger().info("Publishing to: /tracked_person/status")
        self.get_logger().info(
            "Publishing to: /person_following_robot/tracked_person/position"
        )
        self.get_logger().info("Publishing to: /tracked_person/detection_image")

    def publish_status(
        self,
        is_tracked: bool,
        x: float,
        z: float,
        mode: str,
        mode_duration: float,
        approached: bool = False,
        operation_mode: str = "greeting",
        num_persons: int = 0,
        tracking_source: str = "none",
    ):
        """
        Publish tracked person status to ROS topics.

        ALWAYS publishes to both /tracked_person/status (JSON) and
        /person_following_robot/tracked_person/position (PoseStamped).
        The robot controller can use 'mode' and 'is_tracked' fields
        from the status topic to decide how to use the position data.

        Parameters
        ----------
        is_tracked : bool
            Whether target is currently being tracked.
        x : float
            Lateral offset in meters (positive = right).
        z : float
            Distance forward in meters.
        mode : str
            Current state (INACTIVE, TRACKING_ACTIVE, SEARCHING, SWITCHING).
        mode_duration : float
            Seconds in current state.
        approached : bool, optional
            Whether target has approached within threshold, by default False.
        operation_mode : str, optional
            Operation mode ('greeting' or 'following'), by default 'greeting'.
        num_persons : int, optional
            Number of persons currently detected, by default 0.
        tracking_source : str, optional
            Source of position data ('camera', 'lidar', 'lidar_searching', 'none').
        """
        msg = String()
        msg.data = json.dumps(
            {
                "is_tracked": is_tracked,
                "x": round(x, 3),
                "z": round(z, 3),
                "mode": mode,
                "mode_duration": round(mode_duration, 2),
                "approached": approached,
                "operation_mode": operation_mode,
                "num_persons": num_persons,
                "tracking_source": tracking_source,
            }
        )
        self.publisher.publish(msg)

        # ALWAYS publish PoseStamped so the robot controller gets continuous
        # position updates in all modes (TRACKING, SEARCHING with LiDAR, etc.)
        now = self.get_clock().now()
        pose = PoseStamped()
        pose.header.stamp = now.to_msg()
        pose.header.frame_id = "camera_color_optical_frame"
        pose.pose.position.x = x
        pose.pose.position.y = 0.0
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        self.pose_publisher.publish(pose)

        self.publish_count += 1

    def publish_detection_image(self, image: np.ndarray):
        """
        Publish detection visualization image.

        Parameters
        ----------
        image : np.ndarray
            BGR image with visualization overlays.
        """
        try:
            ros_img = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.header.frame_id = "camera_frame"
            self.image_publisher.publish(ros_img)
        except Exception as e:
            self.get_logger().warning(f"Failed to publish image: {e}")


# Utility Functions
def compute_lateral_offset(bbox, distance: float, fx: float, cx: float) -> float:
    """
    Compute lateral offset of target from camera center.

    Parameters
    ----------
    bbox : tuple of int
        Bounding box (x1, y1, x2, y2).
    distance : float
        Distance to target in meters.
    fx : float
        Camera focal length x.
    cx : float
        Camera principal point x.

    Returns
    -------
    float
        Lateral offset in meters (positive = right of center).
    """
    x1, y1, x2, y2 = bbox
    bbox_cx = (x1 + x2) / 2.0
    pixel_offset = bbox_cx - cx
    return (pixel_offset * distance) / fx


def project_lidar_xy_to_uv(
    xy_pts: np.ndarray,
    R_CL: np.ndarray,
    t_CL: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    W: int,
    H: int,
):
    """
    Project LiDAR XY points (2D, z=0) onto camera image plane.

    Uses the same extrinsic math as Go2ROSCameraLidar._project_scan()
    but works on arbitrary XY coordinates (not a full LaserScan).

    Parameters
    ----------
    xy_pts : np.ndarray
        (K, 2) array of XY points in LiDAR frame.
    R_CL, t_CL : np.ndarray
        Camera-to-LiDAR rotation (3x3) and translation (3x1).
    fx, fy, cx, cy : float
        Camera intrinsics.
    W, H : int
        Image width and height.

    Returns
    -------
    np.ndarray or None
        (M, 2) integer UV coordinates that fall inside the image,
        or None if nothing projects into view.
    """
    if xy_pts is None or len(xy_pts) == 0:
        return None

    N = len(xy_pts)
    P_L = np.vstack(
        [
            xy_pts[:, 0],
            xy_pts[:, 1],
            np.zeros(N),
        ]
    )  # (3, N)

    P_C = R_CL.T @ (P_L - t_CL)
    pts_cv = np.vstack([-P_C[1], -P_C[2], P_C[0]])
    Z = pts_cv[2]
    ok = Z > 0.01
    if not np.any(ok):
        return None

    pts = pts_cv[:, ok]
    K_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    uvw = K_mat @ pts
    uv = (uvw[:2] / uvw[2:3]).T
    uv_int = np.round(uv).astype(np.int32)

    in_img = (
        (uv_int[:, 0] >= 0)
        & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0)
        & (uv_int[:, 1] < H)
    )
    return uv_int[in_img] if np.any(in_img) else None


def _draw_lidar_overlay(display, result, target_bbox, W, H):
    """
    Draw LiDAR overlay: all scan points + tracked cluster + KF position.

    Colors:
      YELLOW (0,255,255) = all projected LiDAR scan points
      GREEN  (0,255,0)   = scan points inside target bbox
      MAGENTA(255,0,255) = KF-tracked cluster points
      CYAN   (255,255,0) = KF estimated position marker
    """
    lidar_uv = result.get("lidar_uv", None)
    tracked_uv = result.get("lidar_tracked_uv", None)
    kf_uv = result.get("lidar_kf_uv", None)
    lidar_source = result.get("lidar_source", "")
    lidar_lost = result.get("lidar_lost_frames", 0)

    # 1) Draw ALL projected scan points (yellow / green inside bbox)
    if lidar_uv is not None:
        uv = np.asarray(lidar_uv)
        if uv.ndim == 2 and uv.shape[1] == 2:
            max_pts = 2000
            step = max(1, len(uv) // max_pts) if len(uv) > max_pts else 1
            for idx in range(0, len(uv), step):
                ui, vi = int(uv[idx, 0]), int(uv[idx, 1])
                if ui < 0 or ui >= W or vi < 0 or vi >= H:
                    continue
                inside = False
                if target_bbox is not None:
                    bx1, by1, bx2, by2 = target_bbox
                    inside = (bx1 <= ui < bx2) and (by1 <= vi < by2)
                color = (0, 255, 0) if inside else (0, 255, 255)
                cv2.circle(display, (ui, vi), 2, color, -1)

    # 2) Draw KF-tracked cluster points (MAGENTA, larger)
    if tracked_uv is not None:
        for pt in tracked_uv:
            cv2.circle(display, (int(pt[0]), int(pt[1])), 4, (255, 0, 255), -1)

    # 3) Draw KF position crosshair (CYAN)
    if kf_uv is not None and len(kf_uv) > 0:
        ku, kv = int(kf_uv[0, 0]), int(kf_uv[0, 1])
        cv2.drawMarker(display, (ku, kv), (255, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.circle(display, (ku, kv), 16, (255, 255, 0), 2)
        cv2.putText(
            display,
            "KF",
            (ku + 18, kv - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )

    # 4) Draw predict-only ring (when KF has no measurement)
    if kf_uv is not None and len(kf_uv) > 0 and lidar_source in ("none", ""):
        ku, kv = int(kf_uv[0, 0]), int(kf_uv[0, 1])
        cv2.circle(display, (ku, kv), 24, (0, 165, 255), 1)
        cv2.putText(
            display,
            f"PREDICT ({lidar_lost}f)",
            (ku + 18, kv + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
        )


def draw_visualization(
    frame,
    result,
    system,
    is_tracked,
    x_offset,
    distance,
    publish_count,
    cmd_url,
    approached=False,
    operation_mode="greeting",
):
    """
    Draw visualization overlays on frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame to draw on.
    result : dict
        Processing result from system.process_frame().
    system : PersonFollowingSystem
        System instance for accessing state.
    is_tracked : bool
        Whether target is currently tracked.
    x_offset : float
        Lateral offset in meters.
    distance : float
        Distance to target in meters.
    publish_count : int
        Number of messages published.
    cmd_url : str
        HTTP command server URL.
    approached : bool, optional
        Whether target has approached, by default False.
    operation_mode : str, optional
        Current operation mode, by default 'greeting'.

    Returns
    -------
    np.ndarray
        Frame with visualization overlays.
    """
    display = frame.copy()
    H, W = display.shape[:2]

    lidar_uv = result.get("lidar_uv", None)
    is_go2_mode = lidar_uv is not None
    cluster_pts = result.get("lidar_cluster_pts", 0)
    bbox_pts = result.get("lidar_bbox_pts", 0)
    target_bbox = result.get("bbox", None) if result.get("target_found") else None
    status = result.get("status", "UNKNOWN")

    # Draw projected LiDAR overlay (Go2 mode)
    if is_go2_mode:
        _draw_lidar_overlay(display, result, target_bbox, W, H)

    # Draw SWITCHING mode visualization (greeting mode only)
    if status == "SWITCHING" and operation_mode == "greeting":
        ss = system.switch_state

        current_cand = ss.get_current_candidate()
        if current_cand and "bbox" in current_cand:
            x1, y1, x2, y2 = current_cand["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 3)

            label = (
                f"Checking #{ss.current_candidate_idx + 1} "
                f"(d={current_cand.get('distance', 0):.1f}m)"
            )
            cv2.putText(
                display,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

        for i in range(ss.current_candidate_idx):
            if i < len(ss.candidates):
                cand = ss.candidates[i]
                if "bbox" in cand:
                    x1, y1, x2, y2 = cand["bbox"]
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.line(display, (x2, y1), (x1, y2), (0, 0, 255), 2)
                    cv2.putText(
                        display,
                        "IN HIST",
                        (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                    )

        for i in range(ss.current_candidate_idx + 1, len(ss.candidates)):
            if i < len(ss.candidates):
                cand = ss.candidates[i]
                if "bbox" in cand:
                    x1, y1, x2, y2 = cand["bbox"]
                    cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(
                        display,
                        f"#{i+1}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (128, 128, 128),
                        1,
                    )

        switch_label = (
            f"SWITCHING: {ss.current_candidate_idx + 1}/{len(ss.candidates)} "
            f"(skipped: {ss.skipped_in_history} hist, {ss.skipped_no_features} no-feat)"
        )
        cv2.putText(
            display,
            switch_label,
            (W // 2 - 120, H // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    # Draw all tracks (for non-SWITCHING modes)
    if status != "SWITCHING":
        for track in result.get("all_tracks", []):
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["track_id"]
            target_id = getattr(getattr(system, "target", None), "track_id", None)

            # Only show green if TRACKING_ACTIVE and ID matches   <-- LINE 790
            if (
                status == "TRACKING_ACTIVE"
                and target_id is not None
                and track_id == target_id
            ):
                color = (0, 255, 0)
                thickness = 2
            elif (
                status == "SEARCHING"
                and target_id is not None
                and track_id == target_id
            ):
                # In SEARCHING mode, show the lost target's ID in orange (not green)
                color = (0, 165, 255)  # Orange - searching for this ID
                thickness = 2
            elif status == "SEARCHING":
                # All visible tracks are potential re-id candidates during search
                color = (0, 200, 255)  # Light orange - candidate
                thickness = 1
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

    # Draw target with distance label
    if result.get("target_found") and "bbox" in result:
        x1, y1, x2, y2 = result["bbox"]
        # Use different color if approached (greeting mode only)
        box_color = (
            (255, 0, 255)
            if (approached and operation_mode == "greeting")
            else (0, 255, 0)
        )
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)

        if is_go2_mode:
            if cluster_pts > 0 and distance > 0.1:
                dist_label = (
                    f"{distance:.2f}m ({cluster_pts}/{bbox_pts}pts) x:{x_offset:+.2f}m"
                )
            elif bbox_pts > 0:
                dist_label = f"...m (0/{bbox_pts}pts) x:{x_offset:+.2f}m"
            else:
                dist_label = f"...m (no LiDAR) x:{x_offset:+.2f}m"
        else:
            if distance > 0.1:
                dist_label = f"{distance:.2f}m x:{x_offset:+.2f}m"
            else:
                dist_label = f"...m x:{x_offset:+.2f}m"

        label_y = min(y2 + 20, H - 60)
        cv2.putText(
            display,
            dist_label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
        )

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(display, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
        cv2.line(display, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)

    # Draw candidates (SEARCHING mode) - orange boxes
    for cand in system.get_candidates_info():
        x1, y1, x2, y2 = cand["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            display,
            "SEARCH",
            (x1, y1 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
        )
        cv2.putText(
            display,
            f"L:{cand['clothing_sim']:.2f} C:{cand['clip_sim']:.2f}",
            (x1, y2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
        )

    # Top left status info
    fps = result.get("fps", 0)
    mode_str = "Go2+LiDAR" if is_go2_mode else "RealSense"

    line_height = 22
    y_pos = 20

    # Show operation mode
    op_mode_color = (0, 255, 255) if operation_mode == "following" else (255, 200, 0)
    cv2.putText(
        display,
        f"[{operation_mode.upper()}]",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        op_mode_color,
        2,
    )

    y_pos += line_height
    status_color = {
        "TRACKING_ACTIVE": (0, 255, 0),
        "SEARCHING": (0, 165, 255),
        "SWITCHING": (255, 255, 0),
        "INACTIVE": (128, 128, 128),
    }.get(status, (255, 255, 255))

    cv2.putText(
        display,
        f"Status: {status}",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
    )

    # Background verification indicator (greeting mode only)
    if operation_mode == "greeting" and status == "TRACKING_ACTIVE":
        bg_pending = result.get("bg_verify_pending", False)
        bg_count = result.get("bg_verify_valid_count", 0)
        bg_attempts = result.get("bg_verify_attempts", 0)
        bg_result = result.get("bg_verify_result")

        if bg_result == "in_history":
            # Flash red — matched history, about to go INACTIVE
            verify_text = "VERIFY: IN HISTORY -> SKIP"
            verify_color = (0, 0, 255)  # red
        elif bg_result == "timeout_assume_new":
            verify_text = "VERIFIED (timeout)"
            verify_color = (0, 200, 200)  # dark yellow
        elif bg_pending:
            verify_text = f"VERIFYING ({bg_count}/2, try {bg_attempts}/5)"
            verify_color = (0, 255, 255)  # yellow
        else:
            verify_text = "VERIFIED"
            verify_color = (0, 255, 0)  # green

        y_pos += line_height
        cv2.putText(
            display,
            verify_text,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            verify_color,
            2,
        )

    y_pos += line_height
    tracking_src = result.get("tracking_source", "")
    if is_tracked and tracking_src in ("lidar_searching", "lidar", "lidar_predict"):
        tracked_color = (255, 0, 255)  # magenta for LiDAR
        tracked_text = f"Tracked: LiDAR (d={distance:.2f}m)"
    elif is_tracked:
        tracked_color = (0, 255, 0)
        tracked_text = f"Tracked: YES (d={distance:.2f}m)"
    else:
        tracked_color = (0, 0, 255)
        tracked_text = "Tracked: NO"
    cv2.putText(
        display,
        tracked_text,
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        tracked_color,
        2,
    )

    # LiDAR out-of-FOV warning
    if result.get("lidar_out_of_fov"):
        y_pos += line_height
        cv2.putText(
            display,
            "LiDAR: OUT OF FOV",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 128, 255),  # orange
            2,
        )

    y_pos += line_height
    cv2.putText(
        display,
        f"Mode: {mode_str}",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    y_pos += line_height
    cv2.putText(
        display,
        f"FPS: {fps:.0f}  Pub: {publish_count}",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    # Number of persons detected
    y_pos += line_height
    num_persons = result.get("num_tracks", 0)
    cv2.putText(
        display,
        f"Persons: {num_persons}",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )

    # LiDAR tracker status
    lidar_active = result.get("lidar_active", False)
    if lidar_active or is_go2_mode:
        y_pos += line_height
        lidar_src = result.get("lidar_source", "n/a")
        lidar_lost = result.get("lidar_lost_frames", 0)
        lidar_kf_pts = result.get("lidar_cluster_pts", 0)

        if lidar_src == "camera_synced":
            lidar_color = (0, 255, 0)
            lidar_label = f"LiDAR: synced ({lidar_kf_pts}pts)"
        elif lidar_src == "lidar":
            lidar_color = (255, 0, 255)
            lidar_label = f"LiDAR: KF tracking ({lidar_kf_pts}pts)"
        elif lidar_src == "none" and lidar_active:
            lidar_color = (0, 165, 255)
            lidar_label = f"LiDAR: predict-only (lost {lidar_lost}f)"
        else:
            lidar_color = (128, 128, 128)
            lidar_label = "LiDAR: inactive"

        cv2.putText(
            display,
            lidar_label,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            lidar_color,
            2,
        )

    # History count (greeting mode only)
    if operation_mode == "greeting":
        y_pos += line_height
        hist_count = system.get_history_count()
        cv2.putText(
            display,
            f"History: {hist_count}/{system.max_history_size}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

        # Approached status (greeting mode only)
        y_pos += line_height
        approached_color = (255, 0, 255) if approached else (128, 128, 128)
        cv2.putText(
            display,
            f"Approached: {'YES' if approached else 'NO'}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            approached_color,
            2,
        )

    # Bottom bar - show different controls based on mode
    cv2.rectangle(display, (0, H - 50), (W, H), (0, 0, 0), -1)
    if operation_mode == "greeting":
        controls = "'e'=enroll | 'w'=switch | 'c'=clear | 'g'=greet_ack | 'm'=mode | 's'=status | 'q'=quit"
    else:
        controls = "'e'=enroll | 'c'=clear | 'm'=mode | 's'=status | 'q'=quit"
    cv2.putText(
        display,
        controls,
        (10, H - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        display,
        f"HTTP: {cmd_url}",
        (10, H - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    if status == "SEARCHING":
        time_lost = result.get("time_lost", 0)
        tracking_src = result.get("tracking_source", "")
        lidar_tracking = tracking_src == "lidar_searching"

        if operation_mode == "greeting":
            searching_timeout = getattr(system, "searching_timeout", 5.0)
            search_text = f"SEARCHING... ({time_lost:.1f}s / {searching_timeout:.1f}s)"
        else:
            search_text = f"SEARCHING... ({time_lost:.1f}s)"

        cv2.putText(
            display,
            search_text,
            (W // 2 - 100, H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

        # Show LiDAR-assisted searching indicator with position
        if lidar_tracking:
            lidar_pos = result.get("lidar_position")
            if lidar_pos:
                ld = lidar_pos.get("z", 0)
                lx = lidar_pos.get("x", 0)
                cv2.putText(
                    display,
                    f"LiDAR tracking: d={ld:.2f}m x={lx:.2f}m (publishing)",
                    (W // 2 - 130, H // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )
            else:
                cv2.putText(
                    display,
                    "LiDAR: lost cluster",
                    (W // 2 - 60, H // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1,
                )

    # Show approached banner (greeting mode only)
    if approached and operation_mode == "greeting":
        cv2.putText(
            display,
            "*** APPROACHED ***",
            (W // 2 - 100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 255),
            3,
        )

    return display


# Main
def main() -> None:
    """
    Main entry point for tracked person publisher node.

    Initializes ROS 2, camera interface, person following system,
    HTTP command server, and runs the main processing loop.
    """
    args = parse_args()

    operation_mode: OperationMode = args.mode

    logger.info("=" * 60)
    logger.info("PERSON FOLLOWING SYSTEM - ROS 2")
    logger.info("=" * 60)

    from person_following.managers.model_manager import ModelManager

    logger.info("Checking and preparing models...")
    model_manager = ModelManager(
        model_dir=MODEL_DIR,
        engine_dir=ENGINE_DIR,
        force_recompile=False,  # Set to True to force recompile
    )

    logger.info("Preparing detection model (YOLO11n)...")
    det_success, det_engine = model_manager.prepare_model("yolo11n")
    if not det_success:
        logger.error("Failed to prepare detection model")
        sys.exit(1)
    args.yolo_det = str(det_engine)
    logger.info(f"✓ Detection model ready: {det_engine}")

    # Prepare segmentation model
    logger.info("Preparing segmentation model (YOLO11s-seg)...")
    seg_success, seg_engine = model_manager.prepare_model("yolo11s-seg")
    if not seg_success:
        logger.error("Failed to prepare segmentation model")
        sys.exit(1)
    args.yolo_seg = str(seg_engine)
    logger.info(f"✓ Segmentation model ready: {seg_engine}")

    logger.info("All models ready!")
    logger.info("=" * 60)

    rclpy.init()
    ros_node = TrackedPersonPublisher()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(ros_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    logger.info("ROS 2 node initialized")

    # Initialize camera
    logger.info("Initializing camera...")
    if args.camera_mode == "realsense":
        camera = RealSenseROSCamera(
            node=ros_node,
            color_topic=args.color_topic,
            depth_topic=args.depth_topic,
            camera_info_topic=args.camera_info_topic,
            depth_scale=args.depth_scale,
        )
    elif args.camera_mode == "rtsp":
        # RTSP camera + ROS LiDAR
        logger.info(f"Loading camera extrinsics from {args.extrinsics_yaml}...")
        tx, ty, tz, rx, ry, rz = load_extrinsics_yaml(args.extrinsics_yaml)
        logger.info(
            f"Loaded extrinsics: tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}, "
            f"rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f}"
        )

        camera = Go2RTSPCameraLidar(
            node=ros_node,
            rtsp_url=args.rtsp_url,
            rtsp_fps=args.rtsp_fps,
            rtsp_decode_format=args.rtsp_decode_format,
            rtsp_reconnect_delay=args.rtsp_reconnect_delay,
            scan_topic=args.scan_topic,
            intrinsics_yaml=args.intrinsics_yaml,
            tx=tx,
            ty=ty,
            tz=tz,
            rx=rx,
            ry=ry,
            rz=rz,
        )
    else:
        # go2 mode: ROS image topic + ROS LiDAR
        logger.info(f"Loading camera extrinsics from {args.extrinsics_yaml}...")
        tx, ty, tz, rx, ry, rz = load_extrinsics_yaml(args.extrinsics_yaml)
        logger.info(
            f"Loaded extrinsics: tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}, "
            f"rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f}"
        )

        camera = Go2ROSCameraLidar(
            node=ros_node,
            image_topic=args.image_topic,
            scan_topic=args.scan_topic,
            intrinsics_yaml=args.intrinsics_yaml,
            tx=tx,
            ty=ty,
            tz=tz,
            rx=rx,
            ry=ry,
            rz=rz,
        )

    if getattr(camera, "width", 0) == 0:
        logger.error("Camera not available. Exiting.")
        ros_node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # Initialize person following system
    logger.info("Initializing person following system...")
    from person_following.nodes.person_following_system import PersonFollowingSystem

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
        lidar_cluster_range_jump=args.lidar_range_jump,
        lidar_cluster_min_size=args.lidar_min_cluster_size,
        # LiDAR KF tracker — enabled for go2 and rtsp camera modes
        use_lidar_tracker=(args.camera_mode in ("go2", "rtsp")),
        # History options
        max_history_size=args.max_history_size,
        history_file=args.history_file,
        auto_load_history=not args.no_auto_load_history,
        auto_save_history=not args.no_auto_save_history,
        # Approach and searching timeout options
        approach_distance=args.approach_distance,
        searching_timeout=args.searching_timeout,
        operation_mode=operation_mode,
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

    # Command server
    cmd_queue: "queue.Queue[Command]" = queue.Queue(maxsize=32)
    shared_status = SharedStatus(initial_mode=operation_mode)
    cmd_server: Optional[CommandServer] = None
    cmd_url = "(disabled)"

    if not args.no_command_server:
        cmd_server = CommandServer(
            args.cmd_host, args.cmd_port, cmd_queue, shared_status
        )
        cmd_server.start()
        cmd_url = cmd_server.url
        logger.info(f"Command API listening at: {cmd_url}")

    stop_event = threading.Event()

    def _sigterm_handler(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    pending_enroll = False
    approached = False  # track approached state
    last_action = {"name": None, "ok": None, "detail": None, "ts": None}

    logger.info("=" * 60)
    logger.info(f"Operation Mode: {operation_mode.upper()}")
    logger.info(f"Auto-enroll: {'ON' if auto_enroll else 'OFF'}")
    logger.info(f"Publish rate: {args.publish_hz} Hz")
    if operation_mode == "greeting":
        logger.info(f"Max history size: {args.max_history_size}")
        logger.info(f"Approach distance: {args.approach_distance}m")
        logger.info(f"Searching timeout: {args.searching_timeout}s")
        logger.info(
            "Controls: 'e'=enroll, 'w'=switch, 'c'=clear, 'g'=greet_ack, 'm'=mode, 'h'=clear_hist, 's'=status, 'q'=quit"
        )
    else:
        logger.info("(Following mode: no search timeout, no history, no switch)")
        logger.info("Controls: 'e'=enroll, 'c'=clear, 'm'=mode, 's'=status, 'q'=quit")
    logger.info(f"HTTP commands: {cmd_url}")
    logger.info("=" * 60)

    def _drain_commands(
        color_frame: np.ndarray, depth_frame: Optional[np.ndarray], aux: Optional[dict]
    ) -> None:
        nonlocal pending_enroll, last_action, approached, operation_mode
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
                    approached = False  # Reset approached when clearing
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
                continue

            if cmd.name == "enroll":
                pending_enroll = True
                approached = False  # Reset approached when enrolling new target
                last_action = {
                    "name": "enroll",
                    "ok": None,
                    "detail": "pending",
                    "ts": cmd.ts,
                }
                logger.info("Enroll requested (HTTP)")
                continue

            # Handle set_mode command
            if cmd.name == "set_mode":
                try:
                    new_mode = cmd.params.get("mode", "greeting")
                    result = system.set_operation_mode(new_mode)
                    if result["changed"]:
                        operation_mode = new_mode
                        shared_status.set_mode(new_mode)
                        approached = False
                    last_action = {
                        "name": "set_mode",
                        "ok": True,
                        "detail": f"{result['old_mode']} -> {result['new_mode']}",
                        "ts": cmd.ts,
                    }
                    logger.info(
                        f"Mode changed: {result['old_mode']} -> {result['new_mode']}"
                    )
                except Exception as e:
                    last_action = {
                        "name": "set_mode",
                        "ok": False,
                        "detail": str(e),
                        "ts": cmd.ts,
                    }
                continue

            # Greeting mode only commands
            if cmd.name == "switch":
                if operation_mode != "greeting":
                    last_action = {
                        "name": "switch",
                        "ok": False,
                        "detail": "not available in following mode",
                        "ts": cmd.ts,
                    }
                    logger.warning("Switch command ignored - not in greeting mode")
                    continue
                try:
                    switch_result = system.request_switch_target(
                        color_frame, depth_frame, aux
                    )
                    approached = False
                    last_action = {
                        "name": "switch",
                        "ok": switch_result.get("started", False),
                        "detail": switch_result.get(
                            "reason",
                            f"{switch_result.get('candidates_count', 0)} candidates",
                        ),
                        "ts": cmd.ts,
                    }
                    if switch_result.get("started"):
                        logger.info(
                            f"Switch started: {switch_result.get('candidates_count')} candidates"
                        )
                    else:
                        logger.warning(f"Switch failed: {switch_result.get('reason')}")
                except Exception as e:
                    last_action = {
                        "name": "switch",
                        "ok": False,
                        "detail": str(e),
                        "ts": cmd.ts,
                    }
                continue

            if cmd.name == "greeting_ack":
                if operation_mode != "greeting":
                    last_action = {
                        "name": "greeting_ack",
                        "ok": False,
                        "detail": "not available in following mode",
                        "ts": cmd.ts,
                    }
                    logger.warning(
                        "greeting_ack command ignored - not in greeting mode"
                    )
                    continue
                try:
                    result = system.handle_greeting_acknowledged()
                    approached = False
                    last_action = {
                        "name": "greeting_ack",
                        "ok": True,
                        "detail": f"saved={result['saved']}, history={result['history_size']}",
                        "ts": cmd.ts,
                    }
                    logger.info(
                        f"Greeting acknowledged - saved={result['saved']}, history_size={result['history_size']}"
                    )
                except Exception as e:
                    last_action = {
                        "name": "greeting_ack",
                        "ok": False,
                        "detail": str(e),
                        "ts": cmd.ts,
                    }
                continue

            if cmd.name == "clear_history":
                if operation_mode != "greeting":
                    last_action = {
                        "name": "clear_history",
                        "ok": False,
                        "detail": "not available in following mode",
                        "ts": cmd.ts,
                    }
                    continue
                try:
                    result = system.clear_history(delete_file=False)
                    last_action = {
                        "name": "clear_history",
                        "ok": True,
                        "detail": f"cleared {result['cleared_count']} entries",
                        "ts": cmd.ts,
                    }
                    logger.info(f"History cleared: {result['cleared_count']} entries")
                except Exception as e:
                    last_action = {
                        "name": "clear_history",
                        "ok": False,
                        "detail": str(e),
                        "ts": cmd.ts,
                    }
                continue

            if cmd.name == "delete_history":
                if operation_mode != "greeting":
                    last_action = {
                        "name": "delete_history",
                        "ok": False,
                        "detail": "not available in following mode",
                        "ts": cmd.ts,
                    }
                    continue
                try:
                    result = system.clear_history(delete_file=True)
                    last_action = {
                        "name": "delete_history",
                        "ok": True,
                        "detail": f"cleared {result['cleared_count']}, file_deleted={result['file_deleted']}",
                        "ts": cmd.ts,
                    }
                    logger.info(f"History deleted: {result}")
                except Exception as e:
                    last_action = {
                        "name": "delete_history",
                        "ok": False,
                        "detail": str(e),
                        "ts": cmd.ts,
                    }
                continue

            if cmd.name == "save_history":
                if operation_mode != "greeting":
                    last_action = {
                        "name": "save_history",
                        "ok": False,
                        "detail": "not available in following mode",
                        "ts": cmd.ts,
                    }
                    continue
                try:
                    result = system.save_history()
                    last_action = {
                        "name": "save_history",
                        "ok": result["success"],
                        "detail": f"saved {result.get('count', 0)} entries",
                        "ts": cmd.ts,
                    }
                    logger.info(f"History saved: {result}")
                except Exception as e:
                    last_action = {
                        "name": "save_history",
                        "ok": False,
                        "detail": str(e),
                        "ts": cmd.ts,
                    }
                continue

            if cmd.name == "load_history":
                if operation_mode != "greeting":
                    last_action = {
                        "name": "load_history",
                        "ok": False,
                        "detail": "not available in following mode",
                        "ts": cmd.ts,
                    }
                    continue
                try:
                    result = system.load_history()
                    last_action = {
                        "name": "load_history",
                        "ok": result["success"],
                        "detail": f"loaded {result.get('loaded_count')}/{result.get('file_count', 0)} entries",
                        "ts": cmd.ts,
                    }
                    logger.info(f"History loaded: {result}")
                except Exception as e:
                    last_action = {
                        "name": "load_history",
                        "ok": False,
                        "detail": str(e),
                        "ts": cmd.ts,
                    }
                continue

            if cmd.name == "set_max_history":
                if operation_mode != "greeting":
                    last_action = {
                        "name": "set_max_history",
                        "ok": False,
                        "detail": "not available in following mode",
                        "ts": cmd.ts,
                    }
                    continue
                try:
                    new_size = cmd.params.get("size", 5)
                    result = system.set_max_history_size(new_size)
                    last_action = {
                        "name": "set_max_history",
                        "ok": True,
                        "detail": f"{result['old_size']} -> {result['new_size']}, trimmed {result['trimmed_count']}",
                        "ts": cmd.ts,
                    }
                    logger.info(f"Max history size set: {result}")
                except Exception as e:
                    last_action = {
                        "name": "set_max_history",
                        "ok": False,
                        "detail": str(e),
                        "ts": cmd.ts,
                    }
                continue

    current_mode = "INACTIVE"
    mode_start_time = time.time()
    last_process_time = 0.0
    # Process at slightly lower rate than publish to leave headroom for display
    process_period = publish_period
    last_result = None

    try:
        while rclpy.ok() and not stop_event.is_set():
            color_frame, depth_frame, aux = camera.get_frames()
            if color_frame is None:
                continue
            if args.camera_mode == "realsense" and depth_frame is None:
                continue

            _drain_commands(color_frame, depth_frame, aux)

            if pending_enroll:
                pending_enroll = False
                approached = False
                try:
                    ok = bool(system.enroll_target(color_frame, depth_frame, aux))
                    last_action = {
                        "name": "enroll",
                        "ok": ok,
                        "detail": None,
                        "ts": time.time(),
                    }
                    if ok:
                        logger.info("Enrolled target")
                    else:
                        logger.warning("Enrollment failed")
                except Exception as e:
                    last_action = {
                        "name": "enroll",
                        "ok": False,
                        "detail": str(e),
                        "ts": time.time(),
                    }

            # Run heavy processing (YOLO+BoxMOT+state) at target rate,
            # not every loop iteration, to keep display responsive.
            current_time = time.time()
            should_process = (current_time - last_process_time) >= process_period

            if should_process:
                last_process_time = current_time
                result = system.process_frame(color_frame, depth_frame, aux)
                last_result = result

                # Inject LiDAR KF visualization data (project XY → image UV)
                if hasattr(camera, "R_CL") and result.get("lidar_active", False):
                    H_img, W_img = color_frame.shape[:2]
                    cluster_xy = result.get("lidar_cluster_xy", None)
                    if cluster_xy is not None:
                        result["lidar_tracked_uv"] = project_lidar_xy_to_uv(
                            cluster_xy,
                            camera.R_CL,
                            camera.t_CL,
                            camera.fx,
                            camera.fy,
                            camera.cx,
                            camera.cy,
                            W_img,
                            H_img,
                        )
                    kf_pos = result.get("lidar_kf_position_xy", None)
                    if kf_pos is not None:
                        kf_arr = np.array([[kf_pos[0], kf_pos[1]]])
                        result["lidar_kf_uv"] = project_lidar_xy_to_uv(
                            kf_arr,
                            camera.R_CL,
                            camera.t_CL,
                            camera.fx,
                            camera.fy,
                            camera.cx,
                            camera.cy,
                            W_img,
                            H_img,
                        )
            else:
                # Skipped processing — reuse last result for display
                if last_result is None:
                    continue
                result = last_result

            status = result.get("status", "INACTIVE")

            current_time = time.time()

            if status != current_mode:
                current_mode = status
                mode_start_time = current_time
                if status == "INACTIVE":
                    approached = False
            mode_duration = current_time - mode_start_time

            # One-time events: only process on fresh results
            if should_process:
                if result.get("bg_verify_result") == "in_history":
                    approached = False
                    logger.info(
                        "[BG-VERIFY] Person matched history → INACTIVE (skipped)"
                    )

                if result.get("timeout_inactive"):
                    approached = False
                    logger.info(
                        f"Searching timeout - went inactive, saved_to_history={result.get('saved_to_history')}"
                    )

                if not result.get("switch_active") and "success" in result:
                    if result.get("success"):
                        logger.info(
                            f"Switch completed: new target track_id={result.get('track_id')}"
                        )
                        last_action = {
                            "name": "switch_complete",
                            "ok": True,
                            "detail": f"track_id={result.get('track_id')}",
                            "ts": current_time,
                        }
                    elif result.get("reason"):
                        logger.warning(f"Switch ended: {result.get('reason')}")
                        last_action = {
                            "name": "switch_complete",
                            "ok": False,
                            "detail": result.get("reason"),
                            "ts": current_time,
                        }

            is_tracked = bool(
                result.get("target_found", False)
                and result.get("bbox") is not None
                and result.get("distance") is not None
            )

            tracking_source = result.get("tracking_source", "none")

            x_offset, distance = 0.0, 0.0
            if is_tracked:
                distance = float(result["distance"])
                if np.isfinite(distance) and distance > 0.1:
                    x_offset = float(
                        compute_lateral_offset(
                            result["bbox"], distance, camera.fx, camera.cx
                        )
                    )
                    tracking_source = "camera"
                    # Approached check: use system's internal state which is
                    # gated on bg_verify completion (greeting mode)
                    approached = system._target_approached
                else:
                    is_tracked = False

            # LiDAR position fallback: use LiDAR position when camera has no bbox.
            # This covers ALL LiDAR modes: "lidar", "lidar_predict", "lidar_searching"
            if not is_tracked and result.get("lidar_position") is not None:
                lidar_pos = result["lidar_position"]
                lz = float(lidar_pos.get("z", 0))  # forward distance (positive)
                lx = float(lidar_pos.get("x", 0))  # lateral offset
                lidar_euc = float(lidar_pos.get("distance", 0))  # euclidean sanity
                # Use z (forward distance) — should be positive after coord fix.
                # Fall back to euclidean if z is somehow still negative.
                ldist = lz if lz > 0.1 else lidar_euc
                if ldist > 0.1:
                    distance = ldist
                    x_offset = lx
                    is_tracked = True
                    tracking_source = result.get("tracking_source", "lidar")
                    if ros_node.publish_count % 15 == 0:
                        logger.info(
                            f"[LIDAR-PUB] {tracking_source}: "
                            f"z={lz:.2f}m x={lx:+.2f}m "
                            f"euc={lidar_euc:.2f}m (status={status})"
                        )

            if auto_enroll and status == "INACTIVE":
                try:
                    if system.enroll_target(color_frame, depth_frame, aux):
                        logger.info("Auto-enrolled nearest person")
                        approached = False
                except Exception:
                    pass

            if current_time - last_publish_time >= publish_period:
                num_persons = result.get("num_tracks", 0)
                ros_node.publish_status(
                    is_tracked,
                    x_offset,
                    distance,
                    current_mode,
                    mode_duration,
                    approached,
                    operation_mode=operation_mode,
                    num_persons=num_persons,
                    tracking_source=tracking_source,
                )
                last_publish_time = current_time

                display = None
                if (
                    (not args.no_publish_detection_image)
                    or args.display
                    or video_writer
                ):
                    display = draw_visualization(
                        color_frame,
                        result,
                        system,
                        is_tracked,
                        x_offset,
                        distance,
                        ros_node.publish_count,
                        cmd_url,
                        approached,
                        operation_mode=operation_mode,
                    )

                    if not args.no_publish_detection_image:
                        ros_node.publish_detection_image(display)

                    if video_writer:
                        video_writer.write(display)

                if args.display and display is not None:
                    cv2.imshow("Person Following - ROS 2", display)

                if ros_node.publish_count % 30 == 0 and is_tracked:
                    logger.info(
                        f"Target: x={x_offset:+.2f}m z={distance:.2f}m approached={approached}"
                    )

            # Always poll keyboard to keep window responsive
            if args.display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                elif key == ord("e"):
                    pending_enroll = True
                    approached = False
                elif key == ord("c"):
                    system.clear_target()
                    approached = False
                    last_action = {
                        "name": "clear",
                        "ok": True,
                        "detail": "keyboard",
                        "ts": time.time(),
                    }
                elif key == ord("w"):
                    # Switch (greeting mode only)
                    if operation_mode == "greeting":
                        try:
                            switch_result = system.request_switch_target(
                                color_frame, depth_frame, aux
                            )
                            approached = False
                            if switch_result.get("started"):
                                logger.info(
                                    f"Switch started (keyboard): {switch_result.get('candidates_count')} candidates"
                                )
                        except Exception as e:
                            logger.error(f"Switch failed (keyboard): {e}")
                    else:
                        logger.warning("Switch not available in following mode")
                elif key == ord("g"):
                    # Greeting ack (greeting mode only)
                    if operation_mode == "greeting":
                        try:
                            result_ack = system.handle_greeting_acknowledged()
                            approached = False
                            logger.info(
                                f"Greeting acknowledged (keyboard): saved={result_ack['saved']}, history={result_ack['history_size']}"
                            )
                        except Exception as e:
                            logger.error(f"Greeting ack failed (keyboard): {e}")
                    else:
                        logger.warning("Greeting ack not available in following mode")
                elif key == ord("h"):
                    # Clear history (greeting mode only)
                    if operation_mode == "greeting":
                        system.clear_history()
                        last_action = {
                            "name": "clear_history",
                            "ok": True,
                            "detail": "keyboard",
                            "ts": time.time(),
                        }
                    else:
                        logger.warning("Clear history not available in following mode")
                elif key == ord("m"):
                    # Toggle mode
                    new_mode = (
                        "following" if operation_mode == "greeting" else "greeting"
                    )
                    try:
                        result_mode = system.set_operation_mode(new_mode)
                        if result_mode["changed"]:
                            operation_mode = new_mode
                            shared_status.set_mode(new_mode)
                            approached = False
                            logger.info(
                                f"Mode toggled (keyboard): {result_mode['old_mode']} -> {result_mode['new_mode']}"
                            )
                    except Exception as e:
                        logger.error(f"Mode toggle failed (keyboard): {e}")
                elif key == ord("s"):
                    status_dict = system.get_status()
                    logger.info("=" * 40)
                    for k, v in status_dict.items():
                        logger.info(f"  {k}: {v}")
                    logger.info(f"  approached: {approached}")
                    logger.info(f"  ROS publish count: {ros_node.publish_count}")
                    logger.info(f"  HTTP: {cmd_url}")
                    logger.info("=" * 40)

            target_id = getattr(getattr(system, "target", None), "track_id", None)
            switch_info = (
                system.switch_state.get_summary()
                if system.switch_state.active
                else None
            )
            num_persons = result.get("num_tracks", 0)

            # Build shared status - include mode-specific fields
            status_data = {
                "ok": True,
                "ts": current_time,
                "operation_mode": operation_mode,
                "status": status,
                "is_tracked": bool(is_tracked),
                "x": float(round(x_offset, 3)),
                "z": float(round(distance, 3)),
                "num_persons": num_persons,
                "fps": float(result.get("fps", 0) or 0),
                "target_track_id": target_id,
                "auto_enroll": bool(auto_enroll),
                "publish_count": int(ros_node.publish_count),
                "last_action": dict(last_action),
                "approach_distance": system.approach_distance,
            }

            # Greeting mode specific fields
            if operation_mode == "greeting":
                status_data.update(
                    {
                        "history_size": system.get_history_count(),
                        "max_history_size": system.max_history_size,
                        "switch_active": system.switch_state.active,
                        "switch_info": switch_info,
                        "approached": bool(approached),
                        "searching_timeout": system.searching_timeout,
                    }
                )

            shared_status.set(status_data)

            # Yield CPU only when no work was done this iteration
            if not should_process and (
                current_time - last_publish_time < publish_period
            ):
                time.sleep(0.001)

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

        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            spin_thread.join(timeout=1.0)
        except Exception:
            pass

        ros_node.destroy_node()
        rclpy.shutdown()
        logger.info("Done.")


if __name__ == "__main__":
    main()
