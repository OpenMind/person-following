#!/usr/bin/env python3
"""
ROS2 node for following a tracked person with OM1 greeting integration and geofencing.

This module implements a person-following robot controller that:
- Subscribes to tracked person positions and publishes velocity commands
- Integrates with OM1 greeting system via Zenoh
- Supports two operation modes: greeting and following
- Implements geofencing to keep robot within a defined radius (greeting mode only)
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal, Optional, Tuple
from uuid import uuid4

import rclpy
import requests
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from om_api.msg import Paths
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from std_msgs.msg import String

from zenoh_msgs import PersonGreetingStatus, open_zenoh_session, prepare_header
from zenoh_msgs import String as ZenohString


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """
    Extract yaw angle from quaternion.

    Parameters
    ----------
    x, y, z, w : float
        Quaternion components.

    Returns
    -------
    float
        Yaw angle in radians.
    """
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class FollowerState(Enum):
    """State machine states for person follower."""

    IDLE = "IDLE"
    SWITCHING = "SWITCHING"
    APPROACHING = "APPROACHING"
    GREETING_IN_PROGRESS = "GREETING_IN_PROGRESS"
    SEARCHING = "SEARCHING"
    FOLLOWING = "FOLLOWING"
    RETURNING_TO_CENTER = "RETURNING_TO_CENTER"


class HandshakeCode:
    """Status codes for /om/person_greeting topic (matching OM1 PersonGreetingStatus)."""

    APPROACHING = 0
    APPROACHED = 1
    SWITCH = 2


class _ModeControlHTTPServer(ThreadingHTTPServer):
    """HTTP server for mode control with reference to PersonFollower node."""

    def __init__(self, addr, handler_cls, node: "PersonFollower"):
        """Initialize HTTP server with PersonFollower node reference."""
        super().__init__(addr, handler_cls)
        self.node = node


class _ModeControlHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mode control endpoints."""

    server: _ModeControlHTTPServer

    def log_message(self, fmt, *args):
        """Suppress default HTTP logging."""
        return

    def _send_json(self, code: int, payload: dict):
        """Send JSON response."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[dict]:
        """Read and parse JSON from request body."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/healthz":
            self._send_json(200, {"ok": True})
            return
        if self.path == "/status":
            node = self.server.node
            self._send_json(
                200,
                {
                    "ok": True,
                    "operation_mode": node.get_operation_mode(),
                    "state": node.state.value,
                },
            )
            return
        if self.path == "/get_mode":
            mode = self.server.node.get_operation_mode()
            self._send_json(200, {"ok": True, "operation_mode": mode})
            return
        if self.path == "/geofence":
            node = self.server.node
            geofence_status = node.get_geofence_status()
            self._send_json(200, geofence_status)
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        """Handle POST requests."""
        data = self._read_json() or {}

        if self.path == "/set_mode":
            mode = data.get("mode")
            if mode not in ("greeting", "following"):
                self._send_json(
                    400,
                    {
                        "ok": False,
                        "error": "invalid_mode",
                        "valid_modes": ["greeting", "following"],
                    },
                )
                return
            success = self.server.node.set_operation_mode(mode)
            self._send_json(200, {"ok": success, "operation_mode": mode})
            return

        if self.path == "/geofence/reset_center":
            # Reset geofence center to current robot position
            node = self.server.node
            success = node.reset_geofence_center()
            self._send_json(
                200,
                {
                    "ok": success,
                    "center": node.geofence_center,
                    "message": "Geofence center reset to current position",
                },
            )
            return

        if self.path == "/geofence/enable":
            node = self.server.node
            node.geofence_enabled = True
            # Auto-set center if not set and odom received
            if node.geofence_center is None and node.odom_received:
                node.geofence_center = node.current_odom_position
                node.get_logger().info(
                    f"[GEOFENCE] Center auto-set on enable: "
                    f"({node.geofence_center[0]:.2f}, {node.geofence_center[1]:.2f})"
                )
            self._send_json(
                200,
                {"ok": True, "geofence_enabled": True, "center": node.geofence_center},
            )
            return

        if self.path == "/geofence/disable":
            node = self.server.node
            node.geofence_enabled = False
            self._send_json(200, {"ok": True, "geofence_enabled": False})
            return

        if self.path == "/command":
            cmd = data.get("cmd")
            if cmd == "set_mode":
                mode = data.get("mode")
                if mode not in ("greeting", "following"):
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": "invalid_mode",
                            "valid_modes": ["greeting", "following"],
                        },
                    )
                    return
                success = self.server.node.set_operation_mode(mode)
                self._send_json(200, {"ok": success, "operation_mode": mode})
                return
            self._send_json(400, {"ok": False, "error": "unknown_command"})
            return

        self._send_json(404, {"ok": False, "error": "not_found"})


class PersonFollower(Node):
    """
    A ROS2 node that follows a tracked person and integrates with OM1 greeting system.

    Features:
    - PD control for smooth following
    - Search behavior when no person is found
    - Obstacle avoidance via /om/paths
    - Geofencing to keep robot within defined radius (greeting mode only)
    - Two operation modes: greeting and following
    """

    OperationMode = Literal["greeting", "following"]

    PATH_ANGLES = [60, 45, 30, 15, 0, -15, -30, -45, -60, 180]
    VALID_FORWARD_PATHS = [4]

    def __init__(self):
        """Initialize the PersonFollower ROS2 node."""
        super().__init__("person_follower")

        self._declare_parameters()
        self._load_parameters()
        self._setup_publishers()
        self._setup_subscribers()
        self._init_state()

        try:
            self.zenoh_session = open_zenoh_session()
            self.zenoh_sub = self.zenoh_session.declare_subscriber(
                "om/person_greeting", self._zenoh_person_greeting_callback
            )
            self.get_logger().info("Zenoh session for om/person_greeting initialized.")
        except Exception as e:
            self.zenoh_session = None
            self.get_logger().error(f"Zenoh session error: {e}")

        self.state_machine_timer = self.create_timer(0.3, self.state_machine_tick)

        self._log_startup_info()

    def _declare_parameters(self):
        """Declare all ROS parameters."""
        self.declare_parameter("target_distance", 0.8)
        self.declare_parameter("max_linear_speed", 0.7)
        self.declare_parameter("max_angular_speed", 1.6)
        self.declare_parameter("linear_kp", 0.6)
        self.declare_parameter("linear_kd", 0.05)
        self.declare_parameter("angular_kp", 1.4)
        self.declare_parameter("angular_kd", 0.1)
        self.declare_parameter("distance_tolerance", 0.2)
        self.declare_parameter("angle_tolerance", 0.35)
        self.declare_parameter("timeout", 2.0)

        self.declare_parameter("cmd_host", "127.0.0.1")
        self.declare_parameter("cmd_port", 2001)
        self.declare_parameter("http_port", 2000)

        self.declare_parameter("search_rotation_angle", 15.0)
        self.declare_parameter("search_rotation_speed", 0.3)
        self.declare_parameter("search_wait_time", 1.0)

        self.declare_parameter("geofence_enabled", True)
        self.declare_parameter("geofence_radius", 30.0)
        self.declare_parameter("geofence_soft_radius", 28.0)
        self.declare_parameter("geofence_return_speed", 0.4)
        self.declare_parameter("geofence_max_search_rotations", 48)

        self.declare_parameter("approach_no_position_timeout", 10.0)

    def _load_parameters(self):
        """Load parameters into instance variables."""
        self.target_distance = self.get_parameter("target_distance").value
        self.max_linear_speed = self.get_parameter("max_linear_speed").value
        self.max_angular_speed = self.get_parameter("max_angular_speed").value
        self.linear_kp = self.get_parameter("linear_kp").value
        self.linear_kd = self.get_parameter("linear_kd").value
        self.angular_kp = self.get_parameter("angular_kp").value
        self.angular_kd = self.get_parameter("angular_kd").value
        self.distance_tolerance = self.get_parameter("distance_tolerance").value
        self.angle_tolerance = self.get_parameter("angle_tolerance").value
        self.timeout = self.get_parameter("timeout").value

        self.cmd_host = self.get_parameter("cmd_host").value
        self.cmd_port = self.get_parameter("cmd_port").value
        self.http_port = self.get_parameter("http_port").value

        self.search_rotation_angle = self.get_parameter("search_rotation_angle").value
        self.search_rotation_speed = self.get_parameter("search_rotation_speed").value
        self.search_wait_time = self.get_parameter("search_wait_time").value

        self.geofence_enabled = self.get_parameter("geofence_enabled").value
        self.geofence_radius = self.get_parameter("geofence_radius").value
        self.geofence_soft_radius = self.get_parameter("geofence_soft_radius").value
        self.geofence_return_speed = self.get_parameter("geofence_return_speed").value
        self.geofence_max_search_rotations = self.get_parameter(
            "geofence_max_search_rotations"
        ).value

        self.approach_no_position_timeout = self.get_parameter(
            "approach_no_position_timeout"
        ).value

        self.cmd_base_url = f"http://{self.cmd_host}:{self.cmd_port}"

    def _setup_publishers(self):
        """Create ROS publishers."""
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.state_publisher = self.create_publisher(
            String, "/person_follower/state", 10
        )

    def _setup_subscribers(self):
        """Create ROS subscribers."""
        self.status_subscription = self.create_subscription(
            String, "/tracked_person/status", self.status_callback, 10
        )
        self.position_subscription = self.create_subscription(
            PoseStamped,
            "/person_following_robot/tracked_person/position",
            self.position_callback,
            10,
        )
        self.paths_subscription = self.create_subscription(
            Paths, "/om/paths", self.paths_callback, 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

    def _init_state(self):
        """Initialize state machine and control variables."""
        self.state = FollowerState.IDLE
        self.state_lock = threading.Lock()
        self.state_entry_time = time.time()

        self.operation_mode: "PersonFollower.OperationMode" = "greeting"
        self.mode_lock = threading.Lock()

        self.tracking_status: Optional[dict] = None
        self.tracking_status_received = False
        self.last_tracking_mode: Optional[str] = None
        self.switch_command_pending = False

        self.last_position: Optional[PoseStamped] = None
        self.last_msg_time: Optional[rclpy.time.Time] = None
        self.last_distance_error = 0.0
        self.last_angle_error = 0.0

        self.search_direction = 1
        self.search_rotation_start_time: Optional[float] = None
        self.search_rotation_duration = 0.0
        self.search_phase = "rotate"
        self.search_pause_start_time: Optional[float] = None
        self.search_rotation_count = 0

        self.safe_paths: list[int] = list(range(10))
        self.blocked_paths: set[int] = set()
        self.last_paths_time: Optional[float] = None
        self.paths_timeout = 1.0

        self.stop_sent = False
        self.is_tracking = False

        self.geofence_center: Optional[Tuple[float, float]] = None
        self.current_odom_position: Tuple[float, float] = (0.0, 0.0)
        self.current_odom_yaw: float = 0.0
        self.odom_received = False
        self.geofence_return_target: Optional[Tuple[float, float]] = None
        self.boundary_stuck_start_time: Optional[float] = None
        self.return_obstruction_start_time: Optional[float] = None

        self.last_valid_position_time: Optional[float] = None

        self._start_http_server()

    def _log_startup_info(self):
        """Log startup information."""
        robot_type = os.environ.get("ROBOT_TYPE", "unknown").lower()
        self.get_logger().info("=" * 60)
        self.get_logger().info("Person Follower with OM1 Integration Started")
        self.get_logger().info(f"  Robot type: {robot_type}")
        self.get_logger().info(f"  Tracking API: {self.cmd_base_url}")
        self.get_logger().info(f"  Target distance: {self.target_distance}m")
        self.get_logger().info(f"  Geofence enabled: {self.geofence_enabled}")
        if self.geofence_enabled:
            self.get_logger().info(f"  Geofence radius: {self.geofence_radius}m")
            self.get_logger().info(
                f"  Geofence soft radius: {self.geofence_soft_radius}m"
            )
        self.get_logger().info("=" * 60)

    def _start_http_server(self):
        """Start the HTTP server for mode control."""
        self._http_server = _ModeControlHTTPServer(
            ("0.0.0.0", self.http_port),
            _ModeControlHandler,
            self,
        )
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever, daemon=True
        )
        self._http_thread.start()
        self.get_logger().info(f"HTTP control server started on port {self.http_port}")

    def _stop_http_server(self):
        """Stop the HTTP server."""
        if hasattr(self, "_http_server") and self._http_server:
            self._http_server.shutdown()
            self._http_server.server_close()

    # =========================================================================
    # Operation Mode
    # =========================================================================
    def get_operation_mode(self) -> "PersonFollower.OperationMode":
        """Get current operation mode (thread-safe)."""
        with self.mode_lock:
            return self.operation_mode

    def set_operation_mode(self, mode: "PersonFollower.OperationMode") -> bool:
        """Set operation mode and notify vision system."""
        with self.mode_lock:
            if self.operation_mode == mode:
                return True
            old_mode = self.operation_mode
            self.operation_mode = mode

        self.get_logger().info(f"Operation mode changed: {old_mode} → {mode}")
        self._notify_vision_mode_change(mode)

        with self.state_lock:
            if mode == "following":
                self._stop_robot()
                self._transition_to(FollowerState.FOLLOWING)
                self.stop_sent = False
                self.is_tracking = False
            else:
                self._stop_robot()
                self._transition_to(FollowerState.IDLE)
                self.tracking_status_received = False

        return True

    def _notify_vision_mode_change(self, mode: "PersonFollower.OperationMode"):
        """Notify vision system of mode change via HTTP."""
        url = f"{self.cmd_base_url}/command"
        try:
            response = requests.post(
                url,
                json={"cmd": "set_mode", "mode": mode},
                timeout=2.0,
            )
            if response.status_code == 200:
                self.get_logger().info(f"Vision system mode set to '{mode}'")
            else:
                self.get_logger().error(
                    f"Failed to set vision mode: {response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Vision mode change error: {e}")

    def get_geofence_status(self) -> dict:
        """Get current geofence status."""
        distance_from_center = 0.0
        if self.geofence_center is not None:
            dx = self.current_odom_position[0] - self.geofence_center[0]
            dy = self.current_odom_position[1] - self.geofence_center[1]
            distance_from_center = math.sqrt(dx * dx + dy * dy)

        return {
            "ok": True,
            "enabled": self.geofence_enabled,
            "center": self.geofence_center,
            "radius": self.geofence_radius,
            "soft_radius": self.geofence_soft_radius,
            "current_position": self.current_odom_position,
            "current_yaw_deg": math.degrees(self.current_odom_yaw),
            "distance_from_center": round(distance_from_center, 2),
            "at_soft_boundary": distance_from_center >= self.geofence_soft_radius,
            "at_hard_boundary": distance_from_center >= self.geofence_radius,
            "odom_received": self.odom_received,
        }

    def reset_geofence_center(self) -> bool:
        """Reset geofence center to current robot position."""
        if not self.odom_received:
            self.get_logger().warn("Cannot reset geofence center: no odom received yet")
            return False

        self.geofence_center = self.current_odom_position
        self.get_logger().info(
            f"Geofence center reset to: ({self.geofence_center[0]:.2f}, {self.geofence_center[1]:.2f})"
        )
        return True

    def _get_distance_from_geofence_center(self) -> float:
        """Calculate robot's distance from geofence center."""
        if self.geofence_center is None:
            return 0.0
        dx = self.current_odom_position[0] - self.geofence_center[0]
        dy = self.current_odom_position[1] - self.geofence_center[1]
        return math.sqrt(dx * dx + dy * dy)

    def _get_person_global_position(
        self, person_x: float, person_z: float
    ) -> Tuple[float, float]:
        """
        Convert person's robot-relative position to global coordinates.

        Robot-relative frame from tracking:
          - x = lateral offset (positive = right)
          - z = forward distance

        Odom frame (ROS convention):
          - x = forward
          - y = left
          - yaw = rotation around z
        """
        robot_frame_x = person_z
        robot_frame_y = -person_x

        cos_yaw = math.cos(self.current_odom_yaw)
        sin_yaw = math.sin(self.current_odom_yaw)

        global_x = self.current_odom_position[0] + (
            robot_frame_x * cos_yaw - robot_frame_y * sin_yaw
        )
        global_y = self.current_odom_position[1] + (
            robot_frame_x * sin_yaw + robot_frame_y * cos_yaw
        )

        return (global_x, global_y)

    def _is_person_inside_geofence(self, person_x: float, person_z: float) -> bool:
        """
        Check if person (in robot-relative coords) is inside geofence.

        Parameters
        ----------
        person_x : float
            Person's lateral offset from robot (positive = right)
        person_z : float
            Person's forward distance from robot

        Returns
        -------
        bool
            True if person is inside geofence radius
        """
        if not self.geofence_enabled or self.geofence_center is None:
            return True

        person_global = self._get_person_global_position(person_x, person_z)

        dx = person_global[0] - self.geofence_center[0]
        dy = person_global[1] - self.geofence_center[1]
        distance = math.sqrt(dx * dx + dy * dy)

        inside = distance <= self.geofence_radius

        if not inside:
            self.get_logger().info(
                f"[GEOFENCE] Person at global ({person_global[0]:.1f}, {person_global[1]:.1f}) "
                f"is outside boundary ({distance:.1f}m > {self.geofence_radius}m)",
                throttle_duration_sec=1.0,
            )

        return inside

    def _apply_geofence_to_velocity(self, cmd: Twist) -> Twist:
        """
        Apply geofence constraints to velocity command.

        Only applies in greeting mode. Scales down speed in soft zone,
        blocks outward movement at hard boundary.
        """
        # Only apply geofence in greeting mode
        if self.get_operation_mode() != "greeting":
            return cmd

        if not self.geofence_enabled or self.geofence_center is None:
            return cmd

        distance_from_center = self._get_distance_from_geofence_center()

        if distance_from_center < self.geofence_soft_radius:
            return cmd

        if distance_from_center > 0.01:
            dx = self.current_odom_position[0] - self.geofence_center[0]
            dy = self.current_odom_position[1] - self.geofence_center[1]
            to_center_x = -dx / distance_from_center
            to_center_y = -dy / distance_from_center
        else:
            return cmd

        robot_forward_x = math.cos(self.current_odom_yaw)
        robot_forward_y = math.sin(self.current_odom_yaw)

        dot = to_center_x * robot_forward_x + to_center_y * robot_forward_y

        if dot > 0 or cmd.linear.x <= 0:
            return cmd

        if distance_from_center >= self.geofence_radius:
            self.get_logger().warn(
                f"[GEOFENCE] At hard boundary ({distance_from_center:.1f}m >= {self.geofence_radius}m), "
                "blocking forward movement",
                throttle_duration_sec=1.0,
            )
            cmd.linear.x = 0.0
            return cmd

        scale = (self.geofence_radius - distance_from_center) / (
            self.geofence_radius - self.geofence_soft_radius
        )
        scale = max(0.0, min(1.0, scale))

        original_speed = cmd.linear.x
        cmd.linear.x *= scale

        self.get_logger().info(
            f"[GEOFENCE] In soft zone ({distance_from_center:.1f}m), "
            f"speed scaled: {original_speed:.2f} → {cmd.linear.x:.2f}",
            throttle_duration_sec=0.5,
        )

        return cmd

    def odom_callback(self, msg: Odometry):
        """Handle odometry updates for geofencing."""
        self.current_odom_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )

        # Extract yaw from quaternion
        orientation = msg.pose.pose.orientation
        self.current_odom_yaw = quaternion_to_yaw(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

        if not self.odom_received:
            self.odom_received = True
            if self.geofence_enabled and self.geofence_center is None:
                self.geofence_center = self.current_odom_position
                self.get_logger().info(
                    f"[GEOFENCE] Center auto-set to: "
                    f"({self.geofence_center[0]:.2f}, {self.geofence_center[1]:.2f})"
                )

    def status_callback(self, msg: String):
        """Handle incoming tracking status updates."""
        try:
            self.tracking_status = json.loads(msg.data)
            self.tracking_status_received = True
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse tracking status: {e}")

    def paths_callback(self, msg: Paths):
        """Handle incoming path availability updates from obstacle detection."""
        self.safe_paths = list(msg.paths)
        self.blocked_paths = set(msg.blocked_by_obstacle_idx) | set(
            msg.blocked_by_hazard_idx
        )
        self.last_paths_time = time.time()

    def position_callback(self, msg: PoseStamped):
        """Handle incoming tracked person position updates."""
        if msg.pose.position.z == 0.0:
            return

        self.last_position = msg

        with self.state_lock:
            current_state = self.state

        if current_state == FollowerState.FOLLOWING:
            self._handle_following_position(msg)
        elif current_state == FollowerState.APPROACHING:
            self._handle_approaching_position(msg)

    def _handle_following_position(self, msg: PoseStamped):
        """Handle position updates in FOLLOWING mode."""
        self.stop_sent = False
        self.is_tracking = True

        current_time = self.get_clock().now()
        dt = 0.0
        if self.last_msg_time is not None:
            dt = (current_time - self.last_msg_time).nanoseconds / 1e9
        self.last_msg_time = current_time

        cmd_vel = self._calculate_following_velocity(msg, dt)
        self.cmd_vel_publisher.publish(cmd_vel)

    def _handle_approaching_position(self, msg: PoseStamped):
        """Handle position updates in APPROACHING mode."""
        self.last_valid_position_time = time.time()

        current_time = self.get_clock().now()
        dt = 0.0
        if self.last_msg_time is not None:
            dt = (current_time - self.last_msg_time).nanoseconds / 1e9
        self.last_msg_time = current_time

        cmd_vel = self.calculate_velocity(msg, dt)
        cmd_vel = self._apply_geofence_to_velocity(cmd_vel)
        self.cmd_vel_publisher.publish(cmd_vel)

    def _zenoh_person_greeting_callback(self, data):
        """Handle incoming PersonGreetingStatus messages from Zenoh."""
        try:
            msg = PersonGreetingStatus.deserialize(data.payload.to_bytes())
            if msg.status == HandshakeCode.SWITCH:
                self.get_logger().info(
                    "[Zenoh] Received SWITCH (2) from OM1"
                )
                with self.state_lock:
                    if self.state == FollowerState.IDLE:
                        # Start person following from IDLE state
                        self.get_logger().info("Starting person following...")
                        self._transition_to(FollowerState.SWITCHING)
                        self._call_switch_command()
                    elif self.state == FollowerState.GREETING_IN_PROGRESS:
                        # History already saved when person approached.
                        # Just start switching to next person.
                        self._transition_to(FollowerState.SWITCHING)
                        self._call_switch_command()
        except Exception as e:
            self.get_logger().error(f"Zenoh callback error: {e}")

    def state_machine_tick(self):
        """Main state machine tick called periodically by timer."""
        state_msg = String()
        state_msg.data = self.state.value
        self.state_publisher.publish(state_msg)

        with self.state_lock:
            if self.state == FollowerState.IDLE:
                self._handle_idle_state()
            elif self.state == FollowerState.SWITCHING:
                self._handle_switching_state()
            elif self.state == FollowerState.APPROACHING:
                self._handle_approaching_state()
            elif self.state == FollowerState.GREETING_IN_PROGRESS:
                pass
            elif self.state == FollowerState.SEARCHING:
                self._handle_searching_state()
            elif self.state == FollowerState.FOLLOWING:
                self._handle_following_state()
            elif self.state == FollowerState.RETURNING_TO_CENTER:
                self._handle_returning_to_center_state()

    def _transition_to(self, new_state: FollowerState):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.state_entry_time = time.time()

        if new_state == FollowerState.SEARCHING:
            self.search_phase = "rotate"
            self.search_rotation_start_time = None
            self.search_pause_start_time = None
            self.search_direction *= 1

        if new_state == FollowerState.APPROACHING:
            self.last_distance_error = 0.0
            self.last_angle_error = 0.0
            self.last_msg_time = None
            self.search_rotation_count = 0
            self.boundary_stuck_start_time = None
            self.last_valid_position_time = time.time()

        if new_state == FollowerState.FOLLOWING:
            self.last_distance_error = 0.0
            self.last_angle_error = 0.0
            self.last_msg_time = None
            self.stop_sent = False
            self.is_tracking = False

        if new_state == FollowerState.RETURNING_TO_CENTER:
            self.search_rotation_count = 0
            self.geofence_return_target = self._generate_random_return_target()
            self.return_obstruction_start_time = None

        self.get_logger().info(
            f"State transition: {old_state.value} → {new_state.value}"
        )

    def _handle_idle_state(self):
        """Handle IDLE state - wait for SWITCH command from om/person_greeting."""
        pass

    def _handle_switching_state(self):
        """Handle SWITCHING state - wait for tracking result."""
        self._stop_robot()

        if self.tracking_status is None:
            return

        mode = self.tracking_status.get("mode", "UNKNOWN")

        if mode == "TRACKING_ACTIVE":
            # Check if person is inside geofence before accepting
            person_x = self.tracking_status.get("x", 0.0)
            person_z = self.tracking_status.get("z", 0.0)

            if self._is_person_inside_geofence(person_x, person_z):
                self.get_logger().info(
                    "Tracking found a person inside geofence, starting approach"
                )
                self.switch_command_pending = False
                self._transition_to(FollowerState.APPROACHING)
            else:
                self.get_logger().warn(
                    "[GEOFENCE] Found person is outside boundary, rejecting and searching..."
                )
                self.switch_command_pending = False
                self._call_clear_command()  # Don't save — never greeted
                self._transition_to(FollowerState.SEARCHING)

        elif mode == "INACTIVE" and self.switch_command_pending:
            self.get_logger().info("No valid candidates found, starting search")
            self.switch_command_pending = False
            self._transition_to(FollowerState.SEARCHING)

        self.last_tracking_mode = mode

    def _handle_approaching_state(self):
        """Handle APPROACHING state - PD control and check for approached."""
        # Check for timeout - no valid position received for too long
        if self.last_valid_position_time is not None:
            time_since_position = time.time() - self.last_valid_position_time
            if time_since_position > self.approach_no_position_timeout:
                self.get_logger().warn(
                    f"[TIMEOUT] No valid position for {time_since_position:.1f}s "
                    f"(> {self.approach_no_position_timeout}s), returning to search"
                )
                self._stop_robot()
                self._call_clear_command()
                self._transition_to(FollowerState.SEARCHING)
                return

        if self.tracking_status is None:
            return

        mode = self.tracking_status.get("mode", "UNKNOWN")
        approached = self.tracking_status.get("approached", False)

        if mode == "SWITCHING":
            self._stop_robot()
            self.get_logger().info("Tracking is switching, stopping robot...")
            self._transition_to(FollowerState.SWITCHING)
            return

        if approached:
            self.get_logger().info(
                "Person approached! Saving to history and signaling OM1"
            )
            self._stop_robot()
            # Save features to history (person is close, good features) + clear target
            self._call_greeting_ack_command()
            self._publish_approached()
            self._transition_to(FollowerState.GREETING_IN_PROGRESS)
            return

        if self._check_boundary_stuck():
            return

        if mode == "SEARCHING":
            self.get_logger().info(
                "Tracking lost person, waiting for re-identification..."
            )
        elif mode == "INACTIVE" and self.last_tracking_mode == "SEARCHING":
            self.get_logger().info("Re-identification failed, starting search")
            self._stop_robot()
            self._transition_to(FollowerState.SEARCHING)

        self.last_tracking_mode = mode

    def _check_boundary_stuck(self) -> bool:
        """
        Check if robot is stuck at geofence boundary while approaching.

        If robot is at hard boundary and person is further out (unreachable),
        abandon target after 10 seconds and return to center.

        Returns
        -------
        bool
            True if state was changed (caller should return), False otherwise.
        """
        if not self.geofence_enabled or self.geofence_center is None:
            return False

        distance_from_center = self._get_distance_from_geofence_center()

        at_boundary = distance_from_center >= (self.geofence_radius - 0.5)

        if not at_boundary:
            self.boundary_stuck_start_time = None
            return False

        person_x = self.tracking_status.get("x", 0.0)
        person_z = self.tracking_status.get("z", 0.0)
        person_global = self._get_person_global_position(person_x, person_z)

        dx = person_global[0] - self.geofence_center[0]
        dy = person_global[1] - self.geofence_center[1]
        person_distance_from_center = math.sqrt(dx * dx + dy * dy)

        person_unreachable = person_distance_from_center > distance_from_center

        if not person_unreachable:
            self.boundary_stuck_start_time = None
            return False

        current_time = time.time()

        if self.boundary_stuck_start_time is None:
            self.boundary_stuck_start_time = current_time
            self.get_logger().warn(
                f"[GEOFENCE] At boundary ({distance_from_center:.1f}m) and person is "
                f"unreachable ({person_distance_from_center:.1f}m from center). "
                "Waiting 10s before abandoning..."
            )
            return False

        stuck_duration = current_time - self.boundary_stuck_start_time

        if stuck_duration >= 10.0:
            self.get_logger().warn(
                f"[GEOFENCE] Stuck at boundary for {stuck_duration:.1f}s. "
                "Abandoning target and returning to safe zone."
            )
            self._stop_robot()
            self._call_clear_command()  # Don't save — never greeted
            self.boundary_stuck_start_time = None
            self._transition_to(FollowerState.RETURNING_TO_CENTER)
            return True

        self.get_logger().info(
            f"[GEOFENCE] Stuck at boundary ({stuck_duration:.1f}s / 10.0s)...",
            throttle_duration_sec=2.0,
        )
        return False

    def _handle_following_state(self):
        """Handle FOLLOWING state - safety check for tracking timeout."""
        if self.last_msg_time is None:
            return

        time_since_update = (
            self.get_clock().now() - self.last_msg_time
        ).nanoseconds / 1e9

        if time_since_update > self.timeout:
            if not self.stop_sent:
                self._stop_robot()
                self.stop_sent = True
                self.is_tracking = False
                self.get_logger().warn(
                    f"No person detected for {time_since_update:.1f}s - stopping robot"
                )

    def _handle_searching_state(self):
        """Handle SEARCHING state - rotate, pause, and search for people."""
        if (
            self.geofence_enabled
            and self.search_rotation_count >= self.geofence_max_search_rotations
        ):
            distance_from_center = self._get_distance_from_geofence_center()
            if distance_from_center > self.geofence_soft_radius * 0.5:
                self.get_logger().info(
                    f"[GEOFENCE] {self.search_rotation_count} rotations without valid target, "
                    f"returning to center (current distance: {distance_from_center:.1f}m)"
                )
                self._stop_robot()
                self._transition_to(FollowerState.RETURNING_TO_CENTER)
                return
            else:
                self.search_rotation_count = 0

        if self.search_phase == "rotate":
            self._do_search_rotation()
        elif self.search_phase == "pause":
            self._do_search_pause()
        elif self.search_phase == "wait":
            self._do_search_wait()

    def _generate_random_return_target(self) -> Optional[Tuple[float, float]]:
        """
        Generate a nearby target point inside the soft radius.

        The point is chosen to be:
        1. Inside the soft radius (safe zone)
        2. As close as possible to current robot position (minimize travel)

        Strategy: Move toward center just enough to get inside the safe zone,
        with a small random offset to vary the search position.

        Returns
        -------
        Tuple[float, float] or None
            Target (x, y) point inside soft radius, or None if no center set.
        """
        if self.geofence_center is None:
            return None

        # Current robot position relative to center
        dx = self.current_odom_position[0] - self.geofence_center[0]
        dy = self.current_odom_position[1] - self.geofence_center[1]
        distance_from_center = math.sqrt(dx * dx + dy * dy)

        if distance_from_center < 0.01:
            random_angle = random.uniform(0, 2 * math.pi)
            random_radius = random.uniform(1.0, 3.0)
            target_x = self.geofence_center[0] + random_radius * math.cos(random_angle)
            target_y = self.geofence_center[1] + random_radius * math.sin(random_angle)
            return (target_x, target_y)

        dir_x = dx / distance_from_center
        dir_y = dy / distance_from_center

        target_distance_from_center = self.geofence_soft_radius * random.uniform(
            0.70, 0.85
        )

        base_target_x = self.geofence_center[0] + dir_x * target_distance_from_center
        base_target_y = self.geofence_center[1] + dir_y * target_distance_from_center

        perp_x = -dir_y
        perp_y = dir_x
        random_offset = random.uniform(-2.0, 2.0)

        target_x = base_target_x + perp_x * random_offset
        target_y = base_target_y + perp_y * random_offset

        travel_distance = math.sqrt(
            (target_x - self.current_odom_position[0]) ** 2
            + (target_y - self.current_odom_position[1]) ** 2
        )

        self.get_logger().info(
            f"[GEOFENCE] Return target: ({target_x:.1f}, {target_y:.1f}), "
            f"travel distance: {travel_distance:.1f}m, "
            f"will be {target_distance_from_center:.1f}m from center"
        )

        return (target_x, target_y)

    def _handle_returning_to_center_state(self):
        """
        Handle RETURNING_TO_CENTER state - move robot to safe point inside geofence.

        Includes obstacle avoidance using /om/paths:
        - If path clear: move toward target
        - If blocked < 5s: wait
        - If blocked 5-30s: try alternative angles
        - If blocked > 30s: give up, do rotation search from current position
        """
        if self.geofence_center is None:
            self.get_logger().warn("No geofence center set, transitioning to SEARCHING")
            self._transition_to(FollowerState.SEARCHING)
            return

        target = (
            self.geofence_return_target
            if self.geofence_return_target
            else self.geofence_center
        )

        dx = target[0] - self.current_odom_position[0]
        dy = target[1] - self.current_odom_position[1]
        distance_to_target = math.sqrt(dx * dx + dy * dy)
        angle_to_target = math.atan2(dy, dx)

        arrival_threshold = 1.0
        if distance_to_target < arrival_threshold:
            self.get_logger().info(
                f"[GEOFENCE] Arrived at return target ({distance_to_target:.1f}m away), starting search"
            )
            self._stop_robot()
            self.geofence_return_target = None
            self.return_obstruction_start_time = None
            self._transition_to(FollowerState.SEARCHING)
            return

        path_clear, alternative_angle = self._check_return_path_safety(angle_to_target)

        current_time = time.time()

        if path_clear:
            self.return_obstruction_start_time = None
            self._move_toward_return_target(angle_to_target, distance_to_target)

        else:
            if self.return_obstruction_start_time is None:
                self.return_obstruction_start_time = current_time
                self.get_logger().info("[RETURNING] Path blocked, waiting...")
                self._stop_robot()
                return

            obstruction_duration = current_time - self.return_obstruction_start_time

            if obstruction_duration < 5.0:
                self._stop_robot()
                self.get_logger().info(
                    f"[RETURNING] Waiting for path to clear ({obstruction_duration:.1f}s / 5.0s)",
                    throttle_duration_sec=1.0,
                )

            elif obstruction_duration < 30.0:
                if alternative_angle is not None:
                    self.get_logger().info(
                        f"[RETURNING] Using alternative path at {math.degrees(alternative_angle):.1f}° "
                        f"({obstruction_duration:.1f}s blocked)",
                        throttle_duration_sec=1.0,
                    )
                    self._move_toward_return_target(
                        alternative_angle, distance_to_target
                    )
                else:
                    self._stop_robot()
                    self.get_logger().info(
                        f"[RETURNING] No clear path, waiting ({obstruction_duration:.1f}s / 30.0s)",
                        throttle_duration_sec=1.0,
                    )

            else:
                self.get_logger().warn(
                    f"[GEOFENCE] Blocked for {obstruction_duration:.1f}s, "
                    "giving up return - searching from current position"
                )
                self._stop_robot()
                self.geofence_return_target = None
                self.return_obstruction_start_time = None
                self._transition_to(FollowerState.SEARCHING)

    def _check_return_path_safety(
        self, target_angle: float
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if the path toward return target is clear.

        Parameters
        ----------
        target_angle : float
            Global angle to target in radians.

        Returns
        -------
        Tuple[bool, Optional[float]]
            (is_direct_path_clear, alternative_angle_if_blocked)
            alternative_angle is in global frame (radians)
        """
        if self.last_paths_time is None or (
            time.time() - self.last_paths_time > self.paths_timeout
        ):
            self.get_logger().warn(
                "[RETURNING] Path data stale or unavailable, assuming blocked",
                throttle_duration_sec=2.0,
            )
            return False, None

        relative_angle = target_angle - self.current_odom_yaw
        relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
        relative_angle_deg = math.degrees(relative_angle)

        target_path_idx = self._angle_to_path_index_for_return(relative_angle_deg)

        if (
            target_path_idx in self.safe_paths
            and target_path_idx not in self.blocked_paths
        ):
            return True, None

        alternative_offsets = [15, -15, 30, -30, 45, -45]

        for offset_deg in alternative_offsets:
            alt_relative_deg = relative_angle_deg + offset_deg
            alt_path_idx = self._angle_to_path_index_for_return(alt_relative_deg)

            if (
                alt_path_idx in self.safe_paths
                and alt_path_idx not in self.blocked_paths
            ):
                alt_global_angle = self.current_odom_yaw + math.radians(
                    alt_relative_deg
                )
                return False, alt_global_angle

        return False, None

    def _angle_to_path_index_for_return(self, angle_deg: float) -> int:
        """
        Convert robot-relative angle to path index for return navigation.

        Parameters
        ----------
        angle_deg : float
            Angle in degrees, robot-relative (0° = forward, positive = left)

        Returns
        -------
        int
            Path index (0-8)
        """
        min_diff = float("inf")
        closest_idx = 4

        for idx in range(9):
            diff = abs(self.PATH_ANGLES[idx] - angle_deg)
            if diff < min_diff:
                min_diff = diff
                closest_idx = idx

        return closest_idx

    def _move_toward_return_target(self, movement_angle: float, distance: float):
        """
        Move robot toward return target at specified angle.

        Parameters
        ----------
        movement_angle : float
            Global angle to move toward (radians)
        distance : float
            Distance to target (for speed scaling)
        """
        # Calculate angle error
        angle_error = movement_angle - self.current_odom_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        cmd = Twist()

        if abs(angle_error) > 0.3:
            cmd.angular.z = 0.5 if angle_error > 0 else -0.5
            cmd.linear.x = 0.0
        else:
            cmd.angular.z = 0.5 * angle_error
            cmd.linear.x = min(self.geofence_return_speed, distance * 0.3)

        self.cmd_vel_publisher.publish(cmd)

        self.get_logger().info(
            f"[RETURNING] Moving: angle_err={math.degrees(angle_error):.1f}°, "
            f"dist={distance:.1f}m, cmd=({cmd.linear.x:.2f}, {cmd.angular.z:.2f})",
            throttle_duration_sec=1.0,
        )

    def _do_search_rotation(self):
        """Execute search rotation phase."""
        if self.search_rotation_start_time is None:
            self.search_rotation_start_time = time.time()
            angle_rad = math.radians(self.search_rotation_angle)
            self.search_rotation_duration = angle_rad / self.search_rotation_speed
            direction = "left" if self.search_direction > 0 else "right"
            self.get_logger().info(
                f"Starting search rotation: {self.search_rotation_angle}° ({direction})"
            )

        elapsed = time.time() - self.search_rotation_start_time

        if elapsed < self.search_rotation_duration:
            cmd = Twist()
            cmd.angular.z = self.search_rotation_speed * self.search_direction
            self.cmd_vel_publisher.publish(cmd)
        else:
            self._stop_robot()
            self.search_phase = "pause"
            self.search_pause_start_time = time.time()
            self.search_rotation_count += 1
            self.get_logger().info(
                f"Search rotation complete, pausing for {self.search_wait_time}s... "
                f"(rotation #{self.search_rotation_count})"
            )

    def _do_search_pause(self):
        """Wait after rotation to let vision stabilize before calling switch."""
        if self.search_pause_start_time is None:
            self.search_pause_start_time = time.time()

        elapsed = time.time() - self.search_pause_start_time

        if elapsed >= self.search_wait_time:
            num_persons = 0
            if self.tracking_status is not None:
                num_persons = self.tracking_status.get("num_persons", 0)

            if num_persons > 0:
                self.search_phase = "wait"
                self.get_logger().info(
                    f"Pause complete, {num_persons} person(s) visible - calling switch..."
                )
                self._call_switch_command()
            else:
                self.get_logger().info(
                    "Pause complete, no one visible - rotating again..."
                )
                self.search_phase = "rotate"
                self.search_rotation_start_time = None

    def _do_search_wait(self):
        """Wait for switch result after rotation."""
        if self.tracking_status is None:
            return

        mode = self.tracking_status.get("mode", "UNKNOWN")

        if mode == "TRACKING_ACTIVE":
            person_x = self.tracking_status.get("x", 0.0)
            person_z = self.tracking_status.get("z", 0.0)

            if self._is_person_inside_geofence(person_x, person_z):
                self.get_logger().info(
                    "Found a person during search (inside geofence)!"
                )
                self.switch_command_pending = False
                self._transition_to(FollowerState.APPROACHING)
            else:
                self.get_logger().warn(
                    "[GEOFENCE] Found person is outside boundary, rejecting..."
                )
                self.switch_command_pending = False
                self._call_clear_command()  # Don't save — never greeted
                self.search_phase = "rotate"
                self.search_rotation_start_time = None

        elif mode == "INACTIVE" and self.switch_command_pending:
            self.get_logger().info("No one found, rotating again...")
            self.switch_command_pending = False
            self.search_phase = "rotate"
            self.search_rotation_start_time = None

        self.last_tracking_mode = mode

    def calculate_velocity(self, pose_msg: PoseStamped, dt: float) -> Twist:
        """Calculate velocity commands using PD control with obstacle avoidance."""
        cmd = Twist()

        x = pose_msg.pose.position.x
        z = pose_msg.pose.position.z

        distance = math.sqrt(x**2 + z**2)
        angle = math.atan2(x, z)
        angle_deg = math.degrees(angle)

        close_to_person = distance < (self.target_distance + 0.7)
        if close_to_person:
            path_safe = True
            alternative_angle = None
        else:
            path_safe, alternative_angle = self._check_path_safety(angle_deg)

        distance_error = distance - self.target_distance
        angle_error = angle

        if not path_safe:
            if alternative_angle is not None:
                alt_angle_rad = math.radians(alternative_angle)
                angle_error = alt_angle_rad
                self.get_logger().info(
                    f"[OBSTACLE] Path blocked at {angle_deg:.1f}°, "
                    f"redirecting to safe path at {alternative_angle:.1f}°",
                    throttle_duration_sec=1.0,
                )
            else:
                self.get_logger().warn(
                    f"[OBSTACLE] No safe path to person at {angle_deg:.1f}°, "
                    "switching to search for another person",
                    throttle_duration_sec=1.0,
                )
                self._call_switch_command()
                self._transition_to(FollowerState.SEARCHING)
                return cmd

        # Angular PD control
        p_ang = -self.angular_kp * angle_error
        d_ang = 0.0
        if dt > 0.001:
            d_ang = -self.angular_kd * (angle_error - self.last_angle_error) / dt
        angular_vel = p_ang + d_ang
        self.last_angle_error = angle_error

        # Linear PD control
        p_lin = self.linear_kp * distance_error
        d_lin = 0.0
        if dt > 0.001:
            d_lin = self.linear_kd * (distance_error - self.last_distance_error) / dt
        linear_vel = p_lin + d_lin
        self.last_distance_error = distance_error

        # Clamp angular velocity
        angular_vel = max(
            -self.max_angular_speed, min(angular_vel, self.max_angular_speed)
        )

        if abs(angle_error) > self.angle_tolerance:
            cmd.angular.z = angular_vel
            cmd.linear.x = 0.0
        else:
            if path_safe:
                cmd.angular.z = angular_vel
                linear_vel = max(
                    -self.max_linear_speed, min(linear_vel, self.max_linear_speed)
                )
                if abs(distance_error) < self.distance_tolerance:
                    linear_vel = 0.0
                cmd.linear.x = linear_vel
            else:
                cmd.angular.z = angular_vel
                cmd.linear.x = 0.0

        safe_status = "SAFE" if path_safe else "BLOCKED"
        self.get_logger().info(
            f"[APPROACHING] Dist: {distance:.2f}m Err:{distance_error:.2f}, "
            f"Angle: {angle_deg:.1f}° [{safe_status}] | "
            f"Cmd: lin={cmd.linear.x:.2f}, ang={cmd.angular.z:.2f}",
            throttle_duration_sec=0.5,
        )

        return cmd

    def _calculate_following_velocity(self, pose_msg: PoseStamped, dt: float) -> Twist:
        """Calculate velocity commands for following mode."""
        cmd = Twist()

        x = pose_msg.pose.position.x
        z = pose_msg.pose.position.z

        distance = math.sqrt(x**2 + z**2)
        angle = math.atan2(x, z)

        distance_error = distance - self.target_distance
        angle_error = angle

        # Angular PD control
        p_ang = -self.angular_kp * angle_error
        d_ang = 0.0
        if dt > 0.001:
            d_ang = -self.angular_kd * (angle_error - self.last_angle_error) / dt
        angular_vel = p_ang + d_ang
        self.last_angle_error = angle_error

        # Linear PD control
        p_lin = self.linear_kp * distance_error
        d_lin = 0.0
        if dt > 0.001:
            d_lin = self.linear_kd * (distance_error - self.last_distance_error) / dt
        linear_vel = p_lin + d_lin
        self.last_distance_error = distance_error

        # Clamp angular velocity
        angular_vel = max(
            -self.max_angular_speed, min(angular_vel, self.max_angular_speed)
        )

        if abs(angle_error) > self.angle_tolerance:
            cmd.angular.z = angular_vel
            cmd.linear.x = 0.0
        else:
            cmd.angular.z = angular_vel
            linear_vel = max(
                -self.max_linear_speed, min(linear_vel, self.max_linear_speed)
            )
            if abs(distance_error) < self.distance_tolerance:
                linear_vel = 0.0
            cmd.linear.x = linear_vel

        self.get_logger().info(
            f"[FOLLOWING] Dist: {distance:.2f}m Err:{distance_error:.2f}, "
            f"Angle: {math.degrees(angle):.1f}° | "
            f"Cmd: lin={cmd.linear.x:.2f}, ang={cmd.angular.z:.2f}",
            throttle_duration_sec=0.5,
        )

        return cmd

    def _check_path_safety(self, angle_deg: float) -> Tuple[bool, Optional[float]]:
        """Check if path at given angle is safe and find alternatives if not.

        Parameters
        ----------
        angle_deg : float
            Target angle in degrees.

        Returns
        -------
        Tuple[bool, Optional[float]]
            (is_safe, alternative_angle_in_degrees)
        """
        if self.last_paths_time is not None:
            if time.time() - self.last_paths_time > self.paths_timeout:
                self.get_logger().warn(
                    "Path data stale, assuming unsafe", throttle_duration_sec=2.0
                )
                return False, None

        target_path_idx = self._angle_to_path_index(angle_deg)

        if target_path_idx in self.safe_paths:
            return True, None

        alternative_idx = self._find_nearest_safe_path(target_path_idx)

        if alternative_idx is not None:
            alternative_angle = self.PATH_ANGLES[alternative_idx]
            return False, alternative_angle

        return False, None

    def _angle_to_path_index(self, angle_deg: float) -> int:
        """Convert angle in degrees to nearest path index.

        Parameters
        ----------
        angle_deg : float
            Angle in degrees.

        Returns
        -------
        int
            Path index.
        """
        min_diff = float("inf")
        closest_idx = 4

        for idx in self.VALID_FORWARD_PATHS:
            diff = abs(self.PATH_ANGLES[idx] - angle_deg)
            if diff < min_diff:
                min_diff = diff
                closest_idx = idx

        return closest_idx

    def _find_nearest_safe_path(self, blocked_idx: int) -> Optional[int]:
        """Find nearest safe path to a blocked path index.

        Parameters
        ----------
        blocked_idx : int
            Index of the blocked path.

        Returns
        -------
        Optional[int]
            Index of nearest safe path, or None if no safe path found.
        """
        for offset in range(1, len(self.VALID_FORWARD_PATHS)):
            left_idx = blocked_idx - offset
            if left_idx in self.VALID_FORWARD_PATHS and left_idx in self.safe_paths:
                return left_idx

            right_idx = blocked_idx + offset
            if right_idx in self.VALID_FORWARD_PATHS and right_idx in self.safe_paths:
                return right_idx

        return None

    def _stop_robot(self):
        """Publish zero velocity to stop the robot."""
        self.cmd_vel_publisher.publish(Twist())

    def _publish_approached(self):
        """Publish APPROACHED status to OM1 via Zenoh."""
        if not self.zenoh_session:
            self.get_logger().error(
                "Zenoh session not available for publishing APPROACHED."
            )
            return
        request_id = str(uuid4())
        msg = PersonGreetingStatus(
            header=prepare_header(),
            request_id=ZenohString(data=request_id),
            status=HandshakeCode.APPROACHED,
        )
        self.zenoh_session.put("om/person_greeting", msg.serialize())
        time.sleep(0.1)
        self.get_logger().info("[Zenoh] Published APPROACHED (1) to om/person_greeting")

    def _call_switch_command(self) -> bool:
        """Call switch command on tracking system via HTTP."""
        self.switch_command_pending = True
        return self._send_http_command("switch")

    def _call_greeting_ack_command(self) -> bool:
        """Call greeting_ack command on tracking system via HTTP.
        This saves the current target to history — only call after greeting is done.
        """
        return self._send_http_command("greeting_ack")

    def _call_clear_command(self) -> bool:
        """Call clear command on tracking system via HTTP.
        Clears target WITHOUT saving to history.
        """
        return self._send_http_command("clear")

    def _send_http_command(self, command: str) -> bool:
        """Send HTTP command to tracking system."""
        url = f"{self.cmd_base_url}/{command}"
        try:
            response = requests.post(url, timeout=2.0)
            if response.status_code == 200:
                self.get_logger().info(f"HTTP command '{command}' succeeded")
                return True
            else:
                self.get_logger().error(
                    f"HTTP command '{command}' failed: {response.status_code}"
                )
                return False
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"HTTP command '{command}' error: {e}")
            return False


def main(args=None):
    """Entry point for the person follower node."""
    rclpy.logging._root_logger.log(
        "Starting person follower node with OM1 integration and geofencing...",
        LoggingSeverity.INFO,
    )

    rclpy.init(args=args)
    person_follower = PersonFollower()

    try:
        rclpy.spin(person_follower)
    except KeyboardInterrupt:
        pass
    finally:
        person_follower.cmd_vel_publisher.publish(Twist())
        person_follower.get_logger().info("Shutting down, robot stopped")

        person_follower._stop_http_server()

        if person_follower.zenoh_session:
            person_follower.zenoh_session.close()
            person_follower.get_logger().info("Zenoh session closed")

        person_follower.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
