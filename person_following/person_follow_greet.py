#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import threading
import time
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal, Optional
from uuid import uuid4

import rclpy
import requests
from geometry_msgs.msg import PoseStamped, Twist
from om_api.msg import Paths
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from std_msgs.msg import String
from unitree_api.msg import Request, RequestHeader, RequestIdentity

from zenoh_msgs import PersonGreetingStatus, open_zenoh_session, prepare_header
from zenoh_msgs import String as ZenohString


class FollowerState(Enum):
    """State machine states for person follower."""

    IDLE = "IDLE"
    SWITCHING = "SWITCHING"
    APPROACHING = "APPROACHING"
    GREETING_IN_PROGRESS = "GREETING_IN_PROGRESS"
    SEARCHING = "SEARCHING"
    FOLLOWING = "FOLLOWING"


class HandshakeCode:
    """Status codes for /om/person_greeting topic (matching OM1 PersonGreetingStatus)."""

    APPROACHING = 0
    APPROACHED = 1
    SWITCH = 2


class _ModeControlHTTPServer(ThreadingHTTPServer):
    """HTTP server for mode control with reference to PersonFollower node."""

    def __init__(self, addr, handler_cls, node: "PersonFollower"):
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
    Uses PD control for smooth following and implements search behavior when no person is found.
    """

    OperationMode = Literal["greeting", "following"]

    PATH_ANGLES = [60, 45, 30, 15, 0, -15, -30, -45, -60, 180]
    VALID_FORWARD_PATHS = [4]

    def __init__(self):
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

        self.state_machine_timer = self.create_timer(0.1, self.state_machine_tick)

        self.get_logger().info("Person Follower with OM1 Integration Started")
        self.get_logger().info(f"Tracking API: {self.cmd_base_url}")
        self.get_logger().info(f"Target distance: {self.target_distance}m")

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

        self.cmd_base_url = f"http://{self.cmd_host}:{self.cmd_port}"

    def _setup_publishers(self):
        """Create ROS publishers."""
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.state_publisher = self.create_publisher(
            String, "/person_follower/state", 10
        )
        self.sport_publisher = self.create_publisher(
            Request, "/api/sport/request", 10
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

        self.safe_paths: list[int] = list(range(10))
        self.blocked_paths: set[int] = set()
        self.last_paths_time: Optional[float] = None
        self.paths_timeout = 1.0

        self.stop_sent = False
        self.is_tracking = False

        self._start_http_server()

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

    def get_operation_mode(self) -> "PersonFollower.OperationMode":
        """Get current operation mode (thread-safe)."""
        with self.mode_lock:
            return self.operation_mode

    def set_operation_mode(self, mode: "PersonFollower.OperationMode") -> bool:
        """
        Set operation mode and notify vision system.

        Parameters
        ----------
        mode : PersonFollower.OperationMode
            New operation mode ('greeting' or 'following').

        Returns
        -------
        bool
            True if mode was changed successfully.
        """
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

    def status_callback(self, msg: String):
        """
        Handle incoming tracking status updates.

        Parameters
        ----------
        msg : std_msgs.msg.String
            JSON string containing tracking status.
        """
        try:
            self.tracking_status = json.loads(msg.data)
            self.tracking_status_received = True
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse tracking status: {e}")

    def paths_callback(self, msg: Paths):
        """
        Handle incoming path availability updates from obstacle detection.

        Parameters
        ----------
        msg : om_api.msg.Paths
            Message containing list of safe path indices.
        """
        self.safe_paths = list(msg.paths)
        self.blocked_paths = set(msg.blocked_by_obstacle_idx) | set(
            msg.blocked_by_hazard_idx
        )
        self.last_paths_time = time.time()
        self.get_logger().debug(
            f"Paths update: safe={self.safe_paths}, blocked={self.blocked_paths}",
            throttle_duration_sec=1.0,
        )

    def position_callback(self, msg: PoseStamped):
        """
        Handle incoming tracked person position updates.
        Processes position in APPROACHING state (greeting mode) or FOLLOWING state (following mode).

        Parameters
        ----------
        msg : geometry_msgs.msg.PoseStamped
            The tracked person's position in robot-centric coordinates.
        """
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
        """Handle position updates in FOLLOWING mode (pure PD control)."""
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
        """Handle position updates in APPROACHING mode (greeting state machine)."""
        current_time = self.get_clock().now()
        dt = 0.0
        if self.last_msg_time is not None:
            dt = (current_time - self.last_msg_time).nanoseconds / 1e9
        self.last_msg_time = current_time

        cmd_vel = self.calculate_velocity(msg, dt)
        self.cmd_vel_publisher.publish(cmd_vel)

    def _zenoh_person_greeting_callback(self, data):
        """Handle incoming PersonGreetingStatus messages from Zenoh (om/person_greeting)."""
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

    def _transition_to(self, new_state: FollowerState):
        """
        Transition to a new state.

        Parameters
        ----------
        new_state : FollowerState
            The state to transition to.
        """
        old_state = self.state
        self.state = new_state
        self.state_entry_time = time.time()

        if new_state == FollowerState.SEARCHING:
            self.search_phase = "rotate"
            self.search_rotation_start_time = None
            self.search_pause_start_time = None
            self.search_direction *= -1

        if new_state == FollowerState.APPROACHING:
            self.last_distance_error = 0.0
            self.last_angle_error = 0.0
            self.last_msg_time = None

        if new_state == FollowerState.FOLLOWING:
            self.last_distance_error = 0.0
            self.last_angle_error = 0.0
            self.last_msg_time = None
            self.stop_sent = False
            self.is_tracking = False

        self.get_logger().info(
            f"State transition: {old_state.value} → {new_state.value}"
        )

        # Enable classical walk mode for movement states, disable for stationary states
        if new_state in (FollowerState.IDLE, FollowerState.GREETING_IN_PROGRESS):
            self._disable_classical_mode()
        else:
            self._enable_classical_mode()

    def _handle_idle_state(self):
        """Handle IDLE state - wait for tracking system to be ready."""
        pass

    def _handle_switching_state(self):
        """Handle SWITCHING state - wait for tracking result."""
        self._stop_robot()

        if self.tracking_status is None:
            return

        mode = self.tracking_status.get("mode", "UNKNOWN")

        if mode == "TRACKING_ACTIVE":
            self.get_logger().info("Tracking found a person, starting approach")
            self.switch_command_pending = False
            self._transition_to(FollowerState.APPROACHING)
        elif mode == "INACTIVE" and self.switch_command_pending:
            self.get_logger().info("No valid candidates found, starting search")
            self.switch_command_pending = False
            self._transition_to(FollowerState.SEARCHING)

        self.last_tracking_mode = mode

    def _handle_approaching_state(self):
        """Handle APPROACHING state - PD control and check for approached."""
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
            self.get_logger().info("Person approached! Stopping and signaling OM1")
            self._stop_robot()
            self._call_greeting_ack_command()
            self._publish_approached()
            self._transition_to(FollowerState.GREETING_IN_PROGRESS)
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
        if self.search_phase == "rotate":
            self._do_search_rotation()
        elif self.search_phase == "pause":
            self._do_search_pause()
        elif self.search_phase == "wait":
            self._do_search_wait()

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
            self.get_logger().info(
                f"Search rotation complete, pausing for {self.search_wait_time}s..."
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
            self.get_logger().info("Found a person during search!")
            self.switch_command_pending = False
            self._transition_to(FollowerState.APPROACHING)
        elif mode == "INACTIVE" and self.switch_command_pending:
            self.get_logger().info("No one found, rotating again...")
            self.switch_command_pending = False
            self.search_phase = "rotate"
            self.search_rotation_start_time = None

        self.last_tracking_mode = mode

    def calculate_velocity(self, pose_msg: PoseStamped, dt: float) -> Twist:
        """
        Calculate velocity commands using PD control with obstacle avoidance.

        Checks if the path toward the person is safe using /om/paths data.
        If blocked, searches for an alternative safe path or stops.

        Parameters
        ----------
        pose_msg : geometry_msgs.msg.PoseStamped
            The tracked person's position.
        dt : float
            Time delta since last update in seconds.

        Returns
        -------
        geometry_msgs.msg.Twist
            Velocity command with linear.x and angular.z populated.
        """
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

        p_ang = -self.angular_kp * angle_error
        d_ang = 0.0
        if dt > 0.001:
            d_ang = -self.angular_kd * (angle_error - self.last_angle_error) / dt
        angular_vel = p_ang + d_ang
        self.last_angle_error = angle_error

        p_lin = self.linear_kp * distance_error
        d_lin = 0.0
        if dt > 0.001:
            d_lin = self.linear_kd * (distance_error - self.last_distance_error) / dt
        linear_vel = p_lin + d_lin
        self.last_distance_error = distance_error

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
        """
        Calculate velocity commands using simple PD control (following mode).

        No obstacle avoidance or state machine - pure following behavior.

        Parameters
        ----------
        pose_msg : geometry_msgs.msg.PoseStamped
            The tracked person's position.
        dt : float
            Time delta since last update in seconds.

        Returns
        -------
        geometry_msgs.msg.Twist
            Velocity command with linear.x and angular.z populated.
        """
        cmd = Twist()

        x = pose_msg.pose.position.x
        z = pose_msg.pose.position.z

        distance = math.sqrt(x**2 + z**2)
        angle = math.atan2(x, z)

        distance_error = distance - self.target_distance
        angle_error = angle

        p_ang = -self.angular_kp * angle_error
        d_ang = 0.0
        if dt > 0.001:
            d_ang = -self.angular_kd * (angle_error - self.last_angle_error) / dt
        angular_vel = p_ang + d_ang
        self.last_angle_error = angle_error

        p_lin = self.linear_kp * distance_error
        d_lin = 0.0
        if dt > 0.001:
            d_lin = self.linear_kd * (distance_error - self.last_distance_error) / dt
        linear_vel = p_lin + d_lin
        self.last_distance_error = distance_error

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

    def _check_path_safety(self, angle_deg: float) -> tuple[bool, Optional[float]]:
        """
        Check if the path at the given angle is safe and find alternatives if not.

        Parameters
        ----------
        angle_deg : float
            The angle to the target in degrees (0° = forward, positive = left).

        Returns
        -------
        tuple[bool, Optional[float]]
            (is_safe, alternative_angle)
            - is_safe: True if direct path is safe
            - alternative_angle: If blocked, the nearest safe alternative angle in degrees,
              or None if no safe path exists
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
        """
        Convert an angle in degrees to the nearest path index.

        Parameters
        ----------
        angle_deg : float
            Angle in degrees (0° = forward, positive = left, negative = right).

        Returns
        -------
        int
            Path index (3, 4, or 5 for forward directions).
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
        """
        Find the nearest safe path to a blocked path index.

        Searches outward from the blocked index within valid forward paths (3, 4, 5).

        Parameters
        ----------
        blocked_idx : int
            The index of the blocked path.

        Returns
        -------
        Optional[int]
            The index of the nearest safe path, or None if all paths are blocked.
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

    def _enable_classical_mode(self):
        """Enable classical walk mode for stable movement with payload."""
        try:
            request_msg = Request()
            request_msg.header = RequestHeader()
            request_msg.header.identity = RequestIdentity()
            request_msg.header.identity.api_id = 2049  # SPORT_API_ID_CLASSICWALK

            request_msg.parameter = json.dumps({"data": True})
            self.sport_publisher.publish(request_msg)
            self.get_logger().debug("Enabled classical walk mode")
        except Exception as e:
            self.get_logger().error(f"Failed to enable classical mode: {e}")

    def _disable_classical_mode(self):
        """Disable classical walk mode."""
        try:
            request_msg = Request()
            request_msg.header = RequestHeader()
            request_msg.header.identity = RequestIdentity()
            request_msg.header.identity.api_id = 2049  # SPORT_API_ID_CLASSICWALK

            request_msg.parameter = json.dumps({"data": False})
            self.sport_publisher.publish(request_msg)
            self.get_logger().debug("Disabled classical walk mode")
        except Exception as e:
            self.get_logger().error(f"Failed to disable classical mode: {e}")

    def _publish_approached(self):
        """Publish APPROACHED status to OM1 via Zenoh om/person_greeting."""
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
        self.get_logger().info(
            f"[Zenoh] Published APPROACHED (1) to om/person_greeting (request_id: {request_id})"
        )

    def _call_switch_command(self) -> bool:
        """
        Call switch command on tracking system via HTTP.

        Returns
        -------
        bool
            True if command succeeded, False otherwise.
        """
        self.switch_command_pending = True
        return self._send_http_command("switch")

    def _call_greeting_ack_command(self) -> bool:
        """
        Call greeting_ack command on tracking system via HTTP.

        Returns
        -------
        bool
            True if command succeeded, False otherwise.
        """
        return self._send_http_command("greeting_ack")

    def _send_http_command(self, command: str) -> bool:
        """
        Send HTTP command to tracking system.

        Parameters
        ----------
        command : str
            Command name (e.g., "switch", "greeting_ack").

        Returns
        -------
        bool
            True if command succeeded, False otherwise.
        """
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
    """
    Main function to run the PersonFollower node.

    Parameters
    ----------
    args : list, optional
        Command line arguments to pass to rclpy.init().
    """
    rclpy.logging._root_logger.log(
        "Starting person follower node with OM1 integration...",
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
