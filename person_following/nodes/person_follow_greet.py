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
import threading
import time
from typing import Literal, Optional
from uuid import uuid4

import rclpy
import requests
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from om_api.msg import Paths
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
import zenoh
from std_msgs.msg import String
from unitree_api.msg import Request, RequestHeader, RequestIdentity
from unitree_go.msg import SportModeState

from person_following.managers.geofence_manager import GeofenceManager
from person_following.utils.http_server import ModeControlHandler, ModeControlHTTPServer
from person_following.controllers.motion_controller import MotionController
from person_following.utils.state_machine import (
    FollowerState,
    ReturnToCenter,
    SearchBehavior,
)
from person_following.utils.zenoh_msgs import (
    PersonGreetingStatus,
    open_zenoh_session,
    prepare_header,
)
from person_following.utils.zenoh_msgs import String as ZenohString


class HandshakeCode:
    """Status codes for /om/person_greeting topic (matching OM1 PersonGreetingStatus)."""

    APPROACHING = 0
    APPROACHED = 1
    SWITCH = 2


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
    SPORT_API_ID_CLASSICWALK = 2049

    def __init__(self):
        """Initialize the PersonFollower ROS2 node."""
        super().__init__("person_follower")

        self._declare_parameters()
        self._load_parameters()

        # Initialize component managers
        self.geofence_manager = GeofenceManager(
            node=self,
            enabled=self.geofence_enabled,
            radius=self.geofence_radius,
            soft_radius=self.geofence_soft_radius,
            return_speed=self.geofence_return_speed,
        )

        self.motion_controller = MotionController(
            node=self,
            target_distance=self.target_distance,
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            linear_kp=self.linear_kp,
            linear_kd=self.linear_kd,
            angular_kp=self.angular_kp,
            angular_kd=self.angular_kd,
            distance_tolerance=self.distance_tolerance,
            angle_tolerance=self.angle_tolerance,
        )

        self.search_behavior = SearchBehavior(
            node=self,
            rotation_angle=self.search_rotation_angle,
            rotation_speed=self.search_rotation_speed,
            wait_time=self.search_wait_time,
        )

        self.return_to_center = ReturnToCenter(
            node=self,
            return_speed=self.geofence_return_speed,
        )

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
        self.sport_publisher = self.create_publisher(Request, "/api/sport/request", 10)

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
        self.sport_mode_subscription = self.create_subscription(
            SportModeState, "/sportmodestate", self.sport_mode_callback, 10
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

        self.last_paths_time: Optional[float] = None
        self.paths_timeout = 1.0

        self.stop_sent = False
        self.is_tracking = False

        self.last_valid_position_time: Optional[float] = None
        self.boundary_stuck_start_time: Optional[float] = None

        self._start_http_server()

    def _log_startup_info(self):
        """Log startup information."""
        robot_type = os.environ.get("ROBOT_TYPE", "unknown").lower()
        self.get_logger().info("=" * 60)
        self.get_logger().info("Person Follower with OM1 Integration Started")
        self.get_logger().info(f"  Robot type: {robot_type}")
        self.get_logger().info(f"  Tracking API: {self.cmd_base_url}")
        self.get_logger().info(f"  Target distance: {self.target_distance}m")
        self.get_logger().info(f"  Geofence enabled: {self.geofence_manager.enabled}")
        if self.geofence_manager.enabled:
            self.get_logger().info(
                f"  Geofence radius: {self.geofence_manager.radius}m"
            )
            self.get_logger().info(
                f"  Geofence soft radius: {self.geofence_manager.soft_radius}m"
            )
        self.get_logger().info("=" * 60)

    def _start_http_server(self):
        """Start the HTTP server for mode control."""
        self._http_server = ModeControlHTTPServer(
            ("0.0.0.0", self.http_port),
            ModeControlHandler,
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
        """
        Get current operation mode (thread-safe).

        Returns
        -------
        PersonFollower.OperationMode
            The current operation mode ("greeting" or "following").
        """
        with self.mode_lock:
            return self.operation_mode

    def set_operation_mode(self, mode: "PersonFollower.OperationMode") -> bool:
        """
        Set operation mode and notify vision system.

        Parameters
        ----------
        mode : PersonFollower.OperationMode
            The operation mode to set ("greeting" or "following").

        Returns
        -------
        bool
            True if mode was changed, False if it was already in the desired mode.
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
        """
        Notify vision system of mode change via HTTP.

        Parameters
        ----------
        mode : PersonFollower.OperationMode
            The new operation mode ("greeting" or "following").
        """
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

    def odom_callback(self, msg: Odometry):
        """
        Handle odometry updates for geofencing.

        Parameters
        ----------
        msg : Odometry
            The incoming odometry message containing the robot's current position.
        """
        self.geofence_manager.handle_odom(msg)

    def status_callback(self, msg: String):
        """
        Handle incoming tracking status updates.

        Parameters
        ----------
        msg : String
            The incoming message containing tracking status in JSON format.
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
        msg : Paths
            The incoming message containing safe paths and blocked paths information.
        """
        safe_paths = list(msg.paths)
        blocked_paths = set(msg.blocked_by_obstacle_idx) | set(
            msg.blocked_by_hazard_idx
        )
        self.last_paths_time = time.time()

        self.motion_controller.update_path_safety(safe_paths, blocked_paths)
        self.return_to_center.update_path_safety(
            safe_paths, blocked_paths, self.last_paths_time
        )

    def sport_mode_callback(self, msg: SportModeState):
        """
        Handle incoming sport mode state messages.

        Parameters
        ----------
        msg : SportModeState
            The incoming sport mode state message containing the current mode of the robot.
        """
        if msg.error_code == 100:
            with self.state_lock:
                current_state = self.state

            if current_state not in (FollowerState.IDLE, FollowerState.GREETING_IN_PROGRESS):
                self.get_logger().warning(
                    f"Robot is in Agile mode during {current_state.value} state, "
                    "switching to classical mode for stability."
                )
                try:
                    request_msg = Request()
                    request_msg.header = RequestHeader()
                    request_msg.header.identity = RequestIdentity()
                    request_msg.header.identity.api_id = self.SPORT_API_ID_CLASSICWALK

                    request_msg.parameter = json.dumps({"data": True})
                    self.sport_publisher.publish(request_msg)
                    self.get_logger().info("Sent command to switch to Classical Walk mode")
                except Exception as e:
                    self.get_logger().error(f"Failed to switch to classical mode: {e}")

    def position_callback(self, msg: PoseStamped):
        """
        Handle incoming tracked person position updates.

        Parameters
        ----------
        msg : PoseStamped
            The incoming message containing the tracked person's position relative to the robot.
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
        """
        Handle position updates in FOLLOWING mode.

        Parameters
        ----------
        msg : PoseStamped
            The incoming message containing the tracked person's position relative to the robot.
        """
        self.stop_sent = False
        self.is_tracking = True

        current_time = self.get_clock().now()
        dt = 0.0
        if self.last_msg_time is not None:
            dt = (current_time - self.last_msg_time).nanoseconds / 1e9
        self.last_msg_time = current_time

        cmd_vel = self.motion_controller.calculate_velocity_following(msg, dt)
        self.cmd_vel_publisher.publish(cmd_vel)

    def _handle_approaching_position(self, msg: PoseStamped):
        """
        Handle position updates in APPROACHING mode.

        Parameters
        ----------
        msg : PoseStamped
            The incoming message containing the tracked person's position relative to the robot.
        """
        self.last_valid_position_time = time.time()

        current_time = self.get_clock().now()
        dt = 0.0
        if self.last_msg_time is not None:
            dt = (current_time - self.last_msg_time).nanoseconds / 1e9
        self.last_msg_time = current_time

        cmd_vel, path_safe = self.motion_controller.calculate_velocity_approaching(
            msg, dt
        )

        if not path_safe:
            self._call_switch_command()
            self._transition_to(FollowerState.SEARCHING)
            return

        cmd_vel = self.geofence_manager.apply_velocity_constraints(
            cmd_vel, self.get_operation_mode() == "greeting"
        )
        self.cmd_vel_publisher.publish(cmd_vel)

    def _zenoh_person_greeting_callback(self, data: zenoh.Sample):
        """
        Handle incoming PersonGreetingStatus messages from Zenoh.

        Parameters
        ----------
        data : ZenohString
            The incoming Zenoh message containing the person greeting status in JSON format.
        """
        try:
            msg = PersonGreetingStatus.deserialize(data.payload.to_bytes())
            if msg.status == HandshakeCode.SWITCH:
                self.get_logger().info("[Zenoh] Received SWITCH (2) from OM1")
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
        """
        Main state machine tick called periodically by timer.
        """
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
            self.search_behavior.reset()

        if new_state == FollowerState.APPROACHING:
            self.motion_controller.reset_errors()
            self.last_msg_time = None
            self.search_behavior.rotation_count = 0
            self.boundary_stuck_start_time = None
            self.last_valid_position_time = time.time()

        if new_state == FollowerState.FOLLOWING:
            self.motion_controller.reset_errors()
            self.last_msg_time = None
            self.stop_sent = False
            self.is_tracking = False

        if new_state == FollowerState.RETURNING_TO_CENTER:
            self.search_behavior.rotation_count = 0
            target = self.geofence_manager.generate_random_return_target()
            self.return_to_center.reset(target)

        self.get_logger().info(
            f"State transition: {old_state.value} → {new_state.value}"
        )

        if new_state in (FollowerState.IDLE, FollowerState.GREETING_IN_PROGRESS):
            self._disable_classical_mode()
        else:
            self._enable_classical_mode()

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

            if self.geofence_manager.is_person_inside_geofence(person_x, person_z):
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
        if not self.geofence_manager.enabled or self.geofence_manager.center is None:
            return False

        distance_from_center = self.geofence_manager.get_distance_from_center()

        at_boundary = distance_from_center >= (self.geofence_manager.radius - 0.5)

        if not at_boundary:
            self.boundary_stuck_start_time = None
            return False

        person_x = self.tracking_status.get("x", 0.0)
        person_z = self.tracking_status.get("z", 0.0)
        person_global = self.geofence_manager.get_person_global_position(
            person_x, person_z
        )

        dx = person_global[0] - self.geofence_manager.center[0]
        dy = person_global[1] - self.geofence_manager.center[1]
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

    def _handle_returning_to_center_state(self):
        """
        Handle RETURNING_TO_CENTER state - move robot to safe point inside geofence.

        Includes obstacle avoidance using /om/paths:
        - If path clear: move toward target
        - If blocked < 5s: wait
        - If blocked 5-30s: try alternative angles
        - If blocked > 30s: give up, do rotation search from current position
        """
        if self.geofence_manager.center is None:
            self.get_logger().warn("No geofence center set, transitioning to SEARCHING")
            self._transition_to(FollowerState.SEARCHING)
            return

        target = (
            self.return_to_center.return_target
            if self.return_to_center.return_target
            else self.geofence_manager.center
        )

        dx = target[0] - self.geofence_manager.current_position[0]
        dy = target[1] - self.geofence_manager.current_position[1]
        distance_to_target = math.sqrt(dx * dx + dy * dy)
        angle_to_target = math.atan2(dy, dx)

        arrival_threshold = 1.0
        if distance_to_target < arrival_threshold:
            self.get_logger().info(
                f"[GEOFENCE] Arrived at return target ({distance_to_target:.1f}m away), starting search"
            )
            self._stop_robot()
            self.return_to_center.reset(None)
            self._transition_to(FollowerState.SEARCHING)
            return

        path_clear, alternative_angle = self.return_to_center.check_path_safety(
            angle_to_target, self.geofence_manager.current_yaw
        )

        current_time = time.time()

        if path_clear:
            self.return_to_center.obstruction_start_time = None
            self.return_to_center.move_toward_target(
                angle_to_target,
                self.geofence_manager.current_yaw,
                distance_to_target,
                self.cmd_vel_publisher,
            )

        else:
            if self.return_to_center.obstruction_start_time is None:
                self.return_to_center.obstruction_start_time = current_time
                self.get_logger().info("[RETURNING] Path blocked, waiting...")
                self._stop_robot()
                return

            obstruction_duration = (
                current_time - self.return_to_center.obstruction_start_time
            )

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
                    self.return_to_center.move_toward_target(
                        alternative_angle,
                        self.geofence_manager.current_yaw,
                        distance_to_target,
                        self.cmd_vel_publisher,
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
                self.return_to_center.reset(None)
                self._transition_to(FollowerState.SEARCHING)

    def _handle_searching_state(self):
        """Handle SEARCHING state - rotate, pause, and search for people."""
        if (
            self.geofence_manager.enabled
            and self.search_behavior.rotation_count
            >= self.geofence_max_search_rotations
        ):
            distance_from_center = self.geofence_manager.get_distance_from_center()
            if distance_from_center > self.geofence_manager.soft_radius * 0.5:
                self.get_logger().info(
                    f"[GEOFENCE] {self.search_behavior.rotation_count} rotations without valid target, "
                    f"returning to center (current distance: {distance_from_center:.1f}m)"
                )
                self._stop_robot()
                self._transition_to(FollowerState.RETURNING_TO_CENTER)
                return
            else:
                self.search_behavior.rotation_count = 0

        if self.search_behavior.phase == "rotate":
            self._do_search_rotation()
        elif self.search_behavior.phase == "pause":
            self._do_search_pause()
        elif self.search_behavior.phase == "wait":
            self._do_search_wait()

    def _do_search_rotation(self):
        """Execute search rotation phase."""
        self.search_behavior.do_rotation_phase(self.cmd_vel_publisher)

    def _do_search_pause(self):
        """Wait after rotation to let vision stabilize before calling switch."""
        if self.search_behavior.do_pause_phase():
            num_persons = 0
            if self.tracking_status is not None:
                num_persons = self.tracking_status.get("num_persons", 0)

            if num_persons > 0:
                self.search_behavior.phase = "wait"
                self.get_logger().info(
                    f"Pause complete, {num_persons} person(s) visible - calling switch..."
                )
                self._call_switch_command()
            else:
                self.get_logger().info(
                    "Pause complete, no one visible - rotating again..."
                )
                self.search_behavior.phase = "rotate"
                self.search_behavior.rotation_start_time = None

    def _do_search_wait(self):
        """Wait for switch result after rotation."""
        if self.tracking_status is None:
            return

        mode = self.tracking_status.get("mode", "UNKNOWN")

        if mode == "TRACKING_ACTIVE":
            person_x = self.tracking_status.get("x", 0.0)
            person_z = self.tracking_status.get("z", 0.0)

            if self.geofence_manager.is_person_inside_geofence(person_x, person_z):
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
                self.search_behavior.phase = "rotate"
                self.search_behavior.rotation_start_time = None

        elif mode == "INACTIVE" and self.switch_command_pending:
            self.get_logger().info("No one found, rotating again...")
            self.switch_command_pending = False
            self.search_behavior.phase = "rotate"
            self.search_behavior.rotation_start_time = None

        self.last_tracking_mode = mode

    def _stop_robot(self):
        """Publish zero velocity to stop the robot."""
        self.cmd_vel_publisher.publish(Twist())

    def _enable_classical_mode(self):
        """Enable classical walk mode for stable movement with payload."""
        try:
            request_msg = Request()
            request_msg.header = RequestHeader()
            request_msg.header.identity = RequestIdentity()
            request_msg.header.identity.api_id = self.SPORT_API_ID_CLASSICWALK

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
            request_msg.header.identity.api_id = self.SPORT_API_ID_CLASSICWALK

            request_msg.parameter = json.dumps({"data": False})
            self.sport_publisher.publish(request_msg)
            self.get_logger().debug("Disabled classical walk mode")
        except Exception as e:
            self.get_logger().error(f"Failed to disable classical mode: {e}")

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
