#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import threading
import time
from enum import Enum
from typing import Optional

import rclpy
import requests
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from std_msgs.msg import String
from unitree_api.msg import GreetingStatus


class FollowerState(Enum):
    """State machine states for person follower."""

    IDLE = "IDLE"
    SWITCHING = "SWITCHING"
    APPROACHING = "APPROACHING"
    GREETING_IN_PROGRESS = "GREETING_IN_PROGRESS"
    SEARCHING = "SEARCHING"


class HandshakeCode:
    """Handshake codes for /greeting_handshake topic."""

    APPROACHED = 1
    SWITCH = 2


class PersonFollower(Node):
    """
    A ROS2 node that follows a tracked person and integrates with OM1 greeting system.
    Uses PD control for smooth following and implements search behavior when no person is found.
    """

    def __init__(self):
        super().__init__("person_follower")

        self._declare_parameters()
        self._load_parameters()
        self._setup_publishers()
        self._setup_subscribers()
        self._init_state()

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
        self.declare_parameter("cmd_port", 8080)
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
        self.search_rotation_angle = self.get_parameter("search_rotation_angle").value
        self.search_rotation_speed = self.get_parameter("search_rotation_speed").value
        self.search_wait_time = self.get_parameter("search_wait_time").value

        self.cmd_base_url = f"http://{self.cmd_host}:{self.cmd_port}"

    def _setup_publishers(self):
        """Create ROS publishers."""
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.handshake_publisher = self.create_publisher(
            GreetingStatus, "/greeting_handshake", 10
        )
        self.state_publisher = self.create_publisher(String, "/person_follower/state", 10)

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
        self.handshake_subscription = self.create_subscription(
            GreetingStatus, "/greeting_handshake", self.handshake_callback, 10
        )

    def _init_state(self):
        """Initialize state machine and control variables."""
        self.state = FollowerState.IDLE
        self.state_lock = threading.Lock()
        self.state_entry_time = time.time()

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

    def position_callback(self, msg: PoseStamped):
        """
        Handle incoming tracked person position updates.
        Only processes position when in APPROACHING state.

        Parameters
        ----------
        msg : geometry_msgs.msg.PoseStamped
            The tracked person's position in robot-centric coordinates.
        """
        if msg.pose.position.z == 0.0:
            return

        self.last_position = msg

        with self.state_lock:
            if self.state != FollowerState.APPROACHING:
                return

        current_time = self.get_clock().now()
        dt = 0.0
        if self.last_msg_time is not None:
            dt = (current_time - self.last_msg_time).nanoseconds / 1e9
        self.last_msg_time = current_time

        cmd_vel = self.calculate_velocity(msg, dt)
        self.cmd_vel_publisher.publish(cmd_vel)

    def handshake_callback(self, msg: GreetingStatus):
        """
        Handle incoming handshake messages from OM1.

        Parameters
        ----------
        msg : unitree_api.msg.GreetingStatus
            Greeting status message (code=2 means conversation finished).
        """
        if msg.code == HandshakeCode.SWITCH:
            self.get_logger().info("Received SWITCH (2) from OM1 - conversation finished")
            with self.state_lock:
                if self.state == FollowerState.GREETING_IN_PROGRESS:
                    self._transition_to(FollowerState.SWITCHING)
                    self._call_switch_command()

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

        self.get_logger().info(f"State transition: {old_state.value} → {new_state.value}")

    def _handle_idle_state(self):
        """Handle IDLE state - wait for tracking system to be ready."""
        if self.tracking_status_received:
            self.get_logger().info("Tracking system ready, calling switch...")
            self._transition_to(FollowerState.SWITCHING)
            self._call_switch_command()

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
            self.get_logger().info("Tracking lost person, waiting for re-identification...")
        elif mode == "INACTIVE" and self.last_tracking_mode == "SEARCHING":
            self.get_logger().info("Re-identification failed, starting search")
            self._stop_robot()
            self._transition_to(FollowerState.SEARCHING)

        self.last_tracking_mode = mode

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
                self.get_logger().info("Pause complete, no one visible - rotating again...")
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
        Calculate velocity commands using PD control.

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

        angular_vel = max(-self.max_angular_speed, min(angular_vel, self.max_angular_speed))

        if abs(angle_error) > self.angle_tolerance:
            cmd.angular.z = angular_vel
            cmd.linear.x = 0.0
        else:
            cmd.angular.z = angular_vel
            linear_vel = max(-self.max_linear_speed, min(linear_vel, self.max_linear_speed))
            if abs(distance_error) < self.distance_tolerance:
                linear_vel = 0.0
            cmd.linear.x = linear_vel

        self.get_logger().info(
            f"[APPROACHING] Dist: {distance:.2f}m Err:{distance_error:.2f}, "
            f"Angle: {math.degrees(angle):.1f}° | "
            f"Cmd: lin={cmd.linear.x:.2f}, ang={cmd.angular.z:.2f}",
            throttle_duration_sec=0.5,
        )

        return cmd

    def _stop_robot(self):
        """Publish zero velocity to stop the robot."""
        self.cmd_vel_publisher.publish(Twist())

    def _publish_approached(self):
        """Publish APPROACHED handshake to OM1."""
        msg = GreetingStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "person_follower"
        msg.code = HandshakeCode.APPROACHED
        self.handshake_publisher.publish(msg)
        self.get_logger().info("Published APPROACHED (1) to OM1")

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
        person_follower.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()