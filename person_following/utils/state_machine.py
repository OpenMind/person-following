"""State machine handlers and search behavior for person following."""

from __future__ import annotations

import math
import time
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

from geometry_msgs.msg import Twist
from rclpy.node import Node

if TYPE_CHECKING:
    pass


class FollowerState(Enum):
    """State machine states for person follower."""

    IDLE = "IDLE"
    SWITCHING = "SWITCHING"
    APPROACHING = "APPROACHING"
    GREETING_IN_PROGRESS = "GREETING_IN_PROGRESS"
    SEARCHING = "SEARCHING"
    FOLLOWING = "FOLLOWING"
    RETURNING_TO_CENTER = "RETURNING_TO_CENTER"


class SearchBehavior:
    """Manages search rotation behavior."""

    def __init__(
        self,
        node: Node,
        rotation_angle: float = 15.0,
        rotation_speed: float = 0.3,
        wait_time: float = 1.0,
    ):
        """
        Initialize search behavior.

        Parameters
        ----------
        node : Node
            ROS2 node for logging
        rotation_angle : float
            Degrees to rotate during search
        rotation_speed : float
            Angular velocity during rotation
        wait_time : float
            Seconds to wait after each rotation
        """
        self.node = node
        self.rotation_angle = rotation_angle
        self.rotation_speed = rotation_speed
        self.wait_time = wait_time

        self.direction = 1
        self.rotation_start_time: Optional[float] = None
        self.rotation_duration = 0.0
        self.phase = "rotate"
        self.pause_start_time: Optional[float] = None
        self.rotation_count = 0

    def reset(self):
        """Reset search state."""
        self.phase = "rotate"
        self.rotation_start_time = None
        self.pause_start_time = None
        self.direction *= 1

    def do_rotation_phase(self, cmd_vel_publisher) -> bool:
        """
        Execute rotation phase.

        Parameters
        ----------
        cmd_vel_publisher
            ROS publisher for velocity commands

        Returns
        -------
        bool
            True if rotation is complete
        """
        if self.rotation_start_time is None:
            self.rotation_start_time = time.time()
            angle_rad = math.radians(self.rotation_angle)
            self.rotation_duration = angle_rad / self.rotation_speed
            direction = "left" if self.direction > 0 else "right"
            self.node.get_logger().info(
                f"Starting search rotation: {self.rotation_angle}° ({direction})"
            )

        elapsed = time.time() - self.rotation_start_time

        if elapsed < self.rotation_duration:
            cmd = Twist()
            cmd.angular.z = self.rotation_speed * self.direction
            cmd_vel_publisher.publish(cmd)
            return False
        else:
            # Rotation complete
            cmd_vel_publisher.publish(Twist())
            self.phase = "pause"
            self.pause_start_time = time.time()
            self.rotation_count += 1
            self.node.get_logger().info(
                f"Search rotation complete, pausing for {self.wait_time}s... "
                f"(rotation #{self.rotation_count})"
            )
            return True

    def do_pause_phase(self) -> bool:
        """
        Wait after rotation.

        Returns
        -------
        bool
            True if pause is complete
        """
        if self.pause_start_time is None:
            self.pause_start_time = time.time()

        elapsed = time.time() - self.pause_start_time
        return elapsed >= self.wait_time

    def increment_count(self):
        """Increment rotation count without doing rotation."""
        self.rotation_count += 1


class ReturnToCenter:
    """Manages return-to-center navigation with obstacle avoidance."""

    PATH_ANGLES = [60, 45, 30, 15, 0, -15, -30, -45, -60, 180]

    def __init__(self, node: Node, return_speed: float = 0.4):
        """
        Initialize return-to-center behavior.

        Parameters
        ----------
        node : Node
            ROS2 node for logging
        return_speed : float
            Speed when returning to center
        """
        self.node = node
        self.return_speed = return_speed

        self.return_target: Optional[Tuple[float, float]] = None
        self.obstruction_start_time: Optional[float] = None

        # Path safety tracking
        self.safe_paths: list[int] = list(range(10))
        self.blocked_paths: set[int] = set()
        self.last_paths_time: Optional[float] = None
        self.paths_timeout = 1.0

    def update_path_safety(
        self, safe_paths: list[int], blocked_paths: set[int], paths_time: float
    ):
        """
        Update path safety information.

        Parameters
        ----------
        safe_paths : list[int]
            List of safe path indices
        blocked_paths : set[int]
            Set of blocked path indices
        paths_time : float
            Timestamp of last path update
        """
        self.safe_paths = safe_paths
        self.blocked_paths = blocked_paths
        self.last_paths_time = paths_time

    def reset(self, target: Optional[Tuple[float, float]]):
        """
        Reset return state with new target.

        Parameters
        ----------
        target : Optional[Tuple[float, float]]
            Target position (x, y)
        """
        self.return_target = target
        self.obstruction_start_time = None

    def check_path_safety(
        self, target_angle: float, current_yaw: float
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if the path toward return target is clear.

        Parameters
        ----------
        target_angle : float
            Global angle to target in radians
        current_yaw : float
            Current robot yaw in radians

        Returns
        -------
        Tuple[bool, Optional[float]]
            (is_direct_path_clear, alternative_angle_if_blocked)
            alternative_angle is in global frame (radians)
        """
        if self.last_paths_time is None or (
            time.time() - self.last_paths_time > self.paths_timeout
        ):
            self.node.get_logger().warn(
                "[RETURNING] Path data stale or unavailable, assuming blocked",
                throttle_duration_sec=2.0,
            )
            return False, None

        relative_angle = target_angle - current_yaw
        relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
        relative_angle_deg = math.degrees(relative_angle)

        target_path_idx = self._angle_to_path_index(relative_angle_deg)

        if (
            target_path_idx in self.safe_paths
            and target_path_idx not in self.blocked_paths
        ):
            return True, None

        alternative_offsets = [15, -15, 30, -30, 45, -45]

        for offset_deg in alternative_offsets:
            alt_relative_deg = relative_angle_deg + offset_deg
            alt_path_idx = self._angle_to_path_index(alt_relative_deg)

            if (
                alt_path_idx in self.safe_paths
                and alt_path_idx not in self.blocked_paths
            ):
                alt_global_angle = current_yaw + math.radians(alt_relative_deg)
                return False, alt_global_angle

        return False, None

    def _angle_to_path_index(self, angle_deg: float) -> int:
        """
        Convert robot-relative angle to path index.

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

    def move_toward_target(
        self,
        movement_angle: float,
        current_yaw: float,
        distance: float,
        cmd_vel_publisher,
    ):
        """
        Move robot toward return target at specified angle.

        Parameters
        ----------
        movement_angle : float
            Global angle to move toward (radians)
        current_yaw : float
            Current robot yaw (radians)
        distance : float
            Distance to target (for speed scaling)
        cmd_vel_publisher
            ROS publisher for velocity commands
        """
        angle_error = movement_angle - current_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        cmd = Twist()

        if abs(angle_error) > 0.3:
            cmd.angular.z = 0.5 if angle_error > 0 else -0.5
            cmd.linear.x = 0.0
        else:
            cmd.angular.z = 0.5 * angle_error
            cmd.linear.x = min(self.return_speed, distance * 0.3)

        cmd_vel_publisher.publish(cmd)

        self.node.get_logger().info(
            f"[RETURNING] Moving: angle_err={math.degrees(angle_error):.1f}°, "
            f"dist={distance:.1f}m, cmd=({cmd.linear.x:.2f}, {cmd.angular.z:.2f})",
            throttle_duration_sec=1.0,
        )
