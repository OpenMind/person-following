"""Motion controller with PD control for person following."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node

if TYPE_CHECKING:
    pass


class MotionController:
    """PD controller for smooth person following."""

    # Path angle definitions
    PATH_ANGLES = [60, 45, 30, 15, 0, -15, -30, -45, -60, 180]
    VALID_FORWARD_PATHS = [4]  # Index 4 = 0 degrees (forward)

    def __init__(
        self,
        node: Node,
        target_distance: float = 0.8,
        max_linear_speed: float = 0.7,
        max_angular_speed: float = 1.6,
        linear_kp: float = 0.6,
        linear_kd: float = 0.05,
        angular_kp: float = 1.4,
        angular_kd: float = 0.1,
        distance_tolerance: float = 0.2,
        angle_tolerance: float = 0.35,
    ):
        """
        Initialize motion controller.

        Parameters
        ----------
        node : Node
            ROS2 node for logging
        target_distance : float
            Desired following distance in meters
        max_linear_speed : float
            Maximum forward/backward speed
        max_angular_speed : float
            Maximum rotation speed
        linear_kp : float
            Proportional gain for distance control
        linear_kd : float
            Derivative gain for distance control
        angular_kp : float
            Proportional gain for angle control
        angular_kd : float
            Derivative gain for angle control
        distance_tolerance : float
            Distance error threshold to stop movement
        angle_tolerance : float
            Angle error threshold to stop rotation
        """
        self.node = node
        self.target_distance = target_distance
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.linear_kp = linear_kp
        self.linear_kd = linear_kd
        self.angular_kp = angular_kp
        self.angular_kd = angular_kd
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance = angle_tolerance

        self.last_distance_error = 0.0
        self.last_angle_error = 0.0

        # Path safety tracking
        self.safe_paths: list[int] = list(range(10))
        self.blocked_paths: set[int] = set()

    def update_path_safety(self, safe_paths: list[int], blocked_paths: set[int]):
        """
        Update path safety information.

        Parameters
        ----------
        safe_paths : list[int]
            List of safe path indices
        blocked_paths : set[int]
            Set of blocked path indices
        """
        self.safe_paths = safe_paths
        self.blocked_paths = blocked_paths

    def calculate_velocity_following(self, pose_msg: PoseStamped, dt: float) -> Twist:
        """
        Calculate velocity commands for following mode (no obstacle avoidance).

        Parameters
        ----------
        pose_msg : PoseStamped
            Person's position relative to robot
        dt : float
            Time delta since last update

        Returns
        -------
        Twist
            Velocity command
        """
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

        self.node.get_logger().info(
            f"[FOLLOWING] Dist: {distance:.2f}m Err:{distance_error:.2f}, "
            f"Angle: {math.degrees(angle):.1f}° | "
            f"Cmd: lin={cmd.linear.x:.2f}, ang={cmd.angular.z:.2f}",
            throttle_duration_sec=0.5,
        )

        return cmd

    def calculate_velocity_approaching(
        self, pose_msg: PoseStamped, dt: float
    ) -> Tuple[Twist, bool]:
        """
        Calculate velocity commands for approaching mode with obstacle avoidance.

        Parameters
        ----------
        pose_msg : PoseStamped
            Person's position relative to robot
        dt : float
            Time delta since last update

        Returns
        -------
        Tuple[Twist, bool]
            (velocity_command, path_is_safe)
            If path_is_safe is False and no alternative found, returns zero velocity
        """
        cmd = Twist()

        x = pose_msg.pose.position.x
        z = pose_msg.pose.position.z

        distance = math.sqrt(x**2 + z**2)
        angle = math.atan2(x, z)
        angle_deg = math.degrees(angle)

        # Skip obstacle avoidance if close to person
        close_to_person = distance < (self.target_distance + 0.7)
        if close_to_person:
            path_safe = True
            alternative_angle = None
        else:
            path_safe, alternative_angle = self._check_path_safety(angle_deg)

        distance_error = distance - self.target_distance
        angle_error = angle

        # Redirect if path is blocked
        if not path_safe:
            if alternative_angle is not None:
                alt_angle_rad = math.radians(alternative_angle)
                angle_error = alt_angle_rad
                self.node.get_logger().info(
                    f"[OBSTACLE] Path blocked at {angle_deg:.1f}°, "
                    f"redirecting to safe path at {alternative_angle:.1f}°",
                    throttle_duration_sec=1.0,
                )
            else:
                self.node.get_logger().warn(
                    f"[OBSTACLE] No safe path to person at {angle_deg:.1f}°",
                    throttle_duration_sec=1.0,
                )
                return cmd, False

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
        self.node.get_logger().info(
            f"[APPROACHING] Dist: {distance:.2f}m Err:{distance_error:.2f}, "
            f"Angle: {angle_deg:.1f}° [{safe_status}] | "
            f"Cmd: lin={cmd.linear.x:.2f}, ang={cmd.angular.z:.2f}",
            throttle_duration_sec=0.5,
        )

        return cmd, True

    def _check_path_safety(self, angle_deg: float) -> Tuple[bool, Optional[float]]:
        """
        Check if path at given angle is safe and find alternatives if not.

        Parameters
        ----------
        angle_deg : float
            Target angle in degrees.

        Returns
        -------
        Tuple[bool, Optional[float]]
            (is_safe, alternative_angle_in_degrees)
        """
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
        Convert angle in degrees to nearest path index.

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
        """
        Find nearest safe path to a blocked path index.

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

    def reset_errors(self):
        """Reset PD controller error tracking."""
        self.last_distance_error = 0.0
        self.last_angle_error = 0.0
