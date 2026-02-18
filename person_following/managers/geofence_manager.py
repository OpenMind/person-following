"""Geofence management for person following robot."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Optional, Tuple

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

if TYPE_CHECKING:
    pass


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
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class GeofenceManager:
    """Manages geofencing logic for person following robot."""

    def __init__(
        self,
        node: Node,
        enabled: bool = True,
        radius: float = 30.0,
        soft_radius: float = 28.0,
        return_speed: float = 0.4,
    ):
        """
        Initialize geofence manager.

        Parameters
        ----------
        node : Node
            ROS2 node for logging
        enabled : bool
            Whether geofencing is enabled
        radius : float
            Hard boundary radius in meters
        soft_radius : float
            Soft boundary radius in meters (speed reduction zone)
        return_speed : float
            Speed when returning to center
        """
        self.node = node
        self.enabled = enabled
        self.radius = radius
        self.soft_radius = soft_radius
        self.return_speed = return_speed

        self.center: Optional[Tuple[float, float]] = None
        self.current_position: Tuple[float, float] = (0.0, 0.0)
        self.current_yaw: float = 0.0
        self.odom_received = False

    def _is_valid_position(self, x: float, y: float) -> bool:
        """
        Check if position values are valid (not NaN or infinity).

        Parameters
        ----------
        x, y : float
            Position coordinates

        Returns
        -------
        bool
            True if position is valid
        """
        return (
            math.isfinite(x)
            and math.isfinite(y)
            and not math.isnan(x)
            and not math.isnan(y)
        )

    def handle_odom(self, msg: Odometry):
        """
        Handle odometry updates.

        Parameters
        ----------
        msg : Odometry
            Odometry message
        """
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y

        if not self._is_valid_position(new_x, new_y):
            self.node.get_logger().warn(
                f"[GEOFENCE] Ignoring invalid odom values: ({new_x}, {new_y})",
                throttle_duration_sec=1.0,
            )
            return

        self.current_position = (new_x, new_y)

        orientation = msg.pose.pose.orientation
        self.current_yaw = quaternion_to_yaw(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

        if not self.odom_received:
            self.odom_received = True
            if self.enabled and self.center is None:
                self.center = self.current_position
                self.node.get_logger().info(
                    f"[GEOFENCE] Center auto-set to: "
                    f"({self.center[0]:.2f}, {self.center[1]:.2f})"
                )

    def enable(self) -> Optional[Tuple[float, float]]:
        """
        Enable geofencing.

        Returns
        -------
        Optional[Tuple[float, float]]
            Current geofence center (set if not already set)
        """
        self.enabled = True
        if self.center is None and self.odom_received:
            if self._is_valid_position(*self.current_position):
                self.center = self.current_position
                self.node.get_logger().info(
                    f"[GEOFENCE] Center auto-set on enable: "
                    f"({self.center[0]:.2f}, {self.center[1]:.2f})"
                )
            else:
                self.node.get_logger().warn(
                    "[GEOFENCE] Cannot set center: current position is invalid"
                )
        return self.center

    def disable(self):
        """Disable geofencing."""
        self.enabled = False

    def reset_center(self) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """
        Reset geofence center to current robot position.

        Returns
        -------
        Tuple[bool, Optional[Tuple[float, float]]]
            (success, new_center)
        """
        if not self.odom_received:
            self.node.get_logger().warn(
                "Cannot reset geofence center: no odom received yet"
            )
            return False, None

        if not self._is_valid_position(*self.current_position):
            self.node.get_logger().warn(
                "Cannot reset geofence center: current position is invalid (NaN/inf)"
            )
            return False, None

        self.center = self.current_position
        self.node.get_logger().info(
            f"Geofence center reset to: ({self.center[0]:.2f}, {self.center[1]:.2f})"
        )
        return True, self.center

    def get_distance_from_center(self) -> float:
        """
        Calculate robot's distance from geofence center.

        Returns
        -------
        float
            Distance in meters
        """
        if self.center is None:
            return 0.0
        dx = self.current_position[0] - self.center[0]
        dy = self.current_position[1] - self.center[1]
        return math.sqrt(dx * dx + dy * dy)

    def get_person_global_position(
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

        Parameters
        ----------
        person_x : float
            Person's lateral offset from robot
        person_z : float
            Person's forward distance from robot

        Returns
        -------
        Tuple[float, float]
            Global (x, y) position
        """
        robot_frame_x = person_z
        robot_frame_y = -person_x

        cos_yaw = math.cos(self.current_yaw)
        sin_yaw = math.sin(self.current_yaw)

        global_x = self.current_position[0] + (
            robot_frame_x * cos_yaw - robot_frame_y * sin_yaw
        )
        global_y = self.current_position[1] + (
            robot_frame_x * sin_yaw + robot_frame_y * cos_yaw
        )

        return (global_x, global_y)

    def is_person_inside_geofence(self, person_x: float, person_z: float) -> bool:
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
        if not self.enabled or self.center is None:
            return True

        person_global = self.get_person_global_position(person_x, person_z)

        dx = person_global[0] - self.center[0]
        dy = person_global[1] - self.center[1]
        distance = math.sqrt(dx * dx + dy * dy)

        inside = distance <= self.radius

        if not inside:
            self.node.get_logger().info(
                f"[GEOFENCE] Person at global ({person_global[0]:.1f}, {person_global[1]:.1f}) "
                f"is outside boundary ({distance:.1f}m > {self.radius}m)",
                throttle_duration_sec=1.0,
            )

        return inside

    def apply_velocity_constraints(self, cmd: Twist, is_greeting_mode: bool) -> Twist:
        """
        Apply geofence constraints to velocity command.

        Only applies in greeting mode. Scales down speed in soft zone,
        blocks outward movement at hard boundary.

        Parameters
        ----------
        cmd : Twist
            Original velocity command
        is_greeting_mode : bool
            Whether robot is in greeting mode

        Returns
        -------
        Twist
            Constrained velocity command
        """
        if not is_greeting_mode:
            return cmd

        if not self.enabled or self.center is None:
            return cmd

        distance_from_center = self.get_distance_from_center()

        if distance_from_center < self.soft_radius:
            return cmd

        if distance_from_center > 0.01:
            dx = self.current_position[0] - self.center[0]
            dy = self.current_position[1] - self.center[1]
            to_center_x = -dx / distance_from_center
            to_center_y = -dy / distance_from_center
        else:
            return cmd

        robot_forward_x = math.cos(self.current_yaw)
        robot_forward_y = math.sin(self.current_yaw)

        dot = to_center_x * robot_forward_x + to_center_y * robot_forward_y

        if dot > 0 or cmd.linear.x <= 0:
            return cmd

        if distance_from_center >= self.radius:
            self.node.get_logger().warn(
                f"[GEOFENCE] At hard boundary ({distance_from_center:.1f}m >= {self.radius}m), "
                "blocking forward movement",
                throttle_duration_sec=1.0,
            )
            cmd.linear.x = 0.0
            return cmd

        scale = (self.radius - distance_from_center) / (self.radius - self.soft_radius)
        scale = max(0.0, min(1.0, scale))

        original_speed = cmd.linear.x
        cmd.linear.x *= scale

        self.node.get_logger().info(
            f"[GEOFENCE] In soft zone ({distance_from_center:.1f}m), "
            f"speed scaled: {original_speed:.2f} → {cmd.linear.x:.2f}",
            throttle_duration_sec=0.5,
        )

        return cmd

    def generate_random_return_target(self) -> Optional[Tuple[float, float]]:
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
        if self.center is None:
            return None

        dx = self.current_position[0] - self.center[0]
        dy = self.current_position[1] - self.center[1]
        distance_from_center = math.sqrt(dx * dx + dy * dy)

        if distance_from_center < 0.01:
            random_angle = random.uniform(0, 2 * math.pi)
            random_radius = random.uniform(1.0, 3.0)
            target_x = self.center[0] + random_radius * math.cos(random_angle)
            target_y = self.center[1] + random_radius * math.sin(random_angle)
            return (target_x, target_y)

        dir_x = dx / distance_from_center
        dir_y = dy / distance_from_center

        target_distance_from_center = self.soft_radius * random.uniform(0.70, 0.85)

        base_target_x = self.center[0] + dir_x * target_distance_from_center
        base_target_y = self.center[1] + dir_y * target_distance_from_center

        perp_x = -dir_y
        perp_y = dir_x
        random_offset = random.uniform(-2.0, 2.0)

        target_x = base_target_x + perp_x * random_offset
        target_y = base_target_y + perp_y * random_offset

        travel_distance = math.sqrt(
            (target_x - self.current_position[0]) ** 2
            + (target_y - self.current_position[1]) ** 2
        )

        self.node.get_logger().info(
            f"[GEOFENCE] Return target: ({target_x:.1f}, {target_y:.1f}), "
            f"travel distance: {travel_distance:.1f}m, "
            f"will be {target_distance_from_center:.1f}m from center"
        )

        return (target_x, target_y)

    def get_status(self) -> dict:
        """
        Get current geofence status.

        Returns
        -------
        dict
            Status dictionary with all relevant geofence information
        """
        distance_from_center = self.get_distance_from_center()

        return {
            "ok": True,
            "enabled": self.enabled,
            "center": self.center,
            "radius": self.radius,
            "soft_radius": self.soft_radius,
            "current_position": self.current_position,
            "current_yaw_deg": math.degrees(self.current_yaw),
            "distance_from_center": round(distance_from_center, 2),
            "at_soft_boundary": distance_from_center >= self.soft_radius,
            "at_hard_boundary": distance_from_center >= self.radius,
            "odom_received": self.odom_received,
        }
