#!/usr/bin/env python3
"""ROS2 node for following a tracked person using PD control.

This module implements a person-following robot controller that subscribes
to tracked person positions and publishes velocity commands to follow them
while maintaining a target distance.
"""

import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.logging import LoggingSeverity
from rclpy.node import Node


class PersonFollower(Node):
    """
    ROS2 node that follows a tracked person using PD control.

    This class implements a person-following controller that:
        * Subscribes to tracked person position updates
        * Computes velocity commands using proportional-derivative (PD) control
        * Uses a "rotate first, then move" behavior for smooth following
        * Includes safety timeout to stop the robot when tracking is lost

    Parameters
    ----------
    target_distance : float, optional
        Desired distance to maintain from the person in meters. Default is 1.0.
    max_linear_speed : float, optional
        Maximum linear velocity in m/s. Default is 0.8.
    max_angular_speed : float, optional
        Maximum angular velocity in rad/s. Default is 1.8.
    linear_kp : float, optional
        Proportional gain for linear control. Default is 0.8.
    linear_kd : float, optional
        Derivative gain for linear control. Default is 0.05.
    angular_kp : float, optional
        Proportional gain for angular control. Default is 1.5.
    angular_kd : float, optional
        Derivative gain for angular control. Default is 0.1.
    distance_tolerance : float, optional
        Distance error threshold for stopping in meters. Default is 0.2.
    angle_tolerance : float, optional
        Angle error threshold for rotation priority in radians. Default is 0.35.
    timeout : float, optional
        Time without updates before stopping the robot in seconds. Default is 2.0.
    """

    def __init__(self):
        """
        Initialize the PersonFollower node.

        Sets up ROS2 subscriptions, publishers, parameters, and timers.
        Configures PD control parameters with default values that can be
        overridden via ROS2 parameter system.
        """
        super().__init__("person_follower")

        # Subscribe to tracked person position
        self.subscription = self.create_subscription(
            PoseStamped,
            "/person_following_robot/tracked_person/position",
            self.position_callback,
            10,
        )

        # Publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Control parameters
        self.declare_parameter("target_distance", 1.0)
        self.declare_parameter("max_linear_speed", 0.7)
        self.declare_parameter("max_angular_speed", 1.6)
        self.declare_parameter("linear_kp", 0.6)
        self.declare_parameter("linear_kd", 0.05)
        self.declare_parameter("angular_kp", 1.4)
        self.declare_parameter("angular_kd", 0.1)
        self.declare_parameter("distance_tolerance", 0.2)
        self.declare_parameter("angle_tolerance", 0.35)
        self.declare_parameter("timeout", 2.0)

        # Get parameters
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

        # State
        self.last_position = None
        self.last_msg_time = None
        self.last_distance_error = 0.0
        self.last_angle_error = 0.0
        self.stop_sent = False
        self.is_tracking = False

        # Safety timer
        self.safety_timer = self.create_timer(0.1, self.safety_check)

        self.get_logger().info("Person follower started with PD Control")

    def position_callback(self, msg):
        """
        Handle incoming tracked person position updates.

        Processes the position message, calculates the time delta since
        the last update, computes velocity commands, and publishes them.

        Parameters
        ----------
        msg : PoseStamped
            The tracked person's position in robot-centric coordinates
            where x is lateral offset and z is forward distance.
        """
        if msg.pose.position.z == 0.0:
            return

        self.last_position = msg
        self.stop_sent = False
        self.is_tracking = True

        current_time = self.get_clock().now()
        dt = 0.0
        if self.last_msg_time is not None:
            dt = (current_time - self.last_msg_time).nanoseconds / 1e9

        self.last_msg_time = current_time

        cmd_vel = self.calculate_velocity(msg, dt)
        self.cmd_vel_publisher.publish(cmd_vel)

    def calculate_velocity(self, pose_msg, dt):
        """
        Calculate velocity commands using PD control.

        Computes linear and angular velocities based on the distance and
        angle to the tracked person. Uses a "rotate first, then move"
        behavior to prioritize alignment before forward motion.

        Parameters
        ----------
        pose_msg : PoseStamped
            The tracked person's position.
        dt : float
            Time delta since last update in seconds.

        Returns
        -------
        Twist
            Velocity command with linear.x and angular.z populated.
        """
        cmd = Twist()

        x = pose_msg.pose.position.x
        z = pose_msg.pose.position.z

        distance = math.sqrt(x**2 + z**2)
        angle = math.atan2(x, z)

        distance_error = distance - self.target_distance
        angle_error = angle

        # Angular Control (PD)
        P_ang = -self.angular_kp * angle_error
        D_ang = 0.0
        if dt > 0.001:
            D_ang = -self.angular_kd * (angle_error - self.last_angle_error) / dt
        angular_vel = P_ang + D_ang
        self.last_angle_error = angle_error

        # Linear Control (PD)
        P_lin = self.linear_kp * distance_error
        D_lin = 0.0
        if dt > 0.001:
            D_lin = self.linear_kd * (distance_error - self.last_distance_error) / dt
        linear_vel = P_lin + D_lin
        self.last_distance_error = distance_error

        # Clamp angular
        angular_vel = max(
            -self.max_angular_speed, min(angular_vel, self.max_angular_speed)
        )

        # Behavior: rotate first, then move
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

        # Log
        turn_dir = "Straight"
        if cmd.angular.z > 0.1:
            turn_dir = "Left"
        elif cmd.angular.z < -0.1:
            turn_dir = "Right"

        move_dir = "Stop"
        if cmd.linear.x > 0.05:
            move_dir = "Forward"
        elif cmd.linear.x < -0.05:
            move_dir = "Backward"

        self.get_logger().info(
            f"Dist: {distance:.2f}m Err:{distance_error:.2f} ({move_dir}), "
            f"Angle: {math.degrees(angle):.1f}° ({turn_dir}) | "
            f"Cmd: lin={cmd.linear.x:.2f} (P:{P_lin:.2f} D:{D_lin:.2f}), "
            f"ang={cmd.angular.z:.2f} (P:{P_ang:.2f} D:{D_ang:.2f})",
            throttle_duration_sec=0.5,
        )

        return cmd

    def safety_check(self):
        """
        Perform periodic safety check and stop robot if tracking is lost.

        Called by a timer to ensure the robot stops if no person position
        updates have been received within the configured timeout period.
        Sends stop command only once, then yields control to allow external
        velocity commands.
        """
        if self.last_msg_time is None:
            return

        time_since_update = (
            self.get_clock().now() - self.last_msg_time
        ).nanoseconds / 1e9

        if time_since_update > self.timeout:
            if not self.stop_sent:
                stop_cmd = Twist()
                self.cmd_vel_publisher.publish(stop_cmd)
                self.stop_sent = True
                self.is_tracking = False
                self.get_logger().warn(
                    f"No person detected for {time_since_update:.1f}s - stopping robot"
                )


def main(args=None):
    """
    Entry point for the person follower node.

    Initializes ROS2, creates the PersonFollower node, and spins until
    shutdown. Ensures the robot is stopped on exit.

    Parameters
    ----------
    args : list, optional
        Command-line arguments passed to rclpy.init(). Default is None.
    """
    rclpy.logging._root_logger.log(
        "Starting person follower node...", LoggingSeverity.INFO
    )

    rclpy.init(args=args)
    person_follower = PersonFollower()

    try:
        rclpy.spin(person_follower)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        person_follower.cmd_vel_publisher.publish(stop_cmd)
        person_follower.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
