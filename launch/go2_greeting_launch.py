#!/usr/bin/env python3
"""
ROS2 Launch file for Go2 Greeting Mode.

Launches:
    1. tracked_person_publisher_ros.py - YOLO-based person tracking (greeting mode)
    2. person_follow_greet_geofence.py - Person follower with OM1 greeting integration and geofencing

Requires colcon build to be run first for om_api and unitree_api packages.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    LogInfo,
    TimerAction,
)


def generate_launch_description():
    """Generate the launch description for Go2 greeting system."""
    project_root = os.environ.get("PROJECT_ROOT", "/opt/person_following")
    install_dir = os.path.join(project_root, "install")

    # Tracked person publisher with greeting mode (port 2001)
    def build_tracker_cmd():
        return [
            "bash",
            "-c",
            f"source /opt/ros/jazzy/setup.bash && "
            f"source {install_dir}/setup.bash && "
            f"python3 {project_root}/src/tracked_person_publisher_ros.py "
            f"--camera-mode go2 "
            f"--scan-topic /scan "
            f"--mode greeting "
            f"--cmd-port 2001 ",
        ]

    tracked_person_publisher_cmd = TimerAction(
        period=0.0,
        actions=[
            ExecuteProcess(
                cmd=build_tracker_cmd(),
                name="tracked_person_publisher",
                output="screen",
            )
        ],
    )

    # Person follower with greeting integration and geofencing
    person_follow_greet_cmd = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-c",
                    f"source /opt/ros/jazzy/setup.bash && "
                    f"source {install_dir}/setup.bash && "
                    f"python3 {project_root}/src/person_follow_greet_geofence.py",
                ],
                name="person_follow_greet",
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            LogInfo(msg="Starting Go2 Greeting System..."),
            LogInfo(msg="Step 1: Starting Tracked Person Publisher (greeting mode)..."),
            tracked_person_publisher_cmd,
            LogInfo(msg="Step 2: Starting Person Follow Greet with Geofencing..."),
            person_follow_greet_cmd,
        ]
    )
