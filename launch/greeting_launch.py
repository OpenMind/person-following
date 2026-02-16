#!/usr/bin/env python3
"""
ROS2 Launch file for Person Following Greeting Mode (multi-robot).

Supports: go2, g1, tron, and other robots via ROBOT_TYPE environment variable.

Launches:
    1. tracked_person_publisher_ros.py - YOLO-based person tracking (greeting mode)
    2. person_follow_greet_geofence.py - Person follower with OM1 greeting integration and geofencing

Parameters are loaded from config/<ROBOT_TYPE>_params.yaml based on the ROBOT_TYPE
environment variable (or defaults to go2 if not set).

Requires colcon build to be run first for om_api and unitree_api packages.

Usage:
    export ROBOT_TYPE=go2              # Choose robot type
    colcon build
    source install/setup.bash
    ros2 launch launch/geeting_launch.py
"""

import os

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    LogInfo,
    TimerAction,
)


def generate_launch_description():
    """
    Generate the launch description for greeting system.

    Supports multiple robots: go2, g1, tron, etc.
    Robot type is read from ROBOT_TYPE environment variable.
    Parameters are loaded from config/<ROBOT_TYPE>_params.yaml.
    """
    launch_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(launch_file_dir)

    install_dir = os.path.join(project_root, "install")
    robot_type = os.environ.get("ROBOT_TYPE", "go2").lower()
    config_file = os.path.join(project_root, "config", f"{robot_type}_params.yaml")

    # Validate config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Robot config not found: {config_file}\n"
            f"Supported robots: go2, g1, tron\n"
            f"Set ROBOT_TYPE environment variable and create config/<ROBOT_TYPE>_params.yaml"
        )

    # USB camera node for tron robot
    usb_cam_cmd = None
    if robot_type == "tron":
        usb_cam_cmd = TimerAction(
            period=0.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "bash",
                        "-c",
                        "source /opt/ros/jazzy/setup.bash && "
                        "ros2 run usb_cam usb_cam_node_exe",
                    ],
                    name="usb_cam",
                    output="screen",
                )
            ],
        )

    # Tracked person publisher with greeting mode (port 2001)
    def build_tracker_cmd():
        return [
            "bash",
            "-c",
            f"source /opt/ros/jazzy/setup.bash && "
            f"source {install_dir}/setup.bash && "
            f"python3 {project_root}/person_following/nodes/tracked_person_publisher_ros.py "
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
                    f"python3 {project_root}/person_following/nodes/person_follow_greet.py "
                    f"--ros-args --params-file {config_file}",
                ],
                name="person_follow_greet",
                output="screen",
            )
        ],
    )

    launch_actions = [
        LogInfo(msg=f"Starting {robot_type.upper()} Greeting System..."),
    ]

    # Add USB camera node for tron robot
    if usb_cam_cmd is not None:
        launch_actions.append(LogInfo(msg="Starting USB Camera Node (tron)..."))
        launch_actions.append(usb_cam_cmd)

    launch_actions.extend(
        [
            LogInfo(msg="Step 1: Starting Tracked Person Publisher (greeting mode)..."),
            tracked_person_publisher_cmd,
            LogInfo(msg="Step 2: Starting Person Follow Greet with Geofencing..."),
            person_follow_greet_cmd,
        ]
    )

    return LaunchDescription(launch_actions)
