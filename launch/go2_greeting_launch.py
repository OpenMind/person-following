#!/usr/bin/env python3
"""
ROS2 Launch file for Go2 Greeting Mode.

Launches:
    1. go2_camera_with_adjustable_publisher.py - Go2 camera publisher
    2. tracked_person_publisher_ros.py - YOLO-based person tracking (greeting mode)
    3. person_follow_greet.py - Person follower with OM1 greeting integration

Requires colcon build to be run first for om_api and unitree_api packages.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    TimerAction,
)


def generate_launch_description():
    """Generate the launch description for Go2 greeting system."""
    project_root = os.environ.get("PROJECT_ROOT", "/opt/person_following")
    install_dir = os.path.join(project_root, "install")

    display_arg = DeclareLaunchArgument(
        "display",
        default_value="false",
        description="Enable visualization window (true/false)",
    )

    yolo_det_arg = DeclareLaunchArgument(
        "yolo_det",
        default_value=os.path.join(project_root, "engine/yolo11n.engine"),
        description="Path to YOLO detection TensorRT engine",
    )

    yolo_seg_arg = DeclareLaunchArgument(
        "yolo_seg",
        default_value=os.path.join(project_root, "engine/yolo11s-seg.engine"),
        description="Path to YOLO segmentation TensorRT engine",
    )

    # Shell command to source workspace and run go2_camera
    go2_camera_cmd = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            f"source /opt/ros/jazzy/setup.bash && "
            f"source {install_dir}/setup.bash && "
            f"python3 {project_root}/src/go2_camera_with_adjustable_publisher.py",
        ],
        name="go2_camera",
        output="screen",
    )

    # Tracked person publisher with greeting mode (port 2001)
    def build_tracker_cmd():
        return [
            "bash",
            "-c",
            f"source /opt/ros/jazzy/setup.bash && "
            f"source {install_dir}/setup.bash && "
            f"python3 {project_root}/src/tracked_person_publisher_ros.py "
            f"--camera-mode go2 "
            f"--image-topic /camera/image_raw "
            f"--scan-topic /scan "
            f"--intrinsics-yaml {project_root}/src/camera_intrinsics.yaml "
            f"--mode greeting "
            f"--cmd-port 2001 "
            f"--yolo-det {project_root}/engine/yolo11n.engine "
            f"--yolo-seg {project_root}/engine/yolo11s-seg.engine",
        ]

    tracked_person_publisher_cmd = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=build_tracker_cmd(),
                name="tracked_person_publisher",
                output="screen",
            )
        ],
    )

    # Person follower with greeting integration
    person_follow_greet_cmd = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-c",
                    f"source /opt/ros/jazzy/setup.bash && "
                    f"source {install_dir}/setup.bash && "
                    f"python3 {project_root}/src/person_follow_greet.py",
                ],
                name="person_follow_greet",
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            display_arg,
            yolo_det_arg,
            yolo_seg_arg,
            LogInfo(msg="Starting Go2 Greeting System..."),
            LogInfo(msg="Step 1: Starting Go2 Camera Publisher..."),
            go2_camera_cmd,
            LogInfo(msg="Step 2: Starting Tracked Person Publisher (greeting mode)..."),
            tracked_person_publisher_cmd,
            LogInfo(msg="Step 3: Starting Person Follow Greet..."),
            person_follow_greet_cmd,
        ]
    )
