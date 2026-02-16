#!/usr/bin/env python3
"""
ROS2 Launch file for Person Following System.

Launches:
    1. tracked_person_publisher_ros.py - YOLO-based person tracking and publishing
    2. person_follower.py - PD controller for following the tracked person

This launch file uses ExecuteProcess to run standalone Python scripts.
It does NOT require a ROS package - just `ros2 launch <path_to_this_file>`.

Camera topics are received from an external container via ROS.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate the launch description for person following system."""
    # Get project root from environment or use default
    project_root = os.environ.get("PROJECT_ROOT", "/opt/person_following")

    # Declare launch arguments
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

    color_topic_arg = DeclareLaunchArgument(
        "color_topic",
        default_value="/camera/realsense2_camera_node/color/image_raw",
        description="ROS color image topic",
    )

    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera/realsense2_camera_node/depth/image_rect_raw",
        description="ROS depth image topic",
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/realsense2_camera_node/color/camera_info",
        description="ROS camera info topic",
    )

    auto_enroll_arg = DeclareLaunchArgument(
        "auto_enroll",
        default_value="false",
        description="Enable auto-enrollment (true/false)",
    )

    display_arg = DeclareLaunchArgument(
        "display",
        default_value="false",
        description="Enable visualization window (true/false)",
    )

    # Build command arguments conditionally
    def build_tracker_cmd(context):
        """Build the tracker command with conditional flags."""
        import os as _os

        _project_root = _os.environ.get("PROJECT_ROOT", "/opt/person_following")
        display_val = context.launch_configurations.get("display", "false")
        auto_enroll_val = context.launch_configurations.get("auto_enroll", "false")

        cmd = [
            "python3",
            _os.path.join(_project_root, "person_following/nodes/tracked_person_publisher_ros.py"),
            "--yolo-det",
            context.launch_configurations["yolo_det"],
            "--yolo-seg",
            context.launch_configurations["yolo_seg"],
            "--color-topic",
            context.launch_configurations["color_topic"],
            "--depth-topic",
            context.launch_configurations["depth_topic"],
            "--camera-info-topic",
            context.launch_configurations["camera_info_topic"],
        ]

        if auto_enroll_val.lower() == "true":
            cmd.append("--auto-enroll")

        if display_val.lower() == "true":
            cmd.append("--display")

        return cmd

    # Tracked person publisher - runs as a standalone Python script
    def launch_tracker(context):
        cmd = build_tracker_cmd(context)
        return [
            ExecuteProcess(
                cmd=cmd,
                name="tracked_person_publisher",
                output="screen",
            )
        ]

    tracked_person_publisher_cmd = OpaqueFunction(function=launch_tracker)

    # Person follower - runs as a standalone Python script
    # Delayed start to ensure tracked_person_publisher is ready
    person_follower_cmd = TimerAction(
        period=2.0,  # Wait 2 seconds before starting
        actions=[
            ExecuteProcess(
                cmd=[
                    "python3",
                    os.path.join(project_root, "person_following/nodes/person_follower.py"),
                ],
                name="person_follower",
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            # Launch arguments
            yolo_det_arg,
            yolo_seg_arg,
            color_topic_arg,
            depth_topic_arg,
            camera_info_topic_arg,
            auto_enroll_arg,
            display_arg,
            # Log info
            LogInfo(msg="Starting Person Following System..."),
            LogInfo(msg=["YOLO Detection Engine: ", LaunchConfiguration("yolo_det")]),
            LogInfo(
                msg=["YOLO Segmentation Engine: ", LaunchConfiguration("yolo_seg")]
            ),
            LogInfo(msg=["Color Topic: ", LaunchConfiguration("color_topic")]),
            LogInfo(msg=["Depth Topic: ", LaunchConfiguration("depth_topic")]),
            # Processes
            tracked_person_publisher_cmd,
            person_follower_cmd,
        ]
    )
