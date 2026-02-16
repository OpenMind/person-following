"""
Person Following System - Lab + OpenCLIP.

Main entry point for the person following robot.

Usage
-----
    python main.py --yolo-det yolo11n.engine --yolo-seg yolo11s-seg.engine
    python main.py --yolo-det yolo11n.engine --yolo-seg yolo11s-seg.engine --no-display
    python main.py --help
"""

import argparse
import logging

import cv2
import numpy as np
import pyrealsense2 as rs

from person_following.nodes.person_following_system import PersonFollowingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments for the person following system.
    """
    p = argparse.ArgumentParser(description="Person Following System - Lab + OpenCLIP")

    # Model paths
    p.add_argument(
        "--yolo-det",
        type=str,
        required=True,
        help="Path to YOLO detection TensorRT engine",
    )
    p.add_argument(
        "--yolo-seg",
        type=str,
        required=True,
        help="Path to YOLO segmentation TensorRT engine",
    )

    # OpenCLIP settings
    p.add_argument(
        "--clip-model", type=str, default="ViT-B-16", help="OpenCLIP model name"
    )
    p.add_argument(
        "--clip-pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="OpenCLIP pretrained weights",
    )

    # Thresholds
    p.add_argument(
        "--clothing-threshold",
        type=float,
        default=0.8,
        help="Lab clothing similarity threshold",
    )
    p.add_argument(
        "--clip-threshold",
        type=float,
        default=0.8,
        help="OpenCLIP similarity threshold",
    )
    p.add_argument(
        "--min-mask-coverage",
        type=float,
        default=35.0,
        help="Minimum mask coverage percentage",
    )
    p.add_argument(
        "--search-interval",
        type=float,
        default=0.33,
        help="Search mode feature extraction interval in seconds (default: 0.33 = ~3fps)",
    )

    # Tracker
    p.add_argument(
        "--tracker",
        type=str,
        default="botsort",
        choices=["botsort", "bytetrack"],
        help="Tracker type",
    )

    # Camera
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)

    # Display
    p.add_argument("--no-display", action="store_true", help="Disable display window")
    p.add_argument(
        "--save-video", type=str, default=None, help="Save output to video file"
    )

    return p.parse_args()


class RealSenseCamera:
    """
    RealSense D400 series camera wrapper.

    This class provides a wrapper for Intel RealSense D400 series depth cameras,
    handling initialization, frame acquisition, and depth-color alignment.

    Parameters
    ----------
    width : int, optional
        Frame width in pixels, by default 640.
    height : int, optional
        Frame height in pixels, by default 480.
    fps : int, optional
        Frames per second, by default 30.

    Attributes
    ----------
    pipeline : rs.pipeline
        RealSense pipeline for streaming.
    config : rs.config
        RealSense configuration.
    profile : rs.pipeline_profile
        Active pipeline profile.
    depth_scale : float
        Depth scale factor to convert raw depth values to meters.
    align : rs.align
        Alignment object to align depth frames to color frames.
    """

    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Start pipeline
        self.profile = self.pipeline.start(self.config)

        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Warm up
        for _ in range(30):
            self.pipeline.wait_for_frames()

        logger.info(f"RealSense camera started ({width}x{height}@{fps}fps)")
        logger.info(f"Depth scale: {self.depth_scale}")

    def get_frames(self):
        """
        Get aligned color and depth frames from the RealSense camera.

        Retrieves the latest color and depth frames from the camera pipeline,
        aligns them, and converts depth values to meters using the depth scale.

        Returns
        -------
        color_image : numpy.ndarray or None
            BGR color image of shape (height, width, 3) with dtype uint8.
            Returns None if frame acquisition fails.
        depth_image : numpy.ndarray or None
            Depth image of shape (height, width) with dtype float64,
            containing depth values in meters.
            Returns None if frame acquisition fails.
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale

        return color_image, depth_image

    def stop(self):
        """
        Stop the RealSense camera pipeline.

        Releases all resources associated with the camera pipeline.
        Should be called when the camera is no longer needed.
        """
        self.pipeline.stop()
        logger.info("RealSense camera stopped")


def draw_visualization(frame, result, system):
    """
    Draw tracking visualization overlay on a frame.

    Renders bounding boxes, track IDs, target information, status bars,
    and search candidates onto the input frame for visualization purposes.

    Parameters
    ----------
    frame : numpy.ndarray
        Input BGR color frame of shape (height, width, 3) with dtype uint8.
    result : dict
        Processing result dictionary containing:
        - 'all_tracks' : list of dict
            List of tracked objects with 'bbox' and 'track_id' keys.
        - 'target_found' : bool
            Whether the target is currently tracked.
        - 'bbox' : tuple of int, optional
            Target bounding box (x1, y1, x2, y2) if target is found.
        - 'distance' : float, optional
            Distance to target in meters.
        - 'status' : str
            Current system status ('TRACKING_ACTIVE', 'SEARCHING', 'INACTIVE').
        - 'fps' : float
            Current processing frame rate.
        - 'time_lost' : float, optional
            Time since target was lost (during search mode).
    system : PersonFollowingSystem
        The person following system instance for accessing target info
        and candidate information.

    Returns
    -------
    numpy.ndarray
        Annotated BGR frame of shape (height, width, 3) with dtype uint8,
        containing visualization overlays.

    Notes
    -----
    The visualization includes:
    - Green boxes for the target, gray boxes for other tracks
    - Orange boxes for search candidates with similarity scores
    - Top status bar with tracking status and FPS
    - Bottom bar with keyboard controls
    - Center crosshair on target when tracking
    """
    display = frame.copy()
    H, W = display.shape[:2]

    # Draw all tracks
    for track in result.get("all_tracks", []):
        x1, y1, x2, y2 = track["bbox"]
        track_id = track["track_id"]

        # Color: green for target, gray for others
        if track_id == system.target.track_id:
            color = (0, 255, 0)
            thickness = 2
        else:
            color = (128, 128, 128)
            thickness = 1

        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            display,
            f"ID:{track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    # Draw target bbox with distance
    if result.get("target_found") and "bbox" in result:
        x1, y1, x2, y2 = result["bbox"]
        distance = result.get("distance", 0)

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

        label = f"TARGET @{distance:.2f}m"
        cv2.putText(
            display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # Draw center crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(display, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
        cv2.line(display, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)

    # Draw candidates info during search
    for cand in system.get_candidates_info():
        x1, y1, x2, y2 = cand["bbox"]
        lab_sim = cand["clothing_sim"]
        clip_sim = cand["clip_sim"]

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            display,
            f"L:{lab_sim:.2f} C:{clip_sim:.2f}",
            (x1, y2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
        )

    # Status bar
    status = result.get("status", "UNKNOWN")
    fps = result.get("fps", 0)

    status_color = {
        "TRACKING_ACTIVE": (0, 255, 0),
        "SEARCHING": (0, 165, 255),
        "INACTIVE": (128, 128, 128),
    }.get(status, (255, 255, 255))

    # Top bar
    cv2.rectangle(display, (0, 0), (W, 35), (0, 0, 0), -1)
    cv2.putText(
        display,
        f"Status: {status}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2,
    )
    cv2.putText(
        display,
        f"FPS: {fps:.1f}",
        (W - 100, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    # Track ID and features
    if system.target.track_id is not None:
        info = (
            f"Track: {system.target.track_id} | {system.target.get_quality_summary()}"
        )
        cv2.putText(
            display, info, (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
        )

    # Bottom bar - instructions
    cv2.rectangle(display, (0, H - 30), (W, H), (0, 0, 0), -1)
    cv2.putText(
        display,
        "'e'=enroll | 'c'=clear | 's'=status | 'q'=quit",
        (10, H - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    # Search status
    if status == "SEARCHING":
        time_lost = result.get("time_lost", 0)
        cv2.putText(
            display,
            f"SEARCHING... ({time_lost:.1f}s)",
            (W // 2 - 80, H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

    return display


def main():
    """
    Main entry point for the person following system.

    Initializes the RealSense camera and person following system,
    then runs the main processing loop with visualization and
    keyboard controls for target enrollment and management.

    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("PERSON FOLLOWING SYSTEM - Lab + OpenCLIP")
    logger.info("=" * 60)

    # Initialize camera
    logger.info("[1/2] Initializing camera...")
    camera = RealSenseCamera(args.width, args.height, args.fps)

    # Initialize person following system
    logger.info("[2/2] Initializing person following system...")

    system = PersonFollowingSystem(
        yolo_detection_engine=args.yolo_det,
        yolo_seg_engine=args.yolo_seg,
        device="cuda",
        tracker_type=args.tracker,
        use_clip=True,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clothing_threshold=args.clothing_threshold,
        clip_threshold=args.clip_threshold,
        min_mask_coverage=args.min_mask_coverage,
        search_interval=args.search_interval,
    )

    # Video writer
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.save_video, fourcc, args.fps, (args.width, args.height)
        )
        logger.info(f"Saving video to: {args.save_video}")

    logger.info("=" * 60)
    logger.info("READY")
    logger.info("=" * 60)
    logger.info("Controls:")
    logger.info("  'e' - Enroll target (closest person)")
    logger.info("  'c' - Clear target")
    logger.info("  's' - Print status")
    logger.info("  'q' - Quit")
    logger.info("=" * 60)

    try:
        while True:
            # Get frames
            color_frame, depth_frame = camera.get_frames()
            if color_frame is None:
                continue

            # Process frame
            result = system.process_frame(color_frame, depth_frame)

            # Visualization
            if not args.no_display:
                display = draw_visualization(color_frame, result, system)
                cv2.imshow("Person Following - Lab + OpenCLIP", display)

                if video_writer:
                    video_writer.write(display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("e"):
                    logger.info("[ENROLL] Enrolling target...")
                    success = system.enroll_target(color_frame, depth_frame)
                    if not success:
                        logger.warning("[ENROLL] Failed - no valid person detected")
                elif key == ord("c"):
                    system.clear_target()
                    logger.info("[CLEAR] Target cleared")
                elif key == ord("s"):
                    status = system.get_status()
                    logger.info("=" * 40)
                    logger.info("STATUS")
                    logger.info("=" * 40)
                    for k, v in status.items():
                        logger.info(f"  {k}: {v}")
                    logger.info("=" * 40)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        camera.stop()
        if video_writer:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.info("Done.")


if __name__ == "__main__":
    main()
