"""
Person Following System - Lab + OpenCLIP Version.

Two-stage matching:
1. Lab color histograms (fast, lighting-robust)
2. OpenCLIP embeddings (semantic matching, cross-view robust)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from clothing_matcher_lab_openclip import ClothingMatcher, SegmentationError
from target_state import TargetState

# Configure logging
logger = logging.getLogger(__name__)


class PersonFollowingSystem:
    """
    Person following system using Lab color histograms and OpenCLIP embeddings.

    This system provides robust person tracking and re-identification using a
    two-stage matching approach: fast Lab color histogram matching followed by
    semantic OpenCLIP embedding verification.

    Parameters
    ----------
    yolo_detection_engine : str
        Path to the YOLO detection TensorRT engine file.
    yolo_seg_engine : str
        Path to the YOLO segmentation TensorRT engine file.
    device : str, optional
        Device to run inference on, by default 'cuda'.
    tracker_type : str, optional
        Type of tracker to use ('botsort' or 'bytetrack'), by default 'botsort'.
    frame_margin_lr : int, optional
        Left/right frame margin in pixels for valid detections, by default 20.
    use_clip : bool, optional
        Whether to use OpenCLIP for verification, by default True.
    clip_model : str, optional
        OpenCLIP model architecture name, by default 'ViT-B-16'.
    clip_pretrained : str, optional
        OpenCLIP pretrained weights identifier, by default 'laion2b_s34b_b88k'.
    seg_conf_thresh : float, optional
        Confidence threshold for segmentation, by default 0.3.
    clothing_threshold : float, optional
        Minimum clothing similarity threshold for matching, by default 0.5.
    clip_threshold : float, optional
        Minimum CLIP similarity threshold for verification, by default 0.7.
    min_mask_coverage : float, optional
        Minimum mask coverage percentage required, by default 35.0.
    bucket_spacing : float, optional
        Distance bucket spacing in meters for feature storage, by default 0.5.
    search_interval : float, optional
        Time interval between feature extractions during search mode in seconds,
        by default 0.33 (~3 fps).

    Attributes
    ----------
    detector : TRTYOLODetector
        YOLO detection model.
    tracker : BotSort or ByteTrack
        Object tracker instance.
    clothing_matcher : ClothingMatcher
        Clothing feature extraction and matching module.
    target : TargetState
        Current target state and feature storage.
    fps_history : list
        Rolling history of frame processing times for FPS calculation.
    all_tracks : list
        All tracked objects in the current frame.
    all_candidates_info : list
        Information about search candidates in the current frame.
    """

    def __init__(
        self,
        yolo_detection_engine: str,
        yolo_seg_engine: str,
        device: str = "cuda",
        tracker_type: str = "botsort",
        frame_margin_lr: int = 20,
        use_clip: bool = True,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        seg_conf_thresh: float = 0.3,
        clothing_threshold: float = 0.5,
        clip_threshold: float = 0.7,
        min_mask_coverage: float = 35.0,
        bucket_spacing: float = 0.5,
        search_interval: float = 0.33,  # Search frequency: ~3 fps during SEARCHING
    ):
        from yolo_detector import TRTYOLODetector

        self.detector = TRTYOLODetector(
            yolo_detection_engine, conf_thresh=0.5, nms_thresh=0.45
        )
        self.tracker = self._init_boxmot_tracker(tracker_type)
        self.tracker_type = tracker_type

        self.clothing_matcher = ClothingMatcher(
            yolo_seg_engine,
            device,
            use_clip=use_clip,
            clip_model=clip_model,
            clip_pretrained=clip_pretrained,
            clip_cache_dir="./model",
        )
        self.use_clip = use_clip and self.clothing_matcher.clip_matcher is not None
        self.seg_conf_thresh = seg_conf_thresh

        self.clothing_threshold = clothing_threshold
        self.clip_threshold = clip_threshold
        self.min_mask_coverage = min_mask_coverage
        self.bucket_spacing = bucket_spacing

        # Search mode throttling
        self.search_interval = (
            search_interval  # seconds between feature extraction in SEARCHING mode
        )
        self.last_search_time = 0.0
        self.cached_search_result = None  # Cache last search result

        self.target = TargetState()
        self.target.FRAME_MARGIN_LR = frame_margin_lr
        self.target.MIN_MASK_COVERAGE = min_mask_coverage
        self.target.MIN_MASK_COVERAGE_FOR_MATCH = min_mask_coverage - 5
        self.target.BUCKET_SPACING = bucket_spacing

        self.frame_width = 640
        self.frame_height = 480
        self.last_frame_time = time.time()
        self.fps_history = []
        self.all_tracks = []
        self.all_candidates_info = []

        logger.info("PersonFollowingSystem (Lab + OpenCLIP)")
        logger.info(f"  - Clothing threshold: {clothing_threshold}")
        logger.info(f"  - CLIP threshold: {clip_threshold}")
        logger.info(f"  - Min mask coverage: {min_mask_coverage}%")
        logger.info(f"  - Bucket spacing: {bucket_spacing}m")
        logger.info(
            f"  - Search interval: {search_interval}s ({1/search_interval:.1f} fps)"
        )

    def _init_boxmot_tracker(self, tracker_type: str):
        """
        Initialize the BoxMOT tracker.

        Parameters
        ----------
        tracker_type : str
            Type of tracker to initialize ('botsort' or 'bytetrack').

        Returns
        -------
        BotSort or ByteTrack
            Initialized tracker instance.
        """
        if tracker_type == "botsort":
            from boxmot import BotSort

            return BotSort(
                reid_weights=None,
                device="cuda",
                half=False,
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=30,
                match_thresh=0.8,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                with_reid=False,
            )
        elif tracker_type == "bytetrack":
            from boxmot import ByteTrack

            return ByteTrack(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
        raise ValueError(f"Unknown tracker: {tracker_type}")

    def _run_tracker(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Run the tracker on detections.

        Parameters
        ----------
        detections : numpy.ndarray
            Detection array of shape (N, 6) with columns [x1, y1, x2, y2, conf, cls].
        frame : numpy.ndarray
            BGR color frame of shape (height, width, 3).

        Returns
        -------
        numpy.ndarray
            Track array of shape (M, 7) with columns [x1, y1, x2, y2, track_id, conf, cls],
            or empty array of shape (0, 7) if no tracks.
        """
        if len(detections) == 0:
            return np.empty((0, 7))
        tracks = self.tracker.update(detections, frame)
        return tracks if len(tracks) > 0 else np.empty((0, 7))

    def _get_distance(
        self, bbox: Tuple[int, int, int, int], depth_frame: np.ndarray
    ) -> float:
        """
        Get the distance to a bounding box using depth data.

        Computes the median depth value in a small region around the bounding box center.

        Parameters
        ----------
        bbox : tuple of int
            Bounding box coordinates (x1, y1, x2, y2).
        depth_frame : numpy.ndarray
            Depth frame of shape (height, width) with depth values in meters.

        Returns
        -------
        float
            Median distance in meters, or 0.0 if no valid depth values.
        """
        x1, y1, x2, y2 = bbox
        H, W = depth_frame.shape[:2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        roi = depth_frame[
            max(0, cy - 10) : min(H, cy + 10), max(0, cx - 10) : min(W, cx + 10)
        ]
        valid = roi[(roi > 0.3) & (roi < 10.0)]
        return float(np.median(valid)) if len(valid) > 0 else 0.0

    def _extract_features(
        self, crop: np.ndarray, extract_clip: bool = True
    ) -> Tuple[
        Optional[np.ndarray], Optional[dict], Optional[np.ndarray], float, Optional[str]
    ]:
        """
        Extract features from a person crop.

        Performs segmentation, clothing feature extraction, and optionally CLIP
        embedding extraction from a cropped person image.

        Parameters
        ----------
        crop : numpy.ndarray
            Cropped person image of shape (height, width, 3) in BGR format.
        extract_clip : bool, optional
            Whether to extract CLIP embeddings, by default True.

        Returns
        -------
        mask : numpy.ndarray or None
            Binary segmentation mask of shape (height, width), or None if failed.
        clothing_feat : dict or None
            Clothing feature dictionary containing Lab histograms, or None if failed.
        clip_emb : numpy.ndarray or None
            CLIP embedding vector, or None if not extracted or failed.
        mask_coverage : float
            Percentage of mask coverage (0-100).
        error_msg : str or None
            Error message if any step failed, or None if successful.
        """
        mask = None
        clothing_feat = None
        clip_emb = None
        mask_coverage = 0.0
        error_msg = None

        try:
            mask = self.clothing_matcher.extract_person_mask_from_crop(
                crop, conf_thresh=self.seg_conf_thresh, validate_centroid=True
            )
            mask_coverage = mask.sum() / mask.size * 100
        except SegmentationError as e:
            return None, None, None, 0.0, f"segmentation: {str(e)}"

        try:
            clothing_feat = self.clothing_matcher.extract_clothing_features(crop, mask)
        except Exception as e:
            return mask, None, None, mask_coverage, f"clothing features: {str(e)}"

        if extract_clip and self.use_clip:
            try:
                clip_emb = self.clothing_matcher.extract_clip_embedding(crop, mask)
            except Exception as e:
                error_msg = f"clip: {str(e)}"

        return mask, clothing_feat, clip_emb, mask_coverage, error_msg

    def enroll_target(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> bool:
        """
        Enroll the closest valid person as the tracking target.

        Detects all persons in the frame, selects the closest one within valid
        margins, extracts features, and initializes tracking.

        Parameters
        ----------
        color_frame : numpy.ndarray
            BGR color frame of shape (height, width, 3) with dtype uint8.
        depth_frame : numpy.ndarray
            Depth frame of shape (height, width) with depth values in meters.

        Returns
        -------
        bool
            True if target was successfully enrolled, False otherwise.
        """
        timestamp = time.time()

        detections = self.detector.detect(color_frame)
        H, W = color_frame.shape[:2]
        self.frame_width, self.frame_height = W, H

        tracks = self._run_tracker(detections, color_frame)

        persons = []
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            bbox = (x1, y1, x2, y2)
            is_valid, _ = self.target.is_within_frame_margin(bbox, W, H)
            if not is_valid:
                continue

            distance = self._get_distance(bbox, depth_frame)
            if distance < 0.3:
                continue

            persons.append({"track_id": track_id, "bbox": bbox, "distance": distance})

        if not persons:
            logger.warning("No valid person detected")
            return False

        persons.sort(key=lambda p: p["distance"])
        target = persons[0]

        x1, y1, x2, y2 = target["bbox"]
        crop = color_frame[y1:y2, x1:x2]

        mask, clothing_feat, clip_emb, mask_coverage, error_msg = (
            self._extract_features(crop)
        )

        if error_msg:
            logger.warning(f"Feature extraction failed: {error_msg}")

        if clothing_feat is None:
            logger.warning("Failed to extract clothing features")
            return False

        if mask_coverage < self.min_mask_coverage:
            logger.warning(f"Mask coverage too low: {mask_coverage:.1f}%")
            return False

        if self.use_clip and clip_emb is None:
            logger.warning("CLIP embedding required but failed")
            return False

        self.target.initialize(target["track_id"], target["distance"], timestamp)

        bucket = self.target._get_bucket(target["distance"])
        saved = self.target.save_feature(
            bucket, "approaching", clothing_feat, clip_emb, mask_coverage, timestamp
        )

        if saved:
            self.target.last_saved_distance = target["distance"]
            self.target.last_saved_direction = "approaching"
            logger.info("Target enrolled!")
            logger.info(
                f"   Track ID: {target['track_id']}, Distance: {target['distance']:.2f}m"
            )
            logger.info(f"   Bucket: {bucket:.1f}m, Mask: {mask_coverage:.1f}%")
            logger.info(f"   CLIP: {'yes' if clip_emb is not None else 'no'}")
            return True

        return False

    def process_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> dict:
        """
        Process a single frame for person tracking.

        Runs detection, tracking, and either active tracking or searching
        depending on the current target state.

        Parameters
        ----------
        color_frame : numpy.ndarray
            BGR color frame of shape (height, width, 3) with dtype uint8.
        depth_frame : numpy.ndarray
            Depth frame of shape (height, width) with depth values in meters.

        Returns
        -------
        dict
            Processing result dictionary containing:
            - 'timestamp' : float
                Frame timestamp.
            - 'status' : str
                Current tracking status ('INACTIVE', 'TRACKING_ACTIVE', 'SEARCHING').
            - 'fps' : float
                Current processing frame rate.
            - 'num_detections' : int
                Number of detections in frame.
            - 'num_tracks' : int
                Number of active tracks.
            - 'all_tracks' : list of dict
                All tracked objects with 'track_id' and 'bbox'.
            - 'target_found' : bool
                Whether target was found (if tracking/searching).
            - 'bbox' : tuple, optional
                Target bounding box if found.
            - 'distance' : float, optional
                Target distance if found.
        """
        current_time = time.time()

        dt = current_time - self.last_frame_time
        if dt > 0:
            self.fps_history.append(1.0 / dt)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
        self.last_frame_time = current_time

        H, W = color_frame.shape[:2]
        self.frame_width, self.frame_height = W, H

        detections = self.detector.detect(color_frame)
        tracks = self._run_tracker(detections, color_frame)

        # Clear candidates info every frame (only populated during SEARCHING)
        self.all_candidates_info = []

        self.all_tracks = []
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                self.all_tracks.append({"track_id": track_id, "bbox": (x1, y1, x2, y2)})

        result = {
            "timestamp": current_time,
            "status": self.target.status,
            "fps": np.mean(self.fps_history) if self.fps_history else 0,
            "num_detections": len(detections),
            "num_tracks": len(tracks),
            "all_tracks": self.all_tracks,
        }

        if self.target.status == "TRACKING_ACTIVE":
            result.update(
                self._process_active_tracking(
                    tracks, color_frame, depth_frame, current_time, H, W
                )
            )
        elif self.target.status == "SEARCHING":
            result.update(
                self._process_searching(
                    tracks, color_frame, depth_frame, current_time, H, W
                )
            )

        return result

    def _process_active_tracking(
        self,
        tracks: np.ndarray,
        color_frame: np.ndarray,
        depth_frame: np.ndarray,
        timestamp: float,
        H: int,
        W: int,
    ) -> dict:
        """
        Process frame during active tracking mode.

        Locates the target track, updates distance, saves features when appropriate,
        and transitions to search mode if target is lost.

        Parameters
        ----------
        tracks : numpy.ndarray
            Track array of shape (M, 7) from the tracker.
        color_frame : numpy.ndarray
            BGR color frame of shape (height, width, 3).
        depth_frame : numpy.ndarray
            Depth frame of shape (height, width) with depth values in meters.
        timestamp : float
            Current timestamp.
        H : int
            Frame height in pixels.
        W : int
            Frame width in pixels.

        Returns
        -------
        dict
            Tracking result dictionary containing:
            - 'target_found' : bool
                Whether target track was found.
            - 'bbox' : tuple, optional
                Target bounding box (x1, y1, x2, y2) if found.
            - 'distance' : float, optional
                Target distance in meters if found.
            - 'direction' : str, optional
                Movement direction ('approaching', 'receding', 'stable').
            - 'feature_saved' : bool, optional
                Whether a new feature was saved this frame.
            - 'within_margin' : bool, optional
                Whether target is within valid frame margins.
        """
        target_track = None
        for track in tracks:
            if int(track[4]) == self.target.track_id:
                target_track = track
                break

        if target_track is None:
            self.target.mark_lost(timestamp)
            logger.info(f"Target lost (track_id={self.target.track_id})")
            logger.info(f"   {self.target.get_quality_summary()}")
            return {"target_found": False}

        x1, y1, x2, y2 = map(int, target_track[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        bbox = (x1, y1, x2, y2)
        current_distance = self._get_distance(bbox, depth_frame)
        direction = self.target.detect_movement_direction(current_distance, timestamp)

        should_save, bucket, save_dir = self.target.should_save_feature(
            current_distance, direction, timestamp, bbox, W, H
        )

        feature_saved = False
        if should_save:
            crop = color_frame[y1:y2, x1:x2]
            mask, clothing_feat, clip_emb, mask_coverage, error_msg = (
                self._extract_features(crop)
            )

            if clothing_feat is not None:
                saved = self.target.save_feature(
                    bucket, save_dir, clothing_feat, clip_emb, mask_coverage, timestamp
                )
                if saved:
                    self.target.last_saved_distance = current_distance
                    self.target.last_saved_direction = save_dir
                    feature_saved = True
                    clip_status = "yes" if clip_emb is not None else "no"
                    logger.info(
                        f"Saved @{bucket:.1f}m [{save_dir}] mask:{mask_coverage:.1f}% CLIP:{clip_status}"
                    )

        self.target.frames_tracked += 1
        is_within_margin, _ = self.target.is_within_frame_margin(bbox, W, H)

        return {
            "target_found": True,
            "bbox": bbox,
            "distance": current_distance,
            "direction": direction,
            "feature_saved": feature_saved,
            "within_margin": is_within_margin,
        }

    def _process_searching(
        self,
        tracks: np.ndarray,
        color_frame: np.ndarray,
        depth_frame: np.ndarray,
        timestamp: float,
        H: int,
        W: int,
    ) -> dict:
        """
        Process frame during search mode.

        Searches for the lost target among all tracked persons using two-stage
        matching (clothing similarity followed by CLIP verification).

        Parameters
        ----------
        tracks : numpy.ndarray
            Track array of shape (M, 7) from the tracker.
        color_frame : numpy.ndarray
            BGR color frame of shape (height, width, 3).
        depth_frame : numpy.ndarray
            Depth frame of shape (height, width) with depth values in meters.
        timestamp : float
            Current timestamp.
        H : int
            Frame height in pixels.
        W : int
            Frame width in pixels.

        Returns
        -------
        dict
            Search result dictionary containing:
            - 'target_found' : bool
                Whether target was re-identified.
            - 'bbox' : tuple, optional
                Target bounding box if found.
            - 'matched_track_id' : int, optional
                Track ID of re-identified target.
            - 'stage' : str
                Matching stage reached ('no_clothing_match', 'no_clip_match', 'verified').
            - 'clothing_sim' : float, optional
                Clothing similarity score if matched.
            - 'clip_sim' : float, optional
                CLIP similarity score if matched.
            - 'time_lost' : float
                Time since target was lost in seconds.
            - 'throttled' : bool, optional
                Whether this result was from cache due to throttling.
        """
        self.all_candidates_info = []

        # Build candidate list (fast, every frame)
        candidates = []
        track_bbox_map = {}  # track_id -> current bbox
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            bbox = (x1, y1, x2, y2)
            track_bbox_map[track_id] = bbox

            is_valid, _ = self.target.is_within_frame_margin(bbox, W, H)
            if not is_valid:
                continue

            distance = self._get_distance(bbox, depth_frame)
            if distance < 0.3:
                continue

            candidates.append(
                {"track_id": track_id, "bbox": bbox, "distance": distance}
            )

        if not candidates:
            self.target.mark_lost(timestamp)
            self.cached_search_result = None
            return {
                "target_found": False,
                "candidates_checked": 0,
                "time_lost": self.target.get_time_lost(timestamp),
            }

        # Throttle: check if we should run feature extraction
        time_since_last_search = timestamp - self.last_search_time
        should_run_search = time_since_last_search >= self.search_interval

        # If cached result exists and throttling, use cached with updated bbox
        if not should_run_search and self.cached_search_result is not None:
            cached = self.cached_search_result

            # Update candidate info with current bbox positions
            for cand_info in cached.get("candidates_info", []):
                tid = cand_info["track_id"]
                if tid in track_bbox_map:
                    cand_info["bbox"] = track_bbox_map[tid]
                    self.all_candidates_info.append(cand_info)

            # Update result bbox if target was found
            result = cached.copy()
            if (
                cached.get("target_found")
                and cached.get("matched_track_id") in track_bbox_map
            ):
                result["bbox"] = track_bbox_map[cached["matched_track_id"]]

            result["time_lost"] = self.target.get_time_lost(timestamp)
            result["throttled"] = True
            return result

        # Run full feature extraction
        self.last_search_time = timestamp
        logger.info(f"[SEARCH] Candidates: {len(candidates)}")

        results = []
        candidates_info_cache = []

        for person in candidates:
            x1, y1, x2, y2 = person["bbox"]
            crop = color_frame[y1:y2, x1:x2]
            query_distance = person["distance"]

            mask, clothing_feat, clip_emb, mask_coverage, error_msg = (
                self._extract_features(crop)
            )

            if error_msg and "segmentation" in error_msg:
                logger.debug(
                    f"   Track {person['track_id']}: segmentation failed @{query_distance:.1f}m"
                )
                continue

            if clothing_feat is None:
                logger.debug(
                    f"   Track {person['track_id']}: feature extraction failed @{query_distance:.1f}m"
                )
                continue

            if mask_coverage < self.min_mask_coverage - 10:
                logger.debug(
                    f"   Track {person['track_id']}: mask too small ({mask_coverage:.1f}%) @{query_distance:.1f}m"
                )
                continue

            ref_features = self.target.get_bucket_features_both_directions(
                query_distance
            )

            if not ref_features:
                logger.debug(
                    f"   Track {person['track_id']}: no reference features available"
                )
                continue

            closest_bucket = ref_features[0]["bucket"]

            clothing_sims = []
            clip_sims = []

            for ref in ref_features:
                c_sim = self.clothing_matcher.compute_clothing_similarity(
                    clothing_feat, ref["clothing"]
                )
                clothing_sims.append(c_sim)

                if clip_emb is not None and ref.get("clip") is not None:
                    clip_sim = self.clothing_matcher.compute_clip_similarity(
                        clip_emb, ref["clip"]
                    )
                    clip_sims.append(clip_sim)

            best_clothing_sim = max(clothing_sims) if clothing_sims else 0
            best_clip_sim = max(clip_sims) if clip_sims else 0

            clip_available = len(clip_sims) > 0

            passed_clothing = best_clothing_sim >= self.clothing_threshold

            if self.use_clip:
                if clip_available:
                    passed_clip = best_clip_sim >= self.clip_threshold
                else:
                    passed_clip = False
                    if passed_clothing:
                        logger.warning(
                            f"   Track {person['track_id']}: CLIP unavailable"
                        )
            else:
                passed_clip = True

            status = "PASS" if passed_clothing else "FAIL"
            clip_str = f"CLIP={best_clip_sim:.3f}" if clip_available else "CLIP=N/A"
            logger.info(
                f"   {status} Track {person['track_id']}: C={best_clothing_sim:.3f} {clip_str} "
                f"M={mask_coverage:.1f}% @{closest_bucket:.1f}m"
            )

            results.append(
                {
                    "person": person,
                    "clothing_sim": best_clothing_sim,
                    "clip_sim": best_clip_sim,
                    "clip_available": clip_available,
                    "mask_coverage": mask_coverage,
                    "closest_bucket": closest_bucket,
                    "passed_clothing": passed_clothing,
                    "passed_clip": passed_clip,
                }
            )

            cand_info = {
                "track_id": person["track_id"],
                "bbox": person["bbox"],
                "clothing_sim": best_clothing_sim,
                "clip_sim": best_clip_sim,
                "mask_coverage": mask_coverage,
                "bucket": closest_bucket,
            }
            self.all_candidates_info.append(cand_info)
            candidates_info_cache.append(cand_info)

        stage1_passed = [r for r in results if r["passed_clothing"]]

        if not stage1_passed:
            self.target.mark_lost(timestamp)
            best = max(results, key=lambda r: r["clothing_sim"]) if results else None
            result = {
                "target_found": False,
                "stage": "no_clothing_match",
                "best_clothing_sim": best["clothing_sim"] if best else 0,
                "candidates_checked": len(results),
                "time_lost": self.target.get_time_lost(timestamp),
                "candidates_info": candidates_info_cache,
            }
            self.cached_search_result = result
            return result

        if self.use_clip:
            clip_passed = [
                r for r in stage1_passed if r["passed_clip"] and r["clip_available"]
            ]

            if not clip_passed:
                self.target.mark_lost(timestamp)
                clip_available = [r for r in stage1_passed if r["clip_available"]]
                if clip_available:
                    best = max(clip_available, key=lambda r: r["clip_sim"])
                    logger.info(
                        f"   No CLIP match (best: {best['clip_sim']:.3f} < {self.clip_threshold})"
                    )
                else:
                    best = max(stage1_passed, key=lambda r: r["clothing_sim"])
                    logger.info("   No CLIP available for comparison")

                result = {
                    "target_found": False,
                    "stage": "no_clip_match",
                    "best_clip_sim": best.get("clip_sim", 0),
                    "best_clothing_sim": best["clothing_sim"],
                    "time_lost": self.target.get_time_lost(timestamp),
                    "candidates_info": candidates_info_cache,
                }
                self.cached_search_result = result
                return result

            clip_passed.sort(key=lambda r: r["clip_sim"], reverse=True)
            best = clip_passed[0]
        else:
            stage1_passed.sort(key=lambda r: r["clothing_sim"], reverse=True)
            best = stage1_passed[0]

        self.target.resume_tracking(best["person"]["track_id"])
        logger.info(f"RE-IDENTIFIED: Track {best['person']['track_id']}")
        logger.info(
            f"   C={best['clothing_sim']:.3f} CLIP={best['clip_sim']:.3f} @{best['closest_bucket']:.1f}m"
        )

        # Clear cache since we found the target and will switch to TRACKING_ACTIVE
        self.cached_search_result = None

        return {
            "target_found": True,
            "bbox": best["person"]["bbox"],
            "matched_track_id": best["person"]["track_id"],
            "stage": "verified",
            "clothing_sim": best["clothing_sim"],
            "clip_sim": best["clip_sim"],
            "bucket": best["closest_bucket"],
            "time_lost": self.target.get_time_lost(timestamp),
        }

    def clear_target(self):
        """
        Clear the current tracking target and reset state.

        Resets the target state to initial values while preserving
        system configuration parameters.
        """
        self.target = TargetState()
        self.target.FRAME_MARGIN_LR = 20
        self.target.MIN_MASK_COVERAGE = self.min_mask_coverage
        self.target.BUCKET_SPACING = self.bucket_spacing
        logger.info("Target cleared")

    def get_status(self) -> dict:
        """
        Get the current system status.

        Returns
        -------
        dict
            Status dictionary containing:
            - 'status' : str
                Current tracking status.
            - 'track_id' : int or None
                Current target track ID.
            - 'features' : int
                Total number of stored features.
            - 'quality' : str
                Quality summary string.
            - 'buckets' : str
                Information about feature buckets.
            - 'thresholds' : str
                Current threshold settings.
        """
        return {
            "status": self.target.status,
            "track_id": self.target.track_id,
            "features": self.target.get_total_features(),
            "quality": self.target.get_quality_summary(),
            "buckets": self.target.get_buckets_info(),
            "thresholds": f"C>{self.clothing_threshold} CLIP>{self.clip_threshold} M>{self.min_mask_coverage}%",
        }

    def get_all_tracks(self) -> List[Dict]:
        """
        Get all current tracks.

        Returns
        -------
        list of dict
            List of all tracked objects, each containing 'track_id' and 'bbox'.
        """
        return self.all_tracks

    def get_candidates_info(self) -> List[Dict]:
        """
        Get information about search candidates.

        Returns
        -------
        list of dict
            List of candidate information dictionaries, each containing:
            - 'track_id' : int
            - 'bbox' : tuple
            - 'clothing_sim' : float
            - 'clip_sim' : float
            - 'mask_coverage' : float
            - 'bucket' : float
        """
        return self.all_candidates_info
