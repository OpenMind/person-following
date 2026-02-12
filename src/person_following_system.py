"""
Person Following System - Lab + OpenCLIP.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np

from clothing_matcher_lab_openclip import ClothingMatcher, SegmentationError
from switch_state import SwitchState
from target_state import TargetState

CLIP_CACHE_DIR = str(Path(__file__).resolve().parents[1] / "model")
DEFAULT_HISTORY_PATH = str(
    Path(__file__).resolve().parents[1] / "history" / "person_following_history.pkl"
)
MAX_SAVE_HISTORY = 100  # Max entries to save in file

logger = logging.getLogger(__name__)


OperationMode = Literal["greeting", "following"]


class PersonFollowingSystem:
    """Person following system with switch_target and history."""

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
        search_interval: float = 0.33,
        lidar_cluster_range_jump: float = 0.35,
        lidar_cluster_min_size: int = 4,
        # History settings
        max_history_size: int = 5,
        history_file: str = None,
        auto_load_history: bool = True,
        auto_save_history: bool = True,
        # Switch settings
        switch_interval: float = 0.33,
        switch_candidate_timeout: float = 3.0,
        # Approach and searching timeout settings
        approach_distance: float = 1.0,
        searching_timeout: float = 5.0,
        operation_mode: OperationMode = "greeting",
    ):
        """
        Initialize the person following system.
        """
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
            clip_cache_dir=CLIP_CACHE_DIR,
        )
        self.use_clip = use_clip and self.clothing_matcher.clip_matcher is not None
        self.seg_conf_thresh = seg_conf_thresh

        self.clothing_threshold = clothing_threshold
        self.clip_threshold = clip_threshold
        self.min_mask_coverage = min_mask_coverage
        self.bucket_spacing = bucket_spacing

        self.search_interval = search_interval
        self.last_search_time = 0.0
        self.cached_search_result = None

        self.lidar_cluster_range_jump = lidar_cluster_range_jump
        self.lidar_cluster_min_size = lidar_cluster_min_size

        # History management
        self.max_history_size = max_history_size
        self.enrolled_history: List[dict] = []
        self.history_file = (
            Path(history_file) if history_file else Path(DEFAULT_HISTORY_PATH)
        )
        self.auto_save_history = auto_save_history

        # Switch state machine
        self.switch_state = SwitchState(
            switch_interval=switch_interval,
            candidate_timeout=switch_candidate_timeout,
        )

        # Approach and searching timeout settings
        self.approach_distance = approach_distance
        self.searching_timeout = searching_timeout

        # Store operation mode
        self.operation_mode = operation_mode

        # Target state
        self.target = TargetState()
        self.target.FRAME_MARGIN_LR = frame_margin_lr
        self.target.MIN_MASK_COVERAGE = min_mask_coverage
        self.target.MIN_MASK_COVERAGE_FOR_MATCH = min_mask_coverage - 5
        self.target.BUCKET_SPACING = bucket_spacing
        self.target.operation_mode = operation_mode  # Set target's mode

        self.frame_width = 640
        self.frame_height = 480
        self.last_frame_time = time.time()
        self.fps_history = []
        self.all_tracks = []
        self.all_candidates_info = []

        logger.info("PersonFollowingSystem initialized")
        logger.info(f"  - Operation mode: {operation_mode}")
        logger.info(f"  - Clothing threshold: {clothing_threshold}")
        logger.info(f"  - CLIP threshold: {clip_threshold}")
        logger.info(
            f"  - Switch interval: {switch_interval}s (~{1/switch_interval:.1f} Hz)"
        )
        logger.info(f"  - Switch timeout: {switch_candidate_timeout}s per candidate")
        logger.info(f"  - Max history size: {max_history_size}")
        logger.info(f"  - History file: {self.history_file}")
        logger.info(f"  - Approach distance: {approach_distance}m")
        logger.info(f"  - Searching timeout: {searching_timeout}s")

        # Auto load history on startup
        if auto_load_history:
            self.load_history()

    # Operation Mode Management
    def set_operation_mode(self, mode: OperationMode) -> dict:
        """
        Change operation mode at runtime.

        Switches between 'greeting' and 'following' modes. When switching:
        - Clears the current target
        - Stops any active switch operation
        - Updates target state's mode
        - Loads history if switching to greeting mode

        Parameters
        ----------
        mode : {'greeting', 'following'}
            New operation mode.

        Returns
        -------
        dict
            Result containing:
            - changed : bool
                Whether mode actually changed.
            - old_mode : str
                Previous operation mode.
            - new_mode : str
                Current operation mode.
            - actions : list of str, optional
                Actions taken during switch (e.g., 'cleared_target', 'stopped_switch').
        """
        if mode not in ("greeting", "following"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'greeting' or 'following'")

        old_mode = self.operation_mode

        if mode == old_mode:
            logger.info(f"[MODE] Already in {mode} mode")
            return {
                "changed": False,
                "old_mode": old_mode,
                "new_mode": mode,
            }

        logger.info(f"[MODE] Switching: {old_mode} -> {mode}")

        actions = []

        # Clear current target
        if self.target.status != "INACTIVE":
            self.clear_target()
            actions.append("cleared_target")

        # Stop switch operation if active
        if self.switch_state.active:
            self.switch_state.stop()
            actions.append("stopped_switch")

        # Update mode
        self.operation_mode = mode
        self.target.operation_mode = mode

        # Load history if switching to greeting mode
        if mode == "greeting":
            self.load_history()
            actions.append("loaded_history")

        logger.info(f"[MODE] Now in {mode.upper()} mode (actions: {actions})")

        return {
            "changed": True,
            "old_mode": old_mode,
            "new_mode": mode,
            "actions": actions,
        }

    def get_operation_mode(self) -> OperationMode:
        """
        Get current operation mode.

        Returns
        -------
        str
            Current operation mode ('greeting' or 'following').
        """
        return self.operation_mode

    # Tracker initialization
    def _init_boxmot_tracker(self, tracker_type: str) -> object:
        """
        Initialize the BoxMOT multi-object tracker.

        Parameters
        ----------
        tracker_type : {'botsort', 'bytetrack'}
            Type of tracker to initialize.

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
        Run tracker update with current detections.

        Parameters
        ----------
        detections : np.ndarray
            Detection array of shape (N, 6) with [x1, y1, x2, y2, conf, cls].
        frame : np.ndarray
            Current BGR frame for tracker update.

        Returns
        -------
        np.ndarray
            Tracked objects array of shape (M, 7) with [x1, y1, x2, y2, track_id, conf, cls].
            Returns empty array if no tracks.
        """
        if len(detections) == 0:
            return np.empty((0, 7))
        tracks = self.tracker.update(detections, frame)
        return tracks if len(tracks) > 0 else np.empty((0, 7))

    # Distance helpers
    def _split_lidar_clusters(
        self,
        scan_idx_sorted,
        ranges_sorted,
        range_jump,
        min_cluster_size,
        max_index_gap=1,
    ) -> List[np.ndarray]:
        """
        Split LiDAR points into clusters based on range discontinuities.

        Parameters
        ----------
        scan_idx_sorted : np.ndarray
            Sorted scan indices of points.
        ranges_sorted : np.ndarray
            Corresponding range values sorted by scan index.
        range_jump : float
            Range difference threshold to split clusters.
        min_cluster_size : int
            Minimum number of points for a valid cluster.
        max_index_gap : int, optional
            Maximum allowed gap in scan indices within a cluster, by default 1.

        Returns
        -------
        list of tuple
            List of (scan_indices, ranges) tuples for each valid cluster.
        """
        if scan_idx_sorted.size == 0:
            return []
        clusters = []
        start = 0
        for i in range(1, len(scan_idx_sorted)):
            idx_gap = int(scan_idx_sorted[i]) - int(scan_idx_sorted[i - 1])
            r_gap = abs(float(ranges_sorted[i]) - float(ranges_sorted[i - 1]))
            if idx_gap <= max_index_gap and r_gap <= range_jump:
                continue
            if (i - start) >= min_cluster_size:
                clusters.append(np.arange(start, i, dtype=np.int32))
            start = i
        if (len(scan_idx_sorted) - start) >= min_cluster_size:
            clusters.append(np.arange(start, len(scan_idx_sorted), dtype=np.int32))
        return clusters

    def _get_distance_from_lidar(self, bbox, aux) -> Tuple[float, int, int]:
        """
        Calculate distance to person using LiDAR points within bounding box.

        Parameters
        ----------
        bbox : tuple of int
            Bounding box (x1, y1, x2, y2) in pixel coordinates.
        aux : dict
            Auxiliary data containing 'lidar_uv', 'lidar_ranges', 'lidar_scan_idx'.

        Returns
        -------
        distance : float
            Median distance of the best cluster, or -1.0 if no valid cluster.
        cluster_pts : int
            Number of points in the selected cluster.
        bbox_pts : int
            Total number of LiDAR points within the bounding box.
        """
        if aux is None:
            return 0.0, 0, 0
        uv = aux.get("lidar_uv")
        scan_idx = aux.get("lidar_scan_idx")
        ranges = aux.get("lidar_ranges")
        if uv is None or scan_idx is None or ranges is None:
            return 0.0, 0, 0
        uv = np.asarray(uv)
        scan_idx = np.asarray(scan_idx)
        ranges = np.asarray(ranges)
        if uv.ndim != 2 or uv.shape[1] != 2:
            return 0.0, 0, 0
        x1, y1, x2, y2 = bbox
        u, v_arr = uv[:, 0].astype(np.int32), uv[:, 1].astype(np.int32)
        in_bbox = (u >= x1) & (u < x2) & (v_arr >= y1) & (v_arr < y2)
        if not np.any(in_bbox):
            return 0.0, 0, 0
        sel_idx = scan_idx[in_bbox].astype(np.int32)
        sel_ranges = ranges[in_bbox].astype(np.float64)
        valid = np.isfinite(sel_ranges) & (sel_ranges > 0.1) & (sel_ranges < 20.0)
        sel_idx, sel_ranges = sel_idx[valid], sel_ranges[valid]
        bbox_pts = len(sel_ranges)
        if sel_ranges.size == 0:
            return 0.0, 0, 0
        order = np.argsort(sel_idx)
        idx_sorted, r_sorted = sel_idx[order], sel_ranges[order]
        clusters = self._split_lidar_clusters(
            idx_sorted,
            r_sorted,
            self.lidar_cluster_range_jump,
            self.lidar_cluster_min_size,
        )
        if clusters:
            cluster_medians = [(c, float(np.median(r_sorted[c]))) for c in clusters]
            best_cluster, best_dist = min(cluster_medians, key=lambda x: x[1])
            return best_dist, len(best_cluster), bbox_pts
        return float(np.median(r_sorted)), 0, bbox_pts

    def _get_distance(self, bbox, depth_frame=None, aux=None) -> Tuple[float, int, int]:
        """
        Get distance to target using LiDAR (if available) or depth frame.

        Parameters
        ----------
        bbox : tuple of int
            Bounding box (x1, y1, x2, y2) in pixel coordinates.
        depth_frame : np.ndarray, optional
            Depth image in millimeters (uint16), by default None.
        aux : dict, optional
            Auxiliary data with LiDAR information, by default None.

        Returns
        -------
        distance : float
            Distance in meters, or -1.0 if unavailable.
        cluster_pts : int
            Number of LiDAR cluster points (0 if using depth).
        bbox_pts : int
            Total LiDAR points in bbox (0 if using depth).
        """
        if depth_frame is not None:
            x1, y1, x2, y2 = bbox
            H, W = depth_frame.shape[:2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            roi = depth_frame[
                max(0, cy - 10) : min(H, cy + 10), max(0, cx - 10) : min(W, cx + 10)
            ]
            valid = roi[(roi > 0.3) & (roi < 10.0)]
            return (float(np.median(valid)) if len(valid) > 0 else 0.0), 0, 0
        if aux is not None:
            return self._get_distance_from_lidar(bbox, aux)
        return 0.0, 0, 0

    def _extract_features(self, crop, extract_clip=True):
        """
        Extract clothing features and CLIP embedding from person crop.

        Parameters
        ----------
        crop : np.ndarray
            BGR image crop of the person.
        extract_clip : bool, optional
            Whether to extract CLIP embedding, by default True.

        Returns
        -------
        mask : np.ndarray or None
            Binary segmentation mask of the person.
        clothing_feat : dict or None
            Clothing feature dictionary with LAB histograms.
        clip_emb : np.ndarray or None
            CLIP embedding vector.
        mask_coverage : float
            Percentage of crop covered by mask (0-100).
        error_msg : str or None
            Error message if extraction failed, None otherwise.
        """
        mask, clothing_feat, clip_emb, mask_coverage, error_msg = (
            None,
            None,
            None,
            0.0,
            None,
        )
        try:
            mask = self.clothing_matcher.extract_person_mask_from_crop(
                crop, conf_thresh=self.seg_conf_thresh, validate_centroid=True
            )
            mask_coverage = mask.sum() / mask.size * 100
        except SegmentationError as e:
            return None, None, None, 0.0, f"segmentation: {e}"
        try:
            clothing_feat = self.clothing_matcher.extract_clothing_features(crop, mask)
        except Exception as e:
            return mask, None, None, mask_coverage, f"clothing: {e}"
        if extract_clip and self.use_clip:
            try:
                clip_emb = self.clothing_matcher.extract_clip_embedding(crop, mask)
            except Exception as e:
                error_msg = f"clip: {e}"
        return mask, clothing_feat, clip_emb, mask_coverage, error_msg

    # History Management
    def _save_current_target_to_history(self) -> bool:
        """
        Save ALL bucket features from current target to history.

        History entry structure:
        {
            'buckets': {
                1.5: {'approaching': {...}, 'leaving': {...}},
                2.0: {'approaching': {...}, 'leaving': {...}},
            },
            'timestamp': float,
            'track_id': int,
        }

        Returns
        -------
        bool
            True if successfully saved, False if no valid features or inactive.
        """
        if self.target.status == "INACTIVE":
            return False

        if not self.target.features:
            logger.warning("No features to save to history")
            return False

        # Copy the bucket structure
        buckets_copy = {}
        feature_count = 0

        for bucket, directions in self.target.features.items():
            buckets_copy[bucket] = {}
            for direction, data in directions.items():
                if data is not None and data.get("clothing") is not None:
                    buckets_copy[bucket][direction] = {
                        "clothing": data["clothing"],
                        "clip": data.get("clip"),
                        "mask_coverage": data.get("mask_coverage", 0),
                    }
                    feature_count += 1

        if feature_count == 0:
            logger.warning("No valid features found in target")
            return False

        entry = {
            "buckets": buckets_copy,
            "timestamp": time.time(),
            "track_id": self.target.track_id,
            "base_distance": self.target.base_distance,
        }

        self.enrolled_history.append(entry)

        if len(self.enrolled_history) > self.max_history_size:
            removed = self.enrolled_history.pop(0)
            logger.info(
                f"History full, removed oldest (track_id={removed.get('track_id')})"
            )

        clip_count = sum(
            1
            for b in buckets_copy.values()
            for d in b.values()
            if d.get("clip") is not None
        )
        logger.info(
            f"Saved to history: {feature_count} features ({clip_count} with CLIP) "
            f"across {len(buckets_copy)} buckets "
            f"(history size: {len(self.enrolled_history)})"
        )
        if clip_count == 0 and self.use_clip:
            logger.warning(
                "[HISTORY] WARNING: Saved entry has NO CLIP embeddings! "
                "This person cannot be reliably identified in history."
            )
        return True

    def _get_closest_bucket_features(
        self, buckets: dict, query_distance: float
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Get features from history entry at closest distance bucket.

        Parameters
        ----------
        history_entry : dict
            History entry containing 'buckets' dict with distance keys.
        query_distance : float
            Query distance in meters.

        Returns
        -------
        list of dict
            List of feature dicts with 'clothing', 'clip', 'bucket', 'direction' keys.
            Empty list if no features found.
        """
        if not buckets:
            return None, None

        # Find closest bucket by absolute distance difference
        closest_bucket = min(buckets.keys(), key=lambda b: abs(b - query_distance))

        directions = buckets.get(closest_bucket, {})
        approaching = directions.get("approaching")
        leaving = directions.get("leaving")

        return approaching, leaving

    def _is_in_history(
        self,
        clothing_feat: dict,
        clip_emb: Optional[np.ndarray],
        query_distance: float,
    ) -> Tuple[bool, float, float]:
        """
        Check if a person's features match anyone in history.

        When use_clip=True, requires BOTH clothing AND CLIP to pass for a
        definitive match.  If CLIP is unavailable on either side the comparison
        is treated as inconclusive (no match) rather than falling back to
        clothing-only, which would cause false positives.

        Parameters
        ----------
        clothing_feat : dict
            Clothing features to match.
        clip_emb : np.ndarray or None
            CLIP embedding to match (optional).
        query_distance : float
            Distance for bucket selection.

        Returns
        -------
        is_match : bool
            True if person matches any history entry.
        best_clothing_sim : float
            Highest clothing similarity found.
        best_clip_sim : float
            Highest CLIP similarity found (0.0 if not used).
        """
        if not self.enrolled_history:
            return False, 0.0, 0.0

        best_clothing_sim = 0.0
        best_clip_sim = 0.0

        for hist_entry in self.enrolled_history:
            buckets = hist_entry.get("buckets", {})

            # Legacy format support (single feature)
            if not buckets and "clothing" in hist_entry:
                buckets = {
                    2.0: {
                        "approaching": {
                            "clothing": hist_entry["clothing"],
                            "clip": hist_entry.get("clip"),
                        }
                    }
                }

            # Get closest bucket's features
            approaching, leaving = self._get_closest_bucket_features(
                buckets, query_distance
            )

            # Compare against approaching and leaving (max 2 comparisons per person)
            for direction_name, feat in [
                ("approaching", approaching),
                ("leaving", leaving),
            ]:
                if feat is None:
                    continue

                hist_clothing = feat.get("clothing")
                hist_clip = feat.get("clip")

                if hist_clothing is None:
                    continue

                # Clothing similarity
                c_sim = self.clothing_matcher.compute_clothing_similarity(
                    clothing_feat, hist_clothing
                )
                best_clothing_sim = max(best_clothing_sim, c_sim)

                # If clothing matches, check CLIP
                if c_sim >= self.clothing_threshold:
                    if self.use_clip:
                        # CLIP is enabled — require CLIP confirmation
                        if clip_emb is not None and hist_clip is not None:
                            clip_sim = self.clothing_matcher.compute_clip_similarity(
                                clip_emb, hist_clip
                            )
                            best_clip_sim = max(best_clip_sim, clip_sim)

                            if clip_sim >= self.clip_threshold:
                                logger.info(
                                    f"[HISTORY] MATCH: hist_track={hist_entry.get('track_id')} "
                                    f"C={c_sim:.3f} CLIP={clip_sim:.3f} ({direction_name})"
                                )
                                return True, best_clothing_sim, best_clip_sim
                            else:
                                logger.debug(
                                    f"[HISTORY] Clothing passed but CLIP failed: "
                                    f"hist_track={hist_entry.get('track_id')} "
                                    f"C={c_sim:.3f} CLIP={clip_sim:.3f} < {self.clip_threshold}"
                                )
                        else:
                            # CLIP unavailable on one or both sides — inconclusive.
                            # Do NOT fall back to clothing-only (causes false positives).
                            logger.debug(
                                f"[HISTORY] Clothing passed but CLIP unavailable: "
                                f"hist_track={hist_entry.get('track_id')} C={c_sim:.3f} "
                                f"(candidate_clip={'yes' if clip_emb is not None else 'NO'}, "
                                f"hist_clip={'yes' if hist_clip is not None else 'NO'}) — INCONCLUSIVE"
                            )
                    else:
                        # CLIP disabled — clothing match is sufficient
                        logger.info(
                            f"[HISTORY] MATCH (CLIP disabled): "
                            f"hist_track={hist_entry.get('track_id')} "
                            f"C={c_sim:.3f} ({direction_name})"
                        )
                        return True, best_clothing_sim, best_clip_sim

        return False, best_clothing_sim, best_clip_sim

    def set_max_history_size(self, new_size: int) -> dict:
        """
        Update maximum history size.

        If new size is smaller than current history, oldest entries are removed.

        Parameters
        ----------
        new_size : int
            New maximum history size (must be >= 1).

        Returns
        -------
        dict
            Result with 'old_size', 'new_size', 'current_count'.
        """
        if new_size < 1:
            raise ValueError("max_history_size must be >= 1")
        old = self.max_history_size
        self.max_history_size = new_size
        trimmed = 0
        if len(self.enrolled_history) > new_size:
            trimmed = len(self.enrolled_history) - new_size
            self.enrolled_history = self.enrolled_history[-new_size:]
        logger.info(f"max_history_size: {old} ’ {new_size}, trimmed {trimmed}")
        return {
            "old_size": old,
            "new_size": new_size,
            "trimmed_count": trimmed,
            "current_count": len(self.enrolled_history),
        }

    def clear_history(self, delete_file: bool = False) -> dict:
        """
        Clear enrolled history from memory.

        Parameters
        ----------
        delete_file : bool, optional
            If True, also delete the history file on disk, by default False.

        Returns
        -------
        dict
            Result with 'cleared_count' and 'file_deleted'.
        """
        count = len(self.enrolled_history)
        self.enrolled_history = []
        deleted = False
        if delete_file and self.history_file.exists():
            try:
                self.history_file.unlink()
                deleted = True
                logger.info(f"Deleted history file: {self.history_file}")
            except Exception as e:
                logger.warning(f"Failed to delete history file: {e}")
        logger.info(f"Cleared history ({count} entries)")
        return {"cleared_count": count, "file_deleted": deleted}

    def save_history(self, filepath=None) -> dict:
        """
        Save enrolled history to pickle file.

        Parameters
        ----------
        filepath : str, optional
            Override path for history file, by default None (uses configured path).

        Returns
        -------
        dict
            Result with 'saved', 'count', 'path' on success, or 'error' on failure.
        """
        path = Path(filepath) if filepath else self.history_file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            to_save = self.enrolled_history[-MAX_SAVE_HISTORY:]
            with open(path, "wb") as f:
                pickle.dump(
                    {"version": 2, "history": to_save, "saved_at": time.time()}, f
                )
            logger.info(f"History saved: {len(to_save)} entries’ {path}")
            return {"success": True, "filepath": str(path), "count": len(to_save)}
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
            return {"success": False, "error": str(e)}

    def load_history(self, filepath=None) -> dict:
        """
        Load enrolled history from pickle file.

        Parameters
        ----------
        filepath : str, optional
            Override path for history file, by default None (uses configured path).

        Returns
        -------
        dict
            Result with 'loaded', 'count', 'path' on success, or 'error'/'not_found' on failure.
        """
        path = Path(filepath) if filepath else self.history_file
        if not path.exists():
            logger.info(f"History file not found: {path} (starting fresh)")
            return {"success": True, "loaded_count": 0, "message": "file_not_found"}
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            all_hist = data["history"] if isinstance(data, dict) else data
            file_count = len(all_hist)
            self.enrolled_history = all_hist[-self.max_history_size :]
            logger.info(
                f"History loaded: {len(self.enrolled_history)}/{file_count} entries (max={self.max_history_size})"
            )
            return {
                "success": True,
                "loaded_count": len(self.enrolled_history),
                "file_count": file_count,
            }
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            self.enrolled_history = []
            return {"success": False, "error": str(e)}

    def get_history_count(self) -> int:
        """
        Get current number of entries in history.

        Returns
        -------
        int
            Number of persons stored in history.
        """
        return len(self.enrolled_history)

    def handle_greeting_acknowledged(self) -> dict:
        """
        Handle greeting acknowledgment from user.

        Saves current target to history and transitions to INACTIVE state.
        Only available in greeting mode.

        Returns
        -------
        dict
            Result with 'success', 'saved_to_history', 'history_size',
            or 'error' if in wrong mode/state.
        """
        # Only available in greeting mode
        if self.operation_mode != "greeting":
            logger.warning(
                "handle_greeting_acknowledged called in following mode - ignoring"
            )
            return {
                "saved": False,
                "history_size": 0,
                "status": self.target.status,
                "error": "not_available_in_following_mode",
            }

        saved = False
        if self.target.status != "INACTIVE":
            saved = self._save_current_target_to_history()
            if self.auto_save_history:
                self.save_history()
            logger.info(f"Greeting acknowledged - saved={saved}, going inactive")

        self.clear_target()

        return {
            "saved": saved,
            "history_size": len(self.enrolled_history),
            "status": "INACTIVE",
        }

    # Switch Target - Request (starts state machine)
    def request_switch_target(
        self,
        color_frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        aux: Optional[dict] = None,
    ) -> dict:
        """
        Request to switch to a different target (not in history).

        Clears current target, finds new candidates, and starts
        SWITCHING state. Does NOT save current target to history —
        history is only updated when greeting completes (greeting_ack).
        Only available in greeting mode.

        Parameters
        ----------
        color_frame : np.ndarray
            Current BGR frame for detection.
        depth_frame : np.ndarray, optional
            Depth frame for distance measurement.
        aux : dict, optional
            Auxiliary data (LiDAR).

        Returns
        -------
        dict
            Result with 'switch_started', 'candidates_count', 'saved_to_history',
            or 'error' if in wrong mode/state.
        """
        # Only available in greeting mode
        if self.operation_mode != "greeting":
            logger.warning("request_switch_target called in following mode - ignoring")
            return {
                "started": False,
                "reason": "not_available_in_following_mode",
            }

        timestamp = time.time()

        # Do NOT save current target to history here.
        # History is only updated via handle_greeting_acknowledged() after
        # a greeting is actually completed.  Saving here would pollute
        # history with un-greeted persons, causing false "all in history".

        # Clear current target
        self.clear_target()

        # Detect candidates
        detections = self.detector.detect(color_frame)
        H, W = color_frame.shape[:2]
        self.frame_width, self.frame_height = W, H
        tracks = self._run_tracker(detections, color_frame)

        candidates = []
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
            dist, _, _ = self._get_distance(bbox, depth_frame, aux)
            if dist < 0.3:
                continue
            candidates.append({"track_id": track_id, "bbox": bbox, "distance": dist})

        if not candidates:
            logger.warning("[SWITCH] No valid candidates found")
            return {
                "started": False,
                "reason": "no_candidates",
            }

        # Sort by distance (closest first)
        candidates.sort(key=lambda p: p["distance"])

        # Start switch state machine
        self.switch_state.start(candidates, timestamp)
        self.target.status = "SWITCHING"

        return {
            "started": True,
            "candidates_count": len(candidates),
            "history_size": len(self.enrolled_history),
        }

    # Process Frame - Main loop
    def process_frame(
        self,
        color_frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        aux: Optional[dict] = None,
    ) -> dict:
        """
        Process a single frame through the tracking pipeline.

        Routes processing based on current state (INACTIVE, TRACKING_ACTIVE,
        SEARCHING, SWITCHING).

        Parameters
        ----------
        color_frame : np.ndarray
            BGR image frame.
        depth_frame : np.ndarray, optional
            Depth image in millimeters (uint16).
        aux : dict, optional
            Auxiliary data (LiDAR projections).

        Returns
        -------
        dict
            Processing result containing:
            - timestamp : float
            - status : str
            - operation_mode : str
            - fps : float
            - num_detections : int
            - num_tracks : int
            - all_tracks : list
            - target_found : bool (if tracking/searching)
            - bbox : tuple (if target found)
            - distance : float (if target found)
            - Additional state-specific fields.
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
            "operation_mode": self.operation_mode,  # Include mode in result
            "fps": np.mean(self.fps_history) if self.fps_history else 0,
            "num_detections": len(detections),
            "num_tracks": len(tracks),
            "all_tracks": self.all_tracks,
        }

        # Only include history info in greeting mode
        if self.operation_mode == "greeting":
            result["history_size"] = len(self.enrolled_history)

        if aux:
            for k in ["lidar_uv", "lidar_ranges", "lidar_scan_idx"]:
                if k in aux:
                    result[k] = aux[k]

        # Route based on status
        if self.target.status == "TRACKING_ACTIVE":
            result.update(
                self._process_active_tracking(
                    tracks, color_frame, depth_frame, aux, current_time, H, W
                )
            )
        elif self.target.status == "SEARCHING":
            result.update(
                self._process_searching(
                    tracks, color_frame, depth_frame, aux, current_time, H, W
                )
            )
        elif self.target.status == "SWITCHING":
            # SWITCHING only happens in greeting mode
            result.update(
                self._process_switching(
                    tracks, color_frame, depth_frame, aux, current_time, H, W
                )
            )

        return result

    # Process Switching
    def _process_switching(
        self,
        tracks: np.ndarray,
        color_frame: np.ndarray,
        depth_frame: Optional[np.ndarray],
        aux: Optional[dict],
        timestamp: float,
        H: int,
        W: int,
    ) -> dict:
        """
        Process frame during SWITCHING mode.

        Logic:
        - match >= 1        skip immediately (in history)
        - 3s + match == 0   accept (new person)
        - 3s + valid == 0   skip (can't determine)

        Parameters
        ----------
        tracks : np.ndarray
            Current tracked objects.
        color_frame : np.ndarray
            BGR frame for feature extraction.
        depth_frame : np.ndarray or None
            Depth frame for distance.
        aux : dict or None
            Auxiliary LiDAR data.
        timestamp : float
            Current timestamp.
        H : int
            Frame height.
        W : int
            Frame width.

        Returns
        -------
        dict
            Switch state result with 'switch_active', 'success'/'skipped'/'checking',
            'current_bbox', 'time_remaining', etc.
        """
        ss = self.switch_state

        if not ss.active:
            self.target.status = "INACTIVE"
            return {"switch_active": False, "reason": "not_active"}

        # Update candidate bboxes with current track positions
        track_map = {t["track_id"]: t["bbox"] for t in self.all_tracks}
        for cand in ss.candidates:
            if cand["track_id"] in track_map:
                cand["bbox"] = track_map[cand["track_id"]]
                dist, _, _ = self._get_distance(cand["bbox"], depth_frame, aux)
                if dist > 0.1:
                    cand["distance"] = dist

        candidate = ss.get_current_candidate()

        if candidate is None:
            ss.stop()
            self.target.status = "INACTIVE"
            logger.warning("[SWITCH] All candidates exhausted")
            return {
                "switch_active": False,
                "success": False,
                "reason": "all_candidates_exhausted",
                "switch_summary": ss.get_summary(),
            }

        # Check timeout (3 seconds)
        if ss.is_timeout(timestamp):
            decision = ss.get_timeout_decision()

            if decision == "accept":
                logger.info(
                    f"[SWITCH] #{ss.current_candidate_idx} timeout, no match found "
                    f"(valid={ss.valid_check_count}), ACCEPTING!"
                )

                x1, y1, x2, y2 = candidate["bbox"]
                crop = color_frame[y1:y2, x1:x2]
                mask, clothing_feat, clip_emb, mask_coverage, _ = (
                    self._extract_features(crop)
                )

                return self._accept_switch_candidate(
                    candidate,
                    timestamp,
                    clothing_feat=clothing_feat,
                    clip_emb=clip_emb,
                    mask_coverage=mask_coverage,
                )

            else:  # skip_no_features
                logger.warning(
                    f"[SWITCH] #{ss.current_candidate_idx} timeout, no valid features, skipping"
                )
                has_more = ss.move_to_next_candidate(timestamp, "no_features")
                if not has_more:
                    ss.stop()
                    self.target.status = "INACTIVE"
                    return {
                        "switch_active": False,
                        "success": False,
                        "reason": "all_candidates_exhausted",
                        "switch_summary": ss.get_summary(),
                    }
                return {
                    "switch_active": True,
                    "skipped": True,
                    "reason": "no_features",
                    "switch_summary": ss.get_summary(),
                }

        # Throttle check (~3 Hz)
        if not ss.should_check_now(timestamp):
            return {
                "switch_active": True,
                "throttled": True,
                "current_candidate_idx": ss.current_candidate_idx,
                "current_track_id": candidate["track_id"],
                "current_bbox": candidate["bbox"],
                "time_remaining": ss.get_time_remaining(timestamp),
                "switch_summary": ss.get_summary(),
            }

        # Extract features
        x1, y1, x2, y2 = candidate["bbox"]
        crop = color_frame[y1:y2, x1:x2]

        mask, clothing_feat, clip_emb, mask_coverage, error_msg = (
            self._extract_features(crop)
        )

        features_valid = (
            clothing_feat is not None and mask_coverage >= self.min_mask_coverage - 10
        )

        if not features_valid:
            ss.record_check(timestamp, False, False)
            return {
                "switch_active": True,
                "feature_valid": False,
                "time_remaining": ss.get_time_remaining(timestamp),
                "switch_summary": ss.get_summary(),
            }

        # Check against history (using candidate's distance for closest bucket)
        is_match, c_sim, clip_sim = self._is_in_history(
            clothing_feat, clip_emb, candidate["distance"]
        )
        ss.record_check(timestamp, True, is_match, c_sim, clip_sim)

        clip_status = "yes" if clip_emb is not None else "NO"
        logger.info(
            f"[SWITCH] #{ss.current_candidate_idx} track={candidate['track_id']}: "
            f"C={c_sim:.3f} CLIP={clip_sim:.3f} clip_avail={clip_status} "
            f"{'MATCH!' if is_match else 'no match'} "
            f"(check {ss.valid_check_count}, time_left={ss.get_time_remaining(timestamp):.1f}s)"
        )

        # If ANY match found skip immediately
        if ss.should_skip_now():
            logger.info(
                f"[SWITCH] #{ss.current_candidate_idx} FOUND IN HISTORY, skipping"
            )
            has_more = ss.move_to_next_candidate(timestamp, "in_history")
            if not has_more:
                ss.stop()
                self.target.status = "INACTIVE"
                return {
                    "switch_active": False,
                    "success": False,
                    "reason": "all_in_history",
                    "switch_summary": ss.get_summary(),
                }
            return {
                "switch_active": True,
                "skipped": True,
                "reason": "in_history",
                "clothing_sim": c_sim,
                "switch_summary": ss.get_summary(),
            }

        # No match yet, continue checking
        return {
            "switch_active": True,
            "checking": True,
            "current_track_id": candidate["track_id"],
            "clothing_sim": c_sim,
            "is_match": False,
            "time_remaining": ss.get_time_remaining(timestamp),
            "switch_summary": ss.get_summary(),
        }

    def _accept_switch_candidate(
        self,
        candidate: dict,
        timestamp: float,
        clothing_feat: dict = None,
        clip_emb: np.ndarray = None,
        mask_coverage: float = 0.0,
    ) -> dict:
        """
        Accept a switch candidate as the new target.

        Initializes tracking on the candidate and saves provided features.

        Parameters
        ----------
        candidate : dict
            Candidate info with 'track_id', 'bbox', 'distance'.
        timestamp : float
            Current timestamp.
        clothing_feat : dict, optional
            Extracted clothing features to save.
        clip_emb : np.ndarray, optional
            CLIP embedding to save.
        mask_coverage : float, optional
            Mask coverage percentage.

        Returns
        -------
        dict
            Result with 'switch_active': False, 'success': True, 'track_id', 'distance'.
        """
        ss = self.switch_state

        # Initialize target
        self.target.initialize(candidate["track_id"], candidate["distance"], timestamp)

        # If we have features, save them
        if clothing_feat is not None:
            bucket = self.target._get_bucket(candidate["distance"])
            self.target.save_feature(
                bucket, "approaching", clothing_feat, clip_emb, mask_coverage, timestamp
            )

        ss.stop()
        self.target.status = "TRACKING_ACTIVE"

        # Auto save history
        if self.auto_save_history:
            self.save_history()

        logger.info("=" * 50)
        logger.info("[SWITCH] SUCCESS!")
        logger.info(f"  New target: track_id={candidate['track_id']}")
        logger.info(f"  Distance: {candidate['distance']:.2f}m")
        logger.info(f"  History size: {len(self.enrolled_history)}")
        logger.info(f"  Skipped (in history): {ss.skipped_in_history}")
        logger.info(f"  Skipped (no features): {ss.skipped_no_features}")
        logger.info("=" * 50)

        return {
            "switch_active": False,
            "success": True,
            "track_id": candidate["track_id"],
            "distance": candidate["distance"],
            "history_size": len(self.enrolled_history),
            "switch_summary": ss.get_summary(),
        }

    # Active Tracking
    def _process_active_tracking(
        self, tracks, color_frame, depth_frame, aux, timestamp, H, W
    ) -> dict:
        """
        Process frame during TRACKING_ACTIVE state.

        Updates target position, extracts and saves features at appropriate
        distance buckets, detects if target is lost.

        Parameters
        ----------
        tracks : np.ndarray
            Current tracked objects.
        color_frame : np.ndarray
            BGR frame for feature extraction.
        depth_frame : np.ndarray or None
            Depth frame for distance.
        aux : dict or None
            Auxiliary LiDAR data.
        timestamp : float
            Current timestamp.
        H : int
            Frame height.
        W : int
            Frame width.

        Returns
        -------
        dict
            Tracking result with 'target_found', 'bbox', 'distance', 'direction',
            'feature_saved', 'within_margin', 'lidar_cluster_pts', 'lidar_bbox_pts'.
        """
        target_track = None
        for track in tracks:
            if int(track[4]) == self.target.track_id:
                target_track = track
                break

        if target_track is None:
            self.target.mark_lost(timestamp)
            logger.info(f"Target lost (track_id={self.target.track_id})")
            return {"target_found": False}

        x1, y1, x2, y2 = map(int, target_track[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        bbox = (x1, y1, x2, y2)

        current_distance, cluster_pts, bbox_pts = self._get_distance(
            bbox, depth_frame, aux
        )
        direction = self.target.detect_movement_direction(current_distance, timestamp)

        should_save, bucket, save_dir = self.target.should_save_feature(
            current_distance, direction, timestamp, bbox, W, H
        )

        feature_saved = False
        if should_save:
            crop = color_frame[y1:y2, x1:x2]
            mask, clothing_feat, clip_emb, mask_coverage, _ = self._extract_features(
                crop
            )
            if clothing_feat:
                if self.target.save_feature(
                    bucket, save_dir, clothing_feat, clip_emb, mask_coverage, timestamp
                ):
                    feature_saved = True
                    logger.info(
                        f"Saved @{bucket:.1f}m [{save_dir}] mask:{mask_coverage:.1f}%"
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
            "lidar_cluster_pts": cluster_pts,
            "lidar_bbox_pts": bbox_pts,
        }

    # Searching
    def _process_searching(
        self, tracks, color_frame, depth_frame, aux, timestamp, H, W
    ) -> dict:
        """
        Process frame during SEARCHING state.

        Attempts to re-identify lost target using clothing and CLIP features.
        In greeting mode, times out after searching_timeout. In following mode,
        searches indefinitely.

        Parameters
        ----------
        tracks : np.ndarray
            Current tracked objects.
        color_frame : np.ndarray
            BGR frame for feature extraction.
        depth_frame : np.ndarray or None
            Depth frame for distance.
        aux : dict or None
            Auxiliary LiDAR data.
        timestamp : float
            Current timestamp.
        H : int
            Frame height.
        W : int
            Frame width.

        Returns
        -------
        dict
            Search result with 'target_found', 'time_lost', 'candidates_checked',
            'bbox'/'distance' if re-identified, 'timeout_inactive' if timed out.
        """
        # Search timeout only in greeting mode
        if self.operation_mode == "greeting":
            time_lost = self.target.get_time_lost(timestamp)
            if time_lost >= self.searching_timeout:
                logger.info(
                    f"Searching timeout ({time_lost:.1f}s >= {self.searching_timeout}s), "
                    "going inactive (not saving to history - person left before greeting)"
                )
                # Don't save to history - person walked away before greeting
                # They can be re-approached if they return
                self.clear_target()
                return {
                    "target_found": False,
                    "timeout_inactive": True,
                    "time_lost": time_lost,
                    "saved_to_history": False,
                }

        self.all_candidates_info = []
        candidates = []
        track_bbox_map = {}

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
            distance, _, _ = self._get_distance(bbox, depth_frame=depth_frame, aux=aux)
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

        time_since_last_search = timestamp - self.last_search_time
        should_run_search = time_since_last_search >= self.search_interval

        if not should_run_search and self.cached_search_result is not None:
            cached = self.cached_search_result
            for cand_info in cached.get("candidates_info", []):
                tid = cand_info["track_id"]
                if tid in track_bbox_map:
                    cand_info["bbox"] = track_bbox_map[tid]
                    self.all_candidates_info.append(cand_info)
            result = cached.copy()
            if (
                cached.get("target_found")
                and cached.get("matched_track_id") in track_bbox_map
            ):
                result["bbox"] = track_bbox_map[cached["matched_track_id"]]
            result["time_lost"] = self.target.get_time_lost(timestamp)
            result["throttled"] = True
            return result

        self.last_search_time = timestamp
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
                continue
            if clothing_feat is None:
                continue
            if mask_coverage < self.min_mask_coverage - 10:
                continue

            ref_features = self.target.get_bucket_features_both_directions(
                query_distance
            )
            if not ref_features:
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
                passed_clip = (
                    best_clip_sim >= self.clip_threshold if clip_available else False
                )
            else:
                passed_clip = True

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
                best = (
                    max(clip_available, key=lambda r: r["clip_sim"])
                    if clip_available
                    else max(stage1_passed, key=lambda r: r["clothing_sim"])
                )
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
        self.cached_search_result = None

        return {
            "target_found": True,
            "bbox": best["person"]["bbox"],
            "distance": best["person"]["distance"],
            "matched_track_id": best["person"]["track_id"],
            "stage": "verified",
            "clothing_sim": best["clothing_sim"],
            "clip_sim": best["clip_sim"],
            "bucket": best["closest_bucket"],
            "time_lost": self.target.get_time_lost(timestamp),
        }

    def enroll_target(self, color_frame, depth_frame=None, aux=None) -> bool:
        """
        Enroll the nearest valid person as tracking target.

        Finds the closest person within frame margins and initializes tracking.

        Parameters
        ----------
        color_frame : np.ndarray
            BGR frame for detection.
        depth_frame : np.ndarray, optional
            Depth frame for distance measurement.
        aux : dict, optional
            Auxiliary LiDAR data.

        Returns
        -------
        bool
            True if enrollment successful, False otherwise.
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
            distance, _, _ = self._get_distance(bbox, depth_frame=depth_frame, aux=aux)
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
            logger.info(
                f"Target enrolled: track_id={target['track_id']}, dist={target['distance']:.2f}m"
            )
            return True
        return False

    def clear_target(self) -> None:
        """Clear the current tracking target."""
        self.target = TargetState()
        self.target.FRAME_MARGIN_LR = 20
        self.target.MIN_MASK_COVERAGE = self.min_mask_coverage
        self.target.BUCKET_SPACING = self.bucket_spacing
        self.target.operation_mode = self.operation_mode
        logger.info("Target cleared")

    def get_status(self) -> dict:
        """
        Get current system status.

        Returns
        -------
        dict
            Status containing 'status', 'track_id', 'operation_mode',
            'history_size' (greeting mode only).
        """
        status = {
            "status": self.target.status,
            "operation_mode": self.operation_mode,
            "track_id": self.target.track_id,
            "features": self.target.get_total_features(),
            "quality": self.target.get_quality_summary(),
            "approach_distance": self.approach_distance,
        }

        # Only include greeting-specific info in greeting mode
        if self.operation_mode == "greeting":
            status["history_size"] = len(self.enrolled_history)
            status["max_history_size"] = self.max_history_size
            status["switch_active"] = self.switch_state.active
            status["switch_summary"] = (
                self.switch_state.get_summary() if self.switch_state.active else None
            )
            status["searching_timeout"] = self.searching_timeout

        return status  # FIX: Added missing return statement

    def get_all_tracks(self) -> List[dict]:
        """
        Get all currently tracked persons.

        Returns
        -------
        list of dict
            List of track info with 'track_id' and 'bbox' for each person.
        """
        return self.all_tracks

    def get_candidates_info(self) -> List[dict]:
        """
        Get candidate information from last search operation.

        Returns
        -------
        list of dict
            List of candidate info with similarity scores and bounding boxes.
        """
        return self.all_candidates_info
