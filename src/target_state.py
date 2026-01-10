"""
Target State Management - OpenCLIP Version.

Stores clothing features and OpenCLIP embeddings for person re-identification.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TargetState:
    """
    State management for tracked target person.

    Manages the state and feature storage for a tracked person, including
    distance-bucketed clothing features and OpenCLIP embeddings for
    re-identification. Features are matched by closest distance bucket.

    Parameters
    ----------
    WALKING_SPEED_THRESHOLD : float
        Speed threshold in m/s to detect movement, by default 0.4.
    TIME_WINDOW : float
        Time window in seconds for movement detection, by default 0.167.
    BUCKET_SPACING : float
        Distance bucket spacing in meters, by default 0.5.
    MIN_MASK_COVERAGE : float
        Minimum mask coverage percentage for saving features, by default 35.0.
    MIN_MASK_COVERAGE_FOR_MATCH : float
        Minimum mask coverage percentage for matching, by default 30.0.
    FRAME_MARGIN_LR : int
        Left/right frame margin in pixels, by default 20.

    Attributes
    ----------
    track_id : int or None
        Current tracker ID of the target.
    base_distance : float
        Initial enrollment distance used as reference for bucketing.
    distance_history : deque
        Rolling history of (timestamp, distance) tuples.
    features : dict
        Nested dictionary storing features by bucket and direction.
        Structure: {bucket: {direction: {'clothing': dict, 'clip': array, 'mask_coverage': float}}}
    status : str
        Current tracking status ('INACTIVE', 'TRACKING_ACTIVE', 'SEARCHING').
    frames_tracked : int
        Total frames where target was successfully tracked.
    frames_lost : int
        Consecutive frames where target was lost.
    time_lost_start : float or None
        Timestamp when target was first lost.
    """

    # === Core Parameters ===
    WALKING_SPEED_THRESHOLD = 0.4  # m/s
    TIME_WINDOW = 0.167  # seconds
    DISTANCE_THRESHOLD = WALKING_SPEED_THRESHOLD * TIME_WINDOW

    # Feature storage - 0.5m spacing
    BUCKET_SPACING = 0.5  # meters

    # Quality thresholds
    MIN_MASK_COVERAGE = 35.0
    MIN_MASK_COVERAGE_FOR_MATCH = 30.0

    # Frame margin
    FRAME_MARGIN_LR = 20  # pixels

    # === State Variables ===
    track_id: Optional[int] = None
    base_distance: float = 0.0

    distance_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=10)
    )

    # Features: {bucket: {direction: {'clothing': dict, 'clip': array, 'mask_coverage': float}}}
    features: Dict[float, Dict[str, Optional[dict]]] = field(default_factory=dict)

    last_saved_distance: float = 0.0
    last_saved_direction: Optional[str] = None
    last_saved_timestamp: float = 0.0

    status: str = "INACTIVE"
    frames_tracked: int = 0
    frames_lost: int = 0
    time_lost_start: Optional[float] = None
    feature_extraction_failures: int = 0

    def initialize(self, track_id: int, distance: float, timestamp: float):
        """
        Initialize target state at enrollment.

        Sets up the initial state when a new target is enrolled, including
        the track ID, base distance for bucketing, and initial feature storage.

        Parameters
        ----------
        track_id : int
            Tracker ID assigned to the target.
        distance : float
            Initial distance to target in meters.
        timestamp : float
            Enrollment timestamp.
        """
        self.track_id = track_id
        self.base_distance = distance
        self.distance_history.append((timestamp, distance))
        self.status = "TRACKING_ACTIVE"
        self.last_saved_timestamp = timestamp
        self.feature_extraction_failures = 0
        self.features = {}

        bucket = self._get_bucket(distance)
        self.features[bucket] = {"approaching": None, "leaving": None}

    def _get_bucket(self, distance: float) -> float:
        """
        Get bucket distance for a given distance.

        Calculates which distance bucket a given distance falls into,
        based on the base distance and bucket spacing.

        Parameters
        ----------
        distance : float
            Distance in meters to bucket.

        Returns
        -------
        float
            Bucket distance in meters.
        """
        offset = distance - self.base_distance
        bucket_index = round(offset / self.BUCKET_SPACING)
        return self.base_distance + bucket_index * self.BUCKET_SPACING

    def _get_closest_bucket(self, target_distance: float) -> Optional[float]:
        """
        Find the closest bucket that has features.

        Searches through all buckets with stored features and returns
        the one closest to the target distance.

        Parameters
        ----------
        target_distance : float
            Distance in meters to find closest bucket for.

        Returns
        -------
        float or None
            Closest bucket distance in meters, or None if no features exist.
        """
        if not self.features:
            return None

        valid_buckets = []
        for bucket, directions in self.features.items():
            if any(d is not None for d in directions.values()):
                valid_buckets.append(bucket)

        if not valid_buckets:
            return None

        closest = min(valid_buckets, key=lambda b: abs(b - target_distance))
        return closest

    def is_within_frame_margin(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        margin_lr: int = None,
    ) -> Tuple[bool, str]:
        """
        Check if bounding box is within valid frame margins.

        Determines whether a detection is too close to the frame edges,
        which could indicate partial visibility.

        Parameters
        ----------
        bbox : tuple of int
            Bounding box coordinates (x1, y1, x2, y2).
        frame_width : int
            Frame width in pixels.
        frame_height : int
            Frame height in pixels.
        margin_lr : int, optional
            Left/right margin override in pixels. If None, uses FRAME_MARGIN_LR.

        Returns
        -------
        is_valid : bool
            True if bounding box is within valid margins.
        reason : str
            Reason string ('within_margin', 'left_edge_cut', or 'right_edge_cut').
        """
        x1, y1, x2, y2 = bbox
        margin = margin_lr if margin_lr is not None else self.FRAME_MARGIN_LR

        if x1 < margin:
            return False, "left_edge_cut"
        if x2 > (frame_width - margin):
            return False, "right_edge_cut"
        return True, "within_margin"

    def detect_movement_direction(
        self, current_distance: float, current_timestamp: float
    ) -> Optional[str]:
        """
        Detect movement direction based on distance history.

        Analyzes recent distance measurements to determine if the target
        is approaching or leaving, based on speed threshold.

        Parameters
        ----------
        current_distance : float
            Current distance to target in meters.
        current_timestamp : float
            Current timestamp.

        Returns
        -------
        str or None
            Movement direction ('approaching', 'leaving'), or None if
            movement is below threshold or insufficient data.
        """
        self.distance_history.append((current_timestamp, current_distance))

        if len(self.distance_history) < 2:
            return None

        cutoff_time = current_timestamp - self.TIME_WINDOW
        recent_samples = [
            (ts, dist) for ts, dist in self.distance_history if ts >= cutoff_time
        ]

        if len(recent_samples) < 2:
            return None

        oldest_ts, oldest_dist = recent_samples[0]
        newest_ts, newest_dist = recent_samples[-1]

        time_elapsed = newest_ts - oldest_ts
        if time_elapsed < 0.05:
            return None

        dist_change = newest_dist - oldest_dist
        speed = abs(dist_change) / time_elapsed

        if speed < self.WALKING_SPEED_THRESHOLD:
            return None

        return "leaving" if dist_change > 0 else "approaching"

    def should_save_feature(
        self,
        current_distance: float,
        current_direction: Optional[str],
        current_timestamp: float,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[bool, float, str]:
        """
        Determine if a feature should be saved.

        Checks various conditions to decide whether to save a new feature,
        including direction validity, frame margins, bucket availability,
        and time since last save.

        Parameters
        ----------
        current_distance : float
            Current distance to target in meters.
        current_direction : str or None
            Current movement direction ('approaching', 'leaving', or None).
        current_timestamp : float
            Current timestamp.
        bbox : tuple of int
            Bounding box coordinates (x1, y1, x2, y2).
        frame_width : int
            Frame width in pixels.
        frame_height : int
            Frame height in pixels.

        Returns
        -------
        should_save : bool
            True if feature should be saved.
        bucket : float
            Target bucket distance in meters.
        direction : str
            Movement direction for the feature.
        """
        if current_direction is None:
            return False, 0.0, ""

        is_valid, _ = self.is_within_frame_margin(bbox, frame_width, frame_height)
        if not is_valid:
            return False, 0.0, ""

        bucket = self._get_bucket(current_distance)

        if bucket not in self.features:
            self.features[bucket] = {"approaching": None, "leaving": None}

        if self.features[bucket][current_direction] is not None:
            return False, bucket, current_direction

        time_since_save = current_timestamp - self.last_saved_timestamp
        if time_since_save < 0.3:
            return False, bucket, current_direction

        return True, bucket, current_direction

    def save_feature(
        self,
        bucket_distance: float,
        direction: str,
        clothing_feature: dict,
        clip_embedding: np.ndarray,
        mask_coverage: float,
        timestamp: float,
    ) -> bool:
        """
        Save feature with quality check.

        Stores clothing features and CLIP embeddings for a specific distance
        bucket and movement direction, with minimum mask coverage validation.

        Parameters
        ----------
        bucket_distance : float
            Distance bucket in meters.
        direction : str
            Movement direction ('approaching' or 'leaving').
        clothing_feature : dict
            Clothing feature dictionary containing Lab histograms.
        clip_embedding : numpy.ndarray
            CLIP embedding vector.
        mask_coverage : float
            Mask coverage percentage (0-100).
        timestamp : float
            Save timestamp.

        Returns
        -------
        bool
            True if feature was saved successfully, False if rejected.
        """
        if mask_coverage < self.MIN_MASK_COVERAGE:
            logger.warning(
                f"Feature rejected: mask {mask_coverage:.1f}% < {self.MIN_MASK_COVERAGE}%"
            )
            return False

        if bucket_distance not in self.features:
            self.features[bucket_distance] = {"approaching": None, "leaving": None}

        if self.features[bucket_distance][direction] is None:
            self.features[bucket_distance][direction] = {
                "clothing": clothing_feature,
                "clip": clip_embedding,
                "mask_coverage": mask_coverage,
            }
            self.last_saved_timestamp = timestamp
            self.feature_extraction_failures = 0
            return True

        return False

    def get_closest_bucket_features(
        self, query_distance: float
    ) -> Tuple[Optional[dict], Optional[np.ndarray], float, float]:
        """
        Get features from the closest bucket to query distance.

        Retrieves the best quality features from the bucket closest to
        the specified distance.

        Parameters
        ----------
        query_distance : float
            Distance in meters to query features for.

        Returns
        -------
        clothing_feature : dict or None
            Clothing feature dictionary, or None if not available.
        clip_embedding : numpy.ndarray or None
            CLIP embedding vector, or None if not available.
        mask_coverage : float
            Mask coverage percentage of the returned features.
        bucket_distance : float
            Distance of the bucket the features came from.
        """
        closest_bucket = self._get_closest_bucket(query_distance)

        if closest_bucket is None:
            return None, None, 0.0, 0.0

        directions = self.features.get(closest_bucket, {})

        best_data = None
        best_coverage = 0.0

        for direction, data in directions.items():
            if data is not None:
                coverage = data.get("mask_coverage", 0.0)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_data = data

        if best_data is None:
            return None, None, 0.0, closest_bucket

        return (
            best_data.get("clothing"),
            best_data.get("clip"),
            best_coverage,
            closest_bucket,
        )

    def get_bucket_features_both_directions(self, query_distance: float) -> List[dict]:
        """
        Get all features from the closest bucket for both directions.

        Retrieves features from both approaching and leaving directions
        from the bucket closest to the query distance.

        Parameters
        ----------
        query_distance : float
            Distance in meters to query features for.

        Returns
        -------
        list of dict
            List of feature dictionaries, each containing:
            - 'clothing' : dict
                Clothing feature dictionary.
            - 'clip' : numpy.ndarray or None
                CLIP embedding vector.
            - 'mask_coverage' : float
                Mask coverage percentage.
            - 'direction' : str
                Movement direction ('approaching' or 'leaving').
            - 'bucket' : float
                Bucket distance in meters.
        """
        closest_bucket = self._get_closest_bucket(query_distance)

        if closest_bucket is None:
            return []

        directions = self.features.get(closest_bucket, {})
        result = []

        for direction, data in directions.items():
            if (
                data is not None
                and data.get("mask_coverage", 0) >= self.MIN_MASK_COVERAGE_FOR_MATCH
            ):
                result.append(
                    {
                        "clothing": data.get("clothing"),
                        "clip": data.get("clip"),
                        "mask_coverage": data.get("mask_coverage", 0),
                        "direction": direction,
                        "bucket": closest_bucket,
                    }
                )

        return result

    def get_all_features_flat(self) -> List[dict]:
        """
        Get all features as a flat list.

        Retrieves all stored features across all buckets and directions
        as a flat list for debugging or inspection.

        Returns
        -------
        list of dict
            List of feature info dictionaries, each containing:
            - 'bucket' : float
                Bucket distance in meters.
            - 'direction' : str
                Movement direction.
            - 'mask_coverage' : float
                Mask coverage percentage.
            - 'has_clothing' : bool
                Whether clothing features are present.
            - 'has_clip' : bool
                Whether CLIP embedding is present.
        """
        result = []
        for bucket, directions in self.features.items():
            for direction, data in directions.items():
                if data is not None:
                    result.append(
                        {
                            "bucket": bucket,
                            "direction": direction,
                            "mask_coverage": data.get("mask_coverage", 0),
                            "has_clothing": data.get("clothing") is not None,
                            "has_clip": data.get("clip") is not None,
                        }
                    )
        return result

    def mark_lost(self, current_timestamp: float):
        """
        Mark target as lost and transition to search mode.

        Updates the state to indicate the target has been lost and
        records the time when tracking was lost.

        Parameters
        ----------
        current_timestamp : float
            Timestamp when target was lost.
        """
        self.frames_lost += 1
        self.status = "SEARCHING"
        if self.time_lost_start is None:
            self.time_lost_start = current_timestamp

    def resume_tracking(self, new_track_id: int):
        """
        Resume tracking with a new track ID.

        Called when the target is re-identified, updates the track ID
        and resets loss-related counters.

        Parameters
        ----------
        new_track_id : int
            New tracker ID assigned to the re-identified target.
        """
        self.track_id = new_track_id
        self.status = "TRACKING_ACTIVE"
        self.frames_lost = 0
        self.time_lost_start = None
        self.feature_extraction_failures = 0

    def get_total_features(self) -> int:
        """
        Get total number of stored features.

        Returns
        -------
        int
            Total count of features across all buckets and directions.
        """
        return sum(
            1
            for dirs in self.features.values()
            for data in dirs.values()
            if data is not None
        )

    def get_time_lost(self, current_timestamp: float) -> float:
        """
        Get duration since target was lost.

        Parameters
        ----------
        current_timestamp : float
            Current timestamp.

        Returns
        -------
        float
            Time in seconds since target was lost, or 0.0 if not lost.
        """
        if self.time_lost_start is None:
            return 0.0
        return current_timestamp - self.time_lost_start

    def get_quality_summary(self) -> str:
        """
        Get summary string of stored features.

        Returns
        -------
        str
            Human-readable summary including feature count, CLIP count,
            distance range, and mask coverage range.
        """
        features = self.get_all_features_flat()
        if not features:
            return "No features"

        coverages = [f["mask_coverage"] for f in features]
        buckets = sorted(set(f["bucket"] for f in features))

        bucket_str = (
            f"{min(buckets):.1f}-{max(buckets):.1f}m"
            if len(buckets) > 1
            else f"{buckets[0]:.1f}m"
        )

        clip_count = sum(1 for f in features if f.get("has_clip"))

        return f"F:{len(features)} CLIP:{clip_count} @{bucket_str} M:{min(coverages):.0f}-{max(coverages):.0f}%"

    def get_buckets_info(self) -> str:
        """
        Get detailed information about all feature buckets.

        Returns
        -------
        str
            Multi-line string with bucket details including distance,
            direction, mask coverage, and CLIP availability.
        """
        lines = []
        for bucket in sorted(self.features.keys()):
            dirs = self.features[bucket]
            parts = []
            for d in ["approaching", "leaving"]:
                if dirs.get(d) is not None:
                    cov = dirs[d].get("mask_coverage", 0)
                    has_clip = "Y" if dirs[d].get("clip") is not None else "N"
                    parts.append(f"{d[0].upper()}:{cov:.0f}%{has_clip}")
            if parts:
                lines.append(f"  {bucket:.1f}m: {', '.join(parts)}")
        return "\n".join(lines) if lines else "  No features"
