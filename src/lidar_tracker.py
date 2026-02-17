"""
LiDAR Cluster Tracker with Kalman Filter.

Modular LiDAR tracking component for person-following system.
Provides KF-based cluster tracking as a fallback when camera bbox is lost,
and drift detection when both sources are available.

Integration
-----------
Called from PersonFollowingSystem.process_frame() via the LidarTracker class.
Only active for camera sources with LiDAR (rtsp, go2 — NOT realsense).

Architecture
------------
Camera (YOLO+BoxMOT) is PRIMARY — determines WHO is being tracked.
LiDAR KF is SECONDARY — determines WHERE the tracked person is when camera
loses the bbox (turned sideways, partial occlusion, etc).

    Camera has bbox + LiDAR cluster  →  camera drives tracking, LiDAR syncs
    Camera lost bbox, LiDAR has KF   →  LiDAR drives position, status stays TRACKING_ACTIVE
    Camera lost bbox, LiDAR lost KF  →  SEARCHING
    LiDAR sudden jump detected       →  re-grab from bbox (if available) or SEARCHING

Race Conditions
---------------
No threading issues — LidarTracker is always called from the same thread
as PersonFollowingSystem.process_frame(). The HTTP command server only
enqueues commands; clear/switch/enroll are processed in the main thread.

Coordinate Frame
----------------
LiDAR operates in the scanner's XY plane (x=forward, y=left).
KF state is [x, y, vx, vy] in this frame.
Distance = sqrt(x² + y²) from the robot.
Lateral offset = y (positive = left of robot).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ── Kalman Filter ─────────────────────────────────────────────────────────


class KF2D:
    """
    Constant-velocity Kalman Filter in 2D.

    State vector: [x, y, vx, vy]
    Measurement:  [x, y]

    Tuned for pedestrian tracking at ~10-30Hz LiDAR update rate.
    Process noise scales with dt for consistent behavior across frame rates.
    """

    def __init__(self):
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag([0.5, 0.5, 1.0, 1.0])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.Q_base = np.diag([0.05, 0.05, 1.0, 1.0])
        self.R = np.diag([0.0009, 0.0009])  # 3cm std measurement noise
        self.ok = False
        self._t: Optional[float] = None

    def predict(self, t: float) -> np.ndarray:
        """Predict state to time t. Returns predicted state."""
        dt = max(0.01, min(t - self._t, 1.0)) if self._t else 0.1
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q_base * dt
        return self.x.copy()

    def update(self, z: np.ndarray, t: float) -> np.ndarray:
        """
        Update KF with measurement z=[x,y] at time t.

        On first call, initializes state directly (no prediction).
        On subsequent calls, predicts then updates.
        """
        z = np.asarray(z, dtype=np.float64)
        if not self.ok:
            self.x[:2] = z
            self.x[2:] = 0.0
            self.ok = True
            self._t = t
            return self.x.copy()
        self.predict(t)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self._t = t
        return self.x.copy()

    @property
    def pos(self) -> Tuple[float, float]:
        """Current position (x, y) in LiDAR frame."""
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> Tuple[float, float]:
        """Current velocity (vx, vy) in LiDAR frame."""
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        """Current speed magnitude in m/s."""
        return math.sqrt(self.x[2] ** 2 + self.x[3] ** 2)

    @property
    def distance(self) -> float:
        """Distance from origin (robot) in meters."""
        return math.sqrt(self.x[0] ** 2 + self.x[1] ** 2)


# ── Calibration ───────────────────────────────────────────────────────────


def load_intrinsics(path: str) -> Tuple[float, float, float, float, int, int]:
    """Load camera intrinsics from YAML. Returns (fx, fy, cx, cy, w, h)."""
    with open(path) as f:
        d = yaml.safe_load(f)
    K = np.array(d["camera_matrix"], dtype=np.float64)
    w, h = d.get("image_size", [640, 480])
    return (
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
        int(w),
        int(h),
    )


def load_extrinsics(path: str) -> Tuple[float, float, float, float, float, float]:
    """Load LiDAR-camera extrinsics from YAML. Returns (tx,ty,tz, roll,pitch,yaw)."""
    with open(path) as f:
        d = yaml.safe_load(f)
    t, r = d.get("translation", {}), d.get("rotation_euler", {})
    return (
        float(t.get("x", 0)),
        float(t.get("y", 0)),
        float(t.get("z", 0)),
        float(r.get("roll", 0)),
        float(r.get("pitch", 0)),
        float(r.get("yaw", 0)),
    )


def euler_to_R(rx: float, ry: float, rz: float) -> np.ndarray:
    """Euler angles (roll, pitch, yaw) to 3x3 rotation matrix."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (
        np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    )


# ── LiDAR Cluster Utilities ──────────────────────────────────────────────


def scan_to_xy(scan) -> np.ndarray:
    """
    Convert LaserScan to XY points in LiDAR frame.

    Parameters
    ----------
    scan : sensor_msgs/LaserScan
        ROS LaserScan message.

    Returns
    -------
    np.ndarray
        (N, 2) array of [x, y] points. Empty (0, 2) if no valid points.
    """
    r = np.array(scan.ranges, dtype=np.float64)
    a = scan.angle_min + np.arange(len(r), dtype=np.float64) * scan.angle_increment
    ok = np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max)
    if not np.any(ok):
        return np.empty((0, 2))
    return np.column_stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])])


def project_scan(scan, shape, R_CL, t_CL, fx, fy, cx, cy) -> Optional[dict]:
    """
    Project LaserScan into camera image.

    Returns dict with:
        uv:  (N, 2) int pixel coords
        idx: (N,)   scan indices
        r:   (N,)   ranges
        xy:  (N, 2) XY coords in LiDAR frame
    """
    H, W = shape
    r_full = np.array(scan.ranges, dtype=np.float64)
    n = len(r_full)
    if n == 0:
        return None
    ang = scan.angle_min + np.arange(n, dtype=np.float64) * scan.angle_increment
    ok = np.isfinite(r_full) & (r_full > scan.range_min) & (r_full < scan.range_max)
    if not np.any(ok):
        return None
    idx = np.where(ok)[0].astype(np.int32)
    r, a = r_full[idx], ang[idx]
    xl, yl = r * np.cos(a), r * np.sin(a)
    P_L = np.vstack([xl, yl, np.zeros_like(xl)])
    P_C = R_CL.T @ (P_L - t_CL)
    pts = np.vstack([-P_C[1], -P_C[2], P_C[0]])
    fwd = pts[2] > 0.01
    if not np.any(fwd):
        return None
    pts, idx, r, xl, yl = pts[:, fwd], idx[fwd], r[fwd], xl[fwd], yl[fwd]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    uvw = K @ pts
    uv = (uvw[:2] / uvw[2:3]).T.astype(np.int32)
    vis = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not np.any(vis):
        return None
    return {
        "uv": uv[vis],
        "idx": idx[vis],
        "r": r[vis],
        "xy": np.column_stack([xl[vis], yl[vis]]),
    }


def split_clusters(sidx, rng, jump=0.35, min_sz=4) -> List[np.ndarray]:
    """Split sorted scan indices into clusters by range discontinuity."""
    if sidx.size == 0:
        return []
    breaks = (
        np.where((np.diff(sidx.astype(np.int64)) > 1) | (np.abs(np.diff(rng)) > jump))[
            0
        ]
        + 1
    )
    return [s for s in np.split(np.arange(len(sidx)), breaks) if len(s) >= min_sz]


def cluster_in_bbox(bbox, aux, jump=0.35, min_sz=4) -> Optional[dict]:
    """
    Extract the closest LiDAR cluster within a camera bounding box.

    Parameters
    ----------
    bbox : tuple
        (x1, y1, x2, y2) pixel coordinates.
    aux : dict
        Projected scan data from project_scan().
    jump : float
        Range discontinuity threshold for clustering.
    min_sz : int
        Minimum cluster size.

    Returns
    -------
    dict or None
        Cluster info with keys: c (centroid), pts (points), rng (median range), n (count).
    """
    uv, si, rng, xy = aux["uv"], aux["idx"], aux["r"], aux["xy"]
    x1, y1, x2, y2 = bbox
    m = (uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2)
    if not np.any(m):
        return None
    si2 = si[m].astype(np.int32)
    r2 = rng[m].astype(np.float64)
    xy2 = xy[m]
    ok = np.isfinite(r2) & (r2 > 0.1) & (r2 < 20.0)
    si2, r2, xy2 = si2[ok], r2[ok], xy2[ok]
    if si2.size == 0:
        return None
    o = np.argsort(si2)
    si2, r2, xy2 = si2[o], r2[o], xy2[o]
    cs = split_clusters(si2, r2, jump, min_sz)
    if not cs:
        if len(xy2) >= 2:
            return {
                "c": np.mean(xy2, 0),
                "pts": xy2,
                "rng": float(np.median(r2)),
                "n": len(xy2),
            }
        return None
    best = min(cs, key=lambda c: float(np.median(r2[c])))
    p = xy2[best]
    return {
        "c": np.mean(p, 0),
        "pts": p,
        "rng": float(np.median(r2[best])),
        "n": len(best),
    }


def find_nearest_cluster(
    all_xy, kf_pos, jump=0.35, min_sz=3, radius=1.0
) -> Optional[dict]:
    """
    Find the nearest LiDAR cluster to the KF predicted position.

    Used when camera bbox is lost — searches all scan points for the
    cluster closest to where the KF predicts the person should be.

    Parameters
    ----------
    all_xy : np.ndarray
        (N, 2) all scan points in LiDAR XY frame.
    kf_pos : array-like
        [x, y] KF predicted position.
    jump : float
        Distance threshold for cluster splitting.
    min_sz : int
        Minimum cluster size.
    radius : float
        Search radius around KF position.

    Returns
    -------
    dict or None
        Cluster info: c, pts, rng, n.
    """
    if all_xy.size == 0:
        return None
    kf_pos = np.asarray(kf_pos)
    d = np.linalg.norm(all_xy - kf_pos, axis=1)
    near = d < radius
    if not np.any(near):
        return None
    pts = all_xy[near]
    # Sort by angle for clustering
    ang = np.arctan2(pts[:, 1] - kf_pos[1], pts[:, 0] - kf_pos[0])
    pts_s = pts[np.argsort(ang)]
    gaps = np.linalg.norm(np.diff(pts_s, axis=0), axis=1)
    breaks = np.where(gaps > jump)[0] + 1
    clusters = [
        pts_s[s] for s in np.split(np.arange(len(pts_s)), breaks) if len(s) >= min_sz
    ]
    if not clusters:
        if len(pts) >= min_sz:
            return {
                "c": np.mean(pts, 0),
                "pts": pts,
                "rng": float(np.median(np.linalg.norm(pts, axis=1))),
                "n": len(pts),
            }
        return None
    best = min(clusters, key=lambda c: np.linalg.norm(np.mean(c, 0) - kf_pos))
    return {
        "c": np.mean(best, 0),
        "pts": best,
        "rng": float(np.median(np.linalg.norm(best, axis=1))),
        "n": len(best),
    }


def compute_bbox_overlap_ratio(
    cluster_pts: np.ndarray, aux: dict, bbox: tuple
) -> float:
    """
    Compute fraction of cluster points that project inside the bbox.

    Used for drift detection: if most cluster points project outside
    the person's bbox, the tracker has drifted to a different object.

    Parameters
    ----------
    cluster_pts : np.ndarray
        (N, 2) cluster XY points in LiDAR frame.
    aux : dict
        Projected scan data with 'uv' and 'xy' keys.
    bbox : tuple
        (x1, y1, x2, y2) pixel coordinates.

    Returns
    -------
    float
        Ratio [0.0, 1.0] of cluster points inside bbox.
    """
    if cluster_pts is None or len(cluster_pts) == 0:
        return 0.0
    a_xy = aux["xy"]
    a_uv = aux["uv"]
    if len(a_xy) == 0:
        return 0.0

    # Match cluster points to projected points (within 5cm)
    diff = cluster_pts[:, None, :] - a_xy[None, :, :]
    d2 = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2
    # For each cluster point, find closest projected point
    min_d2 = np.min(d2, axis=1)
    matched_idx = np.argmin(d2, axis=1)

    # Only consider points that have a close match in projected set
    valid_match = min_d2 < 0.0025  # 5cm²
    if not np.any(valid_match):
        return 0.0

    matched_uv = a_uv[matched_idx[valid_match]]
    x1, y1, x2, y2 = bbox
    inside = (
        (matched_uv[:, 0] >= x1)
        & (matched_uv[:, 0] < x2)
        & (matched_uv[:, 1] >= y1)
        & (matched_uv[:, 1] < y2)
    )
    return float(np.sum(inside)) / float(np.sum(valid_match))


# ── LiDAR Tracker ─────────────────────────────────────────────────────────


@dataclass
class LidarTrackResult:
    """Result from a single LidarTracker.update() call."""

    active: bool = False
    source: str = "none"  # "lidar" | "camera_synced" | "none"
    position_xy: Optional[Tuple[float, float]] = None  # LiDAR frame (x, y)
    distance: float = 0.0  # meters from robot
    lateral_offset: float = 0.0  # y in LiDAR frame (positive = left)
    speed: float = 0.0  # m/s
    cluster_pts: int = 0  # number of LiDAR points in cluster
    cluster_xy: Optional[np.ndarray] = None  # (K,2) XY of tracked cluster
    predict_only: bool = False  # KF predicted but no measurement
    frames_lost: int = 0  # consecutive frames without cluster
    drift_detected: bool = False  # cluster drifted away from bbox
    jump_detected: bool = False  # sudden position jump
    bbox_overlap: float = 1.0  # fraction of cluster pts inside bbox


class LidarTracker:
    """
    LiDAR cluster tracker with Kalman Filter for person following.

    Lifecycle
    ---------
    1. enroll(bbox, aux)     — init KF from cluster inside bbox
    2. update(scan_xy, aux, bbox) — per-frame update, returns LidarTrackResult
    3. clear()               — reset everything

    Drift / Jump Detection
    ----------------------
    - Sudden jump: distance between consecutive KF positions > jump_threshold
    - Drift:       <10% of cluster points project inside person bbox
    - Both trigger re-grab from bbox (if available) or signal loss

    Parameters
    ----------
    range_jump : float
        Range discontinuity for clustering, default 0.35m.
    min_cluster_size : int
        Minimum cluster points, default 4.
    search_radius : float
        KF search radius for nearest cluster, default 1.0m.
    max_lost_frames : int
        Frames without cluster before declaring lost, default 30.
    jump_threshold : float
        Max allowed position jump per frame in meters, default 0.5m.
    drift_overlap_threshold : float
        Min bbox overlap ratio before drift is flagged, default 0.1.
    """

    def __init__(
        self,
        range_jump: float = 0.35,
        min_cluster_size: int = 4,
        search_radius: float = 1.0,
        max_lost_frames: int = 30,
        jump_threshold: float = 0.5,
        drift_overlap_threshold: float = 0.1,
    ):
        self.range_jump = range_jump
        self.min_cluster_size = min_cluster_size
        self.search_radius = search_radius
        self.max_lost_frames = max_lost_frames
        self.jump_threshold = jump_threshold
        self.drift_overlap_threshold = drift_overlap_threshold

        self._kf = KF2D()
        self._enrolled = False
        self._lost_frames = 0
        self._last_cluster: Optional[dict] = None
        self._prev_pos: Optional[Tuple[float, float]] = None
        self._last_source: str = "none"  # last tracking source for viz

    @property
    def is_active(self) -> bool:
        """Whether the tracker has been enrolled and not cleared."""
        return self._enrolled and self._kf.ok

    @property
    def position(self) -> Optional[Tuple[float, float]]:
        """Current KF position (x, y) or None if not active."""
        return self._kf.pos if self.is_active else None

    @property
    def distance(self) -> float:
        """Distance from robot in meters."""
        return self._kf.distance if self.is_active else 0.0

    @property
    def speed(self) -> float:
        """Speed in m/s."""
        return self._kf.speed if self.is_active else 0.0

    def enroll(self, bbox: tuple, aux: dict, timestamp: float = None) -> bool:
        """
        Initialize KF from the closest LiDAR cluster inside a bbox.

        Parameters
        ----------
        bbox : tuple
            (x1, y1, x2, y2) pixel coordinates of person.
        aux : dict
            Projected scan data from project_scan().
        timestamp : float, optional
            Current time. Defaults to time.time().

        Returns
        -------
        bool
            True if cluster found and KF initialized.
        """
        if aux is None:
            return False
        t = timestamp or time.time()
        cluster = cluster_in_bbox(bbox, aux, self.range_jump, self.min_cluster_size)
        if cluster is None:
            logger.debug("[LIDAR] enroll failed: no cluster in bbox")
            return False

        self._kf = KF2D()
        self._kf.update(cluster["c"], t)
        self._enrolled = True
        self._lost_frames = 0
        self._last_cluster = cluster
        self._prev_pos = tuple(cluster["c"])
        logger.info(
            f"[LIDAR] Enrolled: {cluster['n']}pts "
            f"({cluster['c'][0]:.2f}, {cluster['c'][1]:.2f}) "
            f"range={cluster['rng']:.2f}m"
        )
        return True

    def clear(self):
        """Reset tracker to inactive state."""
        self._kf = KF2D()
        self._enrolled = False
        self._lost_frames = 0
        self._last_cluster = None
        self._prev_pos = None
        self._last_source = "none"
        logger.info("[LIDAR] Cleared")

    def update(
        self,
        scan_xy: np.ndarray,
        aux: Optional[dict],
        bbox: Optional[tuple] = None,
        timestamp: float = None,
    ) -> LidarTrackResult:
        """
        Per-frame tracker update.

        Decision tree:
        1. If bbox available + aux available:
           - Extract cluster from bbox → sync KF (camera_synced)
           - Check drift: if <10% overlap → re-grab from bbox
        2. If bbox NOT available:
           - find_nearest_cluster(kf_pos) → KF update (lidar)
           - Check for sudden jump → signal loss
        3. If no cluster found either way → predict only, increment lost

        Parameters
        ----------
        scan_xy : np.ndarray
            (N, 2) all scan XY points from scan_to_xy().
        aux : dict or None
            Projected scan data (needed for bbox-based cluster extraction
            and drift overlap computation).
        bbox : tuple or None
            (x1, y1, x2, y2) camera bbox of tracked person.
            None means camera has lost the person.
        timestamp : float, optional
            Current time. Defaults to time.time().

        Returns
        -------
        LidarTrackResult
            Tracking result with position, source, drift/jump flags.
        """
        if not self._enrolled:
            return LidarTrackResult()

        t = timestamp or time.time()
        result = LidarTrackResult(active=True)
        cluster = None
        jump_detected = False
        drift_detected = False

        # ── Path A: Camera bbox available → extract cluster from bbox ──
        if bbox is not None and aux is not None:
            cluster = cluster_in_bbox(bbox, aux, self.range_jump, self.min_cluster_size)

            if cluster is not None:
                # Check drift BEFORE updating KF — only one update per frame
                overlap = compute_bbox_overlap_ratio(cluster["pts"], aux, bbox)
                result.bbox_overlap = overlap

                if overlap < self.drift_overlap_threshold:
                    # Cluster has drifted — most points are outside bbox.
                    # Re-extract strictly from bbox before touching the KF.
                    drift_detected = True
                    logger.warning(
                        f"[LIDAR] Drift detected: only {overlap:.0%} of "
                        f"{cluster['n']} cluster pts inside bbox — re-grabbing"
                    )
                    fresh = cluster_in_bbox(
                        bbox, aux, self.range_jump, max(2, self.min_cluster_size - 1)
                    )
                    if fresh is not None:
                        cluster = fresh
                    # else: keep original cluster, it's all we have

                # Check for sudden position jump
                jump_detected = self._check_jump(cluster["c"])
                if jump_detected:
                    logger.warning(
                        f"[LIDAR] Jump detected with bbox present — re-syncing. "
                        f"Prev=({self._prev_pos[0]:.2f},{self._prev_pos[1]:.2f}) → "
                        f"New=({cluster['c'][0]:.2f},{cluster['c'][1]:.2f})"
                    )

                # Single KF update with the vetted measurement
                self._kf.update(cluster["c"], t)
                self._prev_pos = self._kf.pos
                self._lost_frames = 0
                self._last_cluster = cluster
                result.source = "camera_synced"
            else:
                # Bbox exists but no LiDAR cluster inside it
                # This is OK — might be too far, or LiDAR gap
                # Fall through to KF predict-only
                pass

        # ── Path B: No camera bbox → find nearest cluster to KF prediction ──
        if cluster is None and self._kf.ok:
            kp = self._kf.predict(t)
            cluster = find_nearest_cluster(
                scan_xy,
                kp[:2],
                self.range_jump,
                max(2, self.min_cluster_size - 1),
                self.search_radius,
            )

            if cluster is not None:
                # Check for sudden jump
                jump_detected = self._check_jump(cluster["c"])

                if jump_detected:
                    # Jump with NO bbox → can't verify → signal loss
                    logger.warning(
                        f"[LIDAR] Jump detected WITHOUT bbox — cluster unreliable. "
                        f"Prev=({self._prev_pos[0]:.2f},{self._prev_pos[1]:.2f}) → "
                        f"Cluster=({cluster['c'][0]:.2f},{cluster['c'][1]:.2f})"
                    )
                    self._lost_frames += 1
                    result.source = "none"
                    result.predict_only = True
                    result.jump_detected = True
                    result.frames_lost = self._lost_frames

                    if self._lost_frames > self.max_lost_frames:
                        logger.info(
                            f"[LIDAR] Lost {self._lost_frames}f after jump → auto-clear"
                        )
                        self.clear()
                        result.active = False
                    # Position from prediction, not the jumped cluster
                    result.position_xy = self._kf.pos
                    result.distance = self._kf.distance
                    result.lateral_offset = self._kf.pos[1]
                    result.speed = self._kf.speed
                    self._last_source = result.source
                    return result

                # Valid cluster, no jump
                self._kf.update(cluster["c"], t)
                self._prev_pos = self._kf.pos
                self._lost_frames = 0
                self._last_cluster = cluster
                result.source = "lidar"
            else:
                # No cluster found anywhere — KF predict only
                self._lost_frames += 1
                result.source = "none"
                result.predict_only = True

                if self._lost_frames > self.max_lost_frames:
                    logger.info(f"[LIDAR] Lost {self._lost_frames}f → auto-clear")
                    self.clear()
                    result.active = False
                    self._last_source = result.source
                    return result

        # Fill result
        result.position_xy = self._kf.pos
        result.distance = self._kf.distance
        result.lateral_offset = self._kf.pos[1]
        result.speed = self._kf.speed
        result.frames_lost = self._lost_frames
        result.drift_detected = drift_detected
        result.jump_detected = jump_detected
        result.cluster_pts = cluster["n"] if cluster else 0
        result.cluster_xy = cluster["pts"] if cluster else None
        self._last_source = result.source

        return result

    def _check_jump(self, new_pos) -> bool:
        """
        Detect sudden position jump.

        A jump is when the distance between consecutive positions exceeds
        the threshold. This indicates occlusion, ID switch, or passerby.
        """
        if self._prev_pos is None:
            return False
        dx = new_pos[0] - self._prev_pos[0]
        dy = new_pos[1] - self._prev_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        return dist > self.jump_threshold

    def get_position_for_publish(self) -> Optional[dict]:
        """
        Get position data formatted for publishing to /tracked_person/position.

        The position uses the convention expected by the follower node:
        - x: lateral offset (positive = right in robot frame)
          In LiDAR frame y is left, so x_publish = -y_lidar
        - z: forward distance
          In LiDAR frame x is forward, so z_publish = x_lidar

        Returns
        -------
        dict or None
            Position dict with 'x', 'z', 'distance', 'speed', 'source',
            or None if tracker not active.
        """
        if not self.is_active:
            return None
        lx, ly = self._kf.pos
        return {
            "x": -ly,  # lateral: LiDAR y (left+) → robot x (right+)
            "z": lx,  # forward: LiDAR x → robot z
            "distance": self._kf.distance,
            "speed": self._kf.speed,
            "source": "lidar",
        }

    def get_status(self) -> dict:
        """Get current tracker status for debugging/logging/visualization."""
        if not self.is_active:
            return {
                "lidar_active": False,
                "lidar_source": "none",
                "lidar_cluster_xy": None,
                "lidar_kf_position_xy": None,
            }
        lx, ly = self._kf.pos
        return {
            "lidar_active": True,
            "lidar_x": round(lx, 3),
            "lidar_y": round(ly, 3),
            "lidar_distance": round(self._kf.distance, 3),
            "lidar_speed": round(self._kf.speed, 3),
            "lidar_lost_frames": self._lost_frames,
            "lidar_cluster_pts": self._last_cluster["n"] if self._last_cluster else 0,
            # Visualization data
            "lidar_cluster_xy": (
                self._last_cluster["pts"] if self._last_cluster else None
            ),
            "lidar_kf_position_xy": (lx, ly),
            "lidar_source": self._last_source,
        }
