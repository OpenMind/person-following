"""
Switch Target State Machine.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SwitchState:
    """
    State machine for switch_target operation.

    Timing:
    - switch_interval = 0.33s → ~3 Hz → ~9 checks in 3 seconds
    - candidate_timeout = 3.0s → confirmation window per candidate

    Decision:
    - match_votes >= 1      → SKIP (found in history)
    - timeout + match == 0  → ACCEPT (not found in history)
    - timeout + no valid    → SKIP (can't determine)

    Parameters
    ----------
    candidates : list of dict
        List of candidate persons with 'track_id', 'bbox', 'distance'.
    timestamp : float
        Start timestamp.
    """

    # Configuration
    switch_interval: float = 0.33  # ~3 Hz
    candidate_timeout: float = 3.0  # 3 seconds per candidate

    # State
    active: bool = False
    candidates: List[dict] = field(default_factory=list)
    current_candidate_idx: int = 0
    candidate_start_time: float = 0.0
    last_check_time: float = 0.0

    # Current candidate stats
    check_count: int = 0  # total attempts
    valid_check_count: int = 0  # successful feature extractions
    match_votes: int = 0  # times matched to history (1 is enough to skip!)

    # Best similarity (for logging)
    best_clothing_sim: float = 0.0
    best_clip_sim: float = 0.0

    # Overall statistics
    total_candidates_checked: int = 0
    skipped_in_history: int = 0
    skipped_no_features: int = 0

    def start(self, candidates: List[dict], timestamp: float) -> None:
        """
        Start switch operation with candidate list.

        Parameters
        ----------
        candidates : list of dict
            List of candidate persons with 'track_id', 'bbox', 'distance'.
        timestamp : float
            Start timestamp.
        """
        self.active = True
        self.candidates = candidates
        self.current_candidate_idx = 0
        self.candidate_start_time = timestamp
        self.last_check_time = 0.0
        self._reset_current_stats()
        self._reset_overall_stats()
        logger.info(f"[SWITCH] Started with {len(candidates)} candidates")

    def _reset_current_stats(self) -> None:
        """
        Reset stats for checking a new candidate.
        """
        self.check_count = 0
        self.valid_check_count = 0
        self.match_votes = 0
        self.best_clothing_sim = 0.0
        self.best_clip_sim = 0.0

    def _reset_overall_stats(self):
        """
        Reset overall switch operation statistics.

        Called when starting a new switch operation.
        """
        self.total_candidates_checked = 0
        self.skipped_in_history = 0
        self.skipped_no_features = 0

    def stop(self):
        """
        Stop switch operation and reset state.
        """
        self.active = False
        self.candidates = []
        self.current_candidate_idx = 0

    def get_current_candidate(self) -> Optional[dict]:
        """
        Get current candidate being evaluated.

        Returns
        -------
        dict or None
            Current candidate info, or None if no active switch.
        """
        if not self.active or self.current_candidate_idx >= len(self.candidates):
            return None
        return self.candidates[self.current_candidate_idx]

    def should_check_now(self, timestamp: float) -> bool:
        """
        Check if enough time passed for next feature extraction (~3 Hz).

        Parameters
        ----------
        timestamp : float
            Current timestamp.

        Returns
        -------
        bool
            True if should perform feature check.
        """
        if not self.active:
            return False
        return (timestamp - self.last_check_time) >= self.switch_interval

    def is_timeout(self, timestamp: float) -> bool:
        """
        Check if current candidate has timed out.

        Parameters
        ----------
        timestamp : float
            Current timestamp.

        Returns
        -------
        bool
            True if candidate timeout exceeded.
        """
        return (timestamp - self.candidate_start_time) >= self.candidate_timeout

    def get_time_remaining(self, timestamp: float) -> float:
        """
        Get time remaining for current candidate.

        Parameters
        ----------
        timestamp : float
            Current timestamp.

        Returns
        -------
        float
            Seconds remaining before timeout.
        """
        return max(0, self.candidate_timeout - (timestamp - self.candidate_start_time))

    def record_check(
        self,
        timestamp: float,
        features_valid: bool,
        is_match: bool,
        clothing_sim: float = 0.0,
        clip_sim: float = 0.0,
    ):
        """
        Record result of a feature check.

        Parameters
        ----------
        timestamp : float
            Check timestamp.
        features_valid : bool
            Whether features were successfully extracted.
        is_match : bool
            Whether candidate matched history.
        clothing_sim : float, optional
            Clothing similarity score, by default 0.0.
        clip_sim : float, optional
            CLIP similarity score, by default 0.0.
        """
        self.last_check_time = timestamp
        self.check_count += 1

        if features_valid:
            self.valid_check_count += 1
            self.best_clothing_sim = max(self.best_clothing_sim, clothing_sim)
            self.best_clip_sim = max(self.best_clip_sim, clip_sim)

            if is_match:
                self.match_votes += 1

    def should_skip_now(self) -> bool:
        """
        Check if current candidate should be skipped immediately.

        Returns
        -------
        bool
            True if any check found match in history.
        """
        return self.match_votes >= 1

    def get_timeout_decision(self) -> str:
        """
        Get decision when candidate times out.

        Returns
        -------
        str
            'accept' if valid checks found no match,
            'skip_no_features' if no valid feature checks.
        """
        if self.valid_check_count == 0:
            return "skip_no_features"
        # Had valid checks but no match found
        return "accept"

    def move_to_next_candidate(self, timestamp: float, reason: str) -> bool:
        """
        Move to next candidate in list.

        Parameters
        ----------
        timestamp : float
            Current timestamp.
        reason : str
            Reason for skipping ('in_history' or 'no_features').

        Returns
        -------
        bool
            True if more candidates available, False if exhausted.
        """
        self.total_candidates_checked += 1

        if reason == "in_history":
            self.skipped_in_history += 1
        elif reason == "no_features":
            self.skipped_no_features += 1

        candidate = self.get_current_candidate()
        if candidate:
            logger.info(
                f"[SWITCH] Skipped #{self.current_candidate_idx} "
                f"(track_id={candidate.get('track_id')}, reason={reason}, "
                f"checks={self.check_count}, valid={self.valid_check_count}, "
                f"match={self.match_votes}, C={self.best_clothing_sim:.3f})"
            )

        self.current_candidate_idx += 1
        self.candidate_start_time = timestamp
        self._reset_current_stats()

        has_more = self.current_candidate_idx < len(self.candidates)

        if has_more:
            next_cand = self.candidates[self.current_candidate_idx]
            logger.info(
                f"[SWITCH] Now checking #{self.current_candidate_idx} "
                f"(track_id={next_cand.get('track_id')}, dist={next_cand.get('distance', 0):.2f}m)"
            )

        return has_more

    def get_summary(self) -> dict:
        """
        Get summary of current switch operation.

        Returns
        -------
        dict
            Summary with candidate index, check counts, skip counts, etc.
        """
        return {
            "active": self.active,
            "total_candidates": len(self.candidates),
            "current_idx": self.current_candidate_idx,
            "checked": self.total_candidates_checked,
            "skipped_in_history": self.skipped_in_history,
            "skipped_no_features": self.skipped_no_features,
            "current_checks": self.check_count,
            "current_valid": self.valid_check_count,
            "current_match": self.match_votes,
        }
