"""
Switch Target State Machine.

Manages candidate iteration during switch_target operation.
Each candidate is checked synchronously (single frame) against history:
  - features valid + match    → SKIP (in history)
  - features valid + no match → ACCEPT (new person)
  - features invalid          → SKIP (can't verify)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SwitchState:
    """
    State machine for switch_target candidate iteration.

    Iterates through candidates sorted by distance. Each candidate is
    checked against history in a single frame (synchronous). No timeout
    or multi-frame voting needed.
    """

    # State
    active: bool = False
    candidates: List[dict] = field(default_factory=list)
    current_candidate_idx: int = 0

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
            Start timestamp (for logging).
        """
        self.active = True
        self.candidates = candidates
        self.current_candidate_idx = 0
        self.total_candidates_checked = 0
        self.skipped_in_history = 0
        self.skipped_no_features = 0
        logger.info(f"[SWITCH] Started with {len(candidates)} candidates")

    def stop(self):
        """Stop switch operation and reset state."""
        self.active = False
        self.candidates = []
        self.current_candidate_idx = 0

    def get_current_candidate(self) -> Optional[dict]:
        """
        Get current candidate being evaluated.

        Returns
        -------
        dict or None
            Current candidate info, or None if no active switch or exhausted.
        """
        if not self.active or self.current_candidate_idx >= len(self.candidates):
            return None
        return self.candidates[self.current_candidate_idx]

    def move_to_next_candidate(self, timestamp: float, reason: str) -> bool:
        """
        Skip current candidate and move to next.

        Parameters
        ----------
        timestamp : float
            Current timestamp (for logging).
        reason : str
            Reason for skipping ('in_history', 'no_features', 'lost_track',
            'previous_target').

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
                f"(track_id={candidate.get('track_id')}, reason={reason})"
            )

        self.current_candidate_idx += 1
        has_more = self.current_candidate_idx < len(self.candidates)

        if has_more:
            next_cand = self.candidates[self.current_candidate_idx]
            logger.info(
                f"[SWITCH] Now checking #{self.current_candidate_idx} "
                f"(track_id={next_cand.get('track_id')}, "
                f"dist={next_cand.get('distance', 0):.2f}m)"
            )

        return has_more

    def get_summary(self) -> dict:
        """
        Get summary of current switch operation.

        Returns
        -------
        dict
            Summary with candidate counts and skip reasons.
        """
        return {
            "active": self.active,
            "total_candidates": len(self.candidates),
            "current_idx": self.current_candidate_idx,
            "checked": self.total_candidates_checked,
            "skipped_in_history": self.skipped_in_history,
            "skipped_no_features": self.skipped_no_features,
        }
