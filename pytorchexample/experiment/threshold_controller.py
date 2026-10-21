"""Runtime Dual-Shamir threshold controller (Step 5.7).

Picks the active share set based on how many shares are available. This is NOT a
modification of any Shamir polynomial degree; it only selects the pre-existing
A=Shamir(10,8) or B=Shamir(10,6) share set.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

from dataclasses import dataclass


@dataclass
class ThresholdDecision:
    mode: str                 # 'normal' | 'emergency' | 'failure'
    active_share_set: str | None   # 'A' | 'B' | None
    active_threshold: int | None
    switch_event: bool


class ThresholdController:
    def __init__(
        self,
        normal_threshold: int = 8,
        emergency_threshold: int = 6,
        static_threshold: int | None = None,
    ) -> None:
        self.normal_threshold = normal_threshold
        self.emergency_threshold = emergency_threshold
        # When set, the controller always stays in normal mode / uses A at this
        # fixed threshold (Static t=8 baseline for the Dynamic-D2 comparison). This
        # is a pure experiment-config override; it does not touch any Shamir degree.
        self.static_threshold = static_threshold
        self._prev_mode: str | None = None

    def decide(self, available_shares: int) -> ThresholdDecision:
        if self.static_threshold is not None:
            # Static baseline: never switch; reuse the previous-mode gate so that a
            # mode change is never reported (mode is constant).
            switch_event = self._prev_mode is not None and "normal" != self._prev_mode
            self._prev_mode = "normal"
            return ThresholdDecision("normal", "A", self.static_threshold, switch_event)
        if available_shares >= self.normal_threshold:
            mode, active, thr = "normal", "A", self.normal_threshold
        elif available_shares >= self.emergency_threshold:
            mode, active, thr = "emergency", "B", self.emergency_threshold
        else:
            mode, active, thr = "failure", None, None
        # A mode *change* is a switch event. Always False on the very first call so
        # round 1 does not spuriously report that we "switched" from nothing.
        switch_event = self._prev_mode is not None and mode != self._prev_mode
        self._prev_mode = mode
        return ThresholdDecision(mode, active, thr, switch_event)
