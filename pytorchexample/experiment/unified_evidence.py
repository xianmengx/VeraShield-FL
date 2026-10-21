"""Attack-independent evidence fusion, shared by the detector and CSV replay."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
from dataclasses import dataclass
import math


EVIDENCE_FIELDS = (
    "evidence_policy", "evidence_norm", "evidence_reference", "evidence_temporal",
    "evidence_cohort", "evidence_direction", "evidence_instantaneous",
    "evidence_previous", "evidence_accumulated", "evidence_strong_threshold",
    "evidence_persistent_threshold", "evidence_updated", "temporal_available",
    "cohort_available", "temporal_threshold", "anchor_admitted", "confidence",
    "reference_center", "reference_scale", "cohort_center", "cohort_scale",
    "temporal_center", "temporal_scale",
    "evidence_beta", "evidence_immediate_policy", "temporal_history_age",
    "temporal_baseline_updated", "temporal_reset_reason", "cohort_size",
    "cohort_required", "anchor_age", "anchor_update_reason",
    "direction_hard_gate",
)


def clip(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("evidence must be finite")
    return min(1.0, max(0.0, value))


def direction_evidence(cosine: float, center: float, threshold: float) -> float:
    """Zero at/above center; unit evidence at the actual rejection boundary."""
    center = max(center, threshold + 1e-6)
    return clip((center - cosine) / (center - threshold))


@dataclass(frozen=True)
class EvidenceDecision:
    instantaneous: float
    previous: float
    accumulated: float
    direction: float
    accepted: bool
    confidence: float
    reason: str


class UnifiedEvidenceState:
    def __init__(self, beta=0.8, strong_threshold=1.0, persistent_threshold=0.8,
                 immediate_policy="all_channels"):
        if not 0 <= beta < 1:
            raise ValueError("beta must lie in [0, 1)")
        if not 0 < persistent_threshold <= strong_threshold <= 1:
            raise ValueError("require 0 < persistent <= strong <= 1")
        self.beta = float(beta)
        self.strong_threshold = float(strong_threshold)
        self.persistent_threshold = float(persistent_threshold)
        if immediate_policy not in {"all_channels", "norm_only"}:
            raise ValueError("unknown immediate evidence policy")
        self.immediate_policy = immediate_policy
        self.scores: dict[int, float] = {}

    def observe(self, client_id, norm, reference, temporal=None, cohort=None):
        available = [clip(x) for x in (temporal, cohort) if x is not None]
        direction = max(clip(reference), min(available)) if available else clip(reference)
        instantaneous = max(clip(norm), direction)
        previous = self.scores.get(client_id, 0.0)
        accumulated = self.beta * previous + (1 - self.beta) * instantaneous
        # Rejected observations update evidence, never trusted vector history.
        self.scores[client_id] = accumulated
        strong = (clip(norm) if self.immediate_policy == "norm_only" else instantaneous) >= self.strong_threshold
        persistent = accumulated >= self.persistent_threshold
        accepted = not (strong or persistent)
        reason = ";".join(x for x, flag in (
            ("strong_evidence", strong), ("persistent_evidence", persistent)
        ) if flag) or "accepted"
        confidence = ((1 - clip(norm)) * (1 - 0.5 * direction) * (1 - accumulated)
                      if self.immediate_policy == "norm_only"
                      else (1 - instantaneous) * (1 - accumulated)) if accepted else 0.0
        return EvidenceDecision(
            instantaneous, previous, accumulated, direction, accepted,
            confidence, reason,
        )
