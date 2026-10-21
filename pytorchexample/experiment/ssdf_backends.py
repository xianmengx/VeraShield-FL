"""Backend adapters for apples-to-apples SSDF comparisons.

The adapters are deliberately isolated from the current Flower strategy.  They
let validation code feed identical model deltas to the plaintext detector and
the logical two-server detector without changing the existing training path.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .logical_dual_server_ssdf import LogicalDualServerSSDF
from .unified_logical_ssdf import UnifiedLogicalSSDF
from .ssdf import SSDFDetector


@dataclass(frozen=True)
class BackendDetection:
    client_id: int
    round_id: int
    pass_flag: bool
    risk: float
    dynamic_weight: float
    reason: str
    norm_ok: bool = True
    reference_ok: bool = True
    temporal_ok: bool = True
    cohort_ok: bool = True
    raw_norm: float = float("nan")
    norm_zscore: float = float("nan")
    reference_cosine: float = float("nan")
    temporal_cosine: float = float("nan")
    anchor_stability: float = float("nan")
    persistent_score: float = float("nan")
    persistent_ok: bool = True
    persistent_threshold: float = float("nan")
    round_reference_cosine: float = float("nan")
    round_reference_threshold: float = float("nan")
    cohort_cosine: float = float("nan")
    cohort_threshold: float = float("nan")
    anomaly_streak: int = 0
    history_updated: bool = False
    evidence_diagnostics: dict = field(default_factory=dict)


def _flatten_delta(delta: Mapping[str, Any]) -> np.ndarray:
    values = []
    for value in delta.values():
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        values.append(tensor.to(torch.float64).reshape(-1).cpu().numpy())
    if not values:
        raise ValueError("delta must be non-empty")
    vector = np.concatenate(values)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("delta must contain finite values")
    return vector


class PlaintextSSDFBackend:
    """Adapter around the existing plaintext SSDF-v2 implementation."""

    def __init__(self, **kwargs: Any) -> None:
        self.detector = SSDFDetector(**kwargs)

    def detect_round(self, updates: Sequence[Mapping[str, Any]]) -> list[BackendDetection]:
        payload = [
            {
                **item,
                "delta": {
                    key: value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                    for key, value in item["delta"].items()
                },
            }
            for item in updates
        ]
        results = self.detector.detect_round(payload)
        return [
            BackendDetection(
                client_id=int(result.client_id),
                round_id=int(result.round),
                pass_flag=bool(result.pass_flag),
                risk=float(result.risk),
                dynamic_weight=float(np.clip(1.0 - result.risk, 0.0, 1.0))
                if result.pass_flag else 0.0,
                reason=str(result.decision_reason),
                norm_ok=bool(result.secjudge_ok),
                reference_ok=bool(result.reference_ok),
                temporal_ok=bool(result.temporal_ok),
                cohort_ok=bool(getattr(result, "cohort_ok", True)),
                raw_norm=float(result.raw_norm),
                norm_zscore=float(result.norm_zscore),
                reference_cosine=float(result.reference_cosine),
                temporal_cosine=float(result.cosine),
            )
            for result in results
        ]


class LogicalDualServerSSDFBackend:
    """Adapter around the logical fixed-point two-server SSDF prototype."""

    def __init__(self, **kwargs: Any) -> None:
        policy = kwargs.pop("evidence_policy", "legacy")
        if policy not in {"legacy", "unified_v1_candidate", "unified_v2_candidate"}:
            raise ValueError("unknown evidence_policy")
        self.evidence_policy = policy
        detector = UnifiedLogicalSSDF if policy != "legacy" else LogicalDualServerSSDF
        if policy != "legacy":
            kwargs["evidence_policy"] = policy
        self.detector = detector(**kwargs)

    def aggregation_reference(self) -> np.ndarray | None:
        """Expose the clean cohort anchor for robust dynamic weighting."""
        return self.detector.aggregation_reference()

    def detect_round(self, updates: Sequence[Mapping[str, Any]]) -> list[BackendDetection]:
        bundles = []
        for item in updates:
            # SSDF's plaintext convention detects g=-delta.  Apply the same
            # convention here so both backends see identical vectors.
            vector = -_flatten_delta(item["delta"])
            bundles.append(
                self.detector.make_bundle(
                    client_id=int(item["client_id"]),
                    round_id=int(item["server_round"]),
                    update=vector,
                )
            )
        results = self.detector.detect_round(bundles)
        return [
            BackendDetection(
                client_id=int(result.client_id),
                round_id=int(result.round_id),
                pass_flag=bool(result.pass_flag),
                risk=float(result.risk),
                dynamic_weight=float(result.dynamic_weight),
                reason=str(result.reason),
                norm_ok=bool(result.norm_ok),
                reference_ok=bool(result.reference_ok),
                temporal_ok=bool(result.temporal_ok),
                cohort_ok=bool(result.cohort_ok),
                raw_norm=float(result.raw_norm),
                norm_zscore=float(result.norm_zscore),
                reference_cosine=float(result.reference_cosine),
                temporal_cosine=float(result.temporal_cosine),
                persistent_score=float(result.persistent_score),
                persistent_ok=bool(result.persistent_ok),
                persistent_threshold=float(result.persistent_threshold),
                round_reference_cosine=float(result.round_reference_cosine),
                round_reference_threshold=float(result.round_reference_threshold),
                cohort_cosine=float(result.cohort_cosine),
                cohort_threshold=float(result.cohort_threshold),
                anomaly_streak=int(result.anomaly_streak),
                history_updated=bool(result.history_updated),
                evidence_diagnostics=dict(result.evidence_diagnostics),
                anchor_stability=float(self.detector.last_anchor_stability),
            )
            for result in results
        ]
