"""Dynamic Weight aggregation (Step 5.9).

Given the set of updates that passed SSDF, compute a per-client weight from the
cosine similarity of each update to the round's mean (reference) update, then
aggregate with those weights instead of FedAvg's num-example weights.

cos is invariant to the g = -delta global sign convention, so the same value is
obtained whether the strategy passes the update ``delta`` or ``g``.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import numpy as np
from flwr.app import ArrayRecord
import torch


def calculate_cosine_similarity(
    vecs: list[np.ndarray], reference: np.ndarray | None = None
) -> list[float]:
    """Compute per-update cosine against an explicit or round-mean reference."""
    if not vecs:
        return []
    arr = np.stack([np.asarray(v, dtype=float).ravel() for v in vecs])
    ref = arr.mean(axis=0) if reference is None else np.asarray(reference, dtype=float).ravel()
    if ref.shape != arr.shape[1:]:
        raise ValueError("reference dimension must match flattened updates")
    nref = float(np.linalg.norm(ref))
    out = []
    for v in arr:
        nv = float(np.linalg.norm(v))
        if nv <= 1e-12 or nref <= 1e-12:
            out.append(0.0)
        else:
            out.append(float(np.dot(v, ref) / (nv * nref)))
    return out


def calculate_dynamic_weights(cos: list[float]) -> list[float]:
    """lambda_i = max(0, cos_i) / sum_j max(0, cos_j); zero if sum is zero."""
    if not cos:
        return []
    alpha = [max(0.0, float(c)) for c in cos]
    s = float(sum(alpha))
    if s <= 1e-12:
        return [0.0] * len(cos)
    return [a / s for a in alpha]


def calculate_bounded_dynamic_weights(
    cos: list[float], *, max_weight_multiplier: float = 2.0
) -> list[float]:
    """Normalize positive cosine scores while bounding one client's influence."""
    if not cos:
        return []
    n = len(cos)
    if max_weight_multiplier < 1.0:
        raise ValueError("max_weight_multiplier must be at least 1")
    scores = np.maximum(0.0, np.asarray(cos, dtype=np.float64))
    if not np.isfinite(scores).all():
        raise ValueError("cosine scores must be finite")
    if float(scores.sum()) <= 1e-12:
        return [1.0 / n] * n

    cap = min(1.0, float(max_weight_multiplier) / n)
    weights = np.zeros(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    remaining = 1.0
    while active.any() and remaining > 1e-12:
        active_scores = scores[active]
        score_sum = float(active_scores.sum())
        proposal = (
            np.full(int(active.sum()), remaining / int(active.sum()))
            if score_sum <= 1e-12
            else remaining * active_scores / score_sum
        )
        capped = proposal >= cap - 1e-15
        active_indices = np.flatnonzero(active)
        if not capped.any():
            weights[active_indices] = proposal
            remaining = 0.0
            break
        capped_indices = active_indices[capped]
        weights[capped_indices] = cap
        active[capped_indices] = False
        remaining = 1.0 - float(weights.sum())
    if remaining > 1e-10:
        weights += remaining / n
    return weights.tolist()


def calculate_mixed_dynamic_weights(
    cos: list[float], *, adaptive_blend: float = 0.25,
    max_weight_multiplier: float = 2.0,
    confidence: list[float] | None = None,
) -> list[float]:
    """Blend bounded anchor weights with a uniform utility-preserving floor."""
    if not cos:
        return []
    blend = float(adaptive_blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("adaptive_blend must be in [0, 1]")
    if confidence is not None:
        if len(confidence) != len(cos):
            raise ValueError("one confidence value is required per accepted client")
        quality = np.asarray(confidence, dtype=np.float64)
        if not np.isfinite(quality).all() or np.any(quality < 0) or np.any(quality > 1):
            raise ValueError("confidence must be finite in [0, 1]")
        scores = np.maximum(0, np.asarray(cos, dtype=np.float64))
        if not np.isfinite(scores).all():
            raise ValueError("cosine scores must be finite")
        # Apply confidence to BOTH the uniform and directional components.
        # Otherwise a uniform floor gives suspicious clients benign-sized mass.
        directional = scores / scores.sum() if scores.sum() > 1e-12 else np.full(len(cos), 1 / len(cos))
        scores = quality * ((1 - blend) / len(cos) + blend * directional)
        if scores.sum() <= 1e-12:
            raise ValueError("accepted clients have no positive confidence")
        return calculate_bounded_dynamic_weights(
            scores.tolist(), max_weight_multiplier=max_weight_multiplier
        )
    adaptive = calculate_bounded_dynamic_weights(cos, max_weight_multiplier=max_weight_multiplier)
    uniform = 1.0 / len(cos)
    return [
        (1.0 - blend) * uniform + blend * weight
        for weight in adaptive
    ]


def weighted_aggregate(
    records: list, lambdas: list[float]
) -> ArrayRecord | None:
    """Aggregate local ArrayRecords with the given (sum=1) weights.

    Returns ``sum(lambda_i * local_model_i)`` as an ArrayRecord. Because the
    lambdas sum to 1, this equals ``global + sum(lambda_i * delta_i)`` -- i.e. the
    dynamic-weighted update aggregation described by the design.
    """
    if not records:
        return None
    total: dict[str, torch.Tensor] | None = None
    for rec, lam in zip(records, lambdas, strict=True):
        sd = rec["arrays"].to_torch_state_dict()
        w = float(lam)
        for k, v in sd.items():
            if total is None:
                total = {}
            val = v.detach().float() * w
            if k in total:
                total[k] = total[k] + val
            else:
                total[k] = val
    if total is None:
        return None
    # ArrayRecord accepts a PyTorch state_dict (dict of tensors).
    return ArrayRecord(total)
