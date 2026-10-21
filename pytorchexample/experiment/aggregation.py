"""Unified delta-based aggregation baselines (FedAvg / Krum / Trimmed Mean).

All baselines operate on the SAME update representation: ``delta_i = local_model_i -
global_model`` (model-delta, NOT gradient). The client uploads its local model; the
server derives the delta. Every baseline receives the identical delta vectors and
produces the aggregated model ``global + aggregated_delta``.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import numpy as np

from flwr.app import ArrayRecord
from .recovery import flatten_state, unflatten_state


def _mk_basis(stack: np.ndarray):
    """stack: (n, d) flattened deltas."""
    return stack


def krum_select(
    stack: np.ndarray, f: int, num_byzantine: int | None = None
) -> tuple[int, float]:
    """Single-Krum: return (index, score) of the update minimizing the Krum score.

    score(i) = sum of the squared distances from update i to its (n - f - 2) nearest
    neighbours. This is the Blanchard et al. single-Krum rule. Only ``f`` (the assumed
    number of Byzantine clients from the experiment config) is used -- never the true
    malicious set.
    """
    n = stack.shape[0]
    d = stack.shape[1]
    # pairwise squared Euclidean distances (n x n)
    D = ((stack[:, None, :] - stack[None, :, :]) ** 2).sum(axis=2)
    m = n - f - 2  # number of nearest neighbours to sum
    best_idx, best_score = 0, float("inf")
    for i in range(n):
        row = D[i].copy()
        row[i] = float("inf")
        if m < 1:
            return i, float("nan")
        score = float(np.sort(row)[:m].sum())
        if score < best_score:
            best_score, best_idx = score, i
    return best_idx, best_score


def trimmed_mean_aggregate(
    stack: np.ndarray, trim_ratio: float
) -> tuple[np.ndarray, int, int]:
    """Coordinate-wise trimmed mean. Returns (aggregated_delta, num_removed_low, num_removed_high)."""
    n = stack.shape[0]
    k = int(trim_ratio * n / 2.0)
    srt = np.sort(stack, axis=0)
    if k > 0 and n - 2 * k >= 1:
        trimmed = srt[k : n - k, :]
    else:
        trimmed = srt
    return trimmed.mean(axis=0), k, k


def fedavg_aggregate(
    stack: np.ndarray, weights: list[float] | None = None
) -> np.ndarray:
    n = stack.shape[0]
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=float)
        s = w.sum()
        w = w / s if s > 0 else np.full(n, 1.0 / n)
    return (stack * w[:, None]).sum(axis=0)


def aggregate(
    global_state: dict,
    client_updates: list[dict],
    method: str = "fedavg",
    krum_f: int = 2,
    trim_ratio: float = 0.2,
    weights: list[float] | None = None,
) -> tuple[ArrayRecord | None, dict]:
    """Unified entry point. Returns (new_global_model, diagnostics)."""
    if not client_updates:
        return None, {"method": method, "num_clients": 0}
    fl = [flatten_state(u)[0] for u in client_updates]
    stack = np.stack(fl)
    meta = flatten_state(client_updates[0])[1]
    diag: dict = {"method": method, "num_clients": len(fl)}
    if method == "krum":
        n = len(fl)
        if n < 2 * krum_f + 3:
            # Standard Krum validity n >= 2f+3 is violated. Do NOT lower f, do NOT
            # fall back to FedAvg. Record the failure and produce no update.
            diag["krum_f"] = krum_f
            diag["krum_valid"] = False
            diag["krum_validity_warning"] = "KRUM_VALIDITY_WARNING"
            diag["aggregation_status"] = "aggregation_failed_due_to_insufficient_clients"
            diag["effective_clients"] = n
            return None, diag
        sel, score = krum_select(stack, krum_f)
        agg_flat = stack[sel]
        diag["krum_f"] = krum_f
        diag["krum_valid"] = True
        diag["krum_selected_index"] = int(sel)
        diag["krum_score"] = score
        diag["krum_validity_warning"] = ""
        diag["aggregation_status"] = "ok"
    elif method == "trimmed_mean":
        agg_flat, klow, khigh = trimmed_mean_aggregate(stack, trim_ratio)
        diag["trim_ratio"] = trim_ratio
        diag["num_removed_low"] = klow
        diag["num_removed_high"] = khigh
        diag["aggregation_status"] = (
            "trim_ineffective_too_few_clients"
            if (trim_ratio > 0 and klow == 0)
            else "ok"
        )
    else:  # fedavg
        agg_flat = fedavg_aggregate(stack, weights)
        diag["weights_source"] = "num-example" if weights is not None else "uniform"
        diag["aggregation_status"] = "ok"
    agg_delta = unflatten_state(agg_flat, meta)
    new_state = {k: global_state[k] + agg_delta[k] for k in global_state}
    return ArrayRecord(new_state), diag
