"""VL lightweight aggregation verification (Step 5.12).

A homomorphic accumulator over the update vectors. The commitment of an update
``v`` is ``H(v) = v . g`` (dot product with a deterministic pseudo-random vector
``g``). The group here is the additive group of the reals, so the multiplicative
form ``H(sum_i lambda_i G_i) = prod_i H(G_i)^{lambda_i}`` reads as the additive
homomorphic form ``H(sum_i lambda_i G_i) = sum_i lambda_i H(G_i)`` -- exact up to
floating-point round-off. This is a *lightweight* verifier (O(d) per update), not
cryptographic security: it detects honest-vs-tampered aggregation consistency.

Usage is server-side; ``g`` is derived deterministically from a seed so the whole
run is reproducible.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import time

import numpy as np

HASH_EPS = 1e-9
# The client reconstructs the aggregate delta from the transmitted float32
# model, while the server accumulator is computed before that round-trip. A
# relative tolerance is therefore required for benign serialization error;
# protocol tampering produces errors orders of magnitude larger.
VERIFY_TOL = 5e-2


class VLVerification:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._g: np.ndarray | None = None
        self._g_dim: int | None = None

    def _vector(self, dim: int) -> np.ndarray:
        if self._g is None or self._g_dim != dim:
            rng = np.random.default_rng([self.seed, 0x5EED])
            self._g = rng.normal(0.0, 1.0, size=dim).astype(np.float64)
            self._g_dim = dim
        return self._g

    def hash_update(self, update: np.ndarray) -> float:
        """commitment h = H(update) = update . g."""
        v = np.asarray(update, dtype=np.float64).ravel()
        return float(np.dot(v, self._vector(v.shape[0])))

    def aggregate_hash(self, commitments: list[float], weights: list[float]) -> float:
        """Expected H(sum_i lambda_i G_i) = sum_i lambda_i H(G_i)."""
        return float(np.dot(np.asarray(weights, dtype=float), np.asarray(commitments, dtype=float)))

    def hash_aggregate(self, agg_update: np.ndarray) -> float:
        """H of the server's actual aggregated update."""
        return self.hash_update(agg_update)

    def verify_aggregation(
        self, agg_update: np.ndarray, commitments: list[float], weights: list[float]
    ) -> dict:
        t0 = time.perf_counter()
        h_agg = self.hash_aggregate(agg_update)
        h_expected = self.aggregate_hash(commitments, weights)
        scale = max(abs(h_expected), HASH_EPS)
        err = abs(h_agg - h_expected) / scale
        ok = bool(err < VERIFY_TOL)
        dt = time.perf_counter() - t0
        return {
            "verification_success": ok,
            "hash_error": err,
            "verification_time": dt,
            "h_agg": h_agg,
            "h_expected": h_expected,
        }


def commit_size_bytes() -> int:
    """One scalar float64 commitment per client."""
    return np.float64(0.0).nbytes
