"""Application-level logical Shamir recovery for the Flower pilot (Step 5.3.6/7).

This is a LOGICAL server-side recovery simulation. It does NOT rewrite Flower's
message lifecycle; the client uploads its model normally, and the server-side
adapter reconstructs the update from persisted Shamir shares for clients marked
``post_share``-unavailable.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .secret_sharing import (
    Share,
    fixed_point_decode,
    fixed_point_encode,
    reconstruct_vector,
    share_vector,
    vector_share_bytes,
)
from .logical_servers import LogicalTwoServerStore

SCALE = 2**16
EPS = 1e-8


@dataclass
class ShareTopology:
    n: int
    t: int
    as1: int
    as2: int
    relay: int

    def validate(self) -> bool:
        return (
            self.as1 < self.t
            and self.as2 < self.t
            and self.as1 + self.as2 + self.relay == self.n
            and self.as1 + self.as2 + self.relay >= self.t
        )

    @staticmethod
    def default(n: int, t: int) -> "ShareTopology":
        # Known splits for integration tests (extend for other (n,t) as needed).
        if n == 10 and t == 6:
            return ShareTopology(n, t, 4, 4, 2)
        if n == 5 and t == 3:
            return ShareTopology(n, t, 2, 2, 1)
        if n == 20 and t in (16, 12):
            # Preserve the N=10 40/40/20 holder split while scaling the share set.
            return ShareTopology(n, t, 8, 8, 4)
        # fallback: each server gets t-2, relay gets the rest (may be <t), requires
        # as1+as2 >= t.
        a = max(1, t - 2)
        return ShareTopology(n, t, a, a, max(0, n - 2 * a))

    def index_sets(self):
        i1 = list(range(0, self.as1))
        i2 = list(range(self.as1, self.as1 + self.as2))
        ir = list(range(self.as1 + self.as2, self.n))
        return i1, i2, ir


@dataclass
class RecoveryStateStore:
    """Logical per-(client,round) store of persisted Shamir shares."""

    shares: dict = field(default_factory=dict)
    logical_servers: dict = field(default_factory=dict)

    def put(self, client_id: int, server_round: int, secret_key: str, shares: list[Share]) -> None:
        self.shares[(client_id, server_round, secret_key)] = shares
        routed = LogicalTwoServerStore[Share]()
        routed.provision(shares)
        self.logical_servers[(client_id, server_round, secret_key)] = routed

    def get(self, client_id: int, server_round: int, secret_key: str) -> list[Share]:
        return self.shares.get((client_id, server_round, secret_key), [])

    def clear_round(self, server_round: int) -> None:
        """Release large per-round share vectors after recovery bookkeeping.

        Shares are only needed while the current aggregation round is being
        reconstructed. Keeping A/B vectors for every historical round causes
        the server process to grow until Ray kills it for OOM.
        """
        keys = [key for key in self.shares if key[1] == server_round]
        for key in keys:
            self.shares.pop(key, None)
            self.logical_servers.pop(key, None)

    def get_logical_server_audit(
        self, client_id: int, server_round: int, secret_key: str
    ) -> dict[str, object]:
        routed = self.logical_servers.get((client_id, server_round, secret_key))
        return routed.audit() if routed is not None else {
            "server_topology": "logical_two_server",
            "server_names": ["C1", "C2"],
            "independent_server_processes": False,
            "c1_share_count": 0,
            "c2_share_count": 0,
            "total_share_count": 0,
        }


def _holder_subset(topology: ShareTopology, outage: str):
    i1, i2, ir = topology.index_sets()
    sel = []
    if "AS1" not in outage:
        sel += i1
    if "AS2" not in outage:
        sel += i2
    if "relay" not in outage:
        sel += ir
    return sel


def recover_g(
    g_flat: np.ndarray,
    topology: ShareTopology,
    outage: str,
    seed: int,
    store: RecoveryStateStore,
    client_id: int,
    server_round: int,
) -> dict:
    """Share g, apply a holder outage, reconstruct, decode. Returns a stats dict."""
    secret = fixed_point_encode(np.asarray(g_flat, dtype=float), SCALE)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    shares = share_vector(secret, topology.n, topology.t, rng)
    store.put(client_id, server_round, "g", shares)
    persisted_shares = store.get(client_id, server_round, "g")
    share_time = time.perf_counter() - t0

    idx = _holder_subset(topology, outage)
    sel = [persisted_shares[i] for i in idx]
    shares_available = len(sel)
    t1 = time.perf_counter()
    if shares_available >= topology.t:
        recon = reconstruct_vector(sel[: topology.t], topology.t)
        g_recon = fixed_point_decode(recon, SCALE)
        status = "recovered"
        success = True
        g_max = float(np.max(np.abs(g_recon - np.asarray(g_flat, dtype=float))))
    else:
        g_recon = None
        status = "insufficient_shares"
        success = False
        g_max = float("nan")
    recon_time = time.perf_counter() - t1
    return {
        "shares_available": shares_available,
        "recovery_status": status,
        "recovery_success": int(success),
        "g_recovered": g_recon,
        "g_max_abs_error": g_max,
        "share_generation_time": share_time,
        "reconstruction_time": recon_time,
        "total_recovery_time": share_time + recon_time,
        "outage": outage,
    }


def recover_g_dual(
    g_flat: np.ndarray,
    normal_n: int,
    normal_t: int,
    emergency_t: int,
    active_share_set: str,
    active_threshold: int,
    available_shares: int,
    seed: int,
    store: RecoveryStateStore | None = None,
    client_id: int | None = None,
    server_round: int | None = None,
) -> dict:
    """Generate A=Shamir(n,normal_t) and B=Shamir(n,emergency_t) of g, reconstruct active."""
    # Both threshold schemes are provisioned up front.  A malformed runtime
    # configuration (for example n=10 with t=16) cannot produce valid shares;
    # fail at the call boundary with an actionable error instead of allowing a
    # later ``None`` reconstruction to reach numeric code.
    if normal_n < max(normal_t, emergency_t):
        raise ValueError(
            "invalid dual-Shamir configuration: share_n must be >= "
            f"max(normal_threshold, emergency_threshold) "
            f"({normal_n} < {max(normal_t, emergency_t)})"
        )
    if normal_t < 1 or emergency_t < 1:
        raise ValueError("invalid dual-Shamir configuration: thresholds must be positive")
    g_arr = np.asarray(g_flat, dtype=float)
    # Defensive guard (not an algorithm change): with no active set (failure mode) or a
    # NaN/inf delta from a diverged model, do not attempt to encode shares -- just report
    # recovery failure. Avoids a hard crash in fixed_point_encode on NaN.
    if active_share_set not in ("A", "B") or np.isnan(g_arr).any() or np.isinf(g_arr).any():
        return {
            "shares_available": available_shares,
            "recovery_status": "insufficient_shares",
            "recovery_success": 0, "g_recovered": None, "g_max_abs_error": float("nan"),
            "share_generation_time": 0.0, "reconstruction_time": 0.0,
            "total_recovery_time": 0.0, "b_share_payload_bytes": 0,
            "g_other": None, "same_secret": 0,
        }
    secret = fixed_point_encode(g_arr, SCALE)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    A = share_vector(secret, normal_n, normal_t, rng)
    B = share_vector(secret, normal_n, emergency_t, rng)
    logical_audit: dict[str, object] = {}
    if store is not None and client_id is not None and server_round is not None:
        store.put(client_id, server_round, "A", A)
        store.put(client_id, server_round, "B", B)
        logical_audit = {
            "A": store.get_logical_server_audit(client_id, server_round, "A"),
            "B": store.get_logical_server_audit(client_id, server_round, "B"),
        }
        A = store.get(client_id, server_round, "A")
        B = store.get(client_id, server_round, "B")
    share_time = time.perf_counter() - t0
    # D2-V2: A and B are BOTH provisioned together (pre-distribution) at the moment the
    # client's share set is created. B is NOT separately distributed when Emergency is
    # triggered. Report both payloads as SETUP/pre-distribution cost.
    a_payload_bytes = vector_share_bytes(A)
    b_payload_bytes = vector_share_bytes(B)
    setup_payload_total = a_payload_bytes + b_payload_bytes
    b_redistributed = 0  # never re-sent at Emergency

    def _reconstruct(shares, threshold):
        if threshold is None or len(shares) < threshold:
            return None
        return fixed_point_decode(reconstruct_vector(shares[:threshold], threshold), SCALE)

    g_recon = _reconstruct(A, normal_t) if active_share_set == "A" else _reconstruct(B, emergency_t)
    g_other = _reconstruct(B, emergency_t) if active_share_set == "A" else _reconstruct(A, normal_t)
    target = np.asarray(g_flat, dtype=float)

    if active_share_set == "A":
        shares, threshold = A, normal_t
    elif active_share_set == "B":
        shares, threshold = B, emergency_t
    else:
        return {
            "shares_available": available_shares,
            "recovery_status": "insufficient_shares",
            "recovery_success": 0, "g_recovered": None, "g_max_abs_error": float("nan"),
            "share_generation_time": share_time, "reconstruction_time": 0.0,
            "total_recovery_time": share_time,
            "b_share_payload_bytes": 0, "g_other": None, "same_secret": 0,
            "a_pre_distribution_payload": a_payload_bytes, "b_pre_distribution_payload": b_payload_bytes,
            "setup_share_payload_total": setup_payload_total, "b_redistributed": 0,
            "logical_server_audit": logical_audit,
        }
    t1 = time.perf_counter()
    if available_shares >= threshold:
        status, success = "recovered", 1
        g_max = float(np.max(np.abs(g_recon - target)))
    else:
        g_recon, status, success, g_max = None, "insufficient_shares", 0, float("nan")
    recon_time = time.perf_counter() - t1
    # same_secret = A and B (when both reconstructable) decode to the same g.
    same_secret = 0
    if g_recon is not None and g_other is not None:
        same_secret = int(float(np.max(np.abs(g_recon - g_other))) <= 1e-6)
    return {
        "shares_available": available_shares,
        "recovery_status": status,
        "recovery_success": success,
        "g_recovered": g_recon,
        "g_max_abs_error": g_max,
        "g_other": g_other,
        "same_secret": same_secret,
        "b_share_payload_bytes": b_payload_bytes,
        # D2-V2 accounting: both share sets are provisioned in setup (pre-distribution).
        "a_pre_distribution_payload": a_payload_bytes,
        "b_pre_distribution_payload": b_payload_bytes,
        "setup_share_payload_total": setup_payload_total,
        "b_redistributed": b_redistributed,
        "logical_server_audit": logical_audit,
        "share_generation_time": share_time,
        "reconstruction_time": recon_time,
        "total_recovery_time": share_time + recon_time,
    }


def flatten_state(state: dict) -> tuple[np.ndarray, dict]:
    """Flatten a state_dict (dict of tensors) to a 1-D float array."""
    entries = []
    meta = {}
    offset = 0
    for k in sorted(state.keys()):
        flat = state[k].detach().cpu().reshape(-1).numpy()
        meta[k] = (state[k].shape, offset, len(flat))
        entries.append(flat)
        offset += len(flat)
    return np.concatenate(entries) if entries else np.zeros(0), meta


def unflatten_state(flat: np.ndarray, meta: dict) -> dict:
    out = {}
    for k, (shape, off, length) in meta.items():
        arr = flat[off : off + length].reshape(shape)
        out[k] = torch.as_tensor(arr, dtype=torch.float32)
    return out
