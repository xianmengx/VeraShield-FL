"""SSDF strategy with application-level logical post-share Shamir recovery."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import time
import hashlib
import json
import copy
import numpy as np
import torch
from flwr.app import ArrayRecord
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from .recovery import (
    RecoveryStateStore,
    ShareTopology,
    flatten_state,
    recover_g,
    recover_g_dual,
    unflatten_state,
)
from .ssdf import SSDFDetector
from .ssdf_backends import BackendDetection
from .ssdf_strategy import SSDFStrategy
from .threshold_controller import ThresholdController
from .dynamic_weight import (
    calculate_bounded_dynamic_weights,
    calculate_cosine_similarity,
    calculate_dynamic_weights,
    calculate_mixed_dynamic_weights,
)
from .vl_verification import VLVerification, commit_size_bytes
from .client_verifier import make_proof, state_digest
from .crypto_vl import (
    ClientHashRecord,
    CryptoVLVerifier,
    decode_weighted_aggregate,
    encode_vector,
    homomorphic_hash,
    make_aggregation_proof,
    recovery_state_digest,
    verify_client_record,
    weighted_aggregate,
)
from .crypto_vl_keys import load_crypto_vl_setup
from .participation import (
    candidate_digest,
    records_digest,
    verify_receipt,
    verify_record,
)


# The model returned by Flower is float32.  The canonical aggregation is kept
# in float64 for SSDF/VL bookkeeping, so the reconstructed model delta must be
# compared in the model's representation rather than against the raw float64
# accumulator (otherwise late-round quantization can be reported as a mismatch).
AGGREGATION_INPUT_RTOL = 1e-5
AGGREGATION_INPUT_ATOL = 1e-6


def _largest_remainder_weights(weights: list[float], scale: int) -> list[int]:
    """Canonical non-negative integer weights summing exactly to ``scale``."""
    raw = [max(0.0, float(weight)) * scale for weight in weights]
    quantized = [int(np.floor(value)) for value in raw]
    remainder = int(scale - sum(quantized))
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - quantized[index], -index),
        reverse=True,
    )
    for index in order[: max(0, remainder)]:
        quantized[index] += 1
    return quantized


def build_vl_aggregation_record(
    *,
    server_round: int,
    method: str,
    valid_client_ids: list[int],
    retained_client_ids: list[int],
    unavailable_ids: set[int],
    available_clients: int,
    aggregation_completed: bool,
    recovery_success: bool,
    verification_enabled: bool,
    verification_success: bool | None,
) -> dict:
    """Build the method-aware common aggregation row for a VL round."""
    if not aggregation_completed:
        status = "aggregation_failed_no_retained_clients"
    elif verification_enabled and verification_success is not True:
        status = "aggregation_completed_verification_failed"
    else:
        status = "ok"
    return {
        "round": server_round,
        "method": method,
        "num_clients": len(retained_client_ids),
        "client_ids": list(retained_client_ids),
        "total_clients": len(valid_client_ids),
        "available_clients": available_clients,
        "dropped_clients": len(unavailable_ids),
        "unavailable_ids": sorted(unavailable_ids),
        "effective_clients": len(retained_client_ids),
        "weights_source": "dynamic_lambda" if method == "dynamic_weight" else "uniform",
        "aggregation_status": status,
        "recovery_status": "success" if recovery_success else "failed",
        "verification_status": (
            "not_enabled"
            if not verification_enabled
            else ("success" if verification_success is True else "failed")
        ),
    }


def _aggregate_retained_deltas(
    global_state: dict,
    state_meta: dict,
    retained_cids: list[int],
    retained_deltas: list[np.ndarray],
    weights: list[float],
    *,
    update_scale: int | None = None,
    weight_scale: int | None = None,
) -> tuple[ArrayRecord, np.ndarray]:
    """Aggregate the canonical post-recovery deltas used by DW and VL."""
    if not (
        len(retained_cids) == len(retained_deltas) == len(weights)
        and len(set(retained_cids)) == len(retained_cids)
        and retained_deltas
    ):
        raise RuntimeError("AGGREGATION_INPUT_MISMATCH")

    dim = retained_deltas[0].size
    if any(delta.ndim != 1 or delta.size != dim for delta in retained_deltas):
        raise RuntimeError("AGGREGATION_INPUT_MISMATCH")

    # Crypto-VL verifies a canonical fixed-point aggregate.  When scales are
    # supplied, aggregate that exact object so model execution and verification
    # cannot diverge because of independent rounding paths.
    if update_scale is not None or weight_scale is not None:
        if not update_scale or not weight_scale:
            raise RuntimeError("AGGREGATION_INPUT_MISMATCH")
        integer_weights = _largest_remainder_weights(weights, int(weight_scale))
        encoded_updates = [
            encode_vector(delta, int(update_scale)) for delta in retained_deltas
        ]
        aggregated_delta = decode_weighted_aggregate(
            weighted_aggregate(encoded_updates, integer_weights),
            update_scale=int(update_scale),
            weight_scale=int(weight_scale),
        )
    else:
        aggregated_delta = np.zeros(dim, dtype=np.float64)
        for delta, weight in zip(retained_deltas, weights, strict=True):
            aggregated_delta += delta * float(weight)

    delta_state = unflatten_state(aggregated_delta, state_meta)
    final_state = {
        key: global_state[key]
        + delta_state[key].to(
            device=global_state[key].device, dtype=global_state[key].dtype
        )
        for key in global_state
    }

    # Ensure the model returned to Flower represents the same aggregate delta.
    actual_delta, _ = flatten_state(
        {key: final_state[key] - global_state[key] for key in global_state}
    )
    actual_delta = np.asarray(actual_delta)
    # Quantize the reference to the same dtype before comparing.  This keeps
    # the guard sensitive to shape/content errors while accepting the expected
    # float64 -> float32 round-trip introduced by model reconstruction.
    reference_delta = np.asarray(aggregated_delta, dtype=actual_delta.dtype)
    if not np.allclose(
        actual_delta,
        reference_delta,
        rtol=AGGREGATION_INPUT_RTOL,
        atol=AGGREGATION_INPUT_ATOL,
    ):
        raise RuntimeError("AGGREGATION_INPUT_MISMATCH")
    return ArrayRecord(final_state), aggregated_delta


class SSDFRecoveryStrategy(SSDFStrategy):
    """SSDF + logical post-share dropout recovery (server side)."""

    def __init__(
        self,
        *args,
        share_n: int = 10,
        share_t: int = 6,
        outage: str = "full",
        dropout_timing: str = "post_share",
        post_share_dropout_ids: set[int] | None = None,
        normal_threshold: int = 8,
        emergency_threshold: int = 6,
        threshold_controller_enabled: bool = False,
        runtime_available_schedule: list[int] | None = None,
        static_threshold: int | None = None,
        dynamic_weight: bool = False,
        dynamic_weight_blend: float = 0.25,
        vl_verification: bool = False,
        vl_enforce: bool = False,
        vl_commit_mode: str = "next_round",
        vl_backend: str = "linear",
        vl_update_scale: int = 1 << 16,
        vl_key_dir: str = "",
        vl_tamper: bool = False,
        vl_tamper_rounds: list[int] | None = None,
        vl_client_tamper_rounds: list[int] | None = None,
        control_run_id: str = "",
        paired_run_id: str = "",
        unavailable_by_round: dict | None = None,
        ssdf_filter_enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.share_n = share_n
        self.share_t = share_t
        self.topology = ShareTopology.default(share_n, share_t)
        assert self.topology.validate()
        self.outage = outage
        self.dropout_timing = dropout_timing
        self.post_share_ids = set(post_share_dropout_ids or [])
        self.normal_threshold = normal_threshold
        self.emergency_threshold = emergency_threshold
        self.threshold_controller_enabled = threshold_controller_enabled
        self.unavailable_by_round = unavailable_by_round or {}
        # Crypto VL can be evaluated in isolation while reusing the mature
        # transcript/recovery path.  In that mode SSDF remains observational
        # only and cannot remove updates from the aggregation set.
        self.ssdf_filter_enabled = bool(ssdf_filter_enabled)
        self.controller = ThresholdController(
            normal_threshold, emergency_threshold, static_threshold=static_threshold
        )
        self.runtime_available_schedule = (
            list(runtime_available_schedule) if runtime_available_schedule else None
        )
        self.dynamic_weight = dynamic_weight
        self.dynamic_weight_blend = float(dynamic_weight_blend)
        if not 0.0 <= self.dynamic_weight_blend <= 1.0:
            raise ValueError("dynamic_weight_blend must be in [0, 1]")
        self.vl_verification = vl_verification
        self.vl_enforce = vl_enforce
        self.vl_backend = str(vl_backend).lower()
        self.vl_commit_mode = str(vl_commit_mode)
        if self.vl_commit_mode not in {"next_round", "precommit"}:
            raise ValueError("unknown vl_commit_mode")
        if self.vl_commit_mode == "precommit" and not (
            vl_verification and vl_enforce and self.vl_backend == "crypto"
        ):
            raise ValueError("precommit requires enforced crypto VL verification")
        if self.vl_backend not in {"linear", "crypto"}:
            raise ValueError("vl_backend must be 'linear' or 'crypto'")
        self.vl_update_scale = int(vl_update_scale)
        if self.vl_update_scale <= 0:
            raise ValueError("vl_update_scale must be positive")
        self.crypto_vl_setup = None
        if self.vl_backend == "crypto":
            if not vl_key_dir:
                raise ValueError("crypto VL requires vl_key_dir")
            self.crypto_vl_setup = load_crypto_vl_setup(vl_key_dir)
        self.vl_tamper = vl_tamper
        self.vl_tamper_rounds = set(int(r) for r in (vl_tamper_rounds or []))
        self.vl_client_tamper_rounds = set(
            int(r) for r in (vl_client_tamper_rounds or [])
        )
        self.control_run_id = str(control_run_id or "")
        self.paired_run_id = str(paired_run_id or "")
        self.vl_verifier = VLVerification(seed=kwargs.get("seed", 42) or 42)
        self.ref_detector = (
            SSDFDetector(
                ema_beta=kwargs.get("ema_beta", 0.9), k=kwargs.get("k", 2.0),
                norm_tol=kwargs.get("norm_tol", 1e-3),
                warmup_rounds=kwargs.get("warmup_rounds", 2),
                eps=kwargs.get("eps", 1e-8), norm_floor=kwargs.get("norm_floor", 1e-12),
                reference_k=kwargs.get("reference_k", 2.5),
                reference_floor=kwargs.get("reference_floor", -0.25),
                norm_k=kwargs.get("norm_k", 2.70),
                history_limit=kwargs.get("history_limit", 20),
                detector_mode=kwargs.get("detector_mode", "v2_full"),
            )
            if self.logical_backend is None else None
        )
        self.recovery_store = RecoveryStateStore()
        self.recovery_records: dict[int, list[dict]] = {}
        self.runtime_threshold_records: dict[int, dict] = {}
        self.dynamic_weight_records: dict[int, list[dict]] = {}
        self.dw_timing_records: dict[int, dict] = {}
        self.vl_verification_records: dict[int, dict] = {}
        # Client-verifiable transcript records and receipts returned in train
        # replies are persisted per round.
        self.client_verification_records: dict[int, dict] = {}
        self.aggregation_records: dict[int, dict] = {}
        self.candidate_sets: dict[int, list[int]] = {}
        self.protocol_trace_records: dict[int, dict] = {}
        self.vl_pre_aggregation_states: dict[int, dict] = {}

    def configure_train(self, server_round, arrays, config, grid):
        """Attach the previous round proof to the next client message."""
        messages = super().configure_train(server_round, arrays, config, grid)
        # Partition IDs are the stable client identity used by the experiment
        # manifest.  Use the full configured population rather than transport
        # node IDs, which are Flower-runtime identifiers and need not equal a
        # partition ID under a remote deployment.
        candidate_ids = list(range(int(self.total_nodes or len(messages))))
        self.candidate_sets[server_round] = candidate_ids
        for message in messages:
            message.content["config"]["vl-candidate-set"] = json.dumps(candidate_ids)
            message.content["config"]["vl-proof-skip"] = 0
        previous = self.client_verification_records.get(server_round - 1)
        if not previous or not messages:
            return messages
        previous_failed = previous.get("verification_success") in (False, 0, "0")
        if int(previous.get("rollback_applied", 0) or 0) or previous_failed:
            # The failed proof describes a candidate that was not committed.
            # Skip it so clients refresh their cache from the rolled-back model
            # before the next independently verifiable round.
            for message in messages:
                message.content["config"]["vl-proof-skip"] = 1
            return messages
        if not previous.get("proof"):
            # No canonical proof was produced in the previous round (for
            # example, because no retained update was available).  Clients
            # must refresh their cache before the next proof-bearing round.
            for message in messages:
                message.content["config"]["vl-proof-skip"] = 1
            return messages
        payload = json.loads(previous["proof"])
        # The client validates the signed participation set as part of the
        # transcript.  It is persisted alongside the proof record, so include
        # it in the transport payload as well.
        payload["participation_records"] = json.loads(
            previous.get("participation_records", "[]")
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        for message in messages:
            message.content["config"]["vl-proof"] = encoded
            message.content["config"]["vl-proof-round"] = int(server_round - 1)
            message.content["config"]["vl-proof-stage"] = (
                "integer_numerator_and_model" if self.vl_backend == "crypto"
                else "model_digest_only"
            )
        if server_round in self.vl_client_tamper_rounds:
            received_state = arrays.to_torch_state_dict()
            flat, meta = flatten_state(received_state)
            rng = np.random.default_rng([self.seed, server_round, 88])
            scale = float(np.linalg.norm(flat)) + 1e-9
            altered = flat + rng.normal(0.0, 1.0, size=flat.shape) * (0.1 * scale)
            altered_state = unflatten_state(altered, meta)
            altered_arrays = ArrayRecord(altered_state)
            for message in messages:
                message.content["arrays"] = altered_arrays
        return messages

    def aggregate_train(self, server_round, replies):
        if self.vl_commit_mode == "next_round":
            return self._aggregate_train_candidate(server_round, replies)
        # Only protocol state is transactional. Attempted-round diagnostics are
        # retained even when the tentative transition is discarded.
        objects = [self.detector, self.ref_detector, self.controller]
        if self.logical_backend is not None:
            objects.append(self.logical_backend.detector)
        snapshots = [(obj, copy.deepcopy(obj.__dict__)) for obj in objects if obj is not None]
        committed = False
        try:
            arrays, metrics = self._aggregate_train_candidate(server_round, replies)
            row = self.vl_verification_records.setdefault(server_round, {"round": server_round})
            committed = bool(row.get("precommit_verification_success", False))
            if not committed:
                arrays = self._current_arrays
            row.update({
                "commit_mode": "precommit",
                "model_committed": int(committed),
                "state_committed": int(committed),
                "candidate_rejected": int(not committed),
                "committed_model_digest": state_digest(arrays.to_torch_state_dict()),
            })
            self.client_verification_records.setdefault(server_round, {}).update(row)
            audit = {key: row[key] for key in (
                "commit_mode", "model_committed", "state_committed",
                "candidate_rejected", "committed_model_digest",
            )}
            self.aggregation_records.setdefault(server_round, {}).update(audit)
            self.protocol_trace_records.setdefault(server_round, {}).update(audit)
            return arrays, metrics
        finally:
            if not committed:
                for obj, state in snapshots:
                    obj.__dict__.clear()
                    obj.__dict__.update(state)
            self.recovery_store.clear_round(server_round)

    def _aggregate_train_candidate(self, server_round, replies):
        self._current_round = server_round
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        client_receipts = []
        rejected_receipts = []
        non_rejected_valid = []
        for msg in valid:
            metrics_record = msg.content["metrics"]
            rejected = bool(int(metrics_record.get("vl-rejected", 0) or 0))
            if not rejected:
                non_rejected_valid.append(msg)
                continue
            raw_receipt = msg.content.get("participation", {}).get("vl_receipt", "")
            try:
                receipt = json.loads(str(raw_receipt))
            except (TypeError, ValueError, json.JSONDecodeError):
                receipt = {"malformed": True}
            cid = int(metrics_record.get("client_partition_id", -1))
            receipt_valid = bool(
                not receipt.get("malformed")
                and int(receipt.get("client_id", -2)) == cid
                and int(receipt.get("round_id", -2)) == server_round - 1
                and receipt.get("verification_success") is False
                and verify_receipt(receipt, seed=int(self.seed))
            )
            receipt_row = {**receipt, "receipt_valid": int(receipt_valid)}
            client_receipts.append(receipt_row)
            if receipt_valid:
                rejected_receipts.append(receipt_row)
        valid = non_rejected_valid
        client_receipt_failure = bool(self.vl_enforce and rejected_receipts)
        rollback_target = None
        if client_receipt_failure:
            rejected_round = int(rejected_receipts[0]["round_id"])
            rollback_target = self.vl_pre_aggregation_states.get(rejected_round)
            # Every update in this round was computed from the model clients
            # just rejected, so none of them may enter the next global state.
            valid = []
        global_state = (
            self._current_arrays.to_torch_state_dict() if self._current_arrays else None
        )
        if global_state is not None and self.vl_verification:
            self.vl_pre_aggregation_states[server_round] = {
                key: value.detach().cpu().clone()
                for key, value in global_state.items()
            }
        _, state_meta = flatten_state(global_state)

        # --- Round-level runtime threshold decision (Dual-Shamir) ---------------
        try:
            participating = int((self.records.get(server_round) or {}).get(
                "participating_clients", 0
            ))
        except (TypeError, ValueError):
            participating = 0
        # Number of UAVs that are online = sampled clients minus post-share-dropout.
        unavailable_ids = set(self.unavailable_by_round.get(server_round, []))
        if (
            self.dropout_timing == "post_share"
            and not unavailable_ids
            and self.post_share_ids
        ):
            # The no-manifest path derives a deterministic outage cohort in
            # server_app.  Reuse it every round so post-share recovery is
            # actually exercised instead of remaining a metadata-only mode.
            unavailable_ids = set(self.post_share_ids)
        available_clients = max(0, participating - len(unavailable_ids))
        # Optional per-round available-share schedule overrides the dropout-derived
        # count, enabling a single run to move between normal and emergency modes.
        if (
            self.threshold_controller_enabled
            and self.runtime_available_schedule
            and 0 <= server_round - 1 < len(self.runtime_available_schedule)
        ):
            available_clients = int(self.runtime_available_schedule[server_round - 1])
        # Logical model: each online UAV contributes one share contribution, so the
        # number of available shares equals the number of online UAVs this round.
        available_shares = available_clients
        if self.threshold_controller_enabled:
            decision = self.controller.decide(available_shares)
            threshold_mode = decision.mode
            active_share_set = decision.active_share_set
            active_threshold = decision.active_threshold
            switch_event = bool(decision.switch_event)
        else:
            decision = None
            threshold_mode = "normal"
            active_share_set = "A"
            active_threshold = self.normal_threshold
            switch_event = False

        rows = []
        retained = []
        retained_cids: list[int] = []
        retained_deltas: list = []
        round_b_payload = 0
        round_a_payload = 0
        round_setup_payload = 0
        round_recovery_attempted = 0
        round_recovery_success = 1
        round_recovery_time = 0.0
        round_decision_equal = 1 if self.ref_detector is not None else float("nan")
        prepared_clients = []
        for msg in valid:
            cid = int(msg.content["metrics"].get("client_partition_id", -1))
            local = msg.content["arrays"].to_torch_state_dict()
            delta_orig = {k: local[k] - global_state[k] for k in global_state}
            delta_flat, meta = flatten_state(delta_orig)
            delta_used = delta_flat
            gt = 1 if cid in self.malicious_client_ids else 0
            active = int(msg.content["metrics"].get("attack_active", 0))
            is_post = (self.dropout_timing == "post_share" and cid in unavailable_ids)
            rec_row = {
                "round": server_round, "client_id": cid,
                "dropout_timing": self.dropout_timing,
                "client_dropout": int(is_post),
                "share_generated": int(is_post),
                "share_distributed": int(is_post),
                "ssdf_pass_reference": 0,
                "shares_available": available_shares,
                "available_clients": available_clients,
                "threshold_mode": threshold_mode,
                "active_share_set": active_share_set,
                "active_threshold": active_threshold,
                "switch_event": int(switch_event),
                "recovery_attempted": 0,
                "recovery_status": "no_dropout",
                "recovery_success": 1,
                "recovery_failure_reason": "",
                "recovery_time": 0.0,
                "g_max_abs_error": 0.0,
                "ssdf_pass_recovered": 0,
                "decision_equal": 0,
                "removed": 0,
            }
            use_runtime_dual = self.threshold_controller_enabled and is_post
            if use_runtime_dual:
                g_flat = -delta_flat
                rr = recover_g_dual(
                    g_flat,
                    normal_n=self.share_n,
                    normal_t=self.normal_threshold,
                    emergency_t=self.emergency_threshold,
                    active_share_set=active_share_set,
                    active_threshold=active_threshold,
                    available_shares=available_shares,
                    seed=1000 + server_round * 31 + cid,
                    store=self.recovery_store,
                    client_id=cid,
                    server_round=server_round,
                )
                round_recovery_attempted = 1
                round_b_payload = max(
                    round_b_payload, int(rr.get("b_share_payload_bytes", 0))
                )
                round_a_payload = max(
                    round_a_payload, int(rr.get("a_pre_distribution_payload", 0))
                )
                round_setup_payload = max(
                    round_setup_payload, int(rr.get("setup_share_payload_total", 0))
                )
                round_recovery_time += float(rr["total_recovery_time"])
                rec_row.update({
                    "shares_available": rr["shares_available"],
                    "recovery_attempted": 1,
                    "recovery_status": rr["recovery_status"],
                    "recovery_success": rr["recovery_success"],
                    "recovery_failure_reason": (
                        "" if rr["recovery_success"]
                        else "insufficient shares (< active threshold)"
                    ),
                    "recovery_time": rr["total_recovery_time"],
                    "g_max_abs_error": rr["g_max_abs_error"],
                    "same_secret": rr.get("same_secret", 0),
                    "share_generation_time": rr.get("share_generation_time", 0.0),
                    "reconstruction_time": rr.get("reconstruction_time", 0.0),
                    "b_share_payload_bytes": rr.get("b_share_payload_bytes", 0),
                    "a_pre_distribution_payload": rr.get("a_pre_distribution_payload", 0),
                    "b_pre_distribution_payload": rr.get("b_pre_distribution_payload", 0),
                    "setup_share_payload_total": rr.get("setup_share_payload_total", 0),
                    "b_redistributed": rr.get("b_redistributed", 0),
                    "logical_server_audit": rr.get("logical_server_audit", {}),
                })
                if rr["recovery_success"]:
                    delta_recon_vec = -rr["g_recovered"]
                    delta_used = delta_recon_vec
                else:
                    round_recovery_success = 0
                    rec_row["recovery_failed_removed"] = 1
            elif is_post:
                g_flat = -delta_flat
                rr = recover_g(
                    g_flat, self.topology, self.outage,
                    seed=1000 + server_round * 31 + cid,
                    store=self.recovery_store, client_id=cid, server_round=server_round,
                )
                rec_row.update({
                    "shares_available": rr["shares_available"],
                    "recovery_attempted": 1,
                    "recovery_status": rr["recovery_status"],
                    "recovery_success": rr["recovery_success"],
                    "recovery_failure_reason": (
                        "" if rr["recovery_success"] else "insufficient shares (< t)"
                    ),
                    "recovery_time": rr["total_recovery_time"],
                    "g_max_abs_error": rr["g_max_abs_error"],
                    "outage": rr["outage"],
                    "share_generation_time": rr.get("share_generation_time", 0.0),
                    "reconstruction_time": rr.get("reconstruction_time", 0.0),
                    "logical_server_audit": self.recovery_store.get_logical_server_audit(
                        cid, server_round, "g"
                    ),
                })
                if rr["recovery_success"]:
                    delta_recon_vec = -rr["g_recovered"]
                    delta_used = delta_recon_vec
                else:
                    round_recovery_success = 0
                    rec_row["recovery_failed_removed"] = 1
            else:
                rec_row["same_secret"] = 1

            prepared_clients.append({
                "client_id": cid,
                "server_round": server_round,
                "delta": {"w": torch.as_tensor(delta_flat, dtype=torch.float32)},
                "production_delta": {
                    "w": torch.as_tensor(delta_used, dtype=torch.float32)
                },
                "is_malicious_gt": gt,
                "attack_active": active,
                "message": msg,
                "delta_used": delta_used,
                "is_post": is_post,
                "recovery_success": bool(rec_row["recovery_success"]),
                "recovery_row": rec_row,
            })

        reference_results = (
            self.ref_detector.detect_round(prepared_clients)
            if self.ref_detector is not None else []
        )
        reference_by_cid = {res.client_id: res for res in reference_results}
        production_candidates = [
            {
                "client_id": item["client_id"],
                "server_round": server_round,
                "delta": item["production_delta"],
                "is_malicious_gt": item["is_malicious_gt"],
                "attack_active": item["attack_active"],
            }
            for item in prepared_clients
            if item["recovery_success"]
        ]
        production_results = (
            self.logical_backend.detect_round(production_candidates)
            if self.logical_backend is not None
            else self.detector.detect_round(production_candidates)
        )
        production_by_cid = {res.client_id: res for res in production_results}
        detection_rows = []

        for item in prepared_clients:
            cid = item["client_id"]
            rec_row = item["recovery_row"]
            ref_res = reference_by_cid.get(cid)
            prod_res = production_by_cid.get(cid)
            if prod_res is None:
                if ref_res is not None:
                    prod_res = ref_res
                else:
                    prod_res = BackendDetection(
                        client_id=cid,
                        round_id=server_round,
                        pass_flag=False,
                        risk=1.0,
                        dynamic_weight=0.0,
                        reason="recovery_unavailable",
                        norm_ok=False,
                        reference_ok=False,
                        temporal_ok=False,
                    )
            eligible = int(item["recovery_success"])
            rec_row["ssdf_pass_reference"] = (
                int(bool(ref_res.pass_flag)) if ref_res is not None else float("nan")
            )
            rec_row["ssdf_pass_recovered"] = int(bool(prod_res.pass_flag))
            if ref_res is not None:
                rec_row["decision_equal"] = int(
                    bool(prod_res.pass_flag) == bool(ref_res.pass_flag)
                )
                round_decision_equal = min(
                    round_decision_equal, rec_row["decision_equal"]
                )
            else:
                rec_row["decision_equal"] = float("nan")
            rec_row["removed"] = int(
                (not item["recovery_success"]) or (not prod_res.pass_flag)
            )
            if self.logical_backend is not None:
                rec_row.update({
                    "secjudge_ok": int(prod_res.norm_ok),
                    "norm_sum": prod_res.raw_norm,
                    "temporal_cosine": prod_res.temporal_cosine,
                    "reference_cosine": prod_res.reference_cosine,
                    "anchor_stability": prod_res.anchor_stability,
                    "persistent_score": prod_res.persistent_score,
                    "persistent_ok": int(prod_res.persistent_ok),
                    "persistent_threshold": prod_res.persistent_threshold,
                    "cohort_ok": int(prod_res.cohort_ok),
                    "round_reference_cosine": prod_res.round_reference_cosine,
                    "round_reference_threshold": prod_res.round_reference_threshold,
                    "cohort_cosine": prod_res.cohort_cosine,
                    "cohort_threshold": prod_res.cohort_threshold,
                    "anomaly_streak": prod_res.anomaly_streak,
                    "threshold": float("nan"),
                    "risk": prod_res.risk,
                    "warmup": int(server_round <= 1),
                    "history_length": 0,
                    "mu": float("nan"),
                    "sigma": float("nan"),
                    "decision_reason": prod_res.reason,
                })
            else:
                rec_row.update({
                    "secjudge_ok": int(prod_res.secjudge_ok),
                    "norm_sum": prod_res.norm_sum,
                    "temporal_cosine": prod_res.cosine,
                    "threshold": prod_res.threshold,
                    "risk": prod_res.risk,
                    "warmup": int(prod_res.warmup),
                    "history_length": prod_res.history_length,
                    "mu": prod_res.mu,
                    "sigma": prod_res.sigma,
                    "decision_reason": prod_res.decision_reason,
                })
            rows.append(rec_row)
            if self.logical_backend is not None:
                detection_row = {
                    "round": int(server_round),
                    "client_id": int(cid),
                    "is_malicious_gt": int(item["is_malicious_gt"]),
                    "attack_active": int(item["attack_active"]),
                    "is_malicious_pred": int(not prod_res.pass_flag),
                    "secjudge_ok": int(prod_res.norm_ok),
                    "norm_sum": prod_res.raw_norm,
                    "cosine": prod_res.temporal_cosine,
                    "threshold": float("nan"),
                    "risk": prod_res.risk,
                    "pass": int(prod_res.pass_flag),
                    "warmup": int(server_round <= 1),
                    "attack_mode": self.attack_mode,
                    "burst_round": self.burst_round,
                    "history_length": 0,
                    "mu": float("nan"),
                    "sigma": float("nan"),
                    "preprocess_time": 0.0,
                    "sec_judge_time": 0.0,
                    "sec_cos_time": 0.0,
                    "threshold_time": 0.0,
                    "decision_time": 0.0,
                    "detection_time": 0.0,
                    "total_ssdf_time": 0.0,
                    "logical_as1_time": 0.0,
                    "logical_as2_time": 0.0,
                    "raw_norm": prod_res.raw_norm,
                    "norm_median": float("nan"),
                    "norm_mad": float("nan"),
                    "norm_zscore": prod_res.norm_zscore,
                    "reference_cosine": prod_res.reference_cosine,
                    "reference_threshold": prod_res.round_reference_threshold,
                    "anchor_stability": prod_res.anchor_stability,
                    "persistent_score": prod_res.persistent_score,
                    "persistent_ok": int(prod_res.persistent_ok),
                    "persistent_threshold": prod_res.persistent_threshold,
                    "cohort_ok": int(prod_res.cohort_ok),
                    "round_reference_cosine": prod_res.round_reference_cosine,
                    "round_reference_threshold": prod_res.round_reference_threshold,
                    "cohort_cosine": prod_res.cohort_cosine,
                    "cohort_threshold": prod_res.cohort_threshold,
                    "anomaly_streak": prod_res.anomaly_streak,
                    "sign_agreement": float("nan"),
                    "sign_threshold": float("nan"),
                    "temporal_ok": int(prod_res.temporal_ok),
                    "reference_ok": int(prod_res.reference_ok),
                    "history_updated": int(prod_res.history_updated),
                    "decision_reason": prod_res.reason,
                    "detection_eligible": 1,
                    "ssdf_version": self.ssdf_detector_mode,
                    "ssdf_backend": self.ssdf_backend,
                }
            else:
                detection_row = self._res_to_dict(
                    prod_res, cid, self.attack_mode, self.burst_round,
                    self.ssdf_detector_mode,
                )
                detection_row["ssdf_backend"] = self.ssdf_backend
            detection_row["detection_eligible"] = eligible
            if self.logical_backend is not None:
                detection_row.update(prod_res.evidence_diagnostics)
            if not eligible:
                detection_row["history_updated"] = 0
                detection_row["decision_reason"] = "recovery_unavailable"
            detection_rows.append(detection_row)
            if not self.ssdf_filter_enabled:
                rec_row["ssdf_pass_reference"] = (
                    1 if ref_res is not None else float("nan")
                )
                rec_row["ssdf_pass_recovered"] = 1
                rec_row["decision_equal"] = (
                    1 if ref_res is not None else float("nan")
                )
                rec_row["removed"] = 0
            if not rec_row["removed"]:
                retained.append(item["message"])
                retained_cids.append(cid)
                # This is the single canonical representation used by Dynamic
                # Weight, final aggregation, and VL commitments. Recovery output
                # is already float64; normal float32 deltas are promoted exactly.
                retained_deltas.append(
                    np.asarray(item["delta_used"], dtype=np.float64).ravel()
                )
        self.recovery_records[server_round] = rows
        self.detection_records[server_round] = detection_rows

        if self.threshold_controller_enabled:
            # recovery_success at the round level: false only when recovery cannot
            # happen at all (available_shares < emergency threshold).
            self.runtime_threshold_records[server_round] = {
                "round": server_round,
                "available_clients": available_clients,
                "available_shares": available_shares,
                "threshold_mode": threshold_mode,
                "active_share_set": active_share_set,
                "active_threshold": active_threshold,
                "switch_event": int(switch_event),
                "recovery_attempted": round_recovery_attempted,
                # Success only if every attempted client recovery this round succeeded.
                "recovery_success": round_recovery_success,
                "recovery_time": round_recovery_time,
                # D2-V2: B is pre-distributed at setup; Emergency does NOT re-distribute.
                "B_distribution_bytes": 0,
                "A_pre_distribution_payload": round_a_payload,
                "B_pre_distribution_payload": round_b_payload,
                "setup_share_payload_total": round_setup_payload,
                "b_redistributed": 0,
                "decision_equal": round_decision_equal,
            }

        # ---- Dynamic Weight vs FedAvg aggregation ------------------------------
        cos_map: dict[int, float] = {}
        lam_map: dict[int, float] = {}
        aggregated_delta: np.ndarray | None = None
        method = "fedavg"
        if retained:
            reply_contents = [m.content for m in retained]
            if self.dynamic_weight and retained_deltas:
                _t0 = time.perf_counter()
                anchor = (
                    self.logical_backend.aggregation_reference()
                    if self.logical_backend is not None else None
                )
                if self.logical_backend is not None and self.logical_backend.evidence_policy in {"unified_v1_candidate", "unified_v2_candidate"}:
                    cos = calculate_cosine_similarity(retained_deltas)
                elif anchor is None:
                    cos = [1.0] * len(retained_deltas)
                else:
                    cos = calculate_cosine_similarity(retained_deltas, anchor)
                _t1 = time.perf_counter()
                lambdas = calculate_mixed_dynamic_weights(
                    cos,
                    adaptive_blend=self.dynamic_weight_blend,
                    max_weight_multiplier=2.0,
                    confidence=(
                        [production_by_cid[cid].dynamic_weight for cid in retained_cids]
                        if self.ssdf_filter_enabled and self.logical_backend is not None
                        and self.logical_backend.evidence_policy in {"unified_v1_candidate", "unified_v2_candidate"}
                        else None
                    ),
                )
                if self.vl_verification and self.vl_backend == "crypto":
                    weight_scale = 1 << 16
                    lambdas = [
                        value / float(weight_scale)
                        for value in _largest_remainder_weights(
                            lambdas, weight_scale
                        )
                    ]
                _t2 = time.perf_counter()
                arrays, aggregated_delta = _aggregate_retained_deltas(
                    global_state,
                    state_meta,
                    retained_cids,
                    retained_deltas,
                    lambdas,
                    update_scale=(
                        self.vl_update_scale
                        if self.vl_verification and self.vl_backend == "crypto"
                        else None
                    ),
                    weight_scale=(
                        1 << 16
                        if self.vl_verification and self.vl_backend == "crypto"
                        else None
                    ),
                )
                _t3 = time.perf_counter()
                metrics = self._aggregate_train_metrics_checked(
                    server_round, reply_contents
                )
                method = "dynamic_weight"
                for cid, c, lam in zip(retained_cids, cos, lambdas, strict=True):
                    cos_map[cid] = c
                    lam_map[cid] = lam
                self.dw_timing_records[server_round] = {
                    "round": server_round,
                    "dw_cosine_time": _t1 - _t0,
                    "dw_weight_time": _t2 - _t1,
                    "dw_aggregate_time": _t3 - _t2,
                    "dw_total_time": _t3 - _t0,
                }
            else:
                arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
                metrics = self._aggregate_train_metrics_checked(
                    server_round, reply_contents
                )
                method = "fedavg"
        else:
            # Fail closed when every statistically valid update is rejected:
            # keep the last committed global model and advance no model state.
            # Returning ``None`` here lets Flower treat the round as a failed
            # aggregation and can otherwise break the next configure phase.
            arrays, metrics = self._current_arrays, None

        # Server tampering is an independent fault injection.  Keep it outside
        # the VL branch so the VL-disabled control receives the same corrupted
        # candidate model and can demonstrate propagation of an undetected fault.
        tamper_scheduled_round = bool(
            server_round in self.vl_tamper_rounds
            if self.vl_tamper_rounds
            else self.vl_tamper
        )
        # Apply tampering only after the honest proof is generated below.  This
        # preserves the signed transcript and makes the client the detector.
        tamper_applied_round = False

        # Per-client dynamic-weight records (also written for FedAvg runs).
        dw_rows = []
        for row in rows:
            cid = row["client_id"]
            is_mal = 1 if cid in self.malicious_client_ids else 0
            if method == "dynamic_weight":
                if not row["removed"] and cid in cos_map:
                    c = cos_map[cid]
                    lam = lam_map[cid]
                else:
                    c = 0.0
                    lam = 0.0
            else:
                c = float("nan")
                lam = float("nan") if not row["removed"] else 0.0
            row["cos_similarity"] = c
            row["lambda_weight"] = lam
            row["aggregation_method"] = method
            dw_rows.append({
                "round": server_round,
                "client_id": cid,
                "cos_similarity": c,
                "lambda_weight": lam,
                "ssdf_pass": row["ssdf_pass_recovered"],
                "removed": row["removed"],
                "malicious_label": is_mal,
                "aggregation_method": method,
            })
        self.dynamic_weight_records[server_round] = dw_rows

        if client_receipt_failure:
            if rollback_target is None:
                raise RuntimeError("VL_ROLLBACK_STATE_MISSING")
            arrays = ArrayRecord(rollback_target)
            reason = "client_rejected_round_" + str(
                rejected_receipts[0]["round_id"]
            )
            self.vl_verification_records[server_round] = {
                "round": server_round,
                "client_num": len(rejected_receipts),
                "aggregation_method": method,
                "verification_success": 0,
                "verification_policy": "enforce",
                "rollback_applied": 1,
                "failure_reason": reason,
                "vl_backend": self.vl_backend,
                "verification_time": 0.0,
                "hash_generation_time": 0.0,
                "hash_size_bytes": 0,
                "hash_error": float("nan"),
                "claimed_tampered": 0,
                "tamper_scheduled": 0,
                "tamper_applied": 0,
                "client_receipts": json.dumps(client_receipts, sort_keys=True),
                "transport_status": "client_receipt_received",
            }
            self.client_verification_records[server_round] = {
                "round": server_round,
                "verification_success": 0,
                "verification_policy": "enforce",
                "rollback_applied": 1,
                "failure_reason": reason,
                "proof": "",
                "transport_status": "client_receipt_received",
                "client_receipts": json.dumps(client_receipts, sort_keys=True),
            }

        # ---- VL lightweight aggregation verification --------------------------
        if (
            self.vl_verification
            and not client_receipt_failure
            and arrays is not None
            and retained_deltas
        ):
            try:
                if method == "dynamic_weight":
                    if aggregated_delta is None:
                        raise RuntimeError("AGGREGATION_INPUT_MISMATCH")
                    delta_agg_flat = aggregated_delta
                else:
                    arrays_torch = arrays.to_torch_state_dict()
                    delta_agg_flat, _ = flatten_state(
                        {k: arrays_torch[k] - global_state[k] for k in global_state}
                    )
                tamper_scheduled = tamper_scheduled_round
                delta_before_tamper = np.array(delta_agg_flat, copy=True)
                state_hash_before = hashlib.sha256(
                    np.asarray(delta_before_tamper, dtype=np.float64).tobytes()
                ).hexdigest()
                # The proof below is intentionally built from this honest
                # aggregate.  A scheduled tamper is applied after proof
                # creation and is detected by clients from the bound digest.
                tamper_applied = False
                state_hash_after = hashlib.sha256(
                    np.asarray(delta_agg_flat, dtype=np.float64).tobytes()
                ).hexdigest()
                _h0 = time.perf_counter()
                commitments = [self.vl_verifier.hash_update(d) for d in retained_deltas]
                if method == "dynamic_weight":
                    weights = [lam_map[cid] for cid in retained_cids]
                    if len(weights) != len(commitments):
                        raise RuntimeError("AGGREGATION_INPUT_MISMATCH")
                else:
                    sample_counts = [
                        float(msg.content["metrics"].get(self.weighted_by_key, 1.0))
                        for msg in retained
                    ]
                    sample_total = sum(sample_counts)
                    if sample_total <= 0.0:
                        raise RuntimeError("AGGREGATION_INPUT_MISMATCH")
                    weights = [count / sample_total for count in sample_counts]

                # Build the first client-verifiable transcript from the exact
                # retained deltas and weights used above.  Quantization uses a
                # largest-remainder correction so the integer weights sum to
                # 2^16 exactly; this avoids silently changing the declared
                # aggregation during verification.
                weight_scale = 1 << 16
                quantized = _largest_remainder_weights(weights, weight_scale)
                quantized_weights = {
                    int(cid): int(q)
                    for cid, q in zip(retained_cids, quantized, strict=True)
                }
                proof_order = sorted(
                    range(len(retained_cids)), key=lambda i: int(retained_cids[i])
                )
                proof_ids = [int(retained_cids[i]) for i in proof_order]
                proof_commitments = [float(commitments[i]) for i in proof_order]
                candidate_ids = self.candidate_sets.get(server_round, [])
                candidate_set_digest = candidate_digest(candidate_ids)
                participation_records = []
                for msg in retained:
                    # Participation is a signed JSON transcript and therefore
                    # cannot be stored in Flower's numeric-only MetricRecord.
                    # Read the dedicated ConfigRecord, retaining a fallback
                    # for messages produced by older clients.
                    participation_content = msg.content.get("participation")
                    if participation_content is not None:
                        raw_record = participation_content.get(
                            "participation_record", ""
                        )
                    else:
                        raw_record = msg.content["metrics"].get(
                            "participation_record", ""
                        )
                    if raw_record:
                        try:
                            record = json.loads(str(raw_record))
                        except (TypeError, ValueError, json.JSONDecodeError):
                            record = {}
                        if (
                            record
                            and verify_record(record)
                            and str(record.get("candidate_set_digest", ""))
                            == candidate_set_digest
                        ):
                            participation_records.append(record)
                participation_records.sort(key=lambda x: int(x["client_id"]))
                participation_digest = records_digest(participation_records)
                client_receipts = []
                for msg in retained:
                    content = msg.content.get("participation")
                    raw_receipt = content.get("vl_receipt", "") if content else ""
                    if raw_receipt:
                        try:
                            client_receipts.append(json.loads(str(raw_receipt)))
                        except (TypeError, ValueError, json.JSONDecodeError):
                            client_receipts.append({"malformed": True})
                client_receipts.sort(key=lambda x: int(x.get("client_id", -1)))
                # Use the same canonical state digest as the client verifier.
                # Hashing a flattened float64 view here would produce a
                # different digest even for the identical model state.
                model_digest = state_digest(global_state)
                task_id = self.control_run_id or "vl-ssdf-uavfl"
                if self.vl_backend == "crypto":
                    records_by_client: dict[int, ClientHashRecord] = {}
                    encoded_by_client: dict[int, tuple[int, ...]] = {}
                    for msg, cid, delta in zip(
                        retained, retained_cids, retained_deltas, strict=True
                    ):
                        content = msg.content.get("participation")
                        raw_crypto = (
                            content.get("crypto_vl_record", "") if content else ""
                        )
                        if not raw_crypto:
                            raise ValueError(f"missing crypto VL record for client {cid}")
                        record = ClientHashRecord.from_dict(
                            json.loads(str(raw_crypto))
                        )
                        encoded_update = encode_vector(delta, self.vl_update_scale)
                        if record.client_id != cid:
                            raise ValueError(
                                f"crypto VL client ID mismatch for client {cid}: "
                                f"record_client_id={record.client_id}"
                            )
                        if not verify_client_record(
                            record, setup=self.crypto_vl_setup
                        ):
                            raise ValueError(
                                f"crypto VL authentication failed for client {cid}"
                            )
                        recomputed_commitment = homomorphic_hash(
                            encoded_update, setup=self.crypto_vl_setup
                        )
                        if record.commitment != recomputed_commitment:
                            raise ValueError(
                                f"crypto VL commitment mismatch for client {cid}"
                            )
                        records_by_client[cid] = record
                        encoded_by_client[cid] = encoded_update

                    ordered_records = [records_by_client[cid] for cid in proof_ids]
                    ordered_updates = [encoded_by_client[cid] for cid in proof_ids]
                    aggregate_vector = weighted_aggregate(
                        ordered_updates,
                        [quantized_weights[cid] for cid in proof_ids],
                    )
                    decoded_expected = decode_weighted_aggregate(
                        aggregate_vector,
                        update_scale=self.vl_update_scale,
                        weight_scale=weight_scale,
                    )
                    if not np.allclose(
                        np.asarray(delta_agg_flat, dtype=np.float64),
                        decoded_expected,
                        rtol=2e-4,
                        atol=3.0 / self.vl_update_scale,
                    ):
                        raise ValueError("CRYPTO_AGGREGATE_MODEL_MISMATCH")

                    recovery_digest = recovery_state_digest({
                        "threshold": active_threshold,
                        "available_shares": available_shares,
                        "recovered_client_ids": [
                            int(row["client_id"])
                            for row in rows
                            if row.get("recovery_attempted")
                            and row.get("recovery_success")
                        ],
                        "failed_client_ids": [
                            int(row["client_id"])
                            for row in rows
                            if row.get("recovery_attempted")
                            and not row.get("recovery_success")
                        ],
                        "status": "success" if round_recovery_success else "failed",
                    })
                    returned_model_digest = state_digest(
                        arrays.to_torch_state_dict()
                    )
                    proof = make_aggregation_proof(
                        setup=self.crypto_vl_setup,
                        task_id=task_id,
                        round_id=server_round,
                        previous_model_version=model_digest,
                        records=ordered_records,
                        quantized_weights=quantized_weights,
                        aggregate_vector=aggregate_vector,
                        recovery_digest=recovery_digest,
                        returned_model_digest=returned_model_digest,
                    )
                    crypto_result = CryptoVLVerifier(
                        setup=self.crypto_vl_setup,
                        task_id=task_id,
                        expected_round_id=server_round,
                        expected_previous_model_version=model_digest,
                    ).verify(
                        proof=proof,
                        aggregate_vector=aggregate_vector,
                        expected_records=ordered_records,
                        expected_recovery_digest=recovery_digest,
                        received_model_digest=returned_model_digest,
                        expected_candidate_set_digest=candidate_set_digest,
                    )
                    vr = {
                        "verification_success": crypto_result["verification_success"],
                        "verification_time": time.perf_counter() - _h0,
                        "hash_error": 0.0 if crypto_result["verification_success"] else 1.0,
                        "h_agg": str(proof.aggregate_commitment),
                        "h_expected": str(proof.aggregate_commitment),
                    }
                else:
                    proof = make_proof(
                        task_id=task_id,
                        round_id=server_round,
                        model_digest=model_digest,
                        accepted_set=proof_ids,
                        quantized_weights=quantized_weights,
                        commitments=proof_commitments,
                        aggregate_commitment=self.vl_verifier.hash_aggregate(
                            delta_before_tamper
                        ),
                        participation_digest=participation_digest,
                        candidate_set_digest=candidate_set_digest,
                    )
                    vr = self.vl_verifier.verify_aggregation(
                        delta_agg_flat, commitments, weights
                    )
                hash_gen_time = time.perf_counter() - _h0
                self.vl_verification_records[server_round] = {
                    "round": server_round,
                    "client_num": len(retained_deltas),
                    "aggregation_method": method,
                    "verification_success": int(vr["verification_success"]),
                    "verification_time": vr["verification_time"],
                    "hash_error": vr["hash_error"],
                    "hash_generation_time": hash_gen_time,
                    "hash_size_bytes": (
                        len(json.dumps(proof.as_dict(), sort_keys=True).encode("utf-8"))
                        if self.vl_backend == "crypto"
                        else commit_size_bytes() * len(retained_deltas)
                    ),
                    "h_agg": vr["h_agg"],
                    "h_expected": vr["h_expected"],
                    "claimed_tampered": int(tamper_applied),
                    "tamper_scheduled": int(tamper_scheduled),
                    "tamper_applied": int(tamper_applied),
                    "control_run_id": self.control_run_id,
                    "paired_run_id": self.paired_run_id,
                    "vl_backend": self.vl_backend,
                    "verification_policy": "enforce" if self.vl_enforce else "observe",
                    "rollback_applied": 0,
                    "aggregation_state_hash_before": state_hash_before,
                    "aggregation_state_hash_after": state_hash_after,
                }
                self.client_verification_records[server_round] = {
                    "round": server_round,
                    "task_id": task_id,
                    "verification_success": int(vr["verification_success"]),
                    "model_digest": model_digest,
                    "accepted_set": json.dumps(sorted(retained_cids)),
                    "quantized_weights": json.dumps(quantized_weights, sort_keys=True),
                    "proof_binding": proof.binding,
                    "proof": json.dumps(proof.as_dict(), sort_keys=True),
                    "transport_status": "client_receipt_returned",
                    "verification_policy": "enforce" if self.vl_enforce else "observe",
                    "participation_digest": participation_digest,
                    "participation_records": json.dumps(
                        participation_records, sort_keys=True
                    ),
                    "client_receipts": json.dumps(client_receipts, sort_keys=True),
                    "candidate_set": json.dumps(candidate_ids),
                    "candidate_set_digest": candidate_set_digest,
                    "vl_backend": self.vl_backend,
                }
                self.vl_verification_records[server_round].update(
                    self.client_verification_records[server_round]
                )
                if self.vl_enforce and not bool(vr["verification_success"]):
                    # Fail closed: the previous global state is the last
                    # verified state available to the server for this round.
                    # Do not expose the rejected candidate to evaluation or
                    # to the next round's client proof transcript.
                    if global_state is not None:
                        arrays = ArrayRecord(global_state)
                    self.vl_verification_records[server_round].update(
                        {
                            "rollback_applied": 1,
                            "failure_reason": "aggregation_proof_mismatch",
                        }
                    )
                    self.client_verification_records[server_round].update(
                        {
                            "rollback_applied": 1,
                            "failure_reason": "aggregation_proof_mismatch",
                        }
                    )
            except Exception as exc:  # pragma: no cover - defensive
                if (
                    isinstance(exc, RuntimeError)
                    and str(exc) == "AGGREGATION_INPUT_MISMATCH"
                ):
                    raise
                self.vl_verification_records[server_round] = {
                    "round": server_round,
                    "client_num": len(retained_deltas),
                    "aggregation_method": method,
                    "verification_success": 0,
                    "verification_time": 0.0,
                    "hash_error": float("nan"),
                    "hash_generation_time": 0.0,
                    "hash_size_bytes": 0,
                    "claimed_tampered": 0,
                    "tamper_scheduled": 0,
                    "tamper_applied": 0,
                    "control_run_id": self.control_run_id,
                    "paired_run_id": self.paired_run_id,
                    "verification_policy": "enforce" if self.vl_enforce else "observe",
                    "rollback_applied": 0,
                    "aggregation_state_hash_before": "",
                    "aggregation_state_hash_after": "",
                    "error": str(exc),
                }
                self.client_verification_records[server_round] = {
                    "round": server_round,
                    "transport_status": "not_generated",
                    "verification_policy": "enforce" if self.vl_enforce else "observe",
                    "rollback_applied": 0,
                    "failure_reason": str(exc),
                    "client_receipts": "[]",
                    "error": str(exc),
                }
                if self.vl_enforce:
                    if global_state is not None:
                        arrays = ArrayRecord(global_state)
                    self.vl_verification_records[server_round].update(
                        {
                            "rollback_applied": int(global_state is not None),
                            "failure_reason": f"verification_error:{exc}",
                        }
                    )
                    self.client_verification_records[server_round].update(
                        {
                            "rollback_applied": int(global_state is not None),
                            "failure_reason": f"verification_error:{exc}",
                        }
                    )
                self.vl_verification_records[server_round].update(
                    self.client_verification_records[server_round]
                )

        # Deliver the tampered model only after the honest proof has been
        # finalized.  Clients will reject it on the next training request
        # because the signed returned-model digest no longer matches.
        if (
            tamper_scheduled_round
            and self.vl_verification
            and arrays is not None
            and retained_deltas
            and int(self.vl_verification_records.get(server_round, {}).get(
                "verification_success", 0
            ) or 0)
        ):
            honest_state = arrays.to_torch_state_dict()
            honest_delta, honest_meta = flatten_state({
                key: honest_state[key] - global_state[key]
                for key in global_state
            })
            noise_rng = np.random.default_rng([self.seed, server_round, 77])
            scale = float(np.linalg.norm(honest_delta)) + 1e-9
            tampered_delta = honest_delta + noise_rng.normal(
                0.0, 1.0, size=honest_delta.shape
            ) * (0.1 * scale)
            tampered_state = unflatten_state(tampered_delta, honest_meta)
            arrays = ArrayRecord({
                key: global_state[key] + tampered_state[key].to(
                    device=global_state[key].device,
                    dtype=global_state[key].dtype,
                )
                for key in global_state
            })
            tamper_applied_round = not np.array_equal(honest_delta, tampered_delta)
            row = self.vl_verification_records[server_round]
            row.update({
                "tamper_applied": int(tamper_applied_round),
                "tamper_delivery": "after_proof",
                "tamper_detection_location": "client_next_round",
                "tampered_model_hash": hashlib.sha256(
                    np.asarray(tampered_delta, dtype=np.float64).tobytes()
                ).hexdigest(),
            })

        if self.vl_commit_mode == "precommit":
            precommit_start = time.perf_counter()
            row = self.vl_verification_records.setdefault(server_round, {"round": server_round})
            success = False
            if row.get("verification_success") and retained_deltas and arrays is not None:
                # Verify the actual candidate AFTER any transport/result tamper,
                # against the canonical numerator and authenticated input records.
                result = CryptoVLVerifier(
                    setup=self.crypto_vl_setup,
                    task_id=task_id,
                    expected_round_id=server_round,
                    expected_previous_model_version=model_digest,
                ).verify(
                    proof=proof,
                    aggregate_vector=aggregate_vector,
                    expected_records=ordered_records,
                    expected_recovery_digest=recovery_digest,
                    received_model_digest=state_digest(arrays.to_torch_state_dict()),
                    expected_candidate_set_digest=candidate_set_digest,
                )
                success = bool(result["verification_success"])
                row["precommit_failure_details"] = json.dumps(result, sort_keys=True)
                row["candidate_model_digest"] = state_digest(arrays.to_torch_state_dict())
            row.update({
                "precommit_verification_time": time.perf_counter() - precommit_start,
                "precommit_verification_success": int(success),
                "verification_success": int(success),
                "rollback_applied": 0,
                "failure_reason": "" if success else "precommit_candidate_rejected",
                "tamper_detection_location": "precommit" if row.get("tamper_applied") else "",
            })
            self.client_verification_records.setdefault(server_round, {}).update(row)
            if not success:
                arrays = self._current_arrays

        verification_row = self.vl_verification_records.get(server_round, {})
        verification_success = (
            bool(int(verification_row.get("verification_success", 0)))
            if verification_row
            else None
        )
        self.aggregation_records[server_round] = build_vl_aggregation_record(
            server_round=server_round,
            method=method,
            valid_client_ids=[
                int(msg.content["metrics"].get("client_partition_id", -1))
                for msg in valid
            ],
            retained_client_ids=retained_cids,
            unavailable_ids=unavailable_ids,
            available_clients=available_clients,
            aggregation_completed=arrays is not None,
            recovery_success=round_recovery_success,
            verification_enabled=self.vl_verification,
            verification_success=verification_success,
        )
        self.aggregation_records[server_round].update(
            {
                "verification_policy": verification_row.get(
                    "verification_policy",
                    "enforce" if self.vl_enforce else "observe",
                ),
                "rollback_applied": int(verification_row.get("rollback_applied", 0) or 0),
                "verification_failure_reason": verification_row.get("failure_reason", ""),
            }
        )

        # Protocol trace is derived from the canonical execution path above.
        # It records state transitions without changing any aggregation inputs.
        candidate_ids = list(self.candidate_sets.get(server_round, []))
        valid_ids = [
            int(msg.content["metrics"].get("client_partition_id", -1))
            for msg in valid
        ]
        if self.dropout_timing == "post_share":
            provisioned_ids = sorted(candidate_ids or valid_ids)
            available_ids = sorted(set(provisioned_ids) - unavailable_ids)
            recovery_mode = (
                str(self.runtime_threshold_records.get(server_round, {}).get(
                    "threshold_mode", "POST_SHARE_RECOVERY"
                ))
                if unavailable_ids else "POST_SHARE_NO_DROPOUT"
            )
        else:
            provisioned_ids = sorted(valid_ids)
            available_ids = sorted(valid_ids)
            recovery_mode = "PRE_SHARE_FAILURE" if len(valid_ids) < len(candidate_ids) else "NONE"
        detection_by_client = {
            int(row.get("client_id", -1)): {
                "pass": int(bool(row.get("ssdf_pass_recovered", row.get("pass", 0)))),
                "temporal_cosine": row.get("temporal_cosine"),
                "threshold": row.get("threshold"),
                "risk": row.get("risk"),
                "removed": int(row.get("removed", 0)),
            }
            for row in self.recovery_records.get(server_round, [])
        }
        weight_by_client = {
            int(row.get("client_id", -1)): {
                "round_reference_cosine": row.get("cos_similarity"),
                "lambda": row.get("lambda_weight"),
            }
            for row in self.dynamic_weight_records.get(server_round, [])
        }
        verification = dict(self.vl_verification_records.get(server_round, {}))
        self.protocol_trace_records[server_round] = {
            "round": server_round,
            "candidate_set": sorted(candidate_ids),
            "provisioned_set": provisioned_ids,
            "available_set": available_ids,
            "unavailable_set": sorted(unavailable_ids),
            "accepted_set": sorted(int(cid) for cid in retained_cids),
            "recovery_mode": recovery_mode,
            "active_threshold": self.runtime_threshold_records.get(server_round, {}).get(
                "active_threshold"
            ),
            "ssdf": detection_by_client,
            "weights": weight_by_client,
            "aggregation_method": method,
            "recovery_success": bool(round_recovery_success),
            "verification_enabled": bool(self.vl_verification),
            "verification_success": verification.get("verification_success"),
            "client_verification_transport": verification.get("transport_status"),
        }

        # RecoveryStateStore contains full A/B share vectors. They are no
        # longer needed once this round's audit rows and protocol trace have
        # been materialized, so release them before the next Flower round.
        self.recovery_store.clear_round(server_round)

        return arrays, metrics
