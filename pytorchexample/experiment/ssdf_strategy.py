"""FedAvg with SSDF core detection (server-side plaintext filter)."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from .recorded_fedavg import RecordingFedAvg
from .ssdf import SSDFDetector
from .ssdf_backends import LogicalDualServerSSDFBackend
from .dynamic_weight import calculate_cosine_similarity, calculate_dynamic_weights, calculate_mixed_dynamic_weights
from .recovery import flatten_state, unflatten_state
import numpy as np
import torch


def torch_flatten_state(state):
    flat, _ = flatten_state(state)
    return np.asarray(flat, dtype=np.float64)


def weighted_delta_aggregate(global_state, messages, weights):
    local0 = messages[0].content["arrays"].to_torch_state_dict()
    _, meta = flatten_state(global_state)
    deltas = []
    for msg in messages:
        local = msg.content["arrays"].to_torch_state_dict()
        delta, _ = flatten_state({k: local[k] - global_state[k] for k in global_state})
        deltas.append(np.asarray(delta, dtype=np.float64))
    aggregate = sum(float(w) * d for w, d in zip(weights, deltas, strict=True))
    delta_state = unflatten_state(aggregate, meta)
    return type(messages[0].content["arrays"])({
        k: global_state[k] + delta_state[k].to(global_state[k].dtype)
        for k in global_state
    })


class SSDFStrategy(RecordingFedAvg):
    """FedAvg that runs SSDF detection and aggregates over retained clients only.

    Detection never feeds the client (no oracle); it only filters the set of inputs
    to the standard FedAvg aggregation.
    """

    def __init__(
        self,
        *args,
        malicious_client_ids: list[int] | None = None,
        ema_beta: float = 0.9,
        k: float = 2.0,
        norm_tol: float = 1e-3,
        warmup_rounds: int = 2,
        norm_floor: float = 1e-12,
        eps: float = 1e-8,
        reference_k: float = 2.5,
        reference_floor: float = -0.25,
        norm_k: float = 2.70,
        history_limit: int = 20,
        detector_mode: str = "v2_full",
        attack_mode: str = "persistent",
        burst_round: int = 0,
        unavailable_by_round: dict | None = None,
        ssdf_backend: str = "plaintext",
        evidence_policy: str = "legacy",
        dynamic_weight: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.malicious_client_ids = set(malicious_client_ids or [])
        self.detector = SSDFDetector(
            ema_beta=ema_beta, k=k, norm_tol=norm_tol,
            warmup_rounds=warmup_rounds, eps=eps, norm_floor=norm_floor,
            reference_k=reference_k, reference_floor=reference_floor,
            norm_k=norm_k, history_limit=history_limit,
            detector_mode=detector_mode,
        )
        self.ssdf_backend = str(ssdf_backend).lower()
        if evidence_policy != "legacy" and self.ssdf_backend != "logical_dual_server":
            raise ValueError("unified evidence requires logical_dual_server")
        self.dynamic_weight = bool(dynamic_weight)
        if self.ssdf_backend not in {"plaintext", "logical_dual_server"}:
            raise ValueError("unknown ssdf_backend")
        self.logical_backend = (
            LogicalDualServerSSDFBackend(
                reference_k=reference_k,
                reference_floor=reference_floor,
                norm_k=norm_k,
                history_limit=history_limit,
                seed=int(kwargs.get("seed", 0) or 0),
                evidence_policy=evidence_policy,
            )
            if self.ssdf_backend == "logical_dual_server" else None
        )
        self.detection_records: dict[int, list[dict]] = {}
        self._current_arrays = None
        self.attack_mode = attack_mode
        self.ssdf_detector_mode = detector_mode
        self.burst_round = burst_round
        self.unavailable_by_round = unavailable_by_round or {}

    def configure_train(self, server_round, arrays, config, grid):
        self._current_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def _aggregate_train_metrics_checked(self, server_round, messages):
        """Validate Flower's metric weights before its division-based reducer."""
        weighted_clients = []
        client_diagnostics = []
        for message in messages:
            content = getattr(message, "content", message)
            metrics = content["metrics"]
            client_id = int(metrics.get("client_partition_id", -1))
            client_diagnostics.append(
                {
                    "client_id": client_id,
                    "invalid_update": metrics.get("invalid_update", "missing"),
                    "train_loss": metrics.get("train_loss", "missing"),
                }
            )
            try:
                weight = float(metrics[self.weighted_by_key])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "TRAIN_METRIC_WEIGHT_INVALID "
                    f"round={server_round} client_id={client_id} "
                    f"key={self.weighted_by_key!r}"
                ) from exc
            weighted_clients.append((client_id, weight))

        total_weight = sum(weight for _, weight in weighted_clients)
        if (
            not weighted_clients
            or not np.isfinite(total_weight)
            or total_weight <= 0.0
            or any(not np.isfinite(weight) or weight < 0.0 for _, weight in weighted_clients)
        ):
            raise RuntimeError(
                "TRAIN_METRIC_WEIGHT_TOTAL_INVALID "
                f"round={server_round} key={self.weighted_by_key!r} "
                f"total={total_weight!r} "
                f"clients={[f'client_id={cid}:weight={weight!r}' for cid, weight in weighted_clients]} "
                f"diagnostics={client_diagnostics!r}"
            )
        return self.train_metrics_aggr_fn(messages, self.weighted_by_key)

    def aggregate_train(self, server_round, replies):
        # Record attack / dropout / time stats for ALL replies (observability).
        self._current_round = server_round
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        unavailable_ids = set(self.unavailable_by_round.get(server_round, []))
        valid = [m for m in valid if int(m.content["metrics"].get("client_partition_id", -1)) not in unavailable_ids]

        global_state = (
            self._current_arrays.to_torch_state_dict() if self._current_arrays else None
        )
        candidates = []
        for msg in valid:
            cid = int(msg.content["metrics"].get("client_partition_id", -1))
            local = msg.content["arrays"].to_torch_state_dict()
            delta = {k: local[k] - global_state[k] for k in global_state}
            is_mal_gt = 1 if cid in self.malicious_client_ids else 0
            attack_active = int(msg.content["metrics"].get("attack_active", 0))
            candidates.append(
                {
                    "client_id": cid,
                    "server_round": server_round,
                    "delta": delta,
                    "is_malicious_gt": is_mal_gt,
                    "attack_active": attack_active,
                    "message": msg,
                }
            )
        if self.logical_backend is not None:
            backend_results = self.logical_backend.detect_round(candidates)
            results = backend_results
        else:
            results = self.detector.detect_round(candidates)
        rows = []
        retained = []
        confidence_by_cid = {res.client_id: res.dynamic_weight for res in results} if self.logical_backend is not None else {}
        for candidate, res in zip(candidates, results, strict=True):
            cid = int(candidate["client_id"])
            if self.logical_backend is not None:
                rows.append({
                    "round": int(server_round), "client_id": cid,
                    "is_malicious_gt": int(candidate.get("is_malicious_gt", 0)),
                    "attack_active": int(candidate.get("attack_active", 0)),
                    "is_malicious_pred": int(not res.pass_flag),
                    "secjudge_ok": int(res.norm_ok), "norm_sum": res.raw_norm,
                    "cosine": res.temporal_cosine, "threshold": float("nan"),
                    "norm_zscore": res.norm_zscore,
                    "reference_cosine": res.reference_cosine,
                    "reference_threshold": res.round_reference_threshold,
                    "round_reference_cosine": res.round_reference_cosine,
                    "round_reference_threshold": res.round_reference_threshold,
                    "cohort_cosine": res.cohort_cosine,
                    "cohort_threshold": res.cohort_threshold,
                    "reference_ok": int(res.reference_ok),
                    "temporal_ok": int(res.temporal_ok),
                    "history_updated": int(res.history_updated),
                    "detection_eligible": 1,
                    "risk": res.risk, "pass": int(res.pass_flag),
                    "warmup": int(server_round <= 2),
                    "history_length": 0, "dynamic_weight": res.dynamic_weight,
                    "decision_reason": res.reason,
                    "ssdf_backend": self.ssdf_backend,
                    **res.evidence_diagnostics,
                })
            else:
                rows.append(self._res_to_dict(
                    res, cid, self.attack_mode, self.burst_round,
                    self.ssdf_detector_mode,
                ))
            if res.pass_flag:
                retained.append(candidate["message"])
        self.detection_records[server_round] = rows

        if retained:
            reply_contents = [m.content for m in retained]
            if self.dynamic_weight and global_state is not None:
                deltas = []
                finite_messages = []
                for msg in retained:
                    local = msg.content["arrays"].to_torch_state_dict()
                    delta = torch_flatten_state({k: local[k] - global_state[k] for k in global_state})
                    if np.isfinite(delta).all():
                        deltas.append(delta)
                        finite_messages.append(msg)
                if deltas:
                    cos = calculate_cosine_similarity(deltas)
                    if self.logical_backend is not None and self.logical_backend.evidence_policy in {"unified_v1_candidate", "unified_v2_candidate"}:
                        lambdas = calculate_mixed_dynamic_weights(
                            cos, confidence=[confidence_by_cid[int(m.content["metrics"]["client_partition_id"])] for m in finite_messages]
                        )
                    else:
                        lambdas = calculate_dynamic_weights(cos)
                    arrays = weighted_delta_aggregate(global_state, finite_messages, lambdas)
                    reply_contents = [m.content for m in finite_messages]
                else:
                    arrays = self._current_arrays
            else:
                arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            metrics = self._aggregate_train_metrics_checked(server_round, reply_contents)
        else:
            # Fail closed without poisoning the global model: preserve the
            # previous global arrays and emit a zero-weight metric record.
            arrays = self._current_arrays
            metrics = None
        return arrays, metrics

    @staticmethod
    def _res_to_dict(
        res, cid, attack_mode="persistent", burst_round=0,
        ssdf_version="v2_full",
    ):
        return {
            "round": res.round,
            "client_id": cid,
            "is_malicious_gt": res.is_malicious_gt,
            "attack_active": res.attack_active,
            "is_malicious_pred": res.is_malicious_pred,
            "secjudge_ok": int(res.secjudge_ok),
            "norm_sum": res.norm_sum,
            "cosine": res.cosine,
            "threshold": res.threshold,
            "risk": res.risk,
            "pass": int(res.pass_flag),
            "warmup": int(res.warmup),
            "attack_mode": attack_mode,
            "burst_round": burst_round,
            "history_length": res.history_length,
            "mu": res.mu,
            "sigma": res.sigma,
            "preprocess_time": res.preprocess_time,
            "sec_judge_time": res.sec_judge_time,
            "sec_cos_time": res.sec_cos_time,
            "threshold_time": res.threshold_time,
            "decision_time": res.decision_time,
            "detection_time": res.detection_time,
            "total_ssdf_time": res.total_ssdf_time,
            "logical_as1_time": res.logical_as1_time,
            "logical_as2_time": res.logical_as2_time,
            "raw_norm": res.raw_norm,
            "norm_median": res.norm_median,
            "norm_mad": res.norm_mad,
            "norm_zscore": res.norm_zscore,
            "reference_cosine": res.reference_cosine,
            "reference_threshold": res.reference_threshold,
            "sign_agreement": res.sign_agreement,
            "sign_threshold": res.sign_threshold,
            "temporal_ok": int(res.temporal_ok),
            "reference_ok": int(res.reference_ok),
            "history_updated": int(res.history_updated),
            "decision_reason": res.decision_reason,
            "detection_eligible": 1,
            "ssdf_version": ssdf_version,
        }
