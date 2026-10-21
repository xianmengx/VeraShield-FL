"""A fully-observational ``FedAvg`` wrapper.

``RecordingFedAvg`` subclasses the real Flower 1.31 ``FedAvg`` and ONLY records
instrumentation (participating clients, successful replies, failures, round
timing, per-client training times, communication payload). Every overridden hook
delegates to ``super()`` unchanged, so the FedAvg aggregation behaviour, client
updates, weight computation and model/dataset are completely untouched.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import math
import time
from statistics import mean

import torch
from flwr.serverapp.strategy import FedAvg

from .attack import ATTACK_TYPE_FROM_CODE
from .utils import model_payload_bytes


# Error code used by the client to signal "simulated client dropout".
DROPOUT_ERROR_CODE = 1000


def learning_rate_for_round(
    *, base_lr: float, min_lr: float, schedule: str,
    server_round: int, total_rounds: int,
) -> float:
    """Return the configured client learning rate for one FL round."""
    if base_lr <= 0 or min_lr < 0 or min_lr > base_lr:
        raise ValueError("learning rates must satisfy 0 <= min_lr <= base_lr")
    if server_round < 1 or total_rounds < 1:
        raise ValueError("round indices must be positive")
    if schedule == "constant" or total_rounds == 1:
        return float(base_lr)
    if schedule != "cosine":
        raise ValueError(f"unsupported learning-rate schedule: {schedule}")
    progress = min(server_round - 1, total_rounds - 1) / (total_rounds - 1)
    return float(
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    )


def _f(value) -> float:
    """Safely coerce a metric value to float (NaN on failure)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class RecordingFedAvg(FedAvg):
    """FedAvg that records observability metrics without changing aggregation."""

    def __init__(
        self, *args, seed: int | None = None,
        base_learning_rate: float = 0.1,
        min_learning_rate: float = 0.0,
        learning_rate_schedule: str = "constant",
        total_rounds: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.base_learning_rate = float(base_learning_rate)
        self.min_learning_rate = float(min_learning_rate)
        self.learning_rate_schedule = str(learning_rate_schedule).lower()
        self.total_rounds = int(total_rounds)
        learning_rate_for_round(
            base_lr=self.base_learning_rate,
            min_lr=self.min_learning_rate,
            schedule=self.learning_rate_schedule,
            server_round=1,
            total_rounds=self.total_rounds,
        )
        self.records: dict[int, dict[str, object]] = {}
        self.total_nodes: int | None = None
        self._t_round_start: dict[int, float] = {}
        self._current_round: int | None = None

    def _rec(self, r: int) -> dict[str, object]:
        return self.records.setdefault(r, {})

    def configure_train(self, server_round, arrays, config, grid):
        """Record round start + participating count, then delegate to FedAvg."""
        effective_lr = learning_rate_for_round(
            base_lr=self.base_learning_rate,
            min_lr=self.min_learning_rate,
            schedule=self.learning_rate_schedule,
            server_round=server_round,
            total_rounds=self.total_rounds,
        )
        config["lr"] = effective_lr
        now = time.perf_counter()
        if server_round > 1 and (server_round - 1) in self._t_round_start:
            # Real training round (r-1) finishes when r's configure_train starts.
            self._rec(server_round - 1)["round_time"] = (
                now - self._t_round_start[server_round - 1]
            )
        self._t_round_start[server_round] = now

        if self.total_nodes is None:
            self.total_nodes = len(list(grid.get_node_ids()))

        messages = super().configure_train(server_round, arrays, config, grid)
        payload = model_payload_bytes(arrays)
        rec = self._rec(server_round)
        rec["num_clients"] = self.total_nodes
        rec["participating_clients"] = len(messages)
        rec["model_size_bytes"] = payload
        rec["comm_payload_down"] = payload * len(messages)
        rec["comm_payload_up"] = 0
        rec["comm_payload_total"] = 0
        rec["learning_rate"] = effective_lr
        return messages

    def aggregate_train(self, server_round, replies):
        self._current_round = server_round
        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(self, server_round, replies):
        self._current_round = server_round
        return super().aggregate_evaluate(server_round, replies)

    def _check_and_log_replies(self, replies, is_train, validate=True):
        """Count successful/failed replies and collect payload + per-client times."""
        valid, errors = super()._check_and_log_replies(replies, is_train, validate)
        received_replies = len(valid) + len(errors)
        invalid_updates = 0
        if is_train:
            finite_valid = []
            for message in valid:
                try:
                    invalid_flag = int(message.content["metrics"].get("invalid_update", 0))
                    tensors = message.content["arrays"].to_torch_state_dict()
                    finite = all(bool(torch.isfinite(t).all()) for t in tensors.values())
                except Exception:
                    invalid_flag, finite = 1, False
                if not invalid_flag and finite:
                    finite_valid.append(message)
                else:
                    invalid_updates += 1
            valid = finite_valid
        r = self._current_round
        if r is None:
            return valid, errors

        rec = self._rec(r)
        if is_train:
            rec['received_replies'] = received_replies
            rec['invalid_update_replies'] = invalid_updates
            rec["successful_replies"] = len(valid)
            rec["failures"] = len(errors)
            dropout_count = 0
            for m in errors:
                try:
                    if m.error is not None and m.error.code == DROPOUT_ERROR_CODE:
                        dropout_count += 1
                except Exception:  # pragma: no cover - defensive
                    pass
            rec["dropout_failures"] = dropout_count
            rec["runtime_failures"] = max(0, len(errors) - dropout_count)
            rec["actual_dropout_clients"] = dropout_count
            up = 0
            times = []
            compute_times = []
            attack_rows = []
            for msg in valid:
                try:
                    up += model_payload_bytes(msg.content["arrays"])
                except Exception:  # pragma: no cover - defensive
                    pass
                try:
                    times.append(float(msg.content["metrics"]["client_train_time"]))
                except Exception:  # pragma: no cover - defensive
                    pass
                try:
                    compute_times.append(
                        float(msg.content["metrics"]["client_compute_time"])
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
                try:
                    m = msg.content["metrics"]
                    attack_rows.append({
                        "client_id": int(m.get("client_partition_id", -1)),
                        "is_malicious": bool(m.get("is_malicious", 0)),
                        "update_norm_before": _f(m.get("update_norm_before")),
                        "update_norm_after": _f(m.get("update_norm_after")),
                        "update_cosine_to_original": _f(
                            m.get("update_cosine_to_original")
                        ),
                        "update_norm_ratio": _f(m.get("update_norm_ratio")),
                        "attack_type": ATTACK_TYPE_FROM_CODE.get(
                            int(m.get("attack_type_code", 0)), "none"
                        ),
                        "attack_strength": _f(m.get("attack_strength")),
                        "noise_l2_norm": _f(m.get("noise_l2_norm")),
                        "noise_mean": _f(m.get("noise_mean")),
                        "noise_std_observed": _f(m.get("noise_std_observed")),
                        "update_noise_ratio": _f(m.get("update_noise_ratio")),
                        "backdoor_poisoned_examples": _f(
                            m.get("backdoor_poisoned_examples")
                        ),
                        "backdoor_poison_rate_observed": _f(
                            m.get("backdoor_poison_rate_observed")
                        ),
                    })
                except Exception:  # pragma: no cover - defensive
                    pass
            if attack_rows:
                rec["attack_statistics"] = attack_rows
            rec["comm_payload_up"] = up
            rec["comm_payload_total"] = int(rec.get("comm_payload_down", 0)) + up
            if times:
                rec["client_train_time_mean"] = mean(times)
                rec["client_train_time_max"] = max(times)
            if compute_times:
                rec["client_compute_time_mean"] = mean(compute_times)
                rec["client_compute_time_max"] = max(compute_times)
                _cmean = mean(compute_times)
                rec["straggler_ratio"] = (
                    float(max(compute_times)) / _cmean if _cmean > 0 else 0.0
                )
        else:
            rec["eval_successful_replies"] = len(valid)
            rec["eval_failures"] = len(errors)
        return valid, errors

    def finalize(self, elapsed: float) -> None:
        """Settle the last round's round_time and the round-0 sentinel records."""
        train_rounds = sorted(r for r in self._t_round_start if r >= 1)
        if train_rounds:
            last = train_rounds[-1]
            settled = sum(
                float(self._rec(r).get("round_time", 0)) for r in train_rounds[:-1]
            )
            self._rec(last)["round_time"] = max(0.0, elapsed - settled)

        if 0 not in self.records:
            self.records[0] = {}
        r0 = self.records[0]
        r0.setdefault("round_time", 0)
        r0.setdefault("num_clients", self.total_nodes or 0)
        r0.setdefault("participating_clients", 0)
        r0.setdefault("successful_replies", 0)
        r0.setdefault("failures", 0)
        r0.setdefault("comm_payload_down", 0)
        r0.setdefault("comm_payload_up", 0)
        r0.setdefault("comm_payload_total", 0)
