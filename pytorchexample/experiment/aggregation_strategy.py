"""Unified baseline strategy for classical and robust model-delta aggregation."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import numpy as np
from flwr.app import ArrayRecord

from .recorded_fedavg import RecordingFedAvg
from .aggregation import aggregate
from .feddmc import FedDMCDetector
from .recovery import flatten_state, unflatten_state


class AggregationStrategy(RecordingFedAvg):
    """FedAvg / Krum / Trimmed Mean / FedDMC on identical model deltas.

    All methods receive the same ``delta_i = local_model_i - global_model``
    (model-delta, not gradient) and the same scenario; only the aggregation rule
    differs. Partial/Fractional participation comes from the scenario manifest.
    """

    def __init__(
        self,
        *args,
        aggregation_method: str = "fedavg",
        krum_f: int = 2,
        trim_ratio: float = 0.2,
        scenario_ids: dict | None = None,
        unavailable_by_round: dict | None = None,
        vl_tamper: bool = False,
        vl_tamper_rounds: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.aggregation_method = aggregation_method
        self.krum_f = krum_f
        self.trim_ratio = trim_ratio
        self.scenario_ids = scenario_ids or {}
        self.unavailable_by_round = unavailable_by_round or {}
        self.vl_tamper = bool(vl_tamper)
        self.vl_tamper_rounds = set(int(r) for r in (vl_tamper_rounds or []))
        self.aggregation_records: dict[int, dict] = {}
        self.feddmc_detector = (
            FedDMCDetector() if aggregation_method == "feddmc" else None
        )

    def configure_train(self, server_round, arrays, config, grid):
        self._current_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        self._current_round = server_round
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        unavailable_ids = set(self.unavailable_by_round.get(server_round, []))
        tamper_scheduled = (
            server_round in self.vl_tamper_rounds
            if self.vl_tamper_rounds else self.vl_tamper
        )
        # Register every round before any early return. No model or policy changes.
        counts = self._rec(server_round)
        common = {
            'round': server_round, 'method': self.aggregation_method,
            'num_clients': 0,
            'total_clients': len(valid), 'available_clients': 0,
            'effective_clients': 0, 'client_ids': [],
            'dropped_clients': len(unavailable_ids),
            'unavailable_ids': sorted(unavailable_ids),
            'tamper_scheduled': int(tamper_scheduled), 'tamper_applied': 0,
            'received_replies': counts.get('received_replies', 0),
            'invalid_update_replies': counts.get('invalid_update_replies', 0),
            'dropout_failures': counts.get('dropout_failures', 0),
            'runtime_failures': counts.get('runtime_failures', 0),
        }
        if self.aggregation_method == 'krum':
            common.update(krum_f=self.krum_f, krum_valid=False)
        elif self.aggregation_method == 'trimmed_mean':
            common.update(trim_ratio=self.trim_ratio, num_removed_low=0, num_removed_high=0)
        self.aggregation_records[server_round] = common
        global_state = (
            self._current_arrays.to_torch_state_dict() if self._current_arrays else None
        )
        if not valid or global_state is None:
            common['aggregation_status'] = (
                'aggregation_failed_missing_global_state' if global_state is None
                else 'aggregation_failed_no_valid_replies'
            )
            return None, None
        client_ids = []
        updates = []
        sizes = []
        for msg in valid:
            cid = int(msg.content["metrics"].get("client_partition_id", -1))
            if cid in unavailable_ids:
                continue  # unavailable client excluded from this round's aggregation
            local = msg.content["arrays"].to_torch_state_dict()
            delta = {k: local[k] - global_state[k] for k in global_state}
            client_ids.append(cid)
            updates.append(delta)
            sizes.append(float(msg.content["metrics"].get("num-examples", 1.0)))

        if not updates:
            common['aggregation_status'] = 'aggregation_failed_due_to_insufficient_clients'
            return None, None

        if self.feddmc_detector is not None:
            try:
                delta_flat, diag = self.feddmc_detector.detect(client_ids, updates)
            except ValueError as exc:
                # A non-VL baseline has no rollback path when a prior server
                # tamper makes subsequent client updates non-finite.  Record
                # the failed aggregation and let Flower continue the run so
                # the baseline remains measurable instead of aborting the
                # complete comparison.
                diag = {
                    "method": "feddmc",
                    "aggregation_status": "aggregation_failed_non_finite_updates",
                    "detector": "pca_hdbscan_cumulative_majority",
                    "clustering_status": "skipped_non_finite_updates",
                    "failure_reason": str(exc),
                    "effective_clients": 0,
                    "retained_client_ids": [],
                }
                arrays = None
            else:
                delta_state = unflatten_state(delta_flat, flatten_state(updates[0])[1])
                arrays = ArrayRecord({
                    key: global_state[key] + delta_state[key].to(
                        device=global_state[key].device,
                        dtype=global_state[key].dtype,
                    )
                    for key in global_state
                })
                diag["weights_source"] = "uniform"
        else:
            weights = sizes if self.aggregation_method == "fedavg" else None
            arrays, diag = aggregate(
                global_state,
                updates,
                method=self.aggregation_method,
                krum_f=self.krum_f,
                trim_ratio=self.trim_ratio,
                weights=weights,
            )
        diag.setdefault('effective_clients', len(client_ids))
        diag = {**common, **diag}
        diag["round"] = server_round
        diag["client_ids"] = client_ids
        diag["total_clients"] = len(valid)
        diag["available_clients"] = len(client_ids)
        diag["dropped_clients"] = len(unavailable_ids)
        diag["unavailable_ids"] = sorted(unavailable_ids)
        if "krum_selected_index" in diag and diag["krum_selected_index"] is not None:
            diag["krum_selected_client"] = client_ids[diag["krum_selected_index"]]
        self.aggregation_records[server_round] = diag

        if arrays is None:
            # Krum validity violated -> no update this round (no f-lowering, no fallback).
            return None, None
        tamper_applied = False
        if tamper_scheduled:
            state = arrays.to_torch_state_dict()
            delta_flat, state_meta = flatten_state(
                {k: state[k] - global_state[k] for k in global_state}
            )
            rng = np.random.default_rng([self.seed, server_round, 77])
            scale = float(np.linalg.norm(delta_flat)) + 1e-9
            tampered = delta_flat + rng.normal(0.0, 1.0, size=delta_flat.shape) * (0.1 * scale)
            tampered_state = unflatten_state(tampered, state_meta)
            arrays = ArrayRecord({
                key: global_state[key] + tampered_state[key].to(
                    device=global_state[key].device, dtype=global_state[key].dtype
                ) for key in global_state
            })
            tamper_applied = True
        diag["tamper_scheduled"] = int(tamper_scheduled)
        diag["tamper_applied"] = int(tamper_applied)
        reply_contents = [m.content for m in valid]
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        return arrays, metrics
