"""Pure result recording / I/O. No algorithm logic lives here."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import csv
import json
import time
import uuid
from pathlib import Path

from .attack import select_malicious_client_ids
from .protocol_profile import build_protocol_profile, validate_protocol_profile
from .utils import model_payload_bytes


COLUMNS = [
    "run_id",
    "seed",
    "round",
    "num_clients",
    "participating_clients",
    "successful_replies",
    "failures",
    "dropout_rate",
    "actual_dropout_clients",
    "dropout_failures",
    "runtime_failures",
    "attack_ratio",
    "malicious_clients",
    "train_loss",
    "learning_rate",
    "eval_loss",
    "eval_acc",
    "accuracy",
    "loss",
    "backdoor_asr",
    "client_train_time_agg",
    "client_train_time_mean",
    "client_train_time_max",
    "client_compute_time_mean",
    "client_compute_time_max",
    "straggler_ratio",
    "round_time",
    "total_time",
    "model_size_bytes",
    "comm_payload_down",
    "comm_payload_up",
    "comm_payload_total",
    "comm_bytes_wire",
]

COMM_DEFINITION = (
    "Communication overhead is measured over the TRAINING phase only: "
    "comm_payload_down = model_payload_bytes * participating_clients (global model "
    "broadcast to every training client); comm_payload_up = sum of the serialized "
    "ArrayRecord payloads of SUCCESSFUL client uploads (failed clients are excluded); "
    "comm_payload_total = down + up. The evaluation phase (model re-broadcast and "
    "scalar-metric responses) is excluded to avoid double-counting the same model "
    "payload. comm_bytes_wire is NOT available and is reported as 'unavailable'; it is "
    "never estimated."
)


def _fmt(value):
    """Make a CSV cell value: numbers numeric, NaN -> 'nan', others as-is."""
    if isinstance(value, float) and value != value:
        return "nan"
    if isinstance(value, (int, float, bool)):
        return value
    return value


def _f(value) -> float:
    """Safely coerce a value to float (NaN on failure)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _na_to_none(obj):
    """Recursively replace NaN / inf with None so json.dump(allow_nan=False) never crashes."""
    if isinstance(obj, dict):
        return {k: _na_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_na_to_none(v) for v in obj]
    if isinstance(obj, float):
        return None if obj != obj or obj in (float("inf"), float("-inf")) else obj
    return obj


class ExperimentRecorder:
    """Write ``config.json``, ``results.csv`` and ``summary.json``."""

    def __init__(self, output_dir: str, run_id: str | None = None, seed: int | None = None):
        self.output_dir = Path(output_dir)
        self.run_id = run_id or f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        result,
        records,
        run_config,
        env,
        total_time,
        detection_records=None,
        malicious_client_ids=None,
        attack_mode="persistent",
        burst_round=0,
        attack_enabled=0,
        attack_ratio=0.0,
        dropout_timing="pre_share",
        share_n=10,
        share_t=6,
        holder_outage="full",
        post_share_dropout_ids=None,
        recovery_records=None,
        runtime_threshold_records=None,
        dynamic_weight_records=None,
        dw_timing_records=None,
        vl_verification_records=None,
        aggregation_records=None,
        protocol_trace_records=None,
    ) -> dict:
        max_round = max(records.keys(), default=0)
        rounds = list(range(0, max_round + 1))
        model_bytes = model_payload_bytes(result.arrays) if result.arrays else 0
        num_clients = self._find_num_clients(records)
        attack_ratio = float(run_config.get("attack-ratio", 0.0))
        _seed = run_config.get("seed", self.seed if self.seed is not None else 42)
        _sel_seed = int(run_config.get("malicious-selection-seed", _seed))
        _mal = select_malicious_client_ids(int(_seed), attack_ratio, num_clients, _sel_seed)
        malicious_count = len(_mal)
        attack_type = str(run_config.get("attack-type", "none"))

        rows = []
        cumulative = 0.0
        for r in rounds:
            rec = records.get(r, {})
            server = result.evaluate_metrics_serverapp.get(r, {}) or {}
            trainm = result.train_metrics_clientapp.get(r, {}) or {}
            evalm = result.evaluate_metrics_clientapp.get(r, {}) or {}

            round_time = float(rec.get("round_time", 0))
            cumulative += round_time

            row = {
                "run_id": self.run_id,
                "seed": self.seed if self.seed is not None else run_config.get("seed", 42),
                "round": r,
                "num_clients": rec.get("num_clients", num_clients),
                "participating_clients": rec.get("participating_clients", 0),
                "successful_replies": rec.get("successful_replies", 0),
                "failures": rec.get("failures", 0),
                "dropout_rate": run_config.get("dropout-rate", 0.0),
                "actual_dropout_clients": rec.get("actual_dropout_clients", 0),
                "dropout_failures": rec.get("dropout_failures", 0),
                "runtime_failures": rec.get("runtime_failures", 0),
                "attack_ratio": attack_ratio,
                "malicious_clients": malicious_count,
                "train_loss": trainm.get("train_loss", float("nan")),
                "learning_rate": rec.get("learning_rate", float("nan")),
                "eval_loss": evalm.get("eval_loss", float("nan")),
                "eval_acc": evalm.get("eval_acc", float("nan")),
                "accuracy": server.get("accuracy", float("nan")),
                "loss": server.get("loss", float("nan")),
                "backdoor_asr": server.get("backdoor_asr", float("nan")),
                "client_train_time_agg": trainm.get("client_train_time", float("nan")),
                "client_train_time_mean": rec.get("client_train_time_mean", float("nan")),
                "client_train_time_max": rec.get("client_train_time_max", float("nan")),
                "client_compute_time_mean": rec.get("client_compute_time_mean", float("nan")),
                "client_compute_time_max": rec.get("client_compute_time_max", float("nan")),
                "straggler_ratio": rec.get("straggler_ratio", 0.0),
                "round_time": round_time,
                "total_time": cumulative,
                "model_size_bytes": model_bytes,
                "comm_payload_down": rec.get("comm_payload_down", 0),
                "comm_payload_up": rec.get("comm_payload_up", 0),
                "comm_payload_total": rec.get("comm_payload_total", 0),
                "comm_bytes_wire": "unavailable",
            }
            rows.append(row)

        self._write_csv(rows)
        self._write_compute_profile(run_config, num_clients)
        self._write_attack_statistics(records, attack_type)
        self._write_detection(
            detection_records, attack_mode, burst_round
        )
        self._write_recovery(recovery_records)
        self._write_runtime_threshold(runtime_threshold_records, result)
        self._write_dynamic_weight(dynamic_weight_records, result, records, run_config)
        self._write_overhead(
            recovery_records, dw_timing_records, runtime_threshold_records, run_config
        )
        self._write_vl_verification(vl_verification_records)
        self._write_aggregation(aggregation_records)
        self._write_protocol_trace(protocol_trace_records)
        self._write_setup_manifest(run_config, env, rows, model_bytes)
        self._write_json(
            self.output_dir / "config.json",
            self._build_config(
                run_config, env, rows, model_bytes, post_share_dropout_ids
            ),
        )
        summary = self._build_summary(rows, total_time, env)
        self._write_json(self.output_dir / "summary.json", summary)

        return {
            "output_dir": str(self.output_dir),
            "run_id": self.run_id,
            "final_accuracy": summary["final_accuracy"],
            "final_loss": summary["final_loss"],
            "total_time": summary["total_time"],
        }

    def _write_setup_manifest(self, run_config, env, rows, model_bytes) -> None:
        """Persist one-time protocol parameters separately from round traces."""
        last = rows[-1] if rows else {}
        profile = build_protocol_profile(run_config)
        manifest = {
            "run_id": self.run_id,
            "protocol_name": profile["name"],
            "protocol_version": profile["version"],
            "seed": self.seed if self.seed is not None else run_config.get("seed", 42),
            "num_clients": last.get("num_clients", 0),
            "model_size_bytes": model_bytes,
            "share_n": run_config.get("share-n", 10),
            "share_t": run_config.get("share-t", 6),
            "normal_threshold": run_config.get("normal-threshold", 8),
            "emergency_threshold": run_config.get("emergency-threshold", 6),
            "quantization_scale": 1 << 16,
            "crypto_vl_setup_id": run_config.get("vl-setup-id", ""),
            "hash_setup": (
                "ta_generated_persistent_delta_rho"
                if str(run_config.get("vl-backend", "linear")).lower() == "crypto"
                else "experiment_seed"
            ),
            "backends": {
                "secjudge": "raw_norm_robust_plaintext_core",
                "seccos": "round_reference_temporal_plaintext_core",
                "recovery": "logical_shamir",
                "aggregation_proof": (
                    "authenticated_homomorphic_hash"
                    if str(run_config.get("vl-backend", "linear")).lower() == "crypto"
                    else "linear_accumulator"
                ),
            },
            "trust_model": profile["trust_model"],
            "implementation_boundary": profile["implementation_boundary"],
            "environment": env,
        }
        self._write_json(self.output_dir / "setup_manifest.json", manifest)

    def _write_protocol_trace(self, protocol_trace_records) -> None:
        """Write one JSON object per executed round for protocol-stage auditing."""
        path = self.output_dir / "protocol_trace.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for round_id in sorted(protocol_trace_records or {}):
                handle.write(json.dumps(
                    _na_to_none(protocol_trace_records[round_id]),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ) + "\n")

    @staticmethod
    def _find_num_clients(records) -> int:
        for rec in records.values():
            if rec.get("num_clients") is not None:
                return int(rec["num_clients"])
        return 0

    def _write_csv(self, rows) -> None:
        path = self.output_dir / "results.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _fmt(v) for k, v in row.items()})

    def _write_compute_profile(self, run_config, num_clients) -> None:
        """Record the fixed UAV compute class / simulated factor per client."""
        try:
            fast = float(run_config.get("compute-fast-factor", 1.0))
            medium = float(run_config.get("compute-medium-factor", 1.0))
            slow = float(run_config.get("compute-slow-factor", 1.0))
        except (TypeError, ValueError):
            fast = medium = slow = 1.0
        path = self.output_dir / "client_compute_profile.csv"
        rows = []
        for cid in range(int(num_clients or 0)):
            cls = "fast" if cid <= 2 else ("medium" if cid <= 6 else "slow")
            factor = {"fast": fast, "medium": medium, "slow": slow}[cls]
            rows.append({"client_id": cid, "compute_class": cls, "compute_factor": factor})
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["client_id", "compute_class", "compute_factor"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def _write_attack_statistics(self, records, attack_type) -> None:
        """Flatten per-round per-client attack statistics into attack_statistics.csv."""
        path = self.output_dir / "attack_statistics.csv"
        fieldnames = [
            "round", "client_id", "is_malicious", "update_norm_before",
            "update_norm_after", "update_norm_ratio", "update_cosine_to_original",
            "attack_type", "attack_strength",
            "noise_l2_norm", "noise_mean", "noise_std_observed", "update_noise_ratio",
            "backdoor_poisoned_examples", "backdoor_poison_rate_observed",
        ]
        all_rows = []
        for r in sorted(records):
            attack_rows = records[r].get("attack_statistics", [])
            for row in attack_rows:
                row = dict(row)
                row["round"] = r
                all_rows.append(row)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    def _write_detection(self, detection_records, attack_mode, burst_round) -> None:
        """Write detection_statistics.csv, threshold_stability.csv, metrics + timing."""
        path = self.output_dir / "detection_statistics.csv"
        fieldnames = [
            "round", "client_id", "is_malicious_gt", "attack_active",
            "is_malicious_pred", "secjudge_ok", "norm_sum", "cosine",
            "threshold", "risk", "pass", "warmup", "attack_mode", "burst_round",
            "history_length", "mu", "sigma",
            "preprocess_time", "sec_judge_time", "sec_cos_time", "threshold_time",
            "decision_time", "detection_time", "total_ssdf_time",
            "logical_as1_time", "logical_as2_time",
            "raw_norm", "norm_median", "norm_mad", "norm_zscore",
            "reference_cosine", "reference_threshold", "sign_agreement",
            "sign_threshold", "temporal_ok", "reference_ok",
            "history_updated", "decision_reason", "detection_eligible",
            "ssdf_version", "ssdf_backend", "anchor_stability",
            "persistent_score", "persistent_ok",
            "persistent_threshold",
            "cohort_ok", "round_reference_cosine",
            "round_reference_threshold", "cohort_cosine",
            "cohort_threshold", "anomaly_streak",
        ]
        from .unified_evidence import EVIDENCE_FIELDS
        fieldnames.extend(EVIDENCE_FIELDS)
        all_rows = []
        if detection_records:
            for r in sorted(detection_records):
                for row in detection_records[r]:
                    all_rows.append(dict(row))
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

        # threshold stability
        stab_path = self.output_dir / "threshold_stability.csv"
        stab_fields = [
            "round", "client_id", "history_length", "mu", "sigma",
            "threshold", "cosine", "reference_cosine", "reference_threshold",
            "norm_zscore", "sign_agreement", "history_updated",
        ]
        with open(stab_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=stab_fields, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in stab_fields})

        # SSDF-v2's round-reference and norm gates are active during temporal
        # warmup, so those decisions remain part of detection metrics. Only
        # updates unavailable to the production detector are excluded.
        det = [
            r for r in all_rows
            if int(r.get("detection_eligible", 1)) == 1
        ]

        def _metrics(get_gt):
            tp = tn = fp = fn = 0
            for r in det:
                gt = int(get_gt(r))
                pred = int(r.get("is_malicious_pred", 0))
                if gt == 1 and pred == 1:
                    tp += 1
                elif gt == 0 and pred == 0:
                    tn += 1
                elif gt == 0 and pred == 1:
                    fp += 1
                else:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            acc = (tp + tn) / len(det) if det else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            fnr = fn / (fn + tp) if (fn + tp) else 0.0
            return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
                    "precision": prec, "recall": rec, "f1": f1,
                    "detection_accuracy": acc, "fpr": fpr, "fnr": fnr}

        common = {
            "attack_mode": attack_mode,
            "burst_round": burst_round,
            "n_detection_instances": len(det),
            "server_model": "logical_dual_server_in_single_simulation",
            "timing_mode": "sequential_logical_roles",
        }
        identity = dict(common)
        identity.update(_metrics(lambda r: r.get("is_malicious_gt", 0)))
        # Evaluate every client only in rounds where the attack is active.
        # ``attack_active`` is client-specific, so using it directly as the
        # class label would include pre-attack calibration observations as TNs
        # and slightly dilute FPR. Within active rounds, identity is the label.
        active_rounds = {
            int(r.get("round", -1))
            for r in det
            if int(r.get("attack_active", 0)) == 1
        }
        active_det = [
            r for r in det if int(r.get("round", -1)) in active_rounds
        ]
        active = dict(common)
        active["n_detection_instances"] = len(active_det)
        original_det = det
        det = active_det
        active.update(_metrics(lambda r: r.get("is_malicious_gt", 0)))
        det = original_det
        with open(self.output_dir / "detection_metrics_identity.json", "w", encoding="utf-8") as f:
            json.dump(identity, f, indent=2, allow_nan=False)
        with open(self.output_dir / "detection_metrics_active.json", "w", encoding="utf-8") as f:
            json.dump(active, f, indent=2, allow_nan=False)

        # server-side SSDF timing aggregates over detection rounds
        timing_keys = [
            "preprocess_time", "sec_judge_time", "sec_cos_time", "threshold_time",
            "decision_time", "detection_time", "total_ssdf_time",
            "logical_as1_time", "logical_as2_time",
        ]
        timing = {}
        for tk in timing_keys:
            vals = [float(r.get(tk, 0.0)) for r in all_rows if r.get(tk) not in (None, "")]
            timing[tk] = {
                "mean": (sum(vals) / len(vals)) if vals else 0.0,
                "max": max(vals) if vals else 0.0,
                "total": sum(vals),
                "n": len(vals),
            }
        timing["_notes"] = (
            "server-side plaintext-core timing (logical AS1/AS2 roles, sequential); "
            "NOT physical dual-server parallel timing."
        )
        with open(self.output_dir / "ssdf_timing.json", "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2, allow_nan=False)

    def _write_recovery(self, recovery_records) -> None:
        """Flatten per-round recovery state into recovery_statistics.csv."""
        path = self.output_dir / "recovery_statistics.csv"
        rows = []
        if recovery_records:
            for r in sorted(recovery_records):
                for row in recovery_records[r]:
                    rows.append(dict(row))
        fieldnames = []
        if rows:
            fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _write_runtime_threshold(self, runtime_threshold_records, result) -> None:
        """Flatten per-round runtime Dual-Shamir threshold state into runtime_threshold.csv."""
        if not runtime_threshold_records:
            return
        server_metrics = (result.evaluate_metrics_serverapp or {}) if result else {}
        fieldnames = [
            "round",
            "available_clients",
            "available_shares",
            "threshold_mode",
            "active_share_set",
            "active_threshold",
            "switch_event",
            "recovery_attempted",
            "recovery_success",
            "recovery_time",
            "B_distribution_bytes",
            "A_pre_distribution_payload",
            "B_pre_distribution_payload",
            "setup_share_payload_total",
            "b_redistributed",
            "decision_equal",
            "accuracy",
            "loss",
        ]
        rows = []
        for r in sorted(runtime_threshold_records):
            row = dict(runtime_threshold_records[r])
            row.setdefault("accuracy", float("nan"))
            row.setdefault("loss", float("nan"))
            m = server_metrics.get(r, {}) or {}
            if m:
                row["accuracy"] = _fmt(m.get("accuracy", float("nan")))
                row["loss"] = _fmt(m.get("loss", float("nan")))
            rows.append(row)
        path = self.output_dir / "runtime_threshold.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _fmt(v) for k, v in row.items()})

    def _write_dynamic_weight(self, dynamic_weight_records, result, records, run_config) -> None:
        """Write dynamic_weight_results.csv and dynamic_weight_summary.csv."""
        if not dynamic_weight_records:
            return
        results_path = self.output_dir / "dynamic_weight_results.csv"
        fieldnames = [
            "round",
            "client_id",
            "cos_similarity",
            "lambda_weight",
            "ssdf_pass",
            "removed",
            "malicious_label",
            "aggregation_method",
        ]
        rows = []
        for r in sorted(dynamic_weight_records):
            for row in dynamic_weight_records[r]:
                rows.append(dict(row))
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: _fmt(v) for k, v in row.items()})

        # Per-run summary: attack_ratio, method, final accuracy/loss, convergence,
        # total removed client-rounds.
        server_metrics = (result.evaluate_metrics_serverapp or {}) if result else {}
        accs = {
            r: m.get("accuracy")
            for r, m in server_metrics.items()
            if int(r) >= 1 and isinstance(m.get("accuracy"), float)
        }
        final_acc = accs[max(accs)] if accs else float("nan")
        final_loss = (
            server_metrics[max(accs)].get("loss", float("nan")) if accs else float("nan")
        )
        conv_round = None
        for r in sorted(accs):
            if abs(accs[r] - final_acc) <= 0.005:
                conv_round = r
                break
        if conv_round is None:
            conv_round = max(accs) if accs else None
        total_removed = sum(
            1
            for rows_ in dynamic_weight_records.values()
            for row in rows_
            if int(row.get("removed", 0)) == 1
        )
        method = (
            "dynamic_weight"
            if str(run_config.get("dynamic-weight", 0)) == "1"
            else "fedavg"
        )
        summary = {
            "attack_ratio": run_config.get("attack-ratio", 0.0),
            "method": method,
            "final_accuracy": final_acc,
            "final_loss": final_loss,
            "convergence_round": conv_round,
            "convergence_definition": (
                "first round r>=1 with |accuracy_r - final_accuracy| <= 0.005; "
                "else the last round."
            ),
            "removed_clients": total_removed,
            "removed_definition": (
                "total client-rounds removed (SSDF filter or post-share recovery "
                "failure) across training rounds."
            ),
        }
        with open(self.output_dir / "dynamic_weight_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            w.writeheader()
            w.writerow(summary)

    def _write_overhead(
        self, recovery_records, dw_timing_records, runtime_threshold_records, run_config
    ) -> None:
        """Write overhead.csv: share generation/recovery/DW costs per round."""
        recovery_records = recovery_records or {}
        dw_timing_records = dw_timing_records or {}
        rt_records = runtime_threshold_records or {}
        all_rounds = sorted(
            set(list(recovery_records.keys()) + list(dw_timing_records.keys()) + list(rt_records.keys()))
        )
        fieldnames = [
            "round",
            "recovered_clients",
            "share_generation_time",
            "reconstruction_time",
            "recovery_time",
            "share_size_bytes",
            "B_distribution_bytes",
            "B_pre_distribution_payload",
            "setup_share_payload_total",
            "switch_latency",
            "dw_cosine_time",
            "dw_weight_time",
            "dw_aggregate_time",
            "dw_total_time",
        ]
        rows = []
        for r in all_rounds:
            rec = recovery_records.get(r, [])
            # Only clients where recovery was actually attempted and succeeded.
            recovered = [
                row for row in rec
                if int(row.get("recovery_attempted", 0)) == 1
                and int(row.get("recovery_success", 0)) == 1
            ]
            share_gen = sum(_f(row.get("share_generation_time", 0.0) or 0.0) for row in recovered)
            recon = sum(_f(row.get("reconstruction_time", 0.0) or 0.0) for row in recovered)
            rec_time = sum(_f(row.get("recovery_time", 0.0) or 0.0) for row in recovered)
            share_sizes = [
                _f(row.get("b_share_payload_bytes", 0.0) or 0.0)
                for row in recovered
                if _f(row.get("b_share_payload_bytes", 0.0) or 0.0) > 0
            ]
            share_size = max(share_sizes) if share_sizes else 0
            rt = rt_records.get(r, {}) or {}
            b_bytes = _f(rt.get("B_distribution_bytes", 0))
            b_pre = _f(rt.get("B_pre_distribution_payload", 0))
            setup_total = sum(
                _f(r.get("setup_share_payload_total", 0.0) or 0.0)
                for r in recovered
            )
            dw = dw_timing_records.get(r, {}) or {}
            rows.append({
                "round": r,
                "recovered_clients": len(recovered),
                "share_generation_time": share_gen,
                "reconstruction_time": recon,
                "recovery_time": rec_time,
                "share_size_bytes": share_size,
                "B_distribution_bytes": b_bytes,
                "B_pre_distribution_payload": b_pre,
                "setup_share_payload_total": setup_total,
                "switch_latency": "missing",
                "dw_cosine_time": _f(dw.get("dw_cosine_time", 0.0)),
                "dw_weight_time": _f(dw.get("dw_weight_time", 0.0)),
                "dw_aggregate_time": _f(dw.get("dw_aggregate_time", 0.0)),
                "dw_total_time": _f(dw.get("dw_total_time", 0.0)),
            })
        with open(self.output_dir / "overhead.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: _fmt(v) for k, v in row.items()})

    def _write_vl_verification(self, vl_verification_records) -> None:
        """Write vl_verification_results.csv."""
        if not vl_verification_records:
            return
        fieldnames = [
            "round",
            "client_num",
            "aggregation_method",
            "vl_backend",
            "verification_success",
            "verification_policy",
            "rollback_applied",
            "failure_reason",
            "verification_time",
            "hash_generation_time",
            "hash_size_bytes",
            "hash_error",
            "h_agg",
            "h_expected",
            "claimed_tampered",
            "tamper_scheduled",
            "tamper_applied",
            "control_run_id",
            "paired_run_id",
            "aggregation_state_hash_before",
            "aggregation_state_hash_after",
            "task_id",
            "model_digest",
            "accepted_set",
            "quantized_weights",
            "proof_binding",
            "proof",
            "transport_status",
            "participation_digest",
            "participation_records",
            "client_receipts",
            "commit_mode",
            "model_committed",
            "state_committed",
            "candidate_rejected",
            "precommit_verification_success",
            "precommit_verification_time",
            "precommit_failure_details",
            "candidate_model_digest",
            "committed_model_digest",
            "tamper_detection_location",
        ]
        rows = []
        for r in sorted(vl_verification_records):
            row = dict(vl_verification_records[r])
            row["round"] = r
            rows.append(row)
        with open(self.output_dir / "vl_verification_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: _fmt(v) for k, v in row.items()})

    def _write_aggregation(self, aggregation_records) -> None:
        """Write method-aware common aggregation diagnostics."""
        if not aggregation_records:
            return
        fieldnames = [
            "round", "method", "num_clients", "krum_f", "krum_valid", "krum_selected_client",
            "krum_score", "krum_validity_warning", "trim_ratio",
            "num_removed_low", "num_removed_high", "client_ids",
            "total_clients", "available_clients", "dropped_clients",
            "unavailable_ids", "aggregation_status",
            "effective_clients", "weights_source", "recovery_status",
            "verification_status",
            "verification_policy", "rollback_applied",
            "verification_failure_reason",
            "commit_mode", "model_committed", "state_committed",
            "candidate_rejected", "committed_model_digest",
            "tamper_scheduled", "tamper_applied",
            "received_replies", "invalid_update_replies",
            "dropout_failures", "runtime_failures",
            "detector", "clustering_status", "pca_components",
            "min_cluster_size", "benign_cluster_label", "raw_suspect_ids",
            "suspected_client_ids", "cluster_labels", "history_vote_counts",
            "history_observation_counts", "retained_client_ids",
            "fallback_all_updates",
        ]
        path = self.output_dir / "aggregation_diagnostics.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in sorted(aggregation_records):
                row = dict(aggregation_records[r])
                row["round"] = r
                w.writerow({k: _fmt(v) for k, v in row.items()})

    def _build_config(self, run_config, env, rows, model_bytes, post_share_dropout_ids=None) -> dict:
        last = rows[-1] if rows else {}
        attack_ratio = float(run_config.get("attack-ratio", 0.0))
        _seed = run_config.get("seed", self.seed if self.seed is not None else 42)
        _sel_seed = int(run_config.get("malicious-selection-seed", _seed))
        _num_clients = int(last.get("num_clients", 0) or 0) if last else 0
        _mal = select_malicious_client_ids(int(_seed), attack_ratio, _num_clients, _sel_seed)
        protocol_profile = build_protocol_profile(run_config)
        profile_errors = validate_protocol_profile(protocol_profile)
        if profile_errors:
            raise ValueError("INVALID_PROTOCOL_PROFILE:" + ",".join(profile_errors))
        return {
            "run_id": self.run_id,
            "protocol_profile": protocol_profile,
            "seed": self.seed if self.seed is not None else run_config.get("seed", 42),
            "client_seed_scheme": "global_seed + partition_id",
            "experiment": run_config.get("experiment", "baseline"),
            "dataset": ("MNIST (ylecun/mnist)" if run_config.get("dataset", "cifar10") == "mnist"
                        else "CIFAR-10 (uoft-cs/cifar10)"),
            "partitioner": run_config.get("partitioner", "iid"),
            "alpha": run_config.get("alpha", 1.0),
            "num_partitions": last.get("num_clients", None),
            "strategy": "FedAvg",
            "fraction_train": run_config.get("fraction-train", 1.0),
            "fraction_evaluate": run_config.get("fraction-evaluate", 1.0),
            "ssdf_enabled": run_config.get("ssdf-enabled", 0),
            "ssdf_version": run_config.get("ssdf-version", "v2"),
            "ssdf_detector_identity": run_config.get(
                "ssdf-detector-mode", "v2_full"
            ),
            "ssdf_evidence_policy": run_config.get("ssdf-evidence-policy", "legacy"),
            "ssdf_backend": run_config.get("ssdf-backend", "plaintext"),
            "ssdf_mode": (
                "logical_dual_server_additive_share_prototype"
                if run_config.get("ssdf-backend", "plaintext") == "logical_dual_server"
                else "plaintext_core"
            ),
            "ssdf_execution_mode": (
                "logical_two_server_sequential_simulation"
                if run_config.get("ssdf-backend", "plaintext") == "logical_dual_server"
                else "server_side_plaintext_core"
            ),
            "ssdf_share_n": run_config.get("share-n", 10),
            "ssdf_share_t": run_config.get("share-t", 6),
            "ssdf_holder_outage": run_config.get("holder-outage", "full"),
            "ssdf_dropout_timing": run_config.get("dropout-timing", "pre_share"),
            "ssdf_post_share_dropout_ids": post_share_dropout_ids or [],
            "ssdf_recovery_injection_mode": "logical_server_side_recovery",
            "ssdf_dropout_model": (
                "logical_post_share_dropout"
                if run_config.get("dropout-timing", "") == "post_share"
                else "pre_share_error_reply"
            ),
            "ssdf_server_model": "logical_dual_server_in_single_simulation",
            "ssdf_non_collusion_assumption": True,
            "ssdf_timing_mode": "sequential_logical_roles",
            "ssdf_communication_mode": "reuse_existing_model_message",
            "ssdf_comm_uav_to_as1": 0,
            "ssdf_comm_uav_to_as2": 0,
            "ssdf_comm_server_to_server": 0,
            "ssdf_uav_ssdf_time": "unavailable",
            "ssdf_ema_beta": run_config.get("ema-beta", 0.9),
            "ssdf_threshold_k": run_config.get("threshold-k", 2.0),
            "ssdf_warmup_rounds": run_config.get("warmup-rounds", 2),
            "ssdf_warmup_mode": "temporal_history_only_reference_and_norm_active",
            "ssdf_norm_tol": run_config.get("norm-tol", 1e-3),
            "ssdf_eps": 1e-8,
            "ssdf_norm_floor": 1e-12,
            "ssdf_reference_k": run_config.get("ssdf-reference-k", 2.5),
            "ssdf_reference_floor": run_config.get("ssdf-reference-floor", -0.25),
            "ssdf_norm_k": run_config.get("ssdf-norm-k", 2.70),
            "ssdf_history_limit": run_config.get("ssdf-history-limit", 20),
            "threshold_controller_enabled": run_config.get(
                "threshold-controller-enabled", 0
            ),
            "normal_threshold": run_config.get("normal-threshold", 8),
            "emergency_threshold": run_config.get("emergency-threshold", 6),
            "static_threshold": run_config.get("static-threshold", 0),
            "runtime_available_schedule": run_config.get(
                "runtime-available-schedule", ""
            ),
            "dual_shamir_semantics": {
                "normal": {
                    "active_share_set": "A",
                    "active_threshold": run_config.get("normal-threshold", 8),
                    "distributed_B": True,  # D2-V2: B provisioned at setup alongside A
                    "normal_B_exposure": 0,
                },
                "emergency": {
                    "active_share_set": "B",
                    "active_threshold": run_config.get("emergency-threshold", 6),
                    "distributed_B": True,
                    "B_redistributed_at_emergency": False,
                    "emergency_B_exposure": 1,
                },
            },
            "threshold_controller_rule": (
                "available >= normal_threshold -> normal/A/normal_threshold; "
                "emergency_threshold <= available < normal_threshold -> emergency/B/"
                "emergency_threshold (select the already-provisioned B set, no "
                "re-distribution); available < "
                "emergency_threshold -> failure (recovery cannot happen). The controller "
                "only SELECTS an existing A/B share set; it never changes any Shamir "
                "polynomial degree. A and B are BOTH generated and pre-distributed during "
                "setup; Emergency does NOT retransmit B."
            ),
            "d2_v2_semantics": (
                f"B=Shamir({run_config.get('share-n', 10)},{run_config.get('emergency-threshold', 6)}) "
                f"is provisioned alongside A=Shamir({run_config.get('share-n', 10)},{run_config.get('normal-threshold', 8)}) at setup "
                "(pre-distribution); Emergency reuses the stored B set, with "
                "B_distribution_bytes = 0 at runtime. Only the reconstruction threshold "
                "is chosen; no shares are re-sent during Emergency. This describes "
                "the configured recovery profile; it is inactive when recovery is disabled."
            ),
            "runtime_threshold_available_model": (
                "available_shares == available_clients == (participating clients - "
                "post-share dropout clients) per round; each online UAV contributes one "
                "logical share contribution."
            ),
            "runtime_threshold_tls_wire_bytes": "unavailable",
            "runtime_threshold_recovery_owner": "logical_coordinator",
            "runtime_threshold_recovery_mode": "plaintext_integration_test",
            "dynamic_weight_enabled": run_config.get("dynamic-weight", 0),
            "dynamic_weight_blend": run_config.get("dynamic-weight-blend", 0.25),
            "dynamic_weight_definition": (
                "Unified candidate: cosine uses the accepted-set mean; confidence "
                "v1 q=(1-E)*(1-S); v2 q=(1-eN)*(1-0.5*direction)*(1-S). "
                "Confidence multiplies the entire uniform/directional blended "
                "score before normalization and the 2/|A_t| cap. Nonpositive "
                "cosines use confidence-only weighting. Crypto-VL and model "
                "aggregation use identical quantized weights."
                if run_config.get("ssdf-evidence-policy", "legacy") in {"unified_v1_candidate", "unified_v2_candidate"}
                else
                "For each SSDF-passing client, cosine is measured against the locked "
                "clean-cohort anchor maintained by logical dual-server PoisonGuard; "
                "before the anchor exists, weights are uniform. Positive-cosine weights "
                "are normalized with lambda_i <= 2/|A_t| and clipped mass is "
                "redistributed. Final weights blend 25% anchor weights with 75% "
                "uniform weights; all-nonpositive scores fall back to uniform. New global = "
                "sum(lambda_i * local_i) = global + "
                "sum(lambda_i * delta_i). SSDF-removed / recovery-failed clients get "
                "lambda 0."
            ),
            "vl_verification_enabled": run_config.get("vl-verification", 0),
            "vl_commit_mode": run_config.get("vl-commit-mode", "next_round"),
            "vl_tamper_enabled": run_config.get("vl-tamper", 0),
            "vl_tamper_rounds": run_config.get("vl-tamper-rounds", ""),
            "control_run_id": run_config.get("control-run-id", ""),
            "paired_run_id": run_config.get("paired-run-id", ""),
            "vl_backend": run_config.get("vl-backend", "linear"),
            "vl_update_scale": run_config.get("vl-update-scale", 1 << 16),
            "crypto_vl_setup_id": run_config.get("vl-setup-id", ""),
            "vl_verification_definition": (
                "Prime-order-group vector homomorphic hash over fixed-point updates; "
                "client Ed25519 commitments and PRF tags; accepted-set, integer-weight, "
                "recovery-state, round/model-version, and returned-model binding; "
                "AS1/AS2 authenticated statements; no Pedersen commitment or ZKP."
                if str(run_config.get("vl-backend", "linear")).lower() == "crypto"
                else
                "Additive-group homomorphic accumulator H(v) = v . g (deterministic "
                "random vector g). Verifies H(weighted_agg_delta) == sum_i lambda_i "
                "H(delta_i) within tolerance; lightweight and non-cryptographic."
            ),
            "vl_mode": (
                "authenticated_homomorphic_hash"
                if str(run_config.get("vl-backend", "linear")).lower() == "crypto"
                else "lightweight_accumulator"
            ),
            "vl_verification_location": (
                "client_side_pre_training_with_server_fail_closed_precheck"
                if str(run_config.get("vl-backend", "linear")).lower() == "crypto"
                else "server_side_post_aggregation_and_client_transcript_check"
            ),
            "aggregation_method": run_config.get("aggregation-method", "fedavg"),
            "krum_f": run_config.get("krum-f", 2),
            "trim_ratio": run_config.get("trim-ratio", 0.2),
            "scenario_id": run_config.get("scenario-id", ""),
            "aggregation_update_def": (
                "delta_i = local_model_i - global_model (model-delta, NOT gradient); "
                "FedAvg/Krum/TrimmedMean all use the identical delta representation."
            ),
            "attack_enabled": run_config.get("attack-enabled", 0),
            "attack_type": run_config.get("attack-type", "none"),
            "attack_ratio": attack_ratio,
            "attack_strength": run_config.get("attack-strength", 1.0),
            "attack_sigma": run_config.get("attack-sigma", 0.05),
            "backdoor_poison_rate": run_config.get("backdoor-poison-rate", 0.3),
            "backdoor_target_label": run_config.get("backdoor-target-label", 0),
            "backdoor_trigger_size": run_config.get("backdoor-trigger-size", 3),
            "attack_mode": run_config.get("attack-mode", "persistent"),
            "attack_start_round": run_config.get("attack-start-round", 1),
            "burst_round": run_config.get("burst-round", 0),
            "noise_distribution": "Gaussian",
            "attack_noise_offset": 7,
            "malicious_client_ids": _mal,
            "malicious_selection_seed": _sel_seed,
            "malicious_selection_mode": "fixed_across_rounds",
            "malicious_client_count": len(_mal),
            "compute_model": "deterministic CPU-bound workload",
            "compute_factor_semantics": "simulated relative compute-capacity parameter",
            "simulated_compute_factor": {
                "fast": run_config.get("compute-fast-factor", 1.0),
                "medium": run_config.get("compute-medium-factor", 1.0),
                "slow": run_config.get("compute-slow-factor", 1.0),
            },
            "compute_work_units": run_config.get("work-units", 0),
            "compute_matrix_size": 256,
            # Record the actual runtime device; this is metadata only and does
            # not affect aggregation or client training semantics.
            "compute_device": (
                "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
            ),
            "compute_class_scheme": "client 0-2 fast, 3-6 medium, 7-9 slow",
            "straggler_ratio_definition": (
                "straggler_ratio = max(client_compute_time) / mean(client_compute_time) "
                "measured per round over participating clients, using client_compute_time "
                "(real training + simulated CPU-bound workload). Describes compute "
                "completion-time imbalance only, NOT an FL performance metric."
            ),
            "dropout_rate": run_config.get("dropout-rate", 0.0),
            "dropout_model": "simulated client unavailability via Flower Error reply",
            "failure_source": "separate_dropout_runtime_and_invalid_update_counts",
            "failure_note": (
                "dropout_failures counts explicit dropout error replies; runtime_failures "
                "counts other error replies. Invalid/non-finite or malformed update replies "
                "are filtered separately and are not dropout failures. Baseline aggregation "
                "diagnostics include received_replies and invalid_update_replies. Logical "
                "post-share unavailability is recorded separately in unavailable_ids."
            ),
            "num_clients": last.get("num_clients", None),
            "num_rounds": last.get("round", None),
            "model_size_bytes": model_bytes,
            "run_config": dict(run_config),
            "env_versions": env,
            "communication_definition": COMM_DEFINITION,
            "round_time_definition": (
                "round_time = wall-clock from the start of this round's configure_train "
                "to the start of the next round's configure_train; for the last round it "
                "extends to the return of strategy.start(). round 0 (initial global "
                "evaluation only) is defined as 0. This is an instrumentation-defined "
                "duration, not a Flower-internal official round duration."
            ),
        }

    def _build_summary(self, rows, total_time, env) -> dict:
        valid = [r for r in rows if r["round"] >= 1]
        final = valid[-1] if valid else (rows[0] if rows else {})
        acc_pairs = [
            (r["accuracy"], r["round"])
            for r in valid
            if isinstance(r["accuracy"], float) and r["accuracy"] == r["accuracy"]
        ]
        best = max(acc_pairs, default=(float("nan"), None))
        train_losses = [
            r["train_loss"]
            for r in valid
            if isinstance(r["train_loss"], float) and r["train_loss"] == r["train_loss"]
        ]
        return {
            "run_id": self.run_id,
            "seed": self.seed if self.seed is not None else None,
            "client_seed_scheme": "global_seed + partition_id",
            "number_of_rounds": final.get("round") if isinstance(final, dict) else None,
            "number_of_clients": final.get("num_clients") if isinstance(final, dict) else None,
            "final_accuracy": final.get("accuracy", float("nan")),
            "final_loss": final.get("loss", float("nan")),
            "final_backdoor_asr": final.get("backdoor_asr", float("nan")),
            "best_accuracy": best[0],
            "best_round": best[1],
            "mean_train_loss": (sum(train_losses) / len(train_losses)) if train_losses else float("nan"),
            "total_time": total_time,
            "env_versions": env,
        }

    @staticmethod
    def _write_json(path: Path, data) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_na_to_none(data), f, indent=2, allow_nan=False)
