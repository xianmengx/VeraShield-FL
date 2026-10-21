
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import csv
import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from pytorchexample.experiment.aggregation_diagnostics import (
    validate_aggregation_diagnostics_path,
    validate_aggregation_record,
)
from pytorchexample.experiment.recorder import ExperimentRecorder
from pytorchexample.experiment.ssdf_recovery_strategy import build_vl_aggregation_record


def test_vl_common_row_uses_real_method_aware_fields():
    row = build_vl_aggregation_record(
        server_round=1,
        method="dynamic_weight",
        valid_client_ids=list(range(20)),
        retained_client_ids=list(range(14)),
        unavailable_ids={14, 15, 16, 17, 18, 19},
        available_clients=14,
        aggregation_completed=True,
        recovery_success=True,
        verification_enabled=True,
        verification_success=True,
    )
    assert validate_aggregation_record(row) == []
    assert row["aggregation_status"] == "ok"
    assert row["effective_clients"] == 14
    assert not any(key.startswith("krum_") for key in row)
    assert "trim_ratio" not in row


def test_vl_failed_verification_is_auditable():
    row = build_vl_aggregation_record(
        server_round=2,
        method="dynamic_weight",
        valid_client_ids=[0],
        retained_client_ids=[0],
        unavailable_ids=set(),
        available_clients=1,
        aggregation_completed=True,
        recovery_success=True,
        verification_enabled=True,
        verification_success=False,
    )
    assert row["aggregation_status"] == "aggregation_completed_verification_failed"
    assert row["verification_status"] == "failed"


def test_recorder_writes_na_as_empty_not_zero():
    row = build_vl_aggregation_record(
        server_round=1,
        method="dynamic_weight",
        valid_client_ids=[0],
        retained_client_ids=[0],
        unavailable_ids=set(),
        available_clients=1,
        aggregation_completed=True,
        recovery_success=True,
        verification_enabled=True,
        verification_success=True,
    )
    with TemporaryDirectory() as directory:
        recorder = ExperimentRecorder(directory)
        recorder._write_aggregation({1: row})
        with open(Path(directory) / "aggregation_diagnostics.csv", newline="", encoding="utf-8") as f:
            written = next(csv.DictReader(f))
        assert written["krum_f"] == ""
        assert written["trim_ratio"] == ""
        assert written["verification_status"] == "success"


def test_recorder_writes_setup_manifest_with_explicit_backend_boundaries():
    with TemporaryDirectory() as directory:
        recorder = ExperimentRecorder(directory, run_id="trace-test", seed=42)
        recorder._write_setup_manifest(
            {
                "ssdf-enabled": 1,
                "dropout-timing": "post_share",
                "dynamic-weight": 1,
                "vl-verification": 1,
                "share-n": 20,
                "share-t": 12,
                "normal-threshold": 16,
                "emergency-threshold": 12,
            },
            {"python": "test"},
            [{"num_clients": 20}],
            1234,
        )
        manifest = json.loads(
            (Path(directory) / "setup_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["protocol_version"] == "prototype-flow-v2"
        assert (
            manifest["backends"]["seccos"]
            == "round_reference_temporal_plaintext_core"
        )
        assert manifest["backends"]["aggregation_proof"] == "linear_accumulator"
        assert manifest["implementation_boundary"]["cryptographic_proof"] is False


def test_recorder_writes_protocol_trace_without_conflating_cosines():
    trace = {
        1: {
            "round": 1,
            "candidate_set": [0, 1],
            "provisioned_set": [0, 1],
            "available_set": [0],
            "accepted_set": [0],
            "ssdf": {0: {"temporal_cosine": 0.75}},
            "weights": {0: {"round_reference_cosine": 0.9, "lambda": 1.0}},
        }
    }
    with TemporaryDirectory() as directory:
        recorder = ExperimentRecorder(directory)
        recorder._write_protocol_trace(trace)
        payload = json.loads(
            (Path(directory) / "protocol_trace.jsonl").read_text(encoding="utf-8")
        )
        assert payload["ssdf"]["0"]["temporal_cosine"] == 0.75
        assert payload["weights"]["0"]["round_reference_cosine"] == 0.9


def test_baseline_method_specific_validation():
    common = {
        "round": 1,
        "num_clients": 20,
        "client_ids": list(range(20)),
        "total_clients": 20,
        "available_clients": 20,
        "dropped_clients": 0,
        "unavailable_ids": [],
        "aggregation_status": "ok",
    }
    assert validate_aggregation_record({**common, "method": "fedavg"}) == []
    assert validate_aggregation_record(
        {**common, "method": "krum", "krum_f": 4, "krum_valid": True}
    ) == []


def test_historical_vl_compatibility_requires_real_dedicated_rows():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for filename, count in {
            "results.csv": 31,
            "runtime_threshold.csv": 30,
            "vl_verification_results.csv": 30,
            "overhead.csv": 30,
        }.items():
            with open(root / filename, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["round"])
                writer.writeheader()
                for round_id in range(count):
                    writer.writerow({"round": round_id})
        result = validate_aggregation_diagnostics_path(
            root,
            method="VL-SSDF-UAVFL",
            expected_rounds=30,
            allow_historical_vl=True,
        )
        assert result == {"status": "HISTORICAL_RESULT_COMPATIBLE", "errors": []}


if __name__ == "__main__":
    unittest.main()
    assert validate_aggregation_record(
        {
            **common,
            "method": "trimmed_mean",
            "trim_ratio": 0.2,
            "num_removed_low": 2,
            "num_removed_high": 2,
        }
    ) == []
