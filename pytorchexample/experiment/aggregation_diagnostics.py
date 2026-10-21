"""Method-aware validation for common aggregation diagnostic rows."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import csv
from pathlib import Path


COMMON_REQUIRED = {
    "round", "method", "num_clients", "client_ids", "total_clients",
    "available_clients", "dropped_clients", "unavailable_ids",
    "aggregation_status",
}


def validate_aggregation_record(row: dict) -> list[str]:
    """Return validation errors without treating inapplicable fields as missing."""
    errors = [f"missing:{key}" for key in sorted(COMMON_REQUIRED) if key not in row]
    method = str(row.get("method", "")).lower().replace("-", "_")
    required = ()
    if method == "krum":
        required = ("krum_f", "krum_valid")
    elif method == "trimmed_mean":
        required = ("trim_ratio", "num_removed_low", "num_removed_high")
    elif method == "dynamic_weight":
        required = ("effective_clients", "weights_source", "recovery_status", "verification_status")
    errors.extend(f"missing:{key}" for key in required if key not in row)
    return errors


def validate_aggregation_diagnostics_path(
    output_dir: str | Path, *, method: str, expected_rounds: int,
    allow_historical_vl: bool = False,
) -> dict:
    """Validate a common file, or recognize an immutable historical VL result."""
    output_dir = Path(output_dir)
    common_path = output_dir / "aggregation_diagnostics.csv"
    normalized = method.lower().replace("-", "_")
    if common_path.exists():
        with open(common_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        errors = []
        for index, row in enumerate(rows, start=1):
            errors.extend(f"row:{index}:{error}" for error in validate_aggregation_record(row))
        if len(rows) != expected_rounds:
            errors.append(f"round_count:{len(rows)}!=expected:{expected_rounds}")
        return {"status": "PASS" if not errors else "FAILED", "errors": errors}
    if normalized not in {"vl_ssdf_uavfl", "dynamic_weight"} or not allow_historical_vl:
        return {"status": "FAILED", "errors": ["missing:aggregation_diagnostics.csv"]}
    required = {
        "results.csv": expected_rounds + 1,
        "runtime_threshold.csv": expected_rounds,
        "vl_verification_results.csv": expected_rounds,
        "overhead.csv": expected_rounds,
    }
    errors = []
    for filename, expected_rows in required.items():
        path = output_dir / filename
        if not path.exists():
            errors.append(f"missing:{filename}")
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        if count != expected_rows:
            errors.append(f"row_count:{filename}:{count}!=expected:{expected_rows}")
    return {"status": "HISTORICAL_RESULT_COMPATIBLE" if not errors else "FAILED", "errors": errors}
