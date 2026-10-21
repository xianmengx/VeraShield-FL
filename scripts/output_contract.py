"""Output directory contract and overwrite protection for experiment runners."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

from pathlib import Path


class OutputContractError(RuntimeError):
    """Raised when an experiment output path is unsafe or already complete."""


REQUIRED_RESULT_FILES = ("summary.json", "results.csv", "config.json", "scenario.json")


def resolve_output_path(output_root: str | Path, scenario_id: str, method: str) -> Path:
    """Resolve ``output_root/scenario_id/method`` exactly once."""
    root = Path(output_root).resolve()
    scenario = str(scenario_id).strip()
    method_name = str(method).strip()
    if not scenario or not method_name:
        raise OutputContractError("INVALID_OUTPUT_ROOT: scenario_id and method are required")
    if root.name == scenario:
        raise OutputContractError("INVALID_OUTPUT_ROOT: output_root already names scenario_id")
    target = root / scenario / method_name
    if all((target / name).is_file() for name in REQUIRED_RESULT_FILES):
        raise OutputContractError(f"EXISTING_RESULT_BLOCKED: {target}")
    return target


def validate_output_dir(output_dir: str | Path, scenario_id: str = "", method: str = "") -> Path:
    """Validate an explicitly supplied final directory before a run."""
    path = Path(output_dir).resolve()
    if scenario_id and path.parent.name == scenario_id and method and path.name == method:
        # This is the expected final path; only the complete-result guard applies.
        pass
    if all((path / name).is_file() for name in REQUIRED_RESULT_FILES):
        raise OutputContractError(f"EXISTING_RESULT_BLOCKED: {path}")
    return path
