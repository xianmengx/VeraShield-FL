"""Validate a scenario manifest (unavailable/malicious/IDs/seed)."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    p = Path(args.scenario)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    scen = json.loads(p.read_text(encoding="utf-8"))
    n = int(scen.get("num_clients", 0))
    dropout = float(scen.get("dropout_ratio", 0.0))
    expect_drop = int(round(dropout * n))
    problems = []

    mal = set(scen.get("malicious_ids", []))
    per_round = scen.get("per_round", [])
    if not per_round:
        problems.append("no per_round")
    for r in per_round:
        ids = r.get("unavailable_ids", [])
        if len(ids) != expect_drop:
            problems.append(f"round {r['round']}: unavailable_ids count {len(ids)} != expected {expect_drop}")
        bad = [x for x in ids if x < 0 or x >= n]
        if bad:
            problems.append(f"round {r['round']}: out-of-range ids {bad}")
        if len(set(ids)) != len(ids):
            problems.append(f"round {r['round']}: duplicate unavailable_ids")
        if len(set(r.get("selected_clients", []))) != len(r.get("selected_clients", [])):
            problems.append(f"round {r['round']}: duplicate selected")
        overlap = set(ids) & mal
        if overlap:
            # report but do not fail: malicious AND unavailable is allowed if documented.
            print(f"INFO: round {r['round']} malicious∩unavailable = {sorted(overlap)}")

    if problems:
        print("SCENARIO_VALIDATION_FAIL")
        for pr in problems:
            print("  -", pr)
        return 1
    print(f"SCENARIO_VALIDATION_PASS")
    print(f"  scenario={scen.get('scenario_id')} seed={scen.get('seed')} clients={n} "
          f"dropout={dropout} available_dropped_per_round={expect_drop} "
          f"malicious={sorted(mal)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
