"""Replay candidate evidence on frozen CSV features, without training.

Old CSVs lack adaptive temporal/cohort baselines. Their norm/reference-only
projections are explicitly incomplete and never pass a full validation gate.
"""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pytorchexample.experiment.unified_evidence import UnifiedEvidenceState, direction_evidence, clip


def number(row, key):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def replay(path):
    rows = sorted(csv.DictReader(Path(path).open(newline="", encoding="utf-8")),
                  key=lambda x: (int(x["round"]), int(x["client_id"])))
    rounds = sorted({int(x["round"]) for x in rows})
    active_rounds = {int(x["round"]) for x in rows if number(x, "attack_active") == 1}
    evaluation_rounds = active_rounds or set(rounds)
    policies = {x.get("evidence_policy") for x in rows}
    full = bool(rows) and len(policies) == 1 and policies <= {"unified_v1_candidate", "unified_v2_candidate"}
    policy = next(iter(policies)) if full else "incomplete"
    first = next((x for x in rows if number(x, "evidence_updated") == 1), {})
    state = UnifiedEvidenceState(
        beta=number(first, "evidence_beta") if number(first, "evidence_beta") is not None else .8,
        strong_threshold=number(first, "evidence_strong_threshold") or 1.,
        persistent_threshold=number(first, "evidence_persistent_threshold") or .8,
        immediate_policy="norm_only" if policy == "unified_v2_candidate" else "all_channels")
    counts = dict(TP=0, TN=0, FP=0, FN=0)
    skipped = mismatches = empty = 0
    for r in rounds:
        batch = [x for x in rows if int(x["round"]) == r]
        refs = [number(x, "round_reference_cosine") for x in batch if number(x, "detection_eligible") != 0]
        refs = [x for x in refs if x is not None]
        center = statistics.median(refs) if refs else None
        accepted = 0
        for row in batch:
            if number(row, "detection_eligible") == 0 or (full and number(row, "evidence_updated") == 0):
                skipped += 1
                continue
            if full:
                n, ref = number(row, "evidence_norm"), number(row, "evidence_reference")
                t, c = number(row, "evidence_temporal"), number(row, "evidence_cohort")
            else:
                z = number(row, "norm_zscore")
                cosine = number(row, "round_reference_cosine")
                threshold = number(row, "round_reference_threshold")
                n = clip(z / 2.70) if z is not None else None
                ref = direction_evidence(cosine, center, threshold) if None not in (cosine, center, threshold) else None
                t = c = None
            if n is None or ref is None:
                skipped += 1
                continue
            decision = state.observe(int(row["client_id"]), n, ref, t, c)
            accepted += decision.accepted
            if full:
                recorded = number(row, "evidence_accumulated")
                mismatches += int(recorded is None or abs(recorded - decision.accumulated) > 1e-10 or (number(row, "pass") == 1) != decision.accepted)
            if r in evaluation_rounds:
                gt = number(row, "is_malicious_gt") == 1
                key = ("FN" if gt else "TN") if decision.accepted else ("TP" if gt else "FP")
                counts[key] += 1
        if r in evaluation_rounds and not accepted:
            empty += 1
    tp, tn, fp, fn = (counts[x] for x in ("TP", "TN", "FP", "FN"))
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    eligible = full and skipped == 0 and mismatches == 0
    return dict(path=str(path), evidence_policy=policy, mode="full_frozen_feature_replay" if full else "incomplete_norm_reference_projection",
                missing_baselines=[] if full else ["temporal_adaptive_baseline", "qualified_cohort_baseline", "new_closed_loop_reference"],
                rounds=len(rounds), evaluation_rounds=sorted(evaluation_rounds), counts=counts,
                recall=recall, fpr=fpr, skipped=skipped, decision_mismatches=mismatches,
                empty_accepted_rounds=empty, full_gate_eligible=eligible,
                gate_pass=eligible and (recall is None or recall >= .9) and fpr is not None and fpr <= .05 and empty == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = {"policy": "read_from_versioned_trace", "parameters": {"norm_k": 2.70, "beta": .8, "strong_threshold": 1., "persistent_threshold": .8},
              "parameter_status": "untuned_candidate; no held-out performance claim", "runs": [replay(p) for p in args.paths]}
    report["full_validation_pass"] = all(x["gate_pass"] for x in report["runs"])
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
