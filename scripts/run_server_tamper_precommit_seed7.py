"""Run and audit the authorized standalone server-tamper precommit pilot."""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

from run_server_tamper_5methods_multiseed import ROOT, base, now, save

BATCH = ROOT / "reproductions/server_tamper_precommit_seed7_20261015"
OUT = BATCH / "verashield_fl"


def command():
    argv = base(7, "verashield_fl", OUT)
    argv[argv.index("--experiment") + 1] = "server_tamper_precommit_verashield_fl_s7"
    return argv + ["--dataset", "cifar10", "--vl-commit-mode", "precommit",
                   "--share-n", "20", "--share-t", "16",
                   "--normal-threshold", "16", "--emergency-threshold", "12"]


def audit():
    # Proofs now carry the exact public aggregate numerator.
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    config = json.loads((OUT / "config.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    rc = config["run_config"]
    assert summary["number_of_rounds"] == 60
    assert summary["number_of_clients"] == 20
    assert config["seed"] == 7 and config["partitioner"] == "iid"
    assert rc["vl-commit-mode"] == "precommit"
    assert rc["attack-enabled"] == 0 and rc["dropout-rate"] == 0
    assert rc["normal-threshold"] == 16 and rc["emergency-threshold"] == 12
    with (OUT / "results.csv").open() as handle:
        curve = list(csv.DictReader(handle))
    assert [int(row["round"]) for row in curve] == list(range(61))
    with (OUT / "vl_verification_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["round"]) for row in rows] == list(range(1, 61))
    attacks = [20, 40, 59]
    assert [int(row["round"]) for row in rows if int(row["tamper_applied"])] == attacks
    rejected = [int(row["round"]) for row in rows if int(row["candidate_rejected"])]
    assert rejected == attacks, rejected
    previous = None
    for row in rows:
        r = int(row["round"])
        assert row["commit_mode"] == "precommit"
        assert int(row["model_committed"]) == int(row["state_committed"]) == int(r not in attacks)
        if r in attacks:
            assert int(row["verification_success"]) == 0
            assert row["tamper_detection_location"] == "precommit"
            assert row["committed_model_digest"] == previous["committed_model_digest"]
        previous = row
    return {"status": "PASS", "final_accuracy": summary["final_accuracy"],
            "tamper_applied_rounds": attacks, "precommit_rejection_rounds": rejected,
            "committed_rounds": 57, "seed": 7, "dataset": "cifar10",
            "boundary": "logical prototype; no remote client confirmation certificate"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    BATCH.mkdir(parents=True, exist_ok=True)
    paths = list((ROOT / "pytorchexample").rglob("*.py"))
    paths += [ROOT / "pyproject.toml", ROOT / "scripts/run_experiment.py", Path(__file__)]
    registration = {"argv": command(), "source_sha256": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)},
        "reference": "results/final_v7/server_tamper_5methods_multiseed/seed7/verashield_fl",
        "changes": ["next_round -> precommit", "share_n 10 -> 20; thresholds 8/6 -> 16/12"],
        "reuse_baselines": True}
    plan = BATCH / "registration.json"
    if plan.exists():
        assert json.loads(plan.read_text()) == registration, "Frozen registration changed"
    else:
        save(plan, registration)
    env = dict(os.environ, FLOWER_VENV=_release_sys.prefix,
               PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               FLOWER_CLIENT_RESOURCES_NUM_CPUS="2", FLOWER_CLIENT_RESOURCES_NUM_GPUS="0.0",
               FLOWER_INIT_ARGS_NUM_GPUS="0")
    if args.check_only:
        subprocess.run(command() + ["--dry-run"], cwd=ROOT, env=env, check=True)
        return
    assert not OUT.exists(), "Output already exists; inspect before retry"
    lock = BATCH / "batch.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(str(os.getpid()))
    state = {"status": "RUNNING", "started_utc": now(), "total": 1,
             "completed": 0, "pid": os.getpid(), "current": "seed7/verashield_fl"}
    save(BATCH / "status.json", state)
    try:
        with (BATCH / "verashield_fl.log").open("ab", buffering=0) as log:
            subprocess.run(command(), cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        report = audit()
        save(OUT / "validation_report.json", report)
        state.update(status="COMPLETED", completed=1, current=None, report=report,
                     completed_utc=now())
    except Exception as exc:
        state.update(status="FAILED", error=str(exc), failed_utc=now())
        raise
    finally:
        save(BATCH / "status.json", state)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
