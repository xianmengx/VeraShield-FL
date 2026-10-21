"""Extend the audited CIFAR-10 precommit pilot to seeds 21 and 42."""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from run_server_tamper_precommit_seed7 import BATCH as PILOT, command as pilot_command
from run_server_tamper_5methods_multiseed import ROOT, now, save

BATCH = ROOT / "reproductions/server_tamper_precommit_multiseed_20261015"
SEEDS = (21, 42)
ATTACKS = [20, 40, 59]


def command(seed: int, out: Path) -> list[str]:
    argv = pilot_command()
    for flag, value in (("--seed", str(seed)), ("--output-dir", str(out)),
                        ("--experiment", f"server_tamper_precommit_verashield_fl_s{seed}")):
        argv[argv.index(flag) + 1] = value
    return argv


def audit(out: Path, seed: int) -> dict:
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    cfg = json.loads((out / "config.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    rc = cfg["run_config"]
    assert summary["number_of_rounds"] == 60 and summary["number_of_clients"] == 20
    assert cfg["seed"] == seed and cfg["partitioner"] == "iid"
    for key, value in {"vl-commit-mode": "precommit", "dataset": "cifar10",
                       "attack-enabled": 0, "dropout-rate": 0,
                       "share-n": 20, "share-t": 16, "normal-threshold": 16,
                       "emergency-threshold": 12, "vl-tamper-rounds": "20,40,59"}.items():
        assert rc[key] == value, (key, rc[key], value)
    def rows(name):
        with (out / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    curve = rows("results.csv")
    assert [int(x["round"]) for x in curve] == list(range(61))
    by_round = {int(x["round"]): x for x in curve}
    for x in curve[1:]:
        assert all(math.isfinite(float(x[k])) for k in ("accuracy", "loss", "train_loss"))
        assert float(x["failures"]) == 0
    vl = rows("vl_verification_results.csv")
    assert [int(x["round"]) for x in vl] == list(range(1, 61))
    assert [int(x["round"]) for x in vl if int(x["tamper_applied"])] == ATTACKS
    assert [int(x["round"]) for x in vl if int(x["candidate_rejected"])] == ATTACKS
    previous = None
    for x in vl:
        r = int(x["round"])
        committed = int(r not in ATTACKS)
        assert x["commit_mode"] == "precommit"
        assert int(x["model_committed"]) == int(x["state_committed"]) == committed
        assert int(x["verification_success"]) == int(x["precommit_verification_success"]) == committed
        if previous:
            assert x["model_digest"] == previous["committed_model_digest"]
        if committed:
            assert x["candidate_model_digest"] == x["committed_model_digest"]
        else:
            assert x["tamper_detection_location"] == "precommit"
            assert x["committed_model_digest"] == previous["committed_model_digest"]
            assert x["candidate_model_digest"] != x["committed_model_digest"]
            assert math.isclose(float(by_round[r]["accuracy"]), float(by_round[r-1]["accuracy"]), abs_tol=1e-12)
        previous = x
    assert math.isclose(float(curve[-1]["accuracy"]), summary["final_accuracy"], abs_tol=1e-12)
    return {"status": "PASS", "seed": seed, "dataset": "cifar10",
            "final_accuracy": summary["final_accuracy"], "committed_rounds": 57,
            "tamper_applied_rounds": ATTACKS, "precommit_rejection_rounds": ATTACKS,
            "boundary": "logical prototype; no remote client confirmation certificate"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    BATCH.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((PILOT / "registration.json").read_text())
    for name, expected in pilot["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f"Pilot source changed: {name}"
    assert json.loads((PILOT / "verashield_fl/validation_report.json").read_text())["status"] == "PASS"
    jobs = [{"seed": seed, "output": str(BATCH / f"seed{seed}/verashield_fl"),
             "argv": command(seed, BATCH / f"seed{seed}/verashield_fl")} for seed in SEEDS]
    registration = {"jobs": jobs, "pilot_reference": str(PILOT.relative_to(ROOT)),
                    "source_sha256": {**pilot["source_sha256"],
                        str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                    "scope": "VeraShield-FL only; reuse completed seed7 and historical baselines"}
    plan = BATCH / "registration.json"
    if plan.exists():
        assert json.loads(plan.read_text()) == registration, "Frozen registration changed"
    else:
        save(plan, registration)
    env = dict(os.environ, FLOWER_VENV=str(Path(jobs[0]["argv"][0]).parent.parent),
               PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               FLOWER_CLIENT_RESOURCES_NUM_CPUS="2", FLOWER_CLIENT_RESOURCES_NUM_GPUS="0.0",
               FLOWER_INIT_ARGS_NUM_GPUS="0")
    if args.check_only:
        for job in jobs:
            subprocess.run(job["argv"] + ["--dry-run"], cwd=ROOT, env=env, check=True)
        print("CHECK PASS: 2 additional seeds; frozen pilot sources unchanged", flush=True)
        return
    # Never overwrite partial or finished outputs, or automatically relaunch a batch.
    assert not (BATCH / "status.json").exists(), "Batch already started; inspect status before recovery"
    assert all(not Path(job["output"]).exists() for job in jobs), "Output already exists"
    lock = BATCH / "batch.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(str(os.getpid()))
    state = {"status": "RUNNING", "started_utc": now(), "pid": os.getpid(),
             "total": 2, "completed": 0, "current": None, "results": []}
    save(BATCH / "status.json", state)
    try:
        for job in jobs:
            state.update(current=f"seed{job['seed']}/verashield_fl", current_started_utc=now())
            save(BATCH / "status.json", state)
            with (BATCH / f"seed{job['seed']}.log").open("ab", buffering=0) as log:
                subprocess.run(job["argv"], cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            report = audit(Path(job["output"]), job["seed"])
            save(Path(job["output"]) / "validation_report.json", report)
            state["results"].append(report)
            state["completed"] += 1
            save(BATCH / "status.json", state)
        state.update(status="COMPLETED", current=None, completed_utc=now())
    except Exception as exc:
        state.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", failed_utc=now())
        raise
    finally:
        save(BATCH / "status.json", state)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
