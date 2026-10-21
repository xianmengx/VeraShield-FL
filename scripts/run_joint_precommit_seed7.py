"""Run the repaired main method on the frozen CIFAR-10 joint scenario."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

from run_joint_main_seed7 import ROOT, cmd, now, save

BATCH = ROOT / "reproductions/joint_precommit_cifar10_seed7_20261015"


def main():
    assert not BATCH.exists(), "Existing output must not be overwritten"
    BATCH.mkdir(parents=True)
    reference = ROOT / "results/final_v7/joint_main_seed7/scenario.json"
    scenario = BATCH / "scenario.json"
    save(scenario, json.loads(reference.read_text()))
    out = BATCH / "verashield_fl"
    argv = cmd("verashield_fl", out, scenario) + ["--dataset", "cifar10", "--vl-commit-mode", "precommit"]
    argv[argv.index("--experiment") + 1] = "joint_precommit_cifar10_verashield_fl_s7"
    paths = list((ROOT / "pytorchexample").rglob("*.py")) + [Path(__file__), ROOT / "scripts/run_experiment.py", ROOT / "pyproject.toml"]
    save(BATCH / "registration.json", {"created_utc": now(), "argv": argv,
        "reference_scenario": str(reference), "scenario_sha256": hashlib.sha256(scenario.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
        "scope": "One seed, repaired VeraShield-FL only; historical baseline outputs unchanged"})
    state = {"status": "RUNNING", "started_utc": now(), "pid": os.getpid(), "total": 1,
             "completed": 0, "current": "cifar10/seed7/verashield_fl"}
    save(BATCH / "status.json", state)
    env = dict(os.environ, FLOWER_VENV=_release_sys.prefix,
        PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        FLOWER_CLIENT_RESOURCES_NUM_CPUS="2", FLOWER_CLIENT_RESOURCES_NUM_GPUS="0.0", FLOWER_INIT_ARGS_NUM_GPUS="0")
    try:
        with (BATCH / "run.log").open("w") as log:
            subprocess.run(argv + ["--dry-run"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
        with (out / "vl_verification_results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert [int(x["round"]) for x in rows] == list(range(1, 61))
        rejected = [int(x["round"]) for x in rows if int(x["candidate_rejected"])]
        assert rejected == [20, 40, 59], rejected
        assert [int(x["round"]) for x in rows if int(x["tamper_applied"])] == [20, 40, 59]
        for row in rows:
            assert int(row["model_committed"]) == int(row["state_committed"]) == int(int(row["round"]) not in rejected)
            assert all(x["verification_success"] for x in json.loads(row["client_receipts"]))
        summary = json.loads((out / "summary.json").read_text())
        assert summary["number_of_rounds"] == 60
        report = {"status": "PASS", "final_accuracy": summary["final_accuracy"], "rejected_rounds": rejected,
                  "scope": "automatic completion checks; full mechanism audit still required"}
        save(out / "validation_report.json", report)
        state.update(status="COMPLETED", completed=1, current=None, report=report, completed_utc=now())
    except Exception as exc:
        state.update(status="FAILED", error=str(exc), failed_utc=now())
        raise
    finally:
        save(BATCH / "status.json", state)


if __name__ == "__main__":
    main()
