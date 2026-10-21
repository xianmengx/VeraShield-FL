"""Run the five-method, three-seed server aggregation tamper matrix."""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import json
import csv
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(_release_sys.executable)
BATCH = ROOT / "reproductions/final_v7/server_tamper_5methods_multiseed"
KEY_DIR = Path(str(_RELEASE_ROOT / "local_keys"))
SEEDS = (7, 21, 42)
METHODS = ("verashield_fl", "fedavg", "krum", "trimmed_mean", "feddmc")
TAMPER_ROUNDS = "20,40,59"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def base(seed: int, method: str, out: Path) -> list[str]:
    cmd = [str(PY), "scripts/run_experiment.py", "--seed", str(seed),
           "--rounds", "60", "--clients", "20", "--fraction-train", "1.0",
           "--local-epochs", "1", "--learning-rate", "0.1",
           "--lr-schedule", "cosine", "--lr-min", "0.001",
           "--partitioner", "iid", "--alpha", "1.0",
           "--attack-enabled", "0", "--attack-type", "none",
           "--attack-ratio", "0", "--attack-strength", "1",
           "--dropout-rate", "0", "--dropout-timing", "post_share",
           "--vl-tamper", "0", "--vl-tamper-rounds", TAMPER_ROUNDS,
           "--experiment", f"server_tamper_{method}_s{seed}",
           "--output-dir", str(out)]
    if method == "verashield_fl":
        cmd += ["--ssdf-enabled", "1", "--ssdf-backend", "logical_dual_server",
                "--ssdf-detector-mode", "v2_full",
                "--ssdf-evidence-policy", "unified_v2_candidate",
                "--ssdf-norm-k", "2.70", "--threshold-controller-enabled", "1",
                "--secret-recovery-enabled", "1", "--dynamic-weight", "1",
                "--dynamic-weight-blend", "0.25", "--vl-verification", "1",
                "--vl-enforce", "1", "--vl-backend", "crypto",
                "--vl-key-dir", str(KEY_DIR), "--aggregation-method", "fedavg"]
    else:
        cmd += ["--ssdf-enabled", "0", "--threshold-controller-enabled", "0",
                "--secret-recovery-enabled", "0", "--dynamic-weight", "0",
                "--vl-verification", "0", "--vl-enforce", "0",
                "--vl-backend", "linear", "--aggregation-method", method,
                "--krum-f", "2", "--trim-ratio", "0.2"]
    return cmd


def audit(out: Path, method: str) -> dict:
    cfg = json.loads((out / "config.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    if summary.get("number_of_rounds") != 60 or summary.get("number_of_clients") != 20:
        raise RuntimeError(f"invalid completed output: {out}")
    rc = cfg['run_config']
    assert cfg['partitioner'] == 'iid' and cfg['attack_enabled'] == 0
    assert cfg['dropout_rate'] == 0 and rc['vl-tamper-rounds'] == TAMPER_ROUNDS
    with (out / 'results.csv').open(newline='') as handle:
        curve = list(csv.DictReader(handle))
    assert [int(r['round']) for r in curve] == list(range(61))
    result = {"seed": cfg['seed'], "method": method, "final_accuracy": summary["final_accuracy"],
              "rounds": summary["number_of_rounds"], "tamper_rounds": [20, 40, 59]}
    scheduled = [20, 40, 59]
    skipped = []
    if method == "verashield_fl":
        rows = []
        with (out / "vl_verification_results.csv").open(encoding="utf-8") as h:
            rows = list(csv.DictReader(h))
        applied = [int(r['round']) for r in rows if int(float(r.get('tamper_applied', 0) or 0))]
        assert applied == [20, 40, 59], applied
        # The unchanged signed proof is checked by clients on the next request.
        selected = [r for r in rows if int(r["round"]) in (21, 41, 60)]
        assert len(selected) == 3
        result['detection_rounds'] = [21, 41, 60]
        result["tamper_verification_success"] = [int(float(r["verification_success"])) for r in selected]
        result["tamper_rollback_applied"] = [int(float(r.get("rollback_applied", 0))) for r in selected]
        result["tamper_detected"] = (all(x == 0 for x in result["tamper_verification_success"])
            and all(x == 1 for x in result['tamper_rollback_applied']))
        result['unexpected_verification_failures'] = [int(r['round']) for r in rows
            if int(float(r['verification_success'])) == 0 and int(r['round']) not in (21,41,60)]
        if not result['tamper_detected'] or result['unexpected_verification_failures']:
            raise RuntimeError(f'server-tamper validation failed: {result}')
        result["crypto_failures"] = sum(int(float(r["verification_success"])) == 0 for r in rows)
    else:
        with (out / 'aggregation_diagnostics.csv').open(newline='') as handle:
            diag = list(csv.DictReader(handle))
        by_round = {int(r['round']): r for r in diag}
        assert len(by_round) == len(diag) and set(by_round).issubset(range(1, 61))
        missing_scheduled = set(scheduled) - set(by_round)
        if missing_scheduled:
            raise RuntimeError(f'missing diagnostics for scheduled tamper: {missing_scheduled}')
        # The existing strategy emits no diagnostic when no valid reply exists.
        # Preserve this evidence gap instead of inventing a failure reason.
        result['missing_aggregation_diagnostic_rounds'] = sorted(set(range(1,61)) - set(by_round))
        applied = [int(r['round']) for r in diag
                   if int(float(r.get('tamper_applied', 0) or 0))]
        if not set(applied).issubset(scheduled):
            raise RuntimeError(f'unscheduled tamper injection: {applied}')
        for round_id in scheduled:
            if round_id in applied:
                continue
            row = by_round[round_id]
            status = row.get('aggregation_status', '')
            # No candidate exists after these explicit aggregation failures.
            # Missing tamper flags on a successful candidate remain an error.
            allowed = ('aggregation_failed_due_to_insufficient_clients',
                       'aggregation_failed_non_finite_updates')
            if status not in allowed:
                raise RuntimeError(f'unexplained skipped tamper round={round_id}: {row}')
            skipped.append({'round': round_id, 'reason': status,
                            'effective_clients': row.get('effective_clients', ''),
                            'available_clients': row.get('available_clients', '')})
        result["tamper_verification_success"] = None
        result["tamper_detected"] = False
        result['aggregation_failure_rounds'] = [int(r['round']) for r in diag
            if r.get('aggregation_status', '').startswith('aggregation_failed_')]
    result['scheduled_tamper_rounds'] = scheduled
    result['applied_tamper_rounds'] = applied
    result['skipped_tamper_rounds'] = skipped
    result['scheduled_tamper_count'] = len(scheduled)
    result['applied_tamper_count'] = len(applied)
    result['all_scheduled_tampers_applied'] = applied == scheduled
    result["config_tamper_rounds"] = cfg.get("vl_tamper_rounds")
    return result


def main() -> None:
    check_only = "--check-only" in sys.argv
    BATCH.mkdir(parents=True, exist_ok=True)
    jobs = [(s, m) for s in SEEDS for m in METHODS]
    commands = []
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT / 'pytorchexample').rglob('*.py'))}
    for name in ('pyproject.toml', 'scripts/run_experiment.py'):
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for seed, method in jobs:
        commands.append({"seed": seed, "method": method,
                         "output": str(BATCH / f"seed{seed}" / method),
                         "argv": base(seed, method, BATCH / f"seed{seed}" / method)})
    plan_path = BATCH / 'planned_commands.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if previous['source_sha256'] != hashes or previous['jobs'] != commands:
            raise RuntimeError('resume rejected: experiment source or commands changed')
    else:
        save(plan_path, {"created_utc": now(), "jobs": commands,
         "source_sha256": hashes,
         "settings": {"partitioner": "iid", "clients": 20, "rounds": 60,
                       "tamper_rounds": [20, 40, 59], "seeds": SEEDS}})
    if check_only:
        print(f"CHECK PASS: {len(jobs)} jobs")
        for x in commands:
            print(x["seed"], x["method"], " ".join(x["argv"]))
        return
    lock = BATCH / "batch.lock"
    if lock.exists():
        raise RuntimeError(f"batch already running: {lock}")
    lock.write_text(str(os.getpid()))
    if (BATCH / 'status.json').exists():
        previous_state = json.loads((BATCH / 'status.json').read_text())
        save(BATCH / 'status_history' / f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}.json', previous_state)
    state = {"status": "RUNNING", "started_utc": now(), "total": len(jobs),
             "completed": 0, "current": None, "results": []}
    save(BATCH / "status.json", state)
    env = dict(os.environ, FLOWER_VENV=_release_sys.prefix,
               PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               FLOWER_CLIENT_RESOURCES_NUM_CPUS="2",
               FLOWER_CLIENT_RESOURCES_NUM_GPUS="0.0", FLOWER_INIT_ARGS_NUM_GPUS="0")
    try:
        for item in commands:
            for name, digest in hashes.items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
            seed, method, out = item["seed"], item["method"], Path(item["output"])
            state["current"] = {"seed": seed, "method": method}
            save(BATCH / "status.json", state)
            if (out / "summary.json").exists():
                report = audit(out, method)
            else:
                if out.exists():
                    raise RuntimeError(f"incomplete output exists; inspect before retry: {out}")
                out.mkdir(parents=True)
                log = out.parent / f"{method}.log"
                with log.open("ab", buffering=0) as handle:
                    rc = subprocess.run(item["argv"], cwd=ROOT, env=env,
                                        stdout=handle, stderr=subprocess.STDOUT).returncode
                if rc:
                    raise RuntimeError(f"job failed seed={seed} method={method} rc={rc}")
                report = audit(out, method)
            save(out / "validation_report.json", report)
            state["results"].append(report)
            state["completed"] += 1
            save(BATCH / "status.json", state)
            print(f"DONE seed={seed} method={method} final={report['final_accuracy']}", flush=True)
        save(BATCH / "REPORT.json", {"status": "COMPLETED", "settings": {
            "seeds": SEEDS, "methods": METHODS, "tamper_rounds": [20, 40, 59]},
            "results": state["results"]})
        state.update(status="COMPLETED", completed_utc=now(), current=None)
        save(BATCH / "status.json", state)
    except Exception as exc:
        state.update(status="FAILED", error=str(exc), failed_utc=now())
        save(BATCH / "status.json", state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
