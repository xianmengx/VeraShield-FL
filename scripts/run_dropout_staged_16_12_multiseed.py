"""Run staged post-share dropout experiments with the 16/12 threshold profile."""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import csv
import json
import os
import subprocess
import sys
import argparse
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(_release_sys.executable)
DEFAULT_BATCH = ROOT / "reproductions/final_v7/dropout_staged_16_12_multiseed"
SEEDS = (7, 21, 42)
METHODS = ("verashield_fl", "without_recovery")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def manifest(seed: int, batch: Path, partitioner: str, alpha: float) -> Path:
    path = batch / "manifests" / f"staged_dropout_s{seed}.json"
    if path.exists():
        return path
    rounds = []
    for r in range(1, 61):
        unavailable = [] if r <= 20 else list(range(4)) if r <= 40 else list(range(8))
        rounds.append({
            "round": r,
            "selected_clients": list(range(20)),
            "unavailable_ids": unavailable,
            "dropped_clients": unavailable,
            "malicious_clients": [],
        })
    save(path, {
        "scenario_id": f"staged_dropout_16_12_iid_s{seed}",
        "seed": seed, "num_clients": 20, "num_rounds": 60,
        "dataset": "CIFAR-10", "partitioner": partitioner, "alpha": alpha,
        "partition_seed": seed, "model_seed": seed,
        "attack_type": "none", "attack_enabled": 0, "attack_ratio": 0.0,
        "attack_strength": 1.0, "attack_start_round": 1,
        "malicious_selection_seed": 42, "malicious_ids": [],
        "dropout_ratio": "staged: 0% rounds 1-20; 20% rounds 21-40; 40% rounds 41-60",
        "dropout_timing": "post_share", "fraction_fit": 1.0,
        "per_round": rounds,
        "training_config": {"local_epochs": 1, "learning_rate": 0.1,
                             "lr_schedule": "cosine", "lr_min": 0.001,
                             "batch_size": 32},
    })
    return path


def command(seed: int, method: str, out: Path, scen: Path, partitioner: str, alpha: float) -> list[str]:
    common = [str(PY), "scripts/run_experiment.py", "--seed", str(seed),
              "--rounds", "60", "--clients", "20", "--fraction-train", "1.0",
              "--local-epochs", "1", "--learning-rate", "0.1",
              "--lr-schedule", "cosine", "--lr-min", "0.001",
              "--partitioner", partitioner, "--alpha", str(alpha),
              "--attack-enabled", "0", "--attack-type", "none",
              "--attack-ratio", "0", "--dropout-rate", "0",
              "--dropout-timing", "post_share", "--scenario-id",
              json.loads(scen.read_text())["scenario_id"], "--scenario-manifest", str(scen),
              "--experiment", f"dropout_staged_{method}_s{seed}", "--output-dir", str(out)]
    if method == "verashield_fl":
        common += ["--ssdf-enabled", "1", "--ssdf-backend", "logical_dual_server",
                   "--ssdf-detector-mode", "v2_full", "--secret-recovery-enabled", "1",
                   "--share-n", "20", "--share-t", "16",
                   "--normal-threshold", "16", "--emergency-threshold", "12",
                   "--dynamic-weight", "1", "--threshold-controller-enabled", "1",
                   "--vl-verification", "0", "--vl-enforce", "0",
                   "--aggregation-method", "fedavg"]
    else:
        common += ["--ssdf-enabled", "0", "--secret-recovery-enabled", "0",
                   "--threshold-controller-enabled", "0", "--dynamic-weight", "0",
                   "--vl-verification", "0", "--vl-enforce", "0",
                   "--aggregation-method", "fedavg"]
    return common


def audit(out: Path, method: str, partitioner: str) -> dict:
    cfg = json.loads((out / "config.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert summary["number_of_rounds"] == 60 and summary["number_of_clients"] == 20
    assert cfg["partitioner"] == partitioner and cfg["attack_enabled"] == 0
    rows = list(csv.DictReader((out / "results.csv").open(newline="")))
    assert [int(r["round"]) for r in rows] == list(range(61))
    result = {"seed": cfg["seed"], "method": method,
              "final_accuracy": summary["final_accuracy"], "rounds": 60}
    if method == "verashield_fl":
        result.update({"normal_threshold": cfg["normal_threshold"],
                       "emergency_threshold": cfg["emergency_threshold"],
                       "share_n": cfg["ssdf_share_n"],
                       "share_t": cfg["ssdf_share_t"]})
        recovery = list(csv.DictReader((out / "recovery_statistics.csv").open(newline="")))
        by_round = {}
        for row in recovery:
            by_round.setdefault(int(row['round']), []).append(row)
        assert set(by_round) == set(range(1, 61))
        counts = {"normal": 0, "emergency": 0, "failure": 0}
        switches = []
        for r, group in sorted(by_round.items()):
            signatures = {(x['threshold_mode'], x['active_share_set'],
                           x['active_threshold'], x['available_clients'],
                           x['switch_event']) for x in group}
            assert len(signatures) == 1, f'inconsistent threshold records round {r}'
            mode, share_set, threshold, available, switched = signatures.pop()
            expected = ('normal', 'A', 16, 20 if r <= 20 else 16) if r <= 40 else ('emergency', 'B', 12, 12)
            assert (mode, share_set, int(threshold), int(available)) == expected
            counts[mode] += 1
            if int(switched):
                switches.append(r)
        result["threshold_mode_counts"] = counts
        result['threshold_switch_rounds'] = switches
        attempted = [x for x in recovery if int(x['recovery_attempted'])]
        result['recovery_attempted_rows'] = len(attempted)
        result['recovery_successful_rows'] = sum(int(x['recovery_success']) == 1 for x in attempted)
        result['recovery_success_rate'] = result['recovery_successful_rows'] / len(attempted) if attempted else None
        result['recovery_max_abs_error'] = max(float(x['g_max_abs_error']) for x in attempted)
        assert len(attempted) == 240 and switches == [41]
        result["recovery_failed_rows"] = sum(float(x.get("recovery_success", 1)) == 0 for x in recovery)
        result["empty_accepted_rounds"] = sum(not x.get("accepted_set", "") for x in
                                                (json.loads(line) for line in
                                                 (out / "protocol_trace.jsonl").read_text().splitlines()))
    return result


def refresh_reports(batch: Path, partitioner: str) -> None:
    state = json.loads((batch / 'status.json').read_text())
    assert state['status'] == 'COMPLETED'
    reports = [(batch / f'seed{seed}' / method, audit(batch / f'seed{seed}' / method, method, partitioner))
               for seed in SEEDS for method in METHODS]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    for path in [batch/'status.json', batch/'REPORT.json'] + [p/'validation_report.json' for p, _ in reports]:
        if path.exists():
            save(batch/'statistics_history'/stamp/path.relative_to(batch), json.loads(path.read_text()))
    for path, report in reports:
        save(path/'validation_report.json', report)
    state.update(results=[r for _, r in reports], statistics_updated_utc=now(),
                 statistics_schema='round_threshold_modes_and_client_round_recovery_v2')
    save(batch/'status.json', state)
    settings = json.loads((batch/'planned_commands.json').read_text())['settings']
    save(batch/'REPORT.json', {'status': 'COMPLETED', 'settings': settings, 'results': state['results'],
                              'statistics_updated_utc': state['statistics_updated_utc']})
    lines = ['# Staged dropout results', '',
             'Threshold modes count rounds; recovery counts client-round events, not unique clients.', '',
             '|Method|Final accuracy (mean ± sample SD)|', '|---|---:|']
    for method in METHODS:
        vals = [r['final_accuracy']*100 for _,r in reports if r['method']==method]
        lines.append(f'|{method}|{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}%|')
    lines += ['', '|Seed|Normal rounds|Emergency rounds|Switch round|Recovery successes/attempts|Empty accepted rounds|',
              '|---|---:|---:|---:|---:|---:|']
    for _, r in reports:
        if r['method'] == 'verashield_fl':
            lines.append(f"|{r['seed']}|{r['threshold_mode_counts']['normal']}|{r['threshold_mode_counts']['emergency']}|41|{r['recovery_successful_rows']}/{r['recovery_attempted_rows']}|{r['empty_accepted_rounds']}|")
    lines += ['', 'Comparison limitation: without_recovery also disables SSDF and dynamic weighting; this is not an isolated recovery-only ablation.']
    (batch/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Refreshed six reports: {batch}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--partitioner', choices=('iid', 'dirichlet'), default='iid')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--batch-dir', default=str(DEFAULT_BATCH))
    parser.add_argument('--summarize-only', action='store_true', help='Refresh existing reports without starting training.')
    args = parser.parse_args()
    check = args.check_only
    BATCH = Path(args.batch_dir)
    if args.summarize_only:
        refresh_reports(BATCH, args.partitioner)
        return
    BATCH.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in SEEDS:
        scen = manifest(seed, BATCH, args.partitioner, args.alpha)
        for method in METHODS:
            jobs.append({"seed": seed, "method": method, "manifest": str(scen),
                         "output": str(BATCH / f"seed{seed}" / method),
                         "argv": command(seed, method, BATCH / f"seed{seed}" / method, scen,
                                          args.partitioner, args.alpha)})
    save(BATCH / "planned_commands.json", {"created_utc": now(), "total": len(jobs),
         "settings": {"clients": 20, "rounds": 60, "partitioner": args.partitioner, "alpha": args.alpha,
                      "stages": {"1-20": "0%", "21-40": "20%", "41-60": "40%"},
                      "dropout_timing": "post_share", "normal_threshold": 16, "emergency_threshold": 12},
         "jobs": jobs})
    if check:
        print(f"CHECK PASS: {len(jobs)} jobs, 16/12 profile")
        return
    lock = BATCH / "batch.lock"
    if lock.exists():
        raise RuntimeError("dropout batch is already running")
    lock.write_text(str(os.getpid()))
    state = {"status": "RUNNING", "total": len(jobs), "completed": 0,
             "current": None, "started_utc": now(), "results": []}
    save(BATCH / "status.json", state)
    env = dict(os.environ, FLOWER_VENV=_release_sys.prefix,
               PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               FLOWER_CLIENT_RESOURCES_NUM_CPUS="2", FLOWER_CLIENT_RESOURCES_NUM_GPUS="0.0",
               FLOWER_INIT_ARGS_NUM_GPUS="0")
    try:
        for job in jobs:
            out = Path(job["output"])
            state["current"] = {"seed": job["seed"], "method": job["method"]}
            save(BATCH / "status.json", state)
            if (out / "summary.json").exists():
                report = audit(out, job["method"], args.partitioner)
            else:
                if out.exists():
                    raise RuntimeError(f"incomplete output exists: {out}")
                out.mkdir(parents=True)
                with (out.parent / f"{job['method']}.log").open("ab", buffering=0) as log:
                    rc = subprocess.run(job["argv"], cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT).returncode
                if rc:
                    raise RuntimeError(f"failed seed={job['seed']} method={job['method']} rc={rc}")
                report = audit(out, job["method"], args.partitioner)
            save(out / "validation_report.json", report)
            state["results"].append(report); state["completed"] += 1
            save(BATCH / "status.json", state)
            print(f"DONE seed={job['seed']} method={job['method']}", flush=True)
        save(BATCH / "REPORT.json", {"status": "COMPLETED", "settings": state,
                                      "results": state["results"]})
        state.update(status="COMPLETED", current=None, completed_utc=now())
        save(BATCH / "status.json", state)
    except Exception as exc:
        state.update(status="FAILED", error=str(exc), failed_utc=now())
        save(BATCH / "status.json", state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
