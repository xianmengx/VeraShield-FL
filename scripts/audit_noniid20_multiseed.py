"""Audit and summarize the final non-IID 20% Byzantine three-seed matrix."""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/final_v2/noniid20_multiseed_summary"
SEEDS = (7, 21, 42)
ATTACKS = {
    "sign_flip": 5.0,
    "random_gradient": 10.0,
    "scaling": 10.0,
}
SCENARIO_SLUG = {"sign_flip": "signflip", "random_gradient": "random_gradient", "scaling": "scaling"}
METHODS = ("verashield_fl", "fedavg", "krum", "trimmed_mean", "feddmc")
LABELS = {
    "verashield_fl": "VeraShield-FL",
    "fedavg": "FedAvg",
    "krum": "Krum (f=2)",
    "trimmed_mean": "Trimmed Mean (0.2)",
    "feddmc": "FedDMC",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_dir(seed: int, attack: str, method: str) -> Path:
    if seed == 7 and attack == "sign_flip":
        if method == "verashield_fl":
            return ROOT / "results/final_v2/noniid20_verashield_s7/signflip20_noniid_s7/verashield_fl"
        if method == "krum":
            return ROOT / "results/final_v2/noniid20_fixedbudget_s7/noniid_signflip20_s7_dirichlet_a05_f2/krum"
        return ROOT / f"results/final_v2/noniid20_baselines_s7/signflip20_noniid_s7/{method}"
    slug = SCENARIO_SLUG[attack]
    return ROOT / f"results/final_v2/noniid20_fixedbudget_s{seed}/noniid_{slug}20_s{seed}_dirichlet_a05/{method}"


def scenario_path(seed: int, attack: str) -> Path:
    if seed == 7 and attack == "sign_flip":
        return ROOT / "results/final_v2/scenarios/scenario_noniid_signflip20_s7_dirichlet_a05_f2.json"
    slug = SCENARIO_SLUG[attack]
    if seed == 7:
        return ROOT / f"results/final_v2/scenarios/scenario_noniid_{slug}20_s7_dirichlet_a05.json"
    return ROOT / f"results/final_v2/scenarios/scenario_noniid_{slug}20_s{seed}_dirichlet_a05.json"


def finite(values):
    return [float(v) for v in values if math.isfinite(float(v))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    result_rows = []
    detection_rows = []
    for seed in SEEDS:
        for attack, strength in ATTACKS.items():
            scenario = read_json(scenario_path(seed, attack))
            for method in METHODS:
                d = run_dir(seed, attack, method)
                summary = read_json(d / "summary.json")
                config = read_json(d / "config.json")
                with (d / "results.csv").open(newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                rounds = [r for r in rows if int(r["round"]) > 0]
                checks = {
                    "summary_rounds": summary.get("number_of_rounds") == 60,
                    "summary_clients": summary.get("number_of_clients") == 20,
                    "csv_rounds": len(rounds) == 60,
                    "partitioner": config.get("partitioner") == "dirichlet",
                    "alpha": float(config.get("alpha")) == 0.5,
                    "seed": int(config.get("seed")) == seed,
                    "attack_type": config.get("attack_type") == attack,
                    "attack_ratio": float(config.get("attack_ratio")) == 0.2,
                    "attack_strength": float(config.get("attack_strength")) == strength,
                    "attack_start": int(config.get("attack_start_round")) == 4,
                    "malicious_count": int(config.get("malicious_client_count")) == 4,
                }
                if method == "krum":
                    checks["krum_f"] = int(config.get("krum_f")) == 2
                if method == "trimmed_mean":
                    checks["trim_ratio"] = float(config.get("trim_ratio")) == 0.2
                result_rows.append({
                    "seed": seed, "attack": attack, "method": method,
                    "label": LABELS[method], "final_accuracy": float(summary["final_accuracy"]),
                    "best_accuracy": float(summary["best_accuracy"]), "best_round": int(summary["best_round"]),
                    "runtime_s": float(summary["total_time"]),
                })
                audit_rows.append({"seed": seed, "attack": attack, "method": method,
                                   "path": str(d), "all_checks_pass": all(checks.values()), **checks})
                if method == "verashield_fl":
                    metrics = read_json(d / "detection_metrics_active.json")
                    vl_rows = list(csv.DictReader((d / "vl_verification_results.csv").open(newline="", encoding="utf-8")))
                    vl_ok = sum(str(r.get("verification_success", "")).lower() in {"1", "true"} for r in vl_rows)
                    detection_rows.append({
                        "seed": seed, "attack": attack, "recall": float(metrics["recall"]),
                        "fpr": float(metrics["fpr"]), "precision": float(metrics["precision"]),
                        "f1": float(metrics["f1"]), "TP": int(metrics["TP"]), "TN": int(metrics["TN"]),
                        "FP": int(metrics["FP"]), "FN": int(metrics["FN"]),
                        "vl_rounds": len(vl_rows), "vl_pass": vl_ok,
                        "detection_gate": float(metrics["recall"]) >= 0.9 and float(metrics["fpr"]) <= 0.05,
                        "vl_gate": vl_ok == len(vl_rows),
                    })

    def write(name, rows):
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            fields = []
            for row in rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    write("configuration_audit.csv", audit_rows)
    write("per_seed_results.csv", result_rows)
    write("verashield_detection.csv", detection_rows)

    stats = []
    for attack in ATTACKS:
        for method in METHODS:
            rows = [r for r in result_rows if r["attack"] == attack and r["method"] == method]
            vals = [r["final_accuracy"] for r in rows]
            stats.append({"attack": attack, "method": method, "label": LABELS[method],
                          "n_seeds": len(vals), "mean": statistics.mean(vals),
                          "sd": statistics.stdev(vals), "values": ";".join(f"{v:.6f}" for v in vals)})
    write("mean_std_results.csv", stats)

    all_ok = all(r["all_checks_pass"] for r in audit_rows)
    lines = ["# Non-IID 20% multi-seed audit", "", f"- Seeds: {SEEDS}", "- Dataset: CIFAR-10", "- Partitioner: Dirichlet, alpha=0.5", "- Clients/rounds: 20/60", "- Attack start: round 4", "- Matrix: 3 attacks × 5 methods × 3 seeds = 45 runs", f"- Configuration audit: {'PASS' if all_ok else 'FAIL'}", ""]
    lines.append("## Final accuracy (mean ± sample SD)")
    lines.append("\n| Attack | " + " | ".join(LABELS.values()) + " |\n|---|" + "---:|" * len(METHODS))
    for attack in ATTACKS:
        vals = [next(r for r in stats if r["attack"] == attack and r["method"] == m) for m in METHODS]
        lines.append("| " + attack + " | " + " | ".join(f"{r['mean']*100:.2f} ± {r['sd']*100:.2f}%" for r in vals) + " |")
    lines.append("\n## VeraShield-FL detection")
    lines.append("\n| Attack | Recall | FPR | Precision | VL pass | Detection gate per seed |\n|---|---:|---:|---:|---:|---:|")
    for attack in ATTACKS:
        rs = [r for r in detection_rows if r["attack"] == attack]
        lines.append(f"| {attack} | {statistics.mean(r['recall'] for r in rs)*100:.2f}% ± {statistics.stdev(r['recall'] for r in rs)*100:.2f}% | {statistics.mean(r['fpr'] for r in rs)*100:.2f}% ± {statistics.stdev(r['fpr'] for r in rs)*100:.2f}% | {statistics.mean(r['precision'] for r in rs)*100:.2f}% ± {statistics.stdev(r['precision'] for r in rs)*100:.2f}% | {sum(r['vl_pass'] for r in rs)}/{sum(r['vl_rounds'] for r in rs)} | {sum(bool(r['detection_gate']) for r in rs)}/3 |")
    failed_detection = [r for r in detection_rows if not r["detection_gate"]]
    failed_vl = [r for r in detection_rows if not r["vl_gate"]]
    lines.extend(["", "## Gate audit", "", "Pre-registered per-run gate: recall >= 0.90, FPR <= 0.05, and all 60 honest VL rounds pass.", ""])
    lines.append(f"- Detection gate: {len(detection_rows)-len(failed_detection)}/{len(detection_rows)} runs pass.")
    for r in failed_detection:
        lines.append(f"  - seed {r['seed']} / {r['attack']}: recall={r['recall']:.6f}, FPR={r['fpr']:.6f}.")
    lines.append(f"- VL gate: {len(detection_rows)-len(failed_vl)}/{len(detection_rows)} runs pass.")
    for r in failed_vl:
        lines.append(f"  - seed {r['seed']} / {r['attack']}: {r['vl_pass']}/{r['vl_rounds']} rounds pass.")
    (OUT / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "AUDIT_REPORT.md")


if __name__ == "__main__":
    main()
