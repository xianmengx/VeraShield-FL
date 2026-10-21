"""Generate scenario manifest JSON (deterministic, reused by all baselines)."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import argparse
import json
from pathlib import Path

import numpy as np


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _app_dir() -> Path:
    return _script_dir().parent


def select_malicious(seed, attack_ratio, num_clients, selection_seed):
    # Keep manifest generation independent of the training stack. This mirrors
    # attack.py's deterministic selection without importing torch.
    del seed
    n_mal = max(0, min(int(round(attack_ratio * num_clients)), num_clients))
    if n_mal == 0:
        return []
    rng = np.random.default_rng([selection_seed])
    return sorted(int(x) for x in rng.choice(num_clients, size=n_mal, replace=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--attack-type", default="sign_flip")
    parser.add_argument("--attack-strength", type=float, default=1.0)
    parser.add_argument("--attack-start-round", type=int, default=4)
    parser.add_argument("--attack-ratio", type=float, default=0.0)
    parser.add_argument("--dropout-ratio", type=float, default=0.0)
    parser.add_argument("--dropout-seed", type=int, default=99)
    parser.add_argument("--malicious-selection-seed", type=int, default=42)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--fraction-fit", type=float, default=1.0)
    parser.add_argument(
        "--partitioner",
        choices=("iid", "dirichlet"),
        default="iid",
        help="Data partitioner recorded in the manifest provenance.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Dirichlet concentration recorded in the manifest provenance.",
    )
    parser.add_argument("--output-dir", default="scenarios")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _app_dir() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"scenario_{args.scenario_id}.json"

    n_mal = int(round(args.attack_ratio * args.num_clients))
    malicious = select_malicious(args.seed, args.attack_ratio, args.num_clients, args.malicious_selection_seed)
    if not malicious:
        malicious = list(range(n_mal)) if n_mal else []

    n_drop = int(round(args.dropout_ratio * args.num_clients))
    if n_drop > 0:
        dropped_rng = np.random.default_rng([args.seed, args.dropout_seed])
        dropped = sorted(int(x) for x in dropped_rng.choice(args.num_clients, size=n_drop, replace=False))
    else:
        dropped = []

    per_round = []
    for r in range(1, args.num_rounds + 1):
        if args.fraction_fit >= 1.0:
            selected = list(range(args.num_clients))
        else:
            part_rng = np.random.default_rng([args.seed, args.num_clients, r])
            k = int(round(args.fraction_fit * args.num_clients))
            selected = sorted(int(x) for x in part_rng.choice(args.num_clients, size=k, replace=False))
        per_round.append({
            "round": r,
            "selected_clients": selected,
            "unavailable_ids": list(dropped),
            "dropped_clients": list(dropped),  # alias (back-compat)
            "malicious_clients": list(malicious),
        })

    scenario = {
        "scenario_id": args.scenario_id,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "dataset": "CIFAR-10",
        "partition_seed": args.partition_seed,
        "model_seed": args.model_seed,
        "attack_type": args.attack_type,
        "attack_strength": args.attack_strength,
        "attack_start_round": args.attack_start_round,
        "attack_ratio": args.attack_ratio,
        "malicious_selection_seed": args.malicious_selection_seed,
        "malicious_ids": list(malicious),
        "dropout_ratio": args.dropout_ratio,
        "dropout_seed": args.dropout_seed,
        "fraction_fit": args.fraction_fit,
        "partitioner": args.partitioner,
        "alpha": args.alpha,
        "per_round": per_round,
        "training_config": {
            "local_epochs": 1,
            "learning_rate": 0.1,
            "batch_size": 32,
            "optimizer": "SGD",
            "model": "Net",
            "partitioner": args.partitioner,
            "alpha": args.alpha,
        },
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2, allow_nan=False)
    print(f"wrote {out_file}")
    print("malicious_ids:", malicious, "dropped_ids:", dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
