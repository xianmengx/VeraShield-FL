"""Measure the current logical client/protocol overhead path.

This deliberately measures the implemented logical two-server backend.  It is
not a real network or UAV benchmark: share generation is client-side, while
joint detection is reported separately as a logical server-side operation.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import csv
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from pytorchexample.experiment.logical_dual_server_ssdf import (  # noqa: E402
    LogicalDualServerSSDF,
)


def share_bytes(bundle) -> int:
    total = 0
    for share in (bundle.vector, bundle.aggregation_vector, bundle.norm_sq):
        if share is None:
            continue
        total += int(share.c1.nbytes + share.c2.nbytes)
    return total


def run(dataset: str, dimension: int, seed: int, clients: int, repeats: int):
    rows = []
    rng = np.random.default_rng(seed)
    updates = [rng.normal(0.0, 0.01, dimension).astype(np.float64) for _ in range(clients)]
    for repeat in range(repeats):
        backend = LogicalDualServerSSDF(seed=seed + repeat, spot_check_count=20)
        share_times = []
        bundles = []
        bundle_bytes = []
        for client_id, update in enumerate(updates):
            t0 = time.perf_counter()
            bundle = backend.make_bundle(client_id, 1, update)
            share_times.append(time.perf_counter() - t0)
            bundles.append(bundle)
            bundle_bytes.append(share_bytes(bundle))
        t0 = time.perf_counter()
        decisions = backend.detect_round(bundles)
        detect_time = time.perf_counter() - t0
        rows.append({
            "dataset": dataset,
            "dimension": dimension,
            "clients": clients,
            "seed": seed,
            "repeat": repeat + 1,
            "client_share_generation_mean_ms": float(np.mean(share_times) * 1000),
            "client_share_generation_p95_ms": float(np.percentile(share_times, 95) * 1000),
            "client_share_payload_mean_bytes": float(np.mean(bundle_bytes)),
            "client_share_payload_total_bytes": int(np.sum(bundle_bytes)),
            "server_joint_detection_ms": detect_time * 1000,
            "detection_outputs": len(decisions),
            "detection_valid_outputs": int(sum(d.pass_flag for d in decisions)),
        })
    return rows


def main() -> None:
    output = ROOT / "reproductions" / "analysis" / "logical_client_overhead_20261020.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, dimension in (("CIFAR-10", 62326), ("MNIST", 44746)):
        rows.extend(run(dataset, dimension, seed=7, clients=20, repeats=3))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
