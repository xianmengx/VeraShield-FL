"""Measure the client-side Crypto-VL prototype cost on real model dimensions.

This is a protocol microbenchmark, not a Flower end-to-end or UAV hardware
experiment. It measures per-client record generation and one client's proof
verification, while separately recording proof serialization size.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pytorchexample.experiment.crypto_vl import (  # noqa: E402
    CryptoAggregationProof,
    FIELD_ORDER,
    encode_vector,
    make_aggregation_proof,
    make_client_record,
    CryptoVLVerifier,
)
from pytorchexample.experiment.crypto_vl_keys import (  # noqa: E402
    make_test_crypto_vl_setup,
)
from pytorchexample.experiment.participation import sign_receipt, sign_record  # noqa: E402


def weights(n: int) -> dict[int, int]:
    base, remainder = divmod(1 << 16, n)
    return {i: base + (1 if i < remainder else 0) for i in range(n)}


def run(dataset: str, dimension: int, seed: int, clients: int, repeats: int) -> list[dict]:
    setup = make_test_crypto_vl_setup(seed, range(clients))
    rng = np.random.default_rng(seed)
    # Small values model a one-round update while keeping fixed-point values in range.
    updates = [encode_vector(rng.normal(0.0, 0.01, dimension), 1 << 16) for _ in range(clients)]
    qweights = weights(clients)
    aggregate = tuple(
        sum(qweights[i] * updates[i][j] for i in range(clients)) % FIELD_ORDER
        for j in range(dimension)
    )

    rows: list[dict] = []
    for repeat in range(repeats):
        record_times = []
        participation_times = []
        receipt_times = []
        records = []
        client_record_bytes = []
        participation_bytes = []
        for client_id in range(clients):
            t0 = time.perf_counter()
            record = make_client_record(
                setup=setup,
                client_id=client_id,
                task_id=f"client-overhead-{dataset}",
                round_id=1,
                previous_model_version="previous-model",
                candidate_set_digest="candidate-set",
                encoded_update=updates[client_id],
            )
            record_times.append(time.perf_counter() - t0)
            records.append(record)
            client_record_bytes.append(len(json.dumps(record.as_dict(), sort_keys=True, separators=(",", ":")).encode()))
            t0 = time.perf_counter()
            participation = sign_record(
                seed=seed,
                round_id=1,
                client_id=client_id,
                model_digest="local-model",
                candidate_set_digest="candidate-set",
            )
            participation_times.append(time.perf_counter() - t0)
            participation_bytes.append(len(json.dumps(participation, sort_keys=True, separators=(",", ":")).encode()))
            t0 = time.perf_counter()
            sign_receipt(seed=seed, receipt={
                "client_id": client_id,
                "round_id": 1,
                "verification_success": True,
                "proof_binding": "binding",
            })
            receipt_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        proof = make_aggregation_proof(
            setup=setup,
            task_id=f"client-overhead-{dataset}",
            round_id=1,
            previous_model_version="previous-model",
            records=records,
            quantized_weights=qweights,
            aggregate_vector=aggregate,
            recovery_digest="recovery-state",
            returned_model_digest="returned-model",
        )
        proof_time = time.perf_counter() - t0
        proof_payload = json.dumps(proof.as_dict(), sort_keys=True, separators=(",", ":")).encode()

        verifier = CryptoVLVerifier(
            setup=setup,
            task_id=f"client-overhead-{dataset}",
            expected_round_id=1,
            expected_previous_model_version="previous-model",
        )
        t0 = time.perf_counter()
        t0 = time.perf_counter()
        decoded_proof = CryptoAggregationProof.from_dict(json.loads(proof_payload))
        proof_decode_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = verifier.verify(
            proof=decoded_proof,
            aggregate_vector=aggregate,
            expected_records=records,
            expected_recovery_digest="recovery-state",
            received_model_digest="returned-model",
            expected_candidate_set_digest="candidate-set",
        )
        verify_time = time.perf_counter() - t0
        rows.append({
            "dataset": dataset,
            "dimension": dimension,
            "clients": clients,
            "seed": seed,
            "repeat": repeat + 1,
            "client_record_generation_mean_ms": float(np.mean(record_times) * 1000),
            "client_record_generation_p95_ms": float(np.percentile(record_times, 95) * 1000),
            "participation_sign_mean_ms": float(np.mean(participation_times) * 1000),
            "receipt_sign_mean_ms": float(np.mean(receipt_times) * 1000),
            "proof_decode_ms": proof_decode_time * 1000,
            "client_verification_ms": verify_time * 1000,
            "server_proof_generation_ms": proof_time * 1000,
            "proof_bytes": len(proof_payload),
            "client_record_bytes_mean": float(np.mean(client_record_bytes)),
            "participation_record_bytes_mean": float(np.mean(participation_bytes)),
            "verification_success": int(result["verification_success"]),
        })
    return rows


def main() -> None:
    output = ROOT / "reproductions" / "analysis" / "client_overhead_microbenchmark_payload_aligned_20261020.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Dimensions match the float32 model payloads recorded in the formal runs.
    # They are not asserted to match the current source model parameter count.
    for dataset, dimension in (("CIFAR-10", 62326), ("MNIST", 44746)):
        rows.extend(run(dataset, dimension, seed=7, clients=20, repeats=3))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
