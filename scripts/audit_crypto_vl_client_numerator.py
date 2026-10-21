"""Read-only checks of the four-round numerator transport regression."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import argparse
import csv
import json
from pathlib import Path

from pytorchexample.experiment.crypto_vl import CryptoAggregationProof, CryptoVLVerifier
from pytorchexample.experiment.crypto_vl_keys import load_crypto_vl_setup
from pytorchexample.experiment.participation import candidate_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "results/crypto_vl_client_numerator_validation"
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    setup = load_crypto_vl_setup(args.key_dir)
    assert setup.setup_id == cfg["crypto_vl_setup_id"]
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))

    def rows(name):
        with (folder / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    vl = rows("vl_verification_results.csv")
    curve = rows("results.csv")
    assert [int(x["round"]) for x in vl] == [1, 2, 3, 4]
    assert [int(x["round"]) for x in curve] == [0, 1, 2, 3, 4]
    assert all(float(x["failures"]) == 0 for x in curve[1:])
    assert curve[3]["accuracy"] == curve[2]["accuracy"]
    matched = 0
    previous = None
    for row in vl:
        r = int(row["round"])
        committed = int(r != 3)
        assert int(row["tamper_applied"]) == int(r == 3)
        assert int(row["candidate_rejected"]) == int(r == 3)
        assert int(row["model_committed"]) == int(row["state_committed"]) == committed
        if previous:
            assert row["model_digest"] == previous
        proof = CryptoAggregationProof.from_dict(json.loads(row["proof"]))
        assert proof.aggregate_numerator is not None
        result = CryptoVLVerifier(
            setup=setup, task_id=row["task_id"], expected_round_id=r,
            expected_previous_model_version=row["model_digest"],
        ).verify(
            proof=proof, aggregate_vector=proof.aggregate_numerator,
            expected_records=proof.client_records,
            expected_recovery_digest=proof.recovery_digest,
            received_model_digest=row["candidate_model_digest"],
            expected_candidate_set_digest=candidate_digest(range(20)),
        )
        assert result["reasons"] == (["returned_model_digest_mismatch"] if r == 3 else [])
        receipts = json.loads(row["client_receipts"])
        assert len(receipts) == 20
        if r in (2, 3):
            for receipt in receipts:
                assert receipt["verification_success"]
                assert receipt["aggregate_vector_check"] == "match"
                assert receipt["reasons"] == []
                assert receipt["checked_round_id"] == r - 1
                matched += 1
        if r == 3:
            assert row["committed_model_digest"] == previous
        else:
            assert row["committed_model_digest"] == row["candidate_model_digest"]
        previous = row["committed_model_digest"]
    print(json.dumps({"status": "PASS", "rounds": 4,
                      "exact_client_receipts": matched, "tamper_rejected_round": 3,
                      "committed_rounds": [1, 2, 4]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
