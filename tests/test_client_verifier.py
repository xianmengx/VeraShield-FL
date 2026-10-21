
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import unittest

import numpy as np

from pytorchexample.experiment.client_verifier import (
    ClientAggregationVerifier,
    make_proof,
    verify_model_receipt,
)
from pytorchexample.experiment.vl_verification import VLVerification
from pytorchexample.experiment.participation import (
    candidate_digest,
    records_digest,
    sign_receipt,
    sign_record,
    verify_receipt,
    verify_record,
)


class ClientAggregationVerifierTests(unittest.TestCase):
    def setUp(self):
        self.ids = [1, 2, 3]
        self.updates = [
            np.array([1.0, 0.0, 0.5]),
            np.array([0.2, 1.0, 0.0]),
            np.array([0.0, 0.3, 1.0]),
        ]
        self.scale = 1 << 16
        self.weights = {1: 32768, 2: 19661, 3: 13107}
        vl = VLVerification(seed=42)
        commitments = [vl.hash_update(x) for x in self.updates]
        float_weights = [self.weights[i] / self.scale for i in self.ids]
        aggregate_commitment = vl.aggregate_hash(commitments, float_weights)
        self.aggregate = sum(
            (weight * update for weight, update in zip(float_weights, self.updates)),
            start=np.zeros(3),
        )
        self.proof = make_proof(
            task_id="flower-demo",
            round_id=7,
            model_digest="model-before-r7",
            accepted_set=self.ids,
            quantized_weights=self.weights,
            commitments=commitments,
            aggregate_commitment=aggregate_commitment,
        )
        self.verifier = ClientAggregationVerifier(
            task_id="flower-demo",
            expected_round_id=7,
            expected_model_digest="model-before-r7",
        )

    def test_honest_transcript_is_accepted(self):
        result = self.verifier.verify(
            proof=self.proof,
            aggregate_update=self.aggregate,
            expected_client_ids=self.ids,
        )
        self.assertTrue(result["verification_success"], result)

    def test_deleted_client_is_rejected(self):
        tampered = make_proof(
            task_id=self.proof.task_id,
            round_id=self.proof.round_id,
            model_digest=self.proof.model_digest,
            accepted_set=[1, 2],
            quantized_weights={1: 32768, 2: 32768},
            commitments=self.proof.commitments[:2],
            aggregate_commitment=self.proof.aggregate_commitment,
        )
        result = self.verifier.verify(
            proof=tampered, aggregate_update=self.aggregate, expected_client_ids=self.ids
        )
        self.assertIn("accepted_set_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_weight_tampering_is_rejected(self):
        tampered = make_proof(
            task_id=self.proof.task_id,
            round_id=self.proof.round_id,
            model_digest=self.proof.model_digest,
            accepted_set=self.ids,
            quantized_weights={1: 40000, 2: 13107, 3: 12429},
            commitments=self.proof.commitments,
            aggregate_commitment=self.proof.aggregate_commitment,
        )
        result = self.verifier.verify(
            proof=tampered, aggregate_update=self.aggregate, expected_client_ids=self.ids
        )
        self.assertIn("aggregate_commitment_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_aggregate_tampering_is_rejected(self):
        tampered = make_proof(
            task_id=self.proof.task_id,
            round_id=self.proof.round_id,
            model_digest=self.proof.model_digest,
            accepted_set=self.ids,
            quantized_weights=self.weights,
            commitments=self.proof.commitments,
            aggregate_commitment=self.proof.aggregate_commitment + 1.0,
        )
        result = self.verifier.verify(
            proof=tampered, aggregate_update=self.aggregate, expected_client_ids=self.ids
        )
        self.assertIn("aggregate_commitment_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_replayed_round_is_rejected(self):
        tampered = make_proof(
            task_id=self.proof.task_id,
            round_id=6,
            model_digest=self.proof.model_digest,
            accepted_set=self.ids,
            quantized_weights=self.weights,
            commitments=self.proof.commitments,
            aggregate_commitment=self.proof.aggregate_commitment,
        )
        result = self.verifier.verify(
            proof=tampered, aggregate_update=self.aggregate, expected_client_ids=self.ids
        )
        self.assertIn("round_id_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_returned_model_receipt_binding(self):
        payload = self.proof.as_dict()
        payload["returned_model_digest"] = "bad-digest"
        result = verify_model_receipt(
            proof_payload=payload,
            received_state={"w": np.array([1.0, 2.0])},
        )
        self.assertIn("returned_model_digest_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_signed_participation_record(self):
        record = sign_record(
            seed=42, round_id=7, client_id=1, model_digest="local-model"
        )
        self.assertTrue(verify_record(record))
        record["client_id"] = 2
        self.assertFalse(verify_record(record))
        self.assertTrue(records_digest([sign_record(
            seed=42, round_id=7, client_id=1, model_digest="local-model"
        )]))

    def test_signed_client_rejection_receipt(self):
        receipt = sign_receipt(
            seed=42,
            receipt={
                "client_id": 2,
                "round_id": 7,
                "client_round": 8,
                "verification_success": False,
                "reasons": ["returned_model_digest_mismatch"],
                "proof_binding": "binding-7",
            },
        )
        self.assertTrue(verify_receipt(receipt, seed=42))
        self.assertFalse(verify_receipt(receipt, seed=43))
        tampered = {**receipt, "reasons": ["different_reason"]}
        self.assertFalse(verify_receipt(tampered, seed=42))

    def test_candidate_set_digest_is_signed(self):
        record = sign_record(
            seed=42,
            round_id=7,
            client_id=1,
            model_digest="local-model",
            candidate_set_digest="candidate-a",
        )
        self.assertTrue(verify_record(record))
        record["candidate_set_digest"] = "candidate-b"
        self.assertFalse(verify_record(record))

    def test_participation_digest_changes_when_record_changes(self):
        records = [
            sign_record(seed=42, round_id=7, client_id=i, model_digest=f"m-{i}")
            for i in self.ids
        ]
        original = records_digest(records)
        records[0]["signature"] = records[0]["signature"][::-1]
        self.assertNotEqual(original, records_digest(records))

    def test_model_digest_tampering_is_rejected(self):
        verifier = ClientAggregationVerifier(
            task_id="flower-demo",
            expected_round_id=7,
            expected_model_digest="different-model",
        )
        result = verifier.verify(
            proof=self.proof,
            aggregate_update=self.aggregate,
            expected_client_ids=self.ids,
        )
        self.assertIn("model_digest_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_core_verifier_checks_candidate_and_participation_records(self):
        digest = candidate_digest([1, 2, 3, 4])
        records = [
            sign_record(
                seed=42,
                round_id=7,
                client_id=client_id,
                model_digest=f"local-{client_id}",
                candidate_set_digest=digest,
            )
            for client_id in self.ids
        ]
        proof = make_proof(
            task_id=self.proof.task_id,
            round_id=self.proof.round_id,
            model_digest=self.proof.model_digest,
            accepted_set=self.ids,
            quantized_weights=self.weights,
            commitments=self.proof.commitments,
            aggregate_commitment=self.proof.aggregate_commitment,
            participation_digest=records_digest(records),
            candidate_set_digest=digest,
        )
        result = self.verifier.verify(
            proof=proof,
            aggregate_update=self.aggregate,
            expected_client_ids=self.ids,
            participation_records=records,
            expected_candidate_set_digest=digest,
        )
        self.assertTrue(result["verification_success"], result)
        records[0]["signature"] = records[0]["signature"][::-1]
        result = self.verifier.verify(
            proof=proof,
            aggregate_update=self.aggregate,
            expected_client_ids=self.ids,
            participation_records=records,
            expected_candidate_set_digest=digest,
        )
        self.assertIn("participation_signature_invalid", result["reasons"])
        self.assertFalse(result["verification_success"])



if __name__ == "__main__":
    unittest.main()
