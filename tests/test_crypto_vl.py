
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import unittest
from dataclasses import replace

from pytorchexample.experiment.crypto_vl import (
    CryptoAggregationProof,
    CryptoVLVerifier,
    FIELD_ORDER,
    encode_vector,
    make_aggregation_proof,
    make_client_record,
    recovery_state_digest,
    weighted_aggregate,
)
from pytorchexample.client_app import _verify_crypto_vl_payload
from pytorchexample.experiment.crypto_vl_keys import make_test_crypto_vl_setup


class CryptoVLTests(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.task_id = "forest-uav"
        self.round_id = 7
        self.model_version = "model-v6:4d79"
        self.client_ids = (1, 2, 3)
        self.setup = make_test_crypto_vl_setup(self.seed, self.client_ids)
        self.weights = {1: 5, 2: 3, 3: 2}
        self.updates = (
            encode_vector([0.25, -0.10, 0.05], 1000),
            encode_vector([-0.05, 0.20, 0.15], 1000),
            encode_vector([0.10, 0.05, -0.20], 1000),
        )
        self.candidate_digest = "candidate-manifest-r7"
        self.records = tuple(
            make_client_record(
                setup=self.setup,
                client_id=client_id,
                task_id=self.task_id,
                round_id=self.round_id,
                previous_model_version=self.model_version,
                candidate_set_digest=self.candidate_digest,
                encoded_update=update,
            )
            for client_id, update in zip(self.client_ids, self.updates)
        )
        self.aggregate = weighted_aggregate(
            self.updates, [self.weights[client_id] for client_id in self.client_ids]
        )
        self.recovery_digest = recovery_state_digest({
            "threshold": 2,
            "available_shares": 3,
            "recovered_client_ids": [3],
            "failed_client_ids": [],
            "status": "recovered",
        })
        self.proof = make_aggregation_proof(
            setup=self.setup,
            task_id=self.task_id,
            round_id=self.round_id,
            previous_model_version=self.model_version,
            records=self.records,
            quantized_weights=self.weights,
            aggregate_vector=self.aggregate,
            recovery_digest=self.recovery_digest,
            returned_model_digest="returned-model-r7",
        )
        self.verifier = CryptoVLVerifier(
            setup=self.setup,
            task_id=self.task_id,
            expected_round_id=self.round_id,
            expected_previous_model_version=self.model_version,
        )

    def verify(self, proof=None, aggregate=None, records=None, recovery_digest=None):
        return self.verifier.verify(
            proof=self.proof if proof is None else proof,
            aggregate_vector=self.aggregate if aggregate is None else aggregate,
            expected_records=self.records if records is None else records,
            expected_recovery_digest=(
                self.recovery_digest if recovery_digest is None else recovery_digest
            ),
            received_model_digest="returned-model-r7",
            expected_candidate_set_digest=self.candidate_digest,
        )

    def test_honest_aggregate_is_accepted(self):
        result = self.verify()
        self.assertTrue(result["verification_success"], result)

    def test_aggregate_vector_tampering_is_rejected(self):
        tampered = list(self.aggregate)
        tampered[0] = (tampered[0] + 1) % FIELD_ORDER
        result = self.verify(aggregate=tuple(tampered))
        self.assertIn("aggregate_vector_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_quantized_weight_tampering_is_rejected(self):
        tampered = replace(
            self.proof, quantized_weights=((1, 4), (2, 4), (3, 2))
        )
        result = self.verify(proof=tampered)
        self.assertIn("weighted_commitment_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_accepted_client_omission_is_rejected(self):
        tampered = replace(
            self.proof,
            accepted_set=(1, 2),
            quantized_weights=((1, 5), (2, 3)),
            client_records=self.proof.client_records[:2],
        )
        result = self.verify(proof=tampered)
        self.assertIn("accepted_set_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_client_commitment_replacement_is_rejected(self):
        changed_record = replace(
            self.proof.client_records[0],
            commitment=self.proof.client_records[0].commitment + 1,
        )
        tampered = replace(
            self.proof,
            client_records=(changed_record,) + self.proof.client_records[1:],
        )
        result = self.verify(proof=tampered)
        self.assertIn("client_record_authentication_failed", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_previous_round_proof_replay_is_rejected(self):
        current_verifier = CryptoVLVerifier(
            setup=self.setup,
            task_id=self.task_id,
            expected_round_id=self.round_id + 1,
            expected_previous_model_version="model-v7:new",
        )
        result = current_verifier.verify(
            proof=self.proof,
            aggregate_vector=self.aggregate,
            expected_records=self.records,
            expected_recovery_digest=self.recovery_digest,
        )
        self.assertIn("round_id_mismatch", result["reasons"])
        self.assertIn("model_version_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_recovery_state_tampering_is_rejected(self):
        tampered_recovery = recovery_state_digest({
            "threshold": 3,
            "available_shares": 3,
            "recovered_client_ids": [],
            "failed_client_ids": [3],
            "status": "failed",
        })
        result = self.verify(recovery_digest=tampered_recovery)
        self.assertIn("recovery_state_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_transport_round_trip_preserves_signed_proof(self):
        transported = CryptoAggregationProof.from_dict(self.proof.as_dict())
        result = self.verify(proof=transported)
        self.assertTrue(result["verification_success"], result)

    def test_returned_model_replacement_is_rejected(self):
        result = self.verifier.verify(
            proof=self.proof,
            aggregate_vector=None,
            expected_records=self.records,
            expected_recovery_digest=self.recovery_digest,
            received_model_digest="different-returned-model",
            expected_candidate_set_digest=self.candidate_digest,
        )
        self.assertIn("returned_model_digest_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])

    def test_client_app_rejects_replaced_returned_model(self):
        result = _verify_crypto_vl_payload(
            proof_payload=self.proof.as_dict(),
            setup=self.setup,
            candidate_set_digest=self.candidate_digest,
            received_model_digest="tampered-client-visible-model",
            expected_task_id=self.proof.task_id, expected_round_id=self.proof.round_id,
        )
        self.assertIn("returned_model_digest_mismatch", result["reasons"])
        self.assertFalse(result["verification_success"])


if __name__ == "__main__":
    unittest.main()
