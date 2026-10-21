
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import json
import os
from pathlib import Path
import tempfile
import unittest

from pytorchexample.experiment.crypto_vl import (
    CryptoVLVerifier,
    encode_vector,
    make_aggregation_proof,
    make_client_record,
    recovery_state_digest,
    weighted_aggregate,
)
from pytorchexample.experiment.crypto_vl_keys import (
    initialize_crypto_vl_setup,
    load_crypto_vl_setup,
)


class CryptoVLKeyLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        load_crypto_vl_setup.cache_clear()
        self.tempdir.cleanup()

    def test_create_load_permissions_and_public_manifest(self):
        manifest = initialize_crypto_vl_setup(self.root / "keys", client_ids=range(3))
        setup = load_crypto_vl_setup(self.root / "keys")
        self.assertEqual(setup.setup_id, manifest["setup_id"])
        self.assertEqual(set(setup.client_public_keys), {0, 1, 2})
        self.assertEqual(os.stat(self.root / "keys").st_mode & 0o777, 0o700)
        self.assertEqual(
            os.stat(self.root / "keys" / "clients").st_mode & 0o777, 0o700
        )
        for path in (self.root / "keys").rglob("*.json"):
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        public_text = (self.root / "keys" / "public_manifest.json").read_text()
        public = json.loads(public_text)
        # Inspect field names recursively. Random Base64 public-key material
        # can legitimately contain substrings such as "prf" or "secret".
        def field_names(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield str(key).lower()
                    yield from field_names(child)
            elif isinstance(value, list):
                for child in value:
                    yield from field_names(child)

        names = set(field_names(public))
        self.assertFalse(any("private" in name for name in names))
        self.assertFalse(any("secret" in name for name in names))
        self.assertFalse(any("prf" in name for name in names))
        self.assertFalse(any("hash_key" in name for name in names))
        self.assertEqual(set(public["aggregation_server_public_keys"]), {"AS1", "AS2"})

    def test_refuses_overwrite_without_force(self):
        initialize_crypto_vl_setup(self.root / "keys", client_ids=range(2))
        with self.assertRaises(FileExistsError):
            initialize_crypto_vl_setup(self.root / "keys", client_ids=range(2))

    def test_proof_from_setup_a_is_rejected_by_setup_b(self):
        initialize_crypto_vl_setup(self.root / "a", client_ids=[0])
        initialize_crypto_vl_setup(self.root / "b", client_ids=[0])
        setup_a = load_crypto_vl_setup(self.root / "a")
        setup_b = load_crypto_vl_setup(self.root / "b")
        update = encode_vector([0.25, -0.5], 100)
        record = make_client_record(
            setup=setup_a, client_id=0, task_id="task", round_id=1,
            previous_model_version="v0", candidate_set_digest="clients:0",
            encoded_update=update,
        )
        recovery = recovery_state_digest({
            "threshold": 1, "available_shares": 1,
            "recovered_client_ids": [], "failed_client_ids": [],
            "status": "success",
        })
        aggregate = weighted_aggregate([update], [1])
        proof = make_aggregation_proof(
            setup=setup_a, task_id="task", round_id=1,
            previous_model_version="v0", records=[record],
            quantized_weights={0: 1}, aggregate_vector=aggregate,
            recovery_digest=recovery, returned_model_digest="v1",
        )
        result = CryptoVLVerifier(
            setup=setup_b, task_id="task", expected_round_id=1,
            expected_previous_model_version="v0",
        ).verify(
            proof=proof, aggregate_vector=aggregate, expected_records=[record],
            expected_recovery_digest=recovery, received_model_digest="v1",
            expected_candidate_set_digest="clients:0",
        )
        self.assertFalse(result["verification_success"])
        self.assertIn("setup_id_mismatch", result["reasons"])


if __name__ == "__main__":
    unittest.main()
