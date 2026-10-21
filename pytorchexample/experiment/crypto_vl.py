"""Cryptographically authenticated homomorphic-hash VL prototype.

This module is intentionally independent of Flower.  It makes the first
protocol step executable on small integer vectors before the backend is wired
into model transport.  Updates and weights are fixed-point integers in
``Z_q``.  The vector hash uses the proposal's trusted-setup ``delta/rho``
material to derive secret PRF coefficients ``a_j`` and computes

    HF(x) = g ** (sum_j a_j * x_j mod q) mod p

in the prime-order subgroup of an RFC 3526 safe-prime group, and therefore
``HF(sum_i a_i x_i) = product_i HF(x_i) ** a_i`` exactly.

Ed25519 signatures authenticate client commitments and the two logical
aggregation-server statements.  HMAC tags model the proposal's PRF material
and all cryptographic material comes from a one-time trusted setup.
Completeness is checked against an authenticated
expected manifest supplied by the client, rather than inferred from a digest.

This is a protocol prototype, not a formal proof system or a production key
management implementation.  It does not use Pedersen commitments or ZKPs.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Any, Mapping, Sequence

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .crypto_vl_keys import CryptoVLSetup


# RFC 3526 MODP group 14.  P = 2Q + 1 and Q is prime.
_MODP_2048_HEX = """
FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1
29024E088A67CC74020BBEA63B139B22514A08798E3404DD
EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245
E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED
EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D
C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F
83655D23DCA3AD961C62F356208552BB9ED529077096966D
670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B
E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9
DE2BCBF6955817183995497CEA956AE515D2261898FA0510
15728E5A8AACAA68FFFFFFFFFFFFFFFF
"""
GROUP_MODULUS = int("".join(_MODP_2048_HEX.split()), 16)
FIELD_ORDER = (GROUP_MODULUS - 1) // 2
PROTOCOL_DOMAIN = "vl-ssdf-uavfl/crypto-vl/v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def encode_vector(values: Sequence[float], scale: int) -> tuple[int, ...]:
    """Encode a signed vector as canonical field elements."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    encoded: list[int] = []
    for value in values:
        integer = int(round(float(value) * scale))
        if abs(integer) >= FIELD_ORDER // 2:
            raise ValueError("fixed-point value exceeds the canonical signed range")
        encoded.append(integer % FIELD_ORDER)
    return tuple(encoded)


def weighted_aggregate(
    updates: Sequence[Sequence[int]], weights: Sequence[int]
) -> tuple[int, ...]:
    """Return the exact weighted numerator in ``Z_q``.

    Division by the fixed-point weight scale belongs to model decoding.  The
    numerator is used here so that the homomorphic verification equation has
    integer exponents and no rounding ambiguity.
    """
    if not updates or len(updates) != len(weights):
        raise ValueError("updates and weights must be non-empty and aligned")
    dimension = len(updates[0])
    if dimension == 0 or any(len(update) != dimension for update in updates):
        raise ValueError("all updates must have the same non-zero dimension")
    return tuple(
        sum(int(weight) * int(update[index]) for update, weight in zip(updates, weights))
        % FIELD_ORDER
        for index in range(dimension)
    )


def decode_weighted_aggregate(
    vector: Sequence[int], *, update_scale: int, weight_scale: int
) -> np.ndarray:
    """Decode a weighted numerator for comparison with a model delta."""
    if update_scale <= 0 or weight_scale <= 0:
        raise ValueError("fixed-point scales must be positive")
    signed = [
        int(value) - FIELD_ORDER if int(value) > FIELD_ORDER // 2 else int(value)
        for value in vector
    ]
    return np.asarray(signed, dtype=np.float64) / (update_scale * weight_scale)


def _generator(index: int) -> int:
    """Hash an index into the order-Q subgroup."""
    counter = 0
    while True:
        raw = hashlib.shake_256(
            f"{PROTOCOL_DOMAIN}/generator/{index}/{counter}".encode("ascii")
        ).digest(256)
        candidate = int.from_bytes(raw, "big") % GROUP_MODULUS
        point = pow(candidate, 2, GROUP_MODULUS)
        if point not in (0, 1):
            return point
        counter += 1


def _projection_scalar(vector: Sequence[int], hash_key: bytes) -> int:
    """Apply the trusted-setup keyed linear projection in ``Z_q``.

    SHAKE expands the protected TA-generated ``delta/rho`` material into
    independent 128-bit coefficients.
    """
    dimension = len(vector)
    stream = hashlib.shake_256(
        hash_key
        + b"/"
        + PROTOCOL_DOMAIN.encode("ascii")
        + b"/dimension/"
        + str(dimension).encode("ascii")
    ).digest(16 * dimension)
    scalar = 0
    for index, value in enumerate(vector):
        coefficient = int.from_bytes(
            stream[index * 16:(index + 1) * 16], "big"
        )
        scalar = (
            scalar + coefficient * (int(value) % FIELD_ORDER)
        ) % FIELD_ORDER
    return scalar


def homomorphic_hash(vector: Sequence[int], *, setup: CryptoVLSetup) -> int:
    if not vector:
        raise ValueError("vector must not be empty")
    return pow(
        _generator(0), _projection_scalar(vector, setup.hash_key), GROUP_MODULUS
    )


def combine_hashes(commitments: Sequence[int], weights: Sequence[int]) -> int:
    if not commitments or len(commitments) != len(weights):
        raise ValueError("commitments and weights must be non-empty and aligned")
    result = 1
    for commitment, weight in zip(commitments, weights):
        result = (
            result
            * pow(int(commitment), int(weight) % FIELD_ORDER, GROUP_MODULUS)
        ) % GROUP_MODULUS
    return result


def recovery_state_digest(state: Mapping[str, Any]) -> str:
    """Bind threshold, available shares, recovered IDs, and failure status."""
    normalized = {
        "threshold": int(state["threshold"]),
        "available_shares": int(state["available_shares"]),
        "recovered_client_ids": sorted(
            int(x) for x in state.get("recovered_client_ids", [])
        ),
        "failed_client_ids": sorted(int(x) for x in state.get("failed_client_ids", [])),
        "status": str(state["status"]),
    }
    return _digest(normalized)


def _public_b64(key: Ed25519PrivateKey) -> str:
    raw = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return base64.b64encode(raw).decode("ascii")

@dataclass(frozen=True)
class ClientHashRecord:
    setup_id: str
    client_id: int
    task_id: str
    round_id: int
    previous_model_version: str
    candidate_set_digest: str
    vector_dimension: int
    commitment: int
    auth_tag: str
    public_key: str
    signature: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "domain": PROTOCOL_DOMAIN,
            "setup_id": self.setup_id,
            "client_id": self.client_id,
            "task_id": self.task_id,
            "round_id": self.round_id,
            "previous_model_version": self.previous_model_version,
            "candidate_set_digest": self.candidate_set_digest,
            "vector_dimension": self.vector_dimension,
            "commitment": str(self.commitment),
            "auth_tag": self.auth_tag,
            "public_key": self.public_key,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "signature": self.signature}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClientHashRecord":
        return cls(
            setup_id=str(payload["setup_id"]),
            client_id=int(payload["client_id"]),
            task_id=str(payload["task_id"]),
            round_id=int(payload["round_id"]),
            previous_model_version=str(payload["previous_model_version"]),
            candidate_set_digest=str(payload["candidate_set_digest"]),
            vector_dimension=int(payload["vector_dimension"]),
            commitment=int(payload["commitment"]),
            auth_tag=str(payload["auth_tag"]),
            public_key=str(payload["public_key"]),
            signature=str(payload["signature"]),
        )


def make_client_record(
    *,
    setup: CryptoVLSetup,
    client_id: int,
    task_id: str,
    round_id: int,
    previous_model_version: str,
    candidate_set_digest: str,
    encoded_update: Sequence[int],
) -> ClientHashRecord:
    commitment = homomorphic_hash(encoded_update, setup=setup)
    key = setup.client_private_key(client_id)
    core = {
        "domain": PROTOCOL_DOMAIN,
        "setup_id": setup.setup_id,
        "client_id": int(client_id),
        "task_id": str(task_id),
        "round_id": int(round_id),
        "previous_model_version": str(previous_model_version),
        "candidate_set_digest": str(candidate_set_digest),
        "vector_dimension": len(encoded_update),
        "commitment": str(commitment),
    }
    tag = hmac.new(
        setup.client_prf_key(client_id), _canonical_bytes(core), hashlib.sha256
    ).hexdigest()
    public_key = _public_b64(key)
    signed = {**core, "auth_tag": tag, "public_key": public_key}
    signature = base64.b64encode(key.sign(_canonical_bytes(signed))).decode("ascii")
    return ClientHashRecord(
        setup_id=setup.setup_id, client_id=int(client_id), task_id=str(task_id), round_id=int(round_id),
        previous_model_version=str(previous_model_version),
        candidate_set_digest=str(candidate_set_digest), vector_dimension=len(encoded_update),
        commitment=commitment, auth_tag=tag, public_key=public_key, signature=signature,
    )


def verify_client_record(record: ClientHashRecord, *, setup: CryptoVLSetup) -> bool:
    try:
        if record.setup_id != setup.setup_id:
            return False
        expected_public = setup.client_public_keys[record.client_id]
        if not hmac.compare_digest(record.public_key, expected_public):
            return False
        core = record.unsigned_dict()
        tag_payload = {
            key: value
            for key, value in core.items()
            if key not in {"auth_tag", "public_key"}
        }
        expected_tag = hmac.new(
            setup.client_prf_key(record.client_id),
            _canonical_bytes(tag_payload), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(record.auth_tag, expected_tag):
            return False
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(expected_public, validate=True)
        )
        public_key.verify(base64.b64decode(record.signature), _canonical_bytes(core))
        return True
    except (KeyError, TypeError, ValueError, InvalidSignature):
        return False


def manifest_digest(records: Sequence[ClientHashRecord]) -> str:
    ordered = sorted(records, key=lambda record: record.client_id)
    if [record.client_id for record in ordered] != sorted(
        set(record.client_id for record in ordered)
    ):
        raise ValueError("manifest client IDs must be unique")
    return _digest([record.as_dict() for record in ordered])


@dataclass(frozen=True)
class CryptoAggregationProof:
    setup_id: str
    task_id: str
    round_id: int
    previous_model_version: str
    accepted_set: tuple[int, ...]
    quantized_weights: tuple[tuple[int, int], ...]
    client_records: tuple[ClientHashRecord, ...]
    expected_manifest_digest: str
    recovery_digest: str
    aggregate_commitment: int
    returned_model_digest: str
    binding: str
    as1_public_key: str
    as1_signature: str
    as2_public_key: str
    as2_signature: str
    # Authenticated by aggregate_commitment (and its signed binding), not by
    # hashing a rounded model delta. None permits reading historical proofs;
    # full client vector verification must reject a missing numerator.
    aggregate_numerator: tuple[int, ...] | None = None

    def weights_dict(self) -> dict[int, int]:
        return dict(self.quantized_weights)

    def statement(self) -> dict[str, Any]:
        return {
            "domain": PROTOCOL_DOMAIN,
            "setup_id": self.setup_id,
            "task_id": self.task_id,
            "round_id": self.round_id,
            "previous_model_version": self.previous_model_version,
            "accepted_set": list(self.accepted_set),
            "quantized_weights": [[client_id, weight] for client_id, weight in self.quantized_weights],
            "client_records": [record.as_dict() for record in self.client_records],
            "expected_manifest_digest": self.expected_manifest_digest,
            "recovery_digest": self.recovery_digest,
            "aggregate_commitment": str(self.aggregate_commitment),
            "returned_model_digest": self.returned_model_digest,
            "binding": self.binding,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": "crypto_homomorphic_hash",
            **self.statement(),
            "as1_public_key": self.as1_public_key,
            "as1_signature": self.as1_signature,
            "as2_public_key": self.as2_public_key,
            "as2_signature": self.as2_signature,
            "aggregate_numerator": (
                None if self.aggregate_numerator is None else [
                    str(value - FIELD_ORDER if value > FIELD_ORDER // 2 else value)
                    for value in self.aggregate_numerator
                ]
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CryptoAggregationProof":
        numerator = payload.get("aggregate_numerator")
        if numerator is not None:
            if not isinstance(numerator, list) or any(
                isinstance(value, bool) or not isinstance(value, (str, int))
                for value in numerator
            ):
                raise ValueError("malformed aggregate numerator")
            signed = tuple(int(value) for value in numerator)
            if any(abs(value) >= FIELD_ORDER // 2 for value in signed):
                raise ValueError("aggregate numerator outside signed range")
            numerator = tuple(value % FIELD_ORDER for value in signed)
        return cls(
            setup_id=str(payload["setup_id"]),
            task_id=str(payload["task_id"]),
            round_id=int(payload["round_id"]),
            previous_model_version=str(payload["previous_model_version"]),
            accepted_set=tuple(int(x) for x in payload["accepted_set"]),
            quantized_weights=tuple(
                (int(item[0]), int(item[1]))
                for item in payload["quantized_weights"]
            ),
            client_records=tuple(
                ClientHashRecord.from_dict(item)
                for item in payload["client_records"]
            ),
            expected_manifest_digest=str(payload["expected_manifest_digest"]),
            recovery_digest=str(payload["recovery_digest"]),
            aggregate_commitment=int(payload["aggregate_commitment"]),
            returned_model_digest=str(payload["returned_model_digest"]),
            binding=str(payload["binding"]),
            as1_public_key=str(payload["as1_public_key"]),
            as1_signature=str(payload["as1_signature"]),
            as2_public_key=str(payload["as2_public_key"]),
            as2_signature=str(payload["as2_signature"]),
            aggregate_numerator=numerator,
        )


def _binding_payload(
    *, setup_id: str, task_id: str, round_id: int, previous_model_version: str,
    accepted_set: Sequence[int], quantized_weights: Sequence[tuple[int, int]],
    expected_manifest_digest: str, recovery_digest: str, aggregate_commitment: int,
    returned_model_digest: str,
) -> dict[str, Any]:
    return {
        "domain": PROTOCOL_DOMAIN,
        "setup_id": str(setup_id),
        "task_id": str(task_id), "round_id": int(round_id),
        "previous_model_version": str(previous_model_version),
        "accepted_set": [int(x) for x in accepted_set],
        "quantized_weights": [[int(k), int(v)] for k, v in quantized_weights],
        "expected_manifest_digest": str(expected_manifest_digest),
        "recovery_digest": str(recovery_digest),
        "aggregate_commitment": str(int(aggregate_commitment)),
        "returned_model_digest": str(returned_model_digest),
    }


def make_aggregation_proof(
    *, setup: CryptoVLSetup, task_id: str, round_id: int, previous_model_version: str,
    records: Sequence[ClientHashRecord], quantized_weights: Mapping[int, int],
    aggregate_vector: Sequence[int], recovery_digest: str,
    returned_model_digest: str,
) -> CryptoAggregationProof:
    ordered_records = tuple(sorted(records, key=lambda record: record.client_id))
    accepted_set = tuple(record.client_id for record in ordered_records)
    if accepted_set != tuple(sorted(set(accepted_set))):
        raise ValueError("accepted set must be sorted and duplicate-free")
    weights = tuple(sorted((int(k), int(v)) for k, v in quantized_weights.items()))
    if tuple(k for k, _ in weights) != accepted_set or any(v < 0 for _, v in weights):
        raise ValueError("weights must be non-negative and match the accepted set")
    if not ordered_records or any(
        not verify_client_record(record, setup=setup) for record in ordered_records
    ):
        raise ValueError("all client records must be valid")
    dimension = ordered_records[0].vector_dimension
    if len(aggregate_vector) != dimension or any(record.vector_dimension != dimension for record in ordered_records):
        raise ValueError("vector dimensions do not match")
    expected = combine_hashes(
        [record.commitment for record in ordered_records], [weight for _, weight in weights]
    )
    actual = homomorphic_hash(aggregate_vector, setup=setup)
    if actual != expected:
        raise ValueError("aggregate vector does not match weighted client commitments")
    expected_manifest = manifest_digest(ordered_records)
    binding = _digest(_binding_payload(
        setup_id=setup.setup_id, task_id=task_id, round_id=round_id,
        previous_model_version=previous_model_version,
        accepted_set=accepted_set, quantized_weights=weights,
        expected_manifest_digest=expected_manifest, recovery_digest=recovery_digest,
        aggregate_commitment=actual, returned_model_digest=returned_model_digest,
    ))
    unsigned = {
        **_binding_payload(
            setup_id=setup.setup_id, task_id=task_id, round_id=round_id,
            previous_model_version=previous_model_version,
            accepted_set=accepted_set, quantized_weights=weights,
            expected_manifest_digest=expected_manifest, recovery_digest=recovery_digest,
            aggregate_commitment=actual, returned_model_digest=returned_model_digest,
        ),
        "client_records": [record.as_dict() for record in ordered_records],
        "binding": binding,
    }
    message = _canonical_bytes(unsigned)
    as1 = setup.as_private_key("AS1")
    as2 = setup.as_private_key("AS2")
    return CryptoAggregationProof(
        setup_id=setup.setup_id, task_id=str(task_id), round_id=int(round_id),
        previous_model_version=str(previous_model_version), accepted_set=accepted_set,
        quantized_weights=weights, client_records=ordered_records,
        expected_manifest_digest=expected_manifest, recovery_digest=str(recovery_digest),
        aggregate_commitment=actual,
        returned_model_digest=str(returned_model_digest), binding=binding,
        as1_public_key=setup.as_public_keys["AS1"],
        as1_signature=base64.b64encode(as1.sign(message)).decode("ascii"),
        as2_public_key=setup.as_public_keys["AS2"],
        as2_signature=base64.b64encode(as2.sign(message)).decode("ascii"),
        aggregate_numerator=tuple(int(value) % FIELD_ORDER for value in aggregate_vector),
    )


class CryptoVLVerifier:
    """Client-side verifier for the isolated cryptographic VL prototype."""

    def __init__(
        self, *, setup: CryptoVLSetup, task_id: str, expected_round_id: int,
        expected_previous_model_version: str,
    ) -> None:
        self.setup = setup
        self.task_id = str(task_id)
        self.expected_round_id = int(expected_round_id)
        self.expected_previous_model_version = str(expected_previous_model_version)

    def verify(
        self, *, proof: CryptoAggregationProof,
        aggregate_vector: Sequence[int] | None,
        expected_records: Sequence[ClientHashRecord], expected_recovery_digest: str,
        received_model_digest: str | None = None,
        expected_candidate_set_digest: str | None = None,
    ) -> dict[str, Any]:
        reasons: list[str] = []

        if proof.setup_id != self.setup.setup_id:
            reasons.append("setup_id_mismatch")
        if proof.task_id != self.task_id:
            reasons.append("task_id_mismatch")
        if proof.round_id != self.expected_round_id:
            reasons.append("round_id_mismatch")
        if proof.previous_model_version != self.expected_previous_model_version:
            reasons.append("model_version_mismatch")
        expected_ids = tuple(sorted(record.client_id for record in expected_records))
        if proof.accepted_set != expected_ids:
            reasons.append("accepted_set_mismatch")
        if proof.accepted_set != tuple(sorted(set(proof.accepted_set))):
            reasons.append("accepted_set_not_canonical")
        if tuple(record.client_id for record in proof.client_records) != proof.accepted_set:
            reasons.append("client_record_set_mismatch")
        try:
            expected_manifest = manifest_digest(expected_records)
            proof_manifest = manifest_digest(proof.client_records)
            if not hmac.compare_digest(proof.expected_manifest_digest, expected_manifest):
                reasons.append("expected_manifest_mismatch")
            if not hmac.compare_digest(proof_manifest, expected_manifest):
                reasons.append("client_record_manifest_mismatch")
        except ValueError:
            reasons.append("malformed_manifest")

        for record in proof.client_records:
            if not verify_client_record(record, setup=self.setup):
                reasons.append("client_record_authentication_failed")
                break
            if (
                record.task_id != proof.task_id
                or record.round_id != proof.round_id
                or record.previous_model_version != proof.previous_model_version
            ):
                reasons.append("client_record_context_mismatch")
                break
            if (
                expected_candidate_set_digest is not None
                and record.candidate_set_digest != expected_candidate_set_digest
            ):
                reasons.append("candidate_set_digest_mismatch")
                break

        weights = proof.weights_dict()
        if tuple(sorted(weights)) != proof.accepted_set or any(value < 0 for value in weights.values()):
            reasons.append("weight_set_mismatch")
        else:
            expected_commitment = combine_hashes(
                [record.commitment for record in proof.client_records],
                [weights[client_id] for client_id in proof.accepted_set],
            )
            if expected_commitment != proof.aggregate_commitment:
                reasons.append("weighted_commitment_mismatch")

        if aggregate_vector is not None:
            try:
                dimensions = {record.vector_dimension for record in proof.client_records}
                if dimensions != {len(aggregate_vector)}:
                    reasons.append("aggregate_dimension_mismatch")
                elif homomorphic_hash(
                    aggregate_vector, setup=self.setup
                ) != proof.aggregate_commitment:
                    reasons.append("aggregate_vector_mismatch")
            except ValueError:
                reasons.append("malformed_aggregate_vector")

        if (
            received_model_digest is not None
            and not hmac.compare_digest(
                proof.returned_model_digest, str(received_model_digest)
            )
        ):
            reasons.append("returned_model_digest_mismatch")

        if not hmac.compare_digest(proof.recovery_digest, str(expected_recovery_digest)):
            reasons.append("recovery_state_mismatch")
        expected_binding = _digest(_binding_payload(
            setup_id=proof.setup_id, task_id=proof.task_id, round_id=proof.round_id,
            previous_model_version=proof.previous_model_version,
            accepted_set=proof.accepted_set, quantized_weights=proof.quantized_weights,
            expected_manifest_digest=proof.expected_manifest_digest,
            recovery_digest=proof.recovery_digest,
            aggregate_commitment=proof.aggregate_commitment,
            returned_model_digest=proof.returned_model_digest,
        ))
        if not hmac.compare_digest(proof.binding, expected_binding):
            reasons.append("binding_mismatch")

        statement = proof.statement()
        for role, public_value, signature_value in (
            ("AS1", proof.as1_public_key, proof.as1_signature),
            ("AS2", proof.as2_public_key, proof.as2_signature),
        ):
            try:
                trusted_public = self.setup.as_public_keys[role]
                if not hmac.compare_digest(public_value, trusted_public):
                    raise InvalidSignature
                expected_key = Ed25519PublicKey.from_public_bytes(
                    base64.b64decode(trusted_public, validate=True)
                )
                expected_key.verify(base64.b64decode(signature_value), _canonical_bytes(statement))
            except (KeyError, TypeError, ValueError, InvalidSignature):
                reasons.append(f"{role.lower()}_signature_invalid")

        return {
            "verification_success": not reasons,
            "reasons": reasons,
            "checked_round_id": proof.round_id,
            "checked_client_ids": list(proof.accepted_set),
            "aggregate_commitment": str(proof.aggregate_commitment),
        }
