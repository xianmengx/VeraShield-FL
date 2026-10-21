"""Client-side verification of a server aggregation response.

This module is the first migration step away from the old server self-check in
``vl_verification.py``.  It verifies the *declared aggregation transcript*:
round/model binding, authenticated client-set metadata, quantized weights, and
the scalar additive relation used by the current prototype.  It is deliberately
not presented as a cryptographic commitment, a zero-knowledge proof, or proof
that the server chose the semantically correct accepted set.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .vl_verification import HASH_EPS, VERIFY_TOL, VLVerification
from .participation import records_digest, verify_record


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_model(model_digest: str) -> str:
    """Normalize a model digest before it is included in the transcript."""
    if not isinstance(model_digest, str) or not model_digest:
        raise ValueError("model_digest must be a non-empty string")
    return model_digest


def state_digest(state: Mapping[str, Any]) -> str:
    """Digest a model state deterministically for message-level binding."""
    h = hashlib.sha256()
    for key in sorted(state):
        raw = state[key].detach().cpu() if hasattr(state[key], "detach") else state[key]
        value = np.asarray(raw)
        h.update(str(key).encode("utf-8"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(value.tobytes())
    return h.hexdigest()


def verify_model_receipt(*, proof_payload: Mapping[str, Any], received_state: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the client-visible returned-model and transcript binding."""
    reasons: list[str] = []
    actual = state_digest(received_state)
    if not proof_payload.get("returned_model_digest") or not hmac.compare_digest(
        str(proof_payload["returned_model_digest"]), actual
    ):
        reasons.append("returned_model_digest_mismatch")
    try:
        weights = {int(k): int(v) for k, v in proof_payload["quantized_weights"].items()}
        expected = build_transcript_binding(
            task_id=str(proof_payload["task_id"]),
            round_id=int(proof_payload["round_id"]),
            model_digest=str(proof_payload["model_digest"]),
            accepted_set=tuple(int(x) for x in proof_payload["accepted_set"]),
            quantized_weights=weights,
            commitments=proof_payload["commitments"],
            aggregate_commitment=float(proof_payload["aggregate_commitment"]),
            participation_digest=str(proof_payload.get("participation_digest", "")),
            candidate_set_digest=str(proof_payload.get("candidate_set_digest", "")),
        )
        if not hmac.compare_digest(expected, str(proof_payload["binding"])):
            reasons.append("binding_mismatch")
    except (KeyError, TypeError, ValueError):
        reasons.append("malformed_proof")
    return {"verification_success": not reasons, "reasons": reasons, "received_model_digest": actual}


def build_transcript_binding(
    *,
    task_id: str,
    round_id: int,
    model_digest: str,
    accepted_set: Sequence[int],
    quantized_weights: Mapping[int | str, int],
    commitments: Sequence[float],
    aggregate_commitment: float,
    participation_digest: str = "",
    candidate_set_digest: str = "",
) -> str:
    """Return a deterministic binding digest for the returned transcript.

    The digest detects transcript changes in this prototype.  It is not a
    signature and does not authenticate the server by itself.
    """
    ids = [int(client_id) for client_id in accepted_set]
    if ids != sorted(set(ids)):
        raise ValueError("accepted_set must be sorted and duplicate-free")
    weights = {str(int(k)): int(v) for k, v in quantized_weights.items()}
    payload = {
        "task_id": str(task_id),
        "round_id": int(round_id),
        "model_digest": digest_model(model_digest),
        "accepted_set": ids,
        "quantized_weights": weights,
        "commitments": [float(x) for x in commitments],
        "aggregate_commitment": float(aggregate_commitment),
        "participation_digest": str(participation_digest),
        "candidate_set_digest": str(candidate_set_digest),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class AggregationProof:
    task_id: str
    round_id: int
    model_digest: str
    accepted_set: tuple[int, ...]
    quantized_weights: dict[int, int]
    commitments: tuple[float, ...]
    aggregate_commitment: float
    binding: str
    participation_digest: str = ""
    candidate_set_digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "round_id": self.round_id,
            "model_digest": self.model_digest,
            "accepted_set": list(self.accepted_set),
            "quantized_weights": self.quantized_weights,
            "commitments": list(self.commitments),
            "aggregate_commitment": self.aggregate_commitment,
            "binding": self.binding,
            "participation_digest": self.participation_digest,
            "candidate_set_digest": self.candidate_set_digest,
        }


def make_proof(
    *,
    task_id: str,
    round_id: int,
    model_digest: str,
    accepted_set: Sequence[int],
    quantized_weights: Mapping[int | str, int],
    commitments: Sequence[float],
    aggregate_commitment: float,
    participation_digest: str = "",
    candidate_set_digest: str = "",
) -> AggregationProof:
    ids = tuple(int(x) for x in accepted_set)
    weights = {int(k): int(v) for k, v in quantized_weights.items()}
    binding = build_transcript_binding(
        task_id=task_id,
        round_id=round_id,
        model_digest=model_digest,
        accepted_set=ids,
        quantized_weights=weights,
        commitments=commitments,
        aggregate_commitment=aggregate_commitment,
        participation_digest=participation_digest,
        candidate_set_digest=candidate_set_digest,
    )
    return AggregationProof(
        task_id=str(task_id),
        round_id=int(round_id),
        model_digest=model_digest,
        accepted_set=ids,
        quantized_weights=weights,
        commitments=tuple(float(x) for x in commitments),
        aggregate_commitment=float(aggregate_commitment),
        binding=binding,
        participation_digest=str(participation_digest),
        candidate_set_digest=str(candidate_set_digest),
    )


class ClientAggregationVerifier:
    """Verify a returned transcript from the client's trust boundary."""

    def __init__(
        self,
        *,
        task_id: str,
        expected_round_id: int,
        expected_model_digest: str,
        hash_seed: int = 42,
        weight_scale: int = 1 << 16,
        tolerance: float = VERIFY_TOL,
    ) -> None:
        self.task_id = str(task_id)
        self.expected_round_id = int(expected_round_id)
        self.expected_model_digest = digest_model(expected_model_digest)
        self.weight_scale = int(weight_scale)
        self.tolerance = float(tolerance)
        self._vl = VLVerification(seed=hash_seed)

    def verify(
        self,
        *,
        proof: AggregationProof,
        aggregate_update: np.ndarray,
        expected_client_ids: Sequence[int],
        participation_records: Sequence[Mapping[str, Any]] | None = None,
        expected_candidate_set_digest: str | None = None,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        ids = list(proof.accepted_set)
        expected_ids = sorted(set(int(x) for x in expected_client_ids))
        if proof.task_id != self.task_id:
            reasons.append("task_id_mismatch")
        if proof.round_id != self.expected_round_id:
            reasons.append("round_id_mismatch")
        if proof.model_digest != self.expected_model_digest:
            reasons.append("model_digest_mismatch")
        if ids != sorted(set(ids)):
            reasons.append("accepted_set_not_canonical")
        if ids != expected_ids:
            reasons.append("accepted_set_mismatch")
        if len(proof.commitments) != len(ids):
            reasons.append("commitment_count_mismatch")
        if set(proof.quantized_weights) != set(ids):
            reasons.append("weight_client_set_mismatch")
        if any(v < 0 for v in proof.quantized_weights.values()):
            reasons.append("negative_weight")
        if sum(proof.quantized_weights.values()) != self.weight_scale:
            reasons.append("weight_sum_mismatch")
        if expected_candidate_set_digest is not None and not hmac.compare_digest(
            str(proof.candidate_set_digest), str(expected_candidate_set_digest)
        ):
            reasons.append("candidate_set_digest_mismatch")
        if participation_records is not None:
            records = [dict(record) for record in participation_records]
            record_ids = sorted(int(record.get("client_id", -1)) for record in records)
            if record_ids != expected_ids:
                reasons.append("participation_record_set_mismatch")
            if any(
                int(record.get("round_id", -1)) != self.expected_round_id
                or str(record.get("candidate_set_digest", ""))
                != str(proof.candidate_set_digest)
                or not verify_record(record)
                for record in records
            ):
                reasons.append("participation_signature_invalid")
            if not hmac.compare_digest(
                records_digest(records), str(proof.participation_digest)
            ):
                reasons.append("participation_digest_mismatch")

        expected_binding = build_transcript_binding(
            task_id=proof.task_id,
            round_id=proof.round_id,
            model_digest=proof.model_digest,
            accepted_set=proof.accepted_set,
            quantized_weights=proof.quantized_weights,
            commitments=proof.commitments,
            aggregate_commitment=proof.aggregate_commitment,
            participation_digest=proof.participation_digest,
            candidate_set_digest=proof.candidate_set_digest,
        )
        if not hmac.compare_digest(expected_binding, proof.binding):
            reasons.append("binding_mismatch")

        weights = [proof.quantized_weights[i] / self.weight_scale for i in ids]
        expected = self._vl.aggregate_hash(list(proof.commitments), weights)
        scale = max(abs(expected), HASH_EPS)
        error = abs(float(proof.aggregate_commitment) - expected) / scale
        if error >= self.tolerance:
            reasons.append("aggregate_commitment_mismatch")
        actual_commitment = self._vl.hash_aggregate(aggregate_update)
        actual_error = abs(actual_commitment - proof.aggregate_commitment) / max(
            abs(proof.aggregate_commitment), HASH_EPS
        )
        if actual_error >= self.tolerance:
            reasons.append("aggregate_update_mismatch")

        return {
            "verification_success": not reasons,
            "reasons": reasons,
            "aggregate_commitment_error": error,
            "aggregate_update_error": actual_error,
            "checked_round_id": proof.round_id,
            "checked_client_ids": ids,
        }
