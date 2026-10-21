"""Deterministic Ed25519 participation records for the simulation prototype."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import base64
import hashlib
import json
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def _payload(round_id: int, client_id: int, model_digest: str) -> bytes:
    return json.dumps(
        {"round_id": int(round_id), "client_id": int(client_id), "model_digest": str(model_digest)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def candidate_digest(candidate_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(set(int(x) for x in candidate_ids))).encode()).hexdigest()


def private_key(seed: int, client_id: int) -> Ed25519PrivateKey:
    raw = hashlib.sha256(f"vl-participation:{int(seed)}:{int(client_id)}".encode()).digest()
    return Ed25519PrivateKey.from_private_bytes(raw)


def sign_record(*, seed: int, round_id: int, client_id: int, model_digest: str, candidate_set_digest: str = "") -> dict[str, Any]:
    key = private_key(seed, client_id)
    message = _payload(round_id, client_id, model_digest) + str(candidate_set_digest).encode()
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {
        "round_id": int(round_id), "client_id": int(client_id), "model_digest": str(model_digest),
        "public_key": base64.b64encode(public).decode("ascii"),
        "signature": base64.b64encode(key.sign(message)).decode("ascii"),
        "candidate_set_digest": str(candidate_set_digest),
    }


def verify_record(record: dict[str, Any]) -> bool:
    try:
        message = _payload(record["round_id"], record["client_id"], record["model_digest"]) + str(record.get("candidate_set_digest", "")).encode()
        public = base64.b64decode(record["public_key"])
        signature = base64.b64decode(record["signature"])
        Ed25519PublicKey.from_public_bytes(public).verify(signature, message)
        return True
    except (KeyError, TypeError, ValueError, InvalidSignature):
        return False


def sign_receipt(*, seed: int, receipt: dict[str, Any]) -> dict[str, Any]:
    """Sign a canonical client verification receipt for this prototype."""
    payload = {
        key: value for key, value in receipt.items()
        if key not in {"public_key", "signature", "receipt_valid"}
    }
    message = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    client_id = int(payload["client_id"])
    key = private_key(seed, client_id)
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return {
        **payload,
        "public_key": base64.b64encode(public).decode("ascii"),
        "signature": base64.b64encode(key.sign(message)).decode("ascii"),
    }


def verify_receipt(receipt: dict[str, Any], *, seed: int) -> bool:
    """Verify a receipt against the deterministic prototype client key."""
    try:
        payload = {
            key: value for key, value in receipt.items()
            if key not in {"public_key", "signature", "receipt_valid"}
        }
        message = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        expected_public = private_key(
            seed, int(payload["client_id"])
        ).public_key()
        if base64.b64decode(receipt["public_key"]) != expected_public.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ):
            return False
        expected_public.verify(
            base64.b64decode(receipt["signature"]), message
        )
        return True
    except (KeyError, TypeError, ValueError, InvalidSignature):
        return False


def records_digest(records: list[dict[str, Any]]) -> str:
    canonical = sorted(records, key=lambda x: int(x["client_id"]))
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
