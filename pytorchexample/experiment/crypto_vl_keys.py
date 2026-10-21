"""Trusted-setup key lifecycle for the cryptographic VL prototype."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import secrets
from typing import Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


SETUP_SCHEMA = "vl-ssdf-uavfl/crypto-vl-setup/v1"
SETUP_VERSION = 1


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value, validate=True)


def _public_b64(private_bytes: bytes) -> str:
    key = Ed25519PrivateKey.from_private_bytes(private_bytes)
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return _b64(public)


def _private_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )


@dataclass(frozen=True)
class CryptoVLSetup:
    """Validated private setup loaded from TA-generated files."""

    setup_id: str
    setup_version: int
    hash_key: bytes
    client_private_keys: Mapping[int, bytes]
    client_public_keys: Mapping[int, str]
    client_prf_keys: Mapping[int, bytes]
    as_private_keys: Mapping[str, bytes]
    as_public_keys: Mapping[str, str]
    source: str

    def client_private_key(self, client_id: int) -> Ed25519PrivateKey:
        try:
            raw = self.client_private_keys[int(client_id)]
        except KeyError as exc:
            raise ValueError(f"client {client_id} is not registered in setup") from exc
        return Ed25519PrivateKey.from_private_bytes(raw)

    def client_prf_key(self, client_id: int) -> bytes:
        try:
            return self.client_prf_keys[int(client_id)]
        except KeyError as exc:
            raise ValueError(f"client {client_id} is not registered in setup") from exc

    def as_private_key(self, role: str) -> Ed25519PrivateKey:
        try:
            raw = self.as_private_keys[str(role)]
        except KeyError as exc:
            raise ValueError(f"unknown aggregation-server role: {role}") from exc
        return Ed25519PrivateKey.from_private_bytes(raw)


def _write_private_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.chmod(path, 0o600)
        actual_mode = path.stat().st_mode & 0o777
        if actual_mode != 0o600:
            raise PermissionError(
                f"key filesystem does not enforce mode 0600: {path} has {actual_mode:04o}"
            )
    except Exception:
        path.unlink(missing_ok=True)
        raise


def initialize_crypto_vl_setup(
    key_dir: str | Path, *, client_ids: Sequence[int], force: bool = False
) -> dict:
    """Perform one-time logical TA setup and persist its key material."""
    root = Path(key_dir).expanduser().resolve()
    ids = sorted({int(client_id) for client_id in client_ids})
    if not ids or ids[0] < 0:
        raise ValueError("client_ids must contain unique non-negative integers")

    managed = [root / "public_manifest.json", root / "server_secrets.json"]
    managed.extend(root / "clients" / f"client_{client_id}.json" for client_id in ids)
    existing = [path for path in managed if path.exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite existing setup file: {existing[0]}")
    if force:
        for path in existing:
            path.unlink()

    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    (root / "clients").mkdir(mode=0o700, exist_ok=True)
    os.chmod(root / "clients", 0o700)
    for directory in (root, root / "clients"):
        actual_mode = directory.stat().st_mode & 0o777
        if actual_mode != 0o700:
            raise PermissionError(
                f"key filesystem does not enforce mode 0700: "
                f"{directory} has {actual_mode:04o}"
            )

    setup_id = hashlib.sha256(secrets.token_bytes(32)).hexdigest()[:32]
    client_private: dict[int, bytes] = {}
    client_public: dict[int, str] = {}
    for client_id in ids:
        raw = _private_bytes(Ed25519PrivateKey.generate())
        client_private[client_id] = raw
        client_public[client_id] = _public_b64(raw)

    as_private = {
        role: _private_bytes(Ed25519PrivateKey.generate()) for role in ("AS1", "AS2")
    }
    as_public = {role: _public_b64(raw) for role, raw in as_private.items()}
    created_at = datetime.now(timezone.utc).isoformat()
    public_manifest = {
        "schema": SETUP_SCHEMA,
        "setup_id": setup_id,
        "setup_version": SETUP_VERSION,
        "created_at": created_at,
        "client_public_keys": {str(k): v for k, v in client_public.items()},
        "aggregation_server_public_keys": as_public,
    }
    server_secrets = {
        "schema": SETUP_SCHEMA,
        "setup_id": setup_id,
        "hash_key_b64": _b64(secrets.token_bytes(32)),
        "aggregation_server_private_keys": {
            role: _b64(raw) for role, raw in as_private.items()
        },
    }
    _write_private_json(root / "public_manifest.json", public_manifest)
    _write_private_json(root / "server_secrets.json", server_secrets)
    for client_id in ids:
        _write_private_json(
            root / "clients" / f"client_{client_id}.json",
            {
                "schema": SETUP_SCHEMA,
                "setup_id": setup_id,
                "client_id": client_id,
                "private_key_b64": _b64(client_private[client_id]),
                "prf_key_b64": _b64(secrets.token_bytes(32)),
            },
        )
    load_crypto_vl_setup.cache_clear()
    return public_manifest


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"setup file must contain an object: {path}")
    return value


@lru_cache(maxsize=16)
def load_crypto_vl_setup(key_dir: str | Path) -> CryptoVLSetup:
    """Load and validate a persisted setup; repeated process-local loads are cached."""
    root = Path(key_dir).expanduser().resolve()
    public = _read_json(root / "public_manifest.json")
    server = _read_json(root / "server_secrets.json")
    if public.get("schema") != SETUP_SCHEMA or server.get("schema") != SETUP_SCHEMA:
        raise ValueError("unsupported crypto VL setup schema")
    setup_id = str(public["setup_id"])
    if str(server.get("setup_id")) != setup_id:
        raise ValueError("setup_id mismatch between public and server files")
    version = int(public["setup_version"])
    if version != SETUP_VERSION:
        raise ValueError(f"unsupported crypto VL setup version: {version}")

    client_public = {
        int(client_id): str(value)
        for client_id, value in dict(public["client_public_keys"]).items()
    }
    client_private: dict[int, bytes] = {}
    client_prf: dict[int, bytes] = {}
    for client_id, expected_public in client_public.items():
        payload = _read_json(root / "clients" / f"client_{client_id}.json")
        if str(payload.get("setup_id")) != setup_id or int(payload["client_id"]) != client_id:
            raise ValueError(f"client setup context mismatch for client {client_id}")
        raw = _unb64(str(payload["private_key_b64"]))
        if _public_b64(raw) != expected_public:
            raise ValueError(f"client public/private key mismatch for client {client_id}")
        client_private[client_id] = raw
        client_prf[client_id] = _unb64(str(payload["prf_key_b64"]))

    as_public = {
        str(role): str(value)
        for role, value in dict(public["aggregation_server_public_keys"]).items()
    }
    as_private = {
        str(role): _unb64(str(value))
        for role, value in dict(server["aggregation_server_private_keys"]).items()
    }
    if set(as_public) != {"AS1", "AS2"} or set(as_private) != set(as_public):
        raise ValueError("setup must register exactly AS1 and AS2")
    for role, raw in as_private.items():
        if _public_b64(raw) != as_public[role]:
            raise ValueError(f"aggregation-server key mismatch for {role}")
    hash_key = _unb64(str(server["hash_key_b64"]))
    if len(hash_key) != 32 or any(len(value) != 32 for value in client_prf.values()):
        raise ValueError("hash and PRF keys must be 32 bytes")
    return CryptoVLSetup(
        setup_id=setup_id,
        setup_version=version,
        hash_key=hash_key,
        client_private_keys=client_private,
        client_public_keys=client_public,
        client_prf_keys=client_prf,
        as_private_keys=as_private,
        as_public_keys=as_public,
        source=str(root),
    )


def make_test_crypto_vl_setup(seed: int, client_ids: Sequence[int]) -> CryptoVLSetup:
    """Create deterministic in-memory material for isolated unit tests only."""
    ids = sorted({int(client_id) for client_id in client_ids})

    def derive(label: str) -> bytes:
        return hashlib.sha256(f"{SETUP_SCHEMA}/test-only/{seed}/{label}".encode()).digest()

    clients = {client_id: derive(f"client:{client_id}") for client_id in ids}
    as_private = {role: derive(f"server:{role}") for role in ("AS1", "AS2")}
    return CryptoVLSetup(
        setup_id=hashlib.sha256(derive("setup-id")).hexdigest()[:32],
        setup_version=SETUP_VERSION,
        hash_key=derive("delta-rho"),
        client_private_keys=clients,
        client_public_keys={k: _public_b64(v) for k, v in clients.items()},
        client_prf_keys={client_id: derive(f"prf:{client_id}") for client_id in ids},
        as_private_keys=as_private,
        as_public_keys={role: _public_b64(raw) for role, raw in as_private.items()},
        source="test-only-in-memory",
    )
