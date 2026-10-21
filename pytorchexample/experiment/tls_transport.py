"""Logical TLS transport abstraction (Step 5.2).

No real sockets are opened. A ``LogicalTLSChannel`` records the serialized share
payload it "sends" and preserves the payload unchanged. ``tls_wire_bytes`` and
``relay_wire_bytes`` are intentionally reported as unavailable (no real TLS frame
overhead is measured).
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import struct

TLS_WIRE_BYTES = "unavailable"
RELAY_WIRE_BYTES = "unavailable"
RELAY_TRANSPORT = "logical_internal_delivery"


def serialize_share(share) -> bytes:
    """Serialize a ``Share`` (x + y vector) into a compact binary payload.

    Layout: uint32 x, then d uint64 field elements (each < 2^61).
    """
    d = len(share.y)
    y_bytes = struct.pack(f">{d}Q", *[int(v) % (2**64) for v in share.y])
    return struct.pack(">I", int(share.x)) + y_bytes


def deserialize_share(payload: bytes) -> tuple[int, list[int]]:
    """Inverse of ``serialize_share``."""
    x = struct.unpack(">I", payload[:4])[0]
    body = payload[4:]
    d = len(body) // 8
    y = list(struct.unpack(f">{d}Q", body))
    return x, y


class LogicalTLSChannel:
    """A logical TLS channel preserving payload and counting its bytes."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.payload_bytes = 0

    def send(self, payload: bytes) -> bytes:
        self.payload_bytes += len(payload)
        return payload  # logical TLS preserves payload (no real encryption)


class LogicalRelayPool:
    """Logical recipient for relay shares (internal delivery, not a physical node)."""

    def __init__(self) -> None:
        self.payload_bytes = 0

    def deliver(self, payload: bytes) -> bytes:
        self.payload_bytes += len(payload)
        return payload
