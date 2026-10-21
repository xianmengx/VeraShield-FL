"""Logical two-server share routing used by the prototype.

This module deliberately models roles and share ownership in one process.  It
does not provide process isolation, network transport, or cryptographic
confidentiality.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

from dataclasses import dataclass, field
from typing import Generic, Iterable, TypeVar


T = TypeVar("T")


@dataclass
class LogicalServer(Generic[T]):
    """One logical server's in-process share store."""

    name: str
    shares: list[T] = field(default_factory=list)

    def put(self, shares: Iterable[T]) -> None:
        self.shares.extend(shares)

    @property
    def share_count(self) -> int:
        return len(self.shares)


@dataclass
class LogicalTwoServerStore(Generic[T]):
    """Deterministically route a share sequence to C1 and C2.

    Round-robin routing keeps the two roles symmetric and makes the ownership
    boundary observable in experiment records without changing the recovery
    algorithm.  ``combined`` returns shares in original order.
    """

    c1: LogicalServer[T] = field(default_factory=lambda: LogicalServer("C1"))
    c2: LogicalServer[T] = field(default_factory=lambda: LogicalServer("C2"))

    def provision(self, shares: Iterable[T]) -> None:
        for index, share in enumerate(shares):
            (self.c1 if index % 2 == 0 else self.c2).put([share])

    @property
    def share_count(self) -> int:
        return self.c1.share_count + self.c2.share_count

    def combined(self) -> list[T]:
        result: list[T] = []
        for index in range(self.share_count):
            server = self.c1 if index % 2 == 0 else self.c2
            local_index = index // 2
            if local_index < server.share_count:
                result.append(server.shares[local_index])
        return result

    def audit(self) -> dict[str, object]:
        return {
            "server_topology": "logical_two_server",
            "server_names": ["C1", "C2"],
            "independent_server_processes": False,
            "c1_share_count": self.c1.share_count,
            "c2_share_count": self.c2.share_count,
            "total_share_count": self.share_count,
        }
