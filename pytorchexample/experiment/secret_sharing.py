"""Shamir Secret Sharing over GF(p) with fixed-point float encoding (Step 5.1).

Standalone correctness module. This is NOT wired to Flower / SSDF yet.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

from dataclasses import dataclass

import numpy as np

MODULUS = 2**61 - 1  # Mersenne prime


@dataclass
class Share:
    x: int
    y: list[int]  # length = dimension


def _modinv(a: int, p: int = MODULUS) -> int:
    return pow(int(a) % p, p - 2, p)


def _eval_poly(coeffs: list[int], x: int, p: int = MODULUS) -> int:
    # Horner: coeffs[0] is the constant term f(0).
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % p
    return acc


def fixed_point_encode(vec: np.ndarray, scale: int) -> list[int]:
    return [int(round(float(v) * scale)) % MODULUS for v in vec]


def fixed_point_decode(secret: list[int], scale: int) -> np.ndarray:
    out = []
    for v in secret:
        v = int(v) % MODULUS
        if v > MODULUS // 2:
            v -= MODULUS
        out.append(v / scale)
    return np.array(out, dtype=float)


def share_vector(
    secret: list[int],
    n: int,
    t: int,
    rng: np.random.Generator,
    x_points: list[int] | None = None,
) -> list[Share]:
    """Share a field-vector of length d into n shares with threshold t."""
    d = len(secret)
    if x_points is None:
        x_points = list(range(1, n + 1))
    # For each position k, a polynomial of degree t-1 with f_k(0)=secret[k].
    coeffs = np.zeros((d, t), dtype=object)
    for k in range(d):
        coeffs[k, 0] = int(secret[k]) % MODULUS
        for deg in range(1, t):
            coeffs[k, deg] = int(rng.integers(0, MODULUS)) % MODULUS
    shares = []
    for xi in x_points:
        y = []
        for k in range(d):
            poly = [int(coeffs[k, deg]) % MODULUS for deg in range(t)]
            y.append(_eval_poly(poly, int(xi)))
        shares.append(Share(x=int(xi), y=y))
    return shares


def reconstruct_vector(
    shares: list[Share], t: int, p: int = MODULUS
) -> list[int]:
    """Reconstruct the secret vector from at least t shares (Lagrange at x=0)."""
    sel = shares[:t]
    xs = [s.x for s in sel]
    d = len(sel[0].y)
    secret = [0] * d
    for i, s in enumerate(sel):
        basis = 1
        denom = 1
        for j, xj in enumerate(xs):
            if i == j:
                continue
            basis = (basis * (-xj)) % p
            denom = (denom * (s.x - xj)) % p
        li0 = (basis * _modinv(denom, p)) % p
        for k in range(d):
            secret[k] = (secret[k] + int(s.y[k]) * li0) % p
    return secret


def share_bytes(share: Share, p: int = MODULUS) -> int:
    # x (int) + d field elements; each field element serialized as 8 bytes.
    return 8 + len(share.y) * 8


def vector_share_bytes(shares: list[Share]) -> int:
    return sum(share_bytes(s) for s in shares)
