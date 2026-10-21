"""Byzantine attack framework (Step 3)."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import numpy as np
import torch


ATTACK_TYPE_CODE = {
    "none": 0,
    "sign_flip": 1,
    "scaling": 2,
    "gaussian_noise": 3,
    "random_gradient": 4,
    "backdoor": 5,
    "coordinated_gaussian": 6,
}
ATTACK_TYPE_FROM_CODE = {v: k for k, v in ATTACK_TYPE_CODE.items()}


def select_malicious_client_ids(
    seed: int,
    attack_ratio: float,
    num_clients: int,
    selection_seed: int | None = None,
) -> list[int]:
    """Deterministically choose malicious client IDs, fixed across all rounds.

    The selection is performed once (in the sense that this deterministic function
    returns the same set to every client and to the server). Rounding: the number of
    malicious clients is ``round(attack_ratio * num_clients)``, capped at
    ``num_clients``.
    """
    sel_seed = selection_seed if selection_seed is not None else seed
    n_mal = int(round(attack_ratio * num_clients))
    n_mal = max(0, min(n_mal, num_clients))
    if n_mal <= 0:
        return []
    rng = np.random.default_rng([sel_seed])
    return sorted(int(x) for x in rng.choice(num_clients, size=n_mal, replace=False))


def update_l2_norm(delta: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for t in delta.values():
        total += float((t.detach().float() ** 2).sum())
    return float(total**0.5)


def update_cosine(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    """Cosine similarity between two parameter updates (safe against zero norm)."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k in a:
        if k not in b:
            continue
        x = a[k].detach().float()
        y = b[k].detach().float()
        dot += float((x * y).sum())
        na += float((x**2).sum())
        nb += float((y**2).sum())
    na = na**0.5
    nb = nb**0.5
    if na <= 0 and nb <= 0:
        return 1.0
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / (na * nb))


def update_ratio(after: float, before: float) -> float:
    """Norm ratio after/before, safe against zero."""
    if before == 0.0:
        return 0.0 if after == 0.0 else float("inf")
    return float(after / before)


def apply_update_attack(
    attack_type: str,
    delta: dict[str, torch.Tensor],
    strength: float,
    noise_rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Return the attacked parameter update ``delta_mal`` (never touches the model).

    attack_type:
      - "none":        delta unchanged
      - "sign_flip":   -strength * delta  (strength = lambda)
      - "scaling":     +strength * delta  (strength = lambda)
      - "noise":       delta + N(0, sigma^2)  (strength = sigma)
      - "random_gradient": replace delta with a random direction whose L2 norm
        is ``strength * ||delta||``. Callers can share the RNG seed across
        malicious clients to model a coordinated, non-cancelling attack.
      - "backdoor": scale the update produced by local trigger poisoning by
        ``strength`` (model replacement amplification).
      - "coordinated_gaussian": retain the original update and add a Gaussian
        direction with L2 norm ``strength * ||delta||``. Callers use the same
        RNG seed for all malicious clients in a round to prevent cancellation.
    """
    if attack_type == "sign_flip":
        return {k: -strength * v for k, v in delta.items()}
    if attack_type == "scaling":
        return {k: strength * v for k, v in delta.items()}
    if attack_type == "gaussian_noise":
        out = {}
        for k, v in delta.items():
            base = torch.zeros_like(v.detach())
            rng = noise_rng if noise_rng is not None else np.random.default_rng(0)
            noise = torch.from_numpy(
                rng.normal(0.0, strength, size=base.numel())
            ).reshape(v.detach().shape)
            out[k] = v + noise.to(dtype=v.dtype)
        return out
    if attack_type == "random_gradient":
        rng = noise_rng if noise_rng is not None else np.random.default_rng(0)
        random_tensors = {}
        random_norm_sq = 0.0
        for k, v in delta.items():
            random_value = torch.from_numpy(
                rng.normal(0.0, 1.0, size=v.numel())
            ).reshape(v.shape).to(dtype=v.dtype)
            random_tensors[k] = random_value
            random_norm_sq += float((random_value.detach().float() ** 2).sum())
        source_norm = update_l2_norm(delta)
        random_norm = random_norm_sq**0.5
        if source_norm <= 0.0 or random_norm <= 0.0:
            return {k: torch.zeros_like(v) for k, v in delta.items()}
        scale = float(strength * source_norm / random_norm)
        return {k: scale * v for k, v in random_tensors.items()}
    if attack_type == "coordinated_gaussian":
        rng = noise_rng if noise_rng is not None else np.random.default_rng(0)
        random_tensors = {}
        random_norm_sq = 0.0
        for k, v in delta.items():
            random_value = torch.from_numpy(
                rng.normal(0.0, 1.0, size=v.numel())
            ).reshape(v.shape).to(dtype=v.dtype)
            random_tensors[k] = random_value
            random_norm_sq += float((random_value.detach().float() ** 2).sum())
        source_norm = update_l2_norm(delta)
        random_norm = random_norm_sq**0.5
        if source_norm <= 0.0 or random_norm <= 0.0:
            return dict(delta)
        scale = float(strength * source_norm / random_norm)
        return {k: delta[k] + scale * random_tensors[k] for k in delta}
    if attack_type == "backdoor":
        return {k: strength * v for k, v in delta.items()}
    return dict(delta)  # none / unknown -> unchanged
