"""Small experiment utilities: seeding, model size, environment versions."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import random
import importlib.metadata as _md
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch RNGs for reproducible experiments.

    On CPU this seeds the relevant generators. If CUDA is available, CUDA
    generators are seeded and cudnn deterministic mode is enabled. This does not
    alter any training algorithm.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def model_payload_bytes(array_record) -> int:
    """Return the serialized payload size (bytes) of an ``ArrayRecord``.

    This is the raw tensor-data size carried by the record (the model
    ``state_dict``). It is the *payload* size, NOT the network wire size.
    """
    return int(sum(len(a.data) for a in array_record.values()))


def env_versions() -> dict[str, str | None]:
    """Capture versions of key libraries used by the experiment."""
    result: dict[str, str | None] = {}
    for name in (
        "python",
        "flwr",
        "torch",
        "torchvision",
        "ray",
        "flwr-datasets",
        "datasets",
        "numpy",
    ):
        if name == "python":
            import sys

            result[name] = sys.version.split()[0]
            continue
        try:
            result[name] = _md.version(name)
        except Exception:  # pragma: no cover - informational only
            result[name] = None
    return result


def compute_class_for_partition(partition_id: int) -> str:
    """Fixed mapping client (partition) id -> compute class (UAV 0-9)."""
    if partition_id <= 2:
        return "fast"
    if partition_id <= 6:
        return "medium"
    return "slow"


def simulated_workload_units(compute_factor: float, work_units: int) -> int:
    """Number of dummy CPU-bound workload units for a given simulated factor."""
    if compute_factor <= 0:
        return 0
    return int(round((1.0 / compute_factor - 1.0) * work_units))


def run_simulated_workload(units: int, matrix_size: int = 256) -> float:
    """Deterministic CPU-bound tensor workload (shared-memory ops on fixed tensors).

    This workload is completely independent of the model: it never touches model
    parameters, the optimizer, gradients, loss or the training data. It merely
    consumes CPU to simulate a slower UAV completing the same work in more time.
    Returns the wall-clock (seconds) spent executing it.
    """
    if units <= 0:
        return 0.0
    a = torch.ones(matrix_size, matrix_size, dtype=torch.float32)
    b = torch.ones(matrix_size, matrix_size, dtype=torch.float32)
    acc = a.clone()
    t0 = time.perf_counter()
    for _ in range(units):
        # Accumulate so the matmul is guaranteed to execute (no dead-code elision).
        acc = acc + torch.mm(a, b)
    _ = acc.sum().item()
    return time.perf_counter() - t0
