
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import numpy as np
import torch

from pytorchexample.experiment.crypto_vl import (
    decode_weighted_aggregate,
    encode_vector,
    weighted_aggregate,
)
from pytorchexample.experiment.recovery import flatten_state
from pytorchexample.experiment.ssdf_recovery_strategy import (
    _aggregate_retained_deltas,
    _largest_remainder_weights,
)


def test_crypto_vl_aggregation_and_model_use_the_same_fixed_point_object():
    update_scale = 1 << 16
    weight_scale = 1 << 16
    global_state = {"w": torch.zeros(4, dtype=torch.float32)}
    _, metadata = flatten_state(global_state)
    client_ids = [7, 2, 11]
    deltas = [
        np.array([0.123456, -0.25, 1.0, -1.5], dtype=np.float64),
        np.array([-0.4, 0.3, 0.1, 2.0], dtype=np.float64),
        np.array([0.05, 0.07, -0.2, 0.4], dtype=np.float64),
    ]
    weights = [0.17321, 0.33333, 0.49346]

    _, model_delta = _aggregate_retained_deltas(
        global_state,
        metadata,
        client_ids,
        deltas,
        weights,
        update_scale=update_scale,
        weight_scale=weight_scale,
    )
    integer_weights = _largest_remainder_weights(weights, weight_scale)
    encoded_updates = [encode_vector(delta, update_scale) for delta in deltas]
    decoded = decode_weighted_aggregate(
        weighted_aggregate(encoded_updates, integer_weights),
        update_scale=update_scale,
        weight_scale=weight_scale,
    )

    assert sum(integer_weights) == weight_scale
    assert np.array_equal(model_delta, decoded)
