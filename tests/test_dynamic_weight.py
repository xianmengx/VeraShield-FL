"""Regression tests for the frozen Dynamic Weight rule."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import unittest

import numpy as np

from pytorchexample.experiment.dynamic_weight import (
    calculate_bounded_dynamic_weights,
    calculate_cosine_similarity,
    calculate_dynamic_weights,
    calculate_mixed_dynamic_weights,
)


class DynamicWeightTests(unittest.TestCase):
    def test_positive_cosines_are_normalized(self) -> None:
        self.assertEqual(calculate_dynamic_weights([0.2, 0.3, 0.5]), [0.2, 0.3, 0.5])


    def test_negative_cosines_receive_zero_weight(self) -> None:
        self.assertEqual(calculate_dynamic_weights([-0.2, 0.3, 0.7]), [0.0, 0.3, 0.7])


    def test_all_non_positive_cosines_receive_zero_weight(self) -> None:
        self.assertEqual(calculate_dynamic_weights([-0.2, -0.1, 0.0]), [0.0, 0.0, 0.0])


    def test_threshold_boundary_is_frozen_at_one_e_minus_twelve(self) -> None:
        self.assertEqual(calculate_dynamic_weights([1e-12, 0.0]), [0.0, 0.0])
        self.assertAlmostEqual(calculate_dynamic_weights([1.0001e-12, 0.0])[0], 1.0)
        self.assertEqual(calculate_dynamic_weights([1.0001e-12, 0.0])[1], 0.0)

    def test_explicit_reference_avoids_a_contaminated_round_mean(self) -> None:
        vecs = [np.array([1.0, 0.0]), np.array([0.0, 10.0])]
        cos = calculate_cosine_similarity(vecs, np.array([1.0, 0.0]))
        self.assertAlmostEqual(cos[0], 1.0)
        self.assertAlmostEqual(cos[1], 0.0)

    def test_bounded_weights_cap_coordinated_minority_influence(self) -> None:
        weights = calculate_bounded_dynamic_weights(
            [1.0, 1.0, 1.0, 1.0] + [0.01] * 16,
            max_weight_multiplier=2.0,
        )
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertLessEqual(max(weights), 0.1 + 1e-12)
        self.assertLessEqual(sum(weights[:4]), 0.4 + 1e-12)

    def test_bounded_weights_fall_back_to_uniform(self) -> None:
        self.assertEqual(
            calculate_bounded_dynamic_weights([-1.0, 0.0, -0.5]),
            [1.0 / 3.0] * 3,
        )

    def test_mixed_weights_keep_a_uniform_floor(self) -> None:
        weights = calculate_mixed_dynamic_weights(
            [1.0, 1.0, 1.0, 1.0] + [0.01] * 16,
            adaptive_blend=0.25,
            max_weight_multiplier=2.0,
        )
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreaterEqual(min(weights), 0.75 / 20 - 1e-12)
        self.assertLessEqual(max(weights), 0.75 / 20 + 0.25 * 0.1 + 1e-12)
        self.assertLessEqual(sum(weights[:4]), 0.25 + 1e-12)


if __name__ == "__main__":
    unittest.main()
