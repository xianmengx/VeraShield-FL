
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import math
import unittest

from pytorchexample.experiment.recorded_fedavg import learning_rate_for_round


class LearningRateScheduleTests(unittest.TestCase):
    def test_constant_schedule_preserves_base_rate(self):
        self.assertEqual(learning_rate_for_round(
            base_lr=0.1, min_lr=0.0, schedule="constant",
            server_round=50, total_rounds=100,
        ), 0.1)

    def test_cosine_schedule_has_expected_endpoints_and_midpoint(self):
        first = learning_rate_for_round(
            base_lr=0.1, min_lr=0.001, schedule="cosine",
            server_round=1, total_rounds=100,
        )
        last = learning_rate_for_round(
            base_lr=0.1, min_lr=0.001, schedule="cosine",
            server_round=100, total_rounds=100,
        )
        middle = learning_rate_for_round(
            base_lr=0.1, min_lr=0.001, schedule="cosine",
            server_round=50, total_rounds=99,
        )
        self.assertEqual(first, 0.1)
        self.assertEqual(last, 0.001)
        self.assertTrue(math.isclose(middle, 0.0505))

    def test_invalid_learning_rate_range_is_rejected(self):
        with self.assertRaises(ValueError):
            learning_rate_for_round(
                base_lr=0.01, min_lr=0.1, schedule="cosine",
                server_round=1, total_rounds=100,
            )


if __name__ == "__main__":
    unittest.main()
