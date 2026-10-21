
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import unittest
from unittest.mock import patch

import numpy as np
import torch
from flwr.app import ArrayRecord, MetricRecord
from types import SimpleNamespace

from pytorchexample.experiment.aggregation_strategy import AggregationStrategy
from pytorchexample.experiment.feddmc import FedDMCDetector


class FedDMCTests(unittest.TestCase):
    @staticmethod
    def make_updates():
        rng = np.random.default_rng(7)
        benign = rng.normal(0.0, 0.05, size=(16, 64))
        malicious = rng.normal(4.0, 0.05, size=(4, 64))
        return [
            {"weight": torch.tensor(row, dtype=torch.float32)}
            for row in np.concatenate([benign, malicious])
        ]

    def test_default_detector_rejects_separated_minority(self):
        detector = FedDMCDetector()
        updates = self.make_updates()

        _, diag = detector.detect(list(range(20)), updates)

        self.assertEqual("feddmc", diag["method"])
        self.assertEqual("pca_hdbscan_cumulative_majority", diag["detector"])
        self.assertTrue(set(diag["raw_suspect_ids"]).issuperset({16, 17, 18, 19}))
        self.assertEqual(16, diag["effective_clients"])

    def test_rolling_majority_stabilizes_detection_across_rounds(self):
        detector = FedDMCDetector()
        updates = [{"weight": torch.tensor([float(i), 0.0])} for i in range(4)]
        with patch("pytorchexample.experiment.feddmc.HDBSCAN") as hdbscan:
            hdbscan.return_value.fit_predict.side_effect = [
                np.array([0, 0, 0, 1]),
                np.array([0, 0, 0, 0]),
            ]
            first = detector.detect([0, 1, 2, 3], updates)[1]
            _, second = detector.detect([0, 1, 2, 3], updates)

        self.assertEqual([3], first["suspected_client_ids"])
        self.assertEqual([], second["suspected_client_ids"])
        self.assertEqual(1, second["history_vote_counts"][3])
        self.assertEqual(2, second["history_observation_counts"][3])

    def test_all_filtered_falls_back_to_all_updates(self):
        detector = FedDMCDetector()
        updates = self.make_updates()[:4]
        labels = [
            np.array([(i + shift) % 4 for i in range(4)])
            for shift in range(4)
        ]
        with patch("pytorchexample.experiment.feddmc.HDBSCAN") as hdbscan:
            hdbscan.return_value.fit_predict.side_effect = labels
            for _ in range(4):
                aggregate, diag = detector.detect([0, 1, 2, 3], updates)

        expected = np.mean(np.stack([u["weight"].numpy() for u in updates]), axis=0)
        np.testing.assert_allclose(aggregate, expected)
        self.assertTrue(diag["fallback_all_updates"])
        self.assertEqual(4, diag["effective_clients"])

    def test_flower_strategy_uses_feddmc_and_records_diagnostics(self):
        updates = self.make_updates()
        strategy = AggregationStrategy(
            fraction_train=1.0,
            fraction_evaluate=0.0,
            min_train_nodes=2,
            min_available_nodes=2,
            aggregation_method="feddmc",
            seed=7,
            train_metrics_aggr_fn=lambda *_: MetricRecord({}),
        )
        strategy._current_arrays = ArrayRecord({"weight": torch.zeros(64)})
        strategy._check_and_log_replies = lambda replies, is_train, validate=True: (
            list(replies), []
        )
        replies = [
            SimpleNamespace(
                content={
                    "arrays": ArrayRecord(update),
                    "metrics": {
                        "client_partition_id": cid,
                        "num-examples": 1,
                    },
                }
            )
            for cid, update in enumerate(updates)
        ]

        arrays, _ = strategy.aggregate_train(1, replies)
        final = arrays.to_torch_state_dict()["weight"].numpy()

        np.testing.assert_allclose(
            final,
            np.mean(np.stack([u["weight"].numpy() for u in updates[:16]]), axis=0),
            atol=1e-5,
        )
        diag = strategy.aggregation_records[1]
        self.assertEqual("feddmc", diag["method"])
        self.assertEqual(16, diag["effective_clients"])


if __name__ == "__main__":
    unittest.main()
