
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from flwr.app import MetricRecord, RecordDict
from types import SimpleNamespace

from pytorchexample.experiment.ssdf import SSDFDetector
from pytorchexample.experiment.ssdf_strategy import SSDFStrategy
from pytorchexample.experiment.recorder import ExperimentRecorder


def candidate(client_id, server_round, values, malicious=0, active=0):
    return {
        "client_id": client_id,
        "server_round": server_round,
        "delta": {"w": torch.tensor(values, dtype=torch.float32)},
        "is_malicious_gt": malicious,
        "attack_active": active,
    }


class SSDFV2Tests(unittest.TestCase):
    def test_metric_aggregation_rejects_zero_weight_with_client_diagnostics(self):
        strategy = object.__new__(SSDFStrategy)
        strategy.weighted_by_key = "num-examples"
        strategy.train_metrics_aggr_fn = lambda messages, key: None
        messages = [
            SimpleNamespace(content=RecordDict({"metrics": MetricRecord({
                "client_partition_id": 4,
                "num-examples": 0,
            })}))
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            r"TRAIN_METRIC_WEIGHT_TOTAL_INVALID round=2 .*client_id=4:weight=0.0",
        ):
            strategy._aggregate_train_metrics_checked(2, messages)

    def test_metric_aggregation_preserves_positive_weight_path(self):
        strategy = object.__new__(SSDFStrategy)
        strategy.weighted_by_key = "num-examples"
        expected = object()
        strategy.train_metrics_aggr_fn = lambda messages, key: expected
        messages = [
            SimpleNamespace(content=RecordDict({"metrics": MetricRecord({
                "client_partition_id": 4,
                "num-examples": 100,
            })}))
        ]

        self.assertIs(
            expected, strategy._aggregate_train_metrics_checked(1, messages)
        )

    def test_ablation_modes_isolate_the_expected_signals(self):
        sign_flip = [candidate(cid, 1, [1.0, 0.0]) for cid in range(4)]
        sign_flip.append(candidate(4, 1, [-1.0, 0.0], malicious=1, active=1))
        scaling = [candidate(cid, 1, [1.0, 0.0]) for cid in range(4)]
        scaling.append(candidate(4, 1, [100.0, 0.0], malicious=1, active=1))

        v1 = SSDFDetector(detector_mode="v1_temporal").detect_round(sign_flip)
        reference = SSDFDetector(
            detector_mode="v2_reference_only"
        ).detect_round(sign_flip)
        norm_on_sign = SSDFDetector(
            detector_mode="v2_norm_only"
        ).detect_round(sign_flip)
        norm_on_scaling = SSDFDetector(
            detector_mode="v2_norm_only"
        ).detect_round(scaling)

        self.assertTrue(v1[-1].pass_flag)
        self.assertTrue(v1[-1].history_updated)
        self.assertFalse(reference[-1].pass_flag)
        self.assertTrue(norm_on_sign[-1].pass_flag)
        self.assertFalse(norm_on_scaling[-1].pass_flag)

    def test_unknown_detector_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            SSDFDetector(detector_mode="not_a_mode")

    def test_persistent_sign_flip_is_rejected_in_first_round(self):
        detector = SSDFDetector()
        updates = [
            candidate(0, 1, [1.0, 0.1]),
            candidate(1, 1, [0.9, 0.2]),
            candidate(2, 1, [1.1, -0.1]),
            candidate(3, 1, [1.0, 0.0]),
            candidate(4, 1, [-1.0, -0.1], malicious=1, active=1),
        ]

        results = detector.detect_round(updates)

        self.assertTrue(all(result.pass_flag for result in results[:4]))
        self.assertFalse(results[4].pass_flag)
        self.assertIn("round_direction_outlier", results[4].decision_reason)

    def test_scaling_outlier_is_rejected_by_raw_norm(self):
        detector = SSDFDetector()
        updates = [candidate(cid, 1, [1.0, 0.1]) for cid in range(4)]
        updates.append(candidate(4, 1, [100.0, 10.0], malicious=1, active=1))

        results = detector.detect_round(updates)

        self.assertFalse(results[4].pass_flag)
        self.assertGreater(results[4].norm_zscore, detector.norm_k)
        self.assertIn("raw_norm_outlier", results[4].decision_reason)

    def test_benign_non_iid_variation_is_retained(self):
        detector = SSDFDetector()
        updates = [
            candidate(0, 1, [1.0, 0.2, 0.0]),
            candidate(1, 1, [0.8, -0.1, 0.2]),
            candidate(2, 1, [1.2, 0.0, -0.1]),
            candidate(3, 1, [0.7, 0.3, 0.1]),
            candidate(4, 1, [1.1, -0.2, 0.0]),
        ]

        results = detector.detect_round(updates)

        self.assertEqual(5, len(results))
        self.assertTrue(all(result.pass_flag for result in results))

    def test_rejected_update_does_not_modify_history(self):
        detector = SSDFDetector()
        updates = [candidate(cid, 1, [1.0, 0.0]) for cid in range(4)]
        updates.append(candidate(4, 1, [-1.0, 0.0], malicious=1, active=1))

        results = detector.detect_round(updates)

        self.assertFalse(results[4].history_updated)
        self.assertIsNone(detector.history[4]["prev_g"])
        self.assertEqual([], detector.history[4]["cos"])

    def test_synchronized_benign_shift_does_not_reject_all_clients(self):
        detector = SSDFDetector(warmup_rounds=1)
        for server_round in range(1, 4):
            results = detector.detect_round(
                [candidate(cid, server_round, [1.0, 0.0]) for cid in range(5)]
            )
            self.assertTrue(all(result.pass_flag for result in results))

        shifted = detector.detect_round(
            [candidate(cid, 4, [0.0, 1.0]) for cid in range(5)]
        )

        self.assertTrue(all(result.pass_flag for result in shifted))
        self.assertTrue(all(result.reference_ok for result in shifted))

    def test_diagnostics_have_one_complete_row_per_client(self):
        detector = SSDFDetector()
        results = detector.detect_round(
            [candidate(cid, 1, [1.0, 0.1 * cid]) for cid in range(5)]
        )
        rows = [SSDFStrategy._res_to_dict(result, result.client_id) for result in results]

        self.assertEqual(5, len(rows))
        self.assertEqual({0, 1, 2, 3, 4}, {row["client_id"] for row in rows})
        required = {
            "raw_norm", "norm_zscore", "reference_cosine",
            "reference_threshold", "history_updated", "decision_reason",
            "ssdf_version",
        }
        self.assertTrue(all(required <= row.keys() for row in rows))

    def test_temporal_warmup_decisions_are_in_detection_metrics(self):
        detector = SSDFDetector()
        results = detector.detect_round(
            [
                candidate(0, 1, [1.0, 0.0]),
                candidate(1, 1, [1.0, 0.0]),
                candidate(2, 1, [-1.0, 0.0], malicious=1, active=1),
            ]
        )
        rows = [SSDFStrategy._res_to_dict(result, result.client_id) for result in results]
        self.assertTrue(all(row["warmup"] == 1 for row in rows))

        with TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory)
            recorder._write_detection({1: rows}, "persistent", 0)
            metrics = json.loads(
                (Path(directory) / "detection_metrics_active.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(3, metrics["n_detection_instances"])
        self.assertEqual(1, metrics["TP"])

    def test_active_metrics_scope_all_clients_to_attack_rounds(self):
        rows = {
            1: [
                {
                    "round": 1, "client_id": 0, "is_malicious_gt": 0,
                    "attack_active": 0, "is_malicious_pred": 0,
                    "detection_eligible": 1,
                },
                {
                    "round": 1, "client_id": 1, "is_malicious_gt": 1,
                    "attack_active": 0, "is_malicious_pred": 0,
                    "detection_eligible": 1,
                },
            ],
            4: [
                {
                    "round": 4, "client_id": 0, "is_malicious_gt": 0,
                    "attack_active": 0, "is_malicious_pred": 1,
                    "detection_eligible": 1,
                },
                {
                    "round": 4, "client_id": 1, "is_malicious_gt": 1,
                    "attack_active": 1, "is_malicious_pred": 1,
                    "detection_eligible": 1,
                },
            ],
        }
        with TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(directory)
            recorder._write_detection(rows, "persistent", 0)
            metrics = json.loads(
                (Path(directory) / "detection_metrics_active.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(2, metrics["n_detection_instances"])
        self.assertEqual(1, metrics["TP"])
        self.assertEqual(1, metrics["FP"])
        self.assertEqual(1.0, metrics["fpr"])

if __name__ == "__main__":
    unittest.main()
