
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import numpy as np

from pytorchexample.experiment.logical_dual_server_ssdf import LogicalDualServerSSDF
from pytorchexample.experiment.ssdf_recovery_strategy import SSDFRecoveryStrategy


def test_detection_shares_reconstruct_without_exposing_plaintext_state():
    detector = LogicalDualServerSSDF(seed=7)
    bundle = detector.make_bundle(2, 1, np.array([1.0, 2.0, 3.0]))
    assert not np.array_equal(bundle.vector.c1, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(bundle.vector.reconstruct(), [1.0, 2.0, 3.0])
    assert np.allclose(bundle.norm_sq.reconstruct(), [14.0])


def test_joint_detector_accepts_consistent_cohort_and_updates_history():
    detector = LogicalDualServerSSDF(seed=7)
    bundles = [detector.make_bundle(i, 1, np.array([1.0, 0.1 * i])) for i in range(4)]
    result = detector.detect_round(bundles)
    assert len(result) == 4
    assert all(item.pass_flag for item in result)
    assert all(item.reason == "accepted" for item in result)


def test_joint_detector_rejects_direction_outlier():
    detector = LogicalDualServerSSDF(seed=7, consecutive_rejections=1)
    clean = [detector.make_bundle(i, 1, np.array([1.0, 0.0])) for i in range(4)]
    detector.detect_round(clean)
    bundles = [
        detector.make_bundle(0, 2, np.array([1.0, 0.0])),
        detector.make_bundle(1, 2, np.array([1.0, 0.0])),
        detector.make_bundle(2, 2, np.array([1.0, 0.0])),
        detector.make_bundle(3, 2, np.array([-1.0, 0.0])),
    ]
    result = detector.detect_round(bundles)
    assert result[-1].pass_flag is False
    assert "round_direction_outlier" in result[-1].reason


def test_secjudge_norm_failure_is_rejected_without_two_hit_delay():
    detector = LogicalDualServerSSDF(seed=7, consecutive_rejections=2)
    calibration = [
        detector.make_bundle(i, 1, np.array([1.0, 0.01 * i]))
        for i in range(6)
    ]
    detector.detect_round(calibration)
    bundles = [
        detector.make_bundle(i, 2, np.array([1.0, 0.01 * i]))
        for i in range(5)
    ]
    bundles.append(detector.make_bundle(5, 2, np.array([100.0, 1.0])))
    result = detector.detect_round(bundles)
    assert result[-1].norm_ok is False
    assert result[-1].pass_flag is False
    assert result[-1].dynamic_weight == 0.0
    assert "raw_norm_outlier" in result[-1].reason
    assert "outlier_pending_confirmation" not in result[-1].reason


def test_tampered_share_is_rejected():
    detector = LogicalDualServerSSDF(seed=7)
    bundle = detector.make_bundle(0, 1, np.array([1.0, 2.0]))
    tampered = type(bundle)(
        client_id=bundle.client_id, round_id=bundle.round_id,
        vector=type(bundle.vector)(bundle.vector.c1 + 1.0, bundle.vector.c2),
        norm_sq=bundle.norm_sq, vector_digest=bundle.vector_digest,
    )
    result = detector.detect_round([tampered])
    assert result[0].pass_flag is False
    assert result[0].reason == "share_integrity_failure"


def test_detection_and_aggregation_shares_are_bound():
    detector = LogicalDualServerSSDF(seed=7)
    clean = detector.make_bundle(0, 1, np.array([1.0, 2.0]))
    other = detector.make_bundle(1, 1, np.array([-1.0, 2.0]))
    decoupled = type(clean)(
        client_id=clean.client_id,
        round_id=clean.round_id,
        vector=clean.vector,
        norm_sq=clean.norm_sq,
        vector_digest=clean.vector_digest,
        aggregation_vector=other.aggregation_vector,
        binding_digest=clean.binding_digest,
    )
    result = detector.detect_round([decoupled])
    assert result[0].pass_flag is False
    assert result[0].reason in {"share_binding_failure", "detection_aggregation_mismatch"}


def test_spot_check_accepts_small_finite_training_updates():
    """Fixed-point disclosure checks must not reject honest low-norm deltas."""
    detector = LogicalDualServerSSDF(seed=7, spot_check_count=20)
    update = np.random.default_rng(11).normal(0.0, 1e-4, 62006)
    bundle = detector.make_bundle(0, 1, update)
    assert detector.spot_check(bundle)


def test_logical_recovery_strategy_does_not_create_plaintext_reference_detector():
    strategy = SSDFRecoveryStrategy(
        ssdf_backend="logical_dual_server",
        vl_backend="linear",
    )
    assert strategy.logical_backend is not None
    assert strategy.ref_detector is None


def test_leave_one_out_median_rejects_persistent_minority_without_benign_rejects():
    detector = LogicalDualServerSSDF(seed=7)
    malicious = {1, 8, 12, 13}
    rng = np.random.default_rng(7)
    malicious_rejections = 0
    benign_rejections = 0
    for round_id in range(1, 7):
        base = rng.normal(size=128)
        base /= np.linalg.norm(base)
        bundles = []
        for client_id in range(20):
            update = base + 0.03 * rng.normal(size=128)
            if client_id in malicious:
                update = -update
            bundles.append(detector.make_bundle(client_id, round_id, update))
        results = detector.detect_round(bundles)
        malicious_rejections += sum(
            not result.pass_flag and result.client_id in malicious
            for result in results
        )
        benign_rejections += sum(
            not result.pass_flag and result.client_id not in malicious
            for result in results
        )
    assert malicious_rejections >= 20
    assert benign_rejections == 0
    assert detector._cohort_members is not None
    assert not (detector._cohort_members & malicious)


def test_persistent_score_is_exposed_and_rejection_freezes_state():
    detector = LogicalDualServerSSDF(
        seed=7, consecutive_rejections=1, persistent_warmup_rounds=1
    )
    bundles = [detector.make_bundle(i, 1, np.array([1.0, 0.01 * i])) for i in range(4)]
    first = detector.detect_round(bundles)
    assert all(np.isfinite(item.persistent_score) for item in first)
    before = detector._persistent_scores[3]
    outlier = detector.make_bundle(3, 2, np.array([-1.0, 0.0]))
    clean = [detector.make_bundle(i, 2, np.array([1.0, 0.01 * i])) for i in range(3)]
    second = detector.detect_round(clean + [outlier])
    assert second[-1].pass_flag is False
    assert detector._persistent_scores[3] == before


def test_persistent_score_is_monitoring_evidence_not_a_hard_gate():
    detector = LogicalDualServerSSDF(
        seed=7, consecutive_rejections=1, persistent_warmup_rounds=1
    )
    clean = [
        detector.make_bundle(i, 1, np.array([1.0, 0.01 * i]))
        for i in range(6)
    ]
    detector.detect_round(clean)
    # Force a low historical score while leaving all current-round evidence
    # benign.  Non-IID score drift must affect risk, not reject by itself.
    detector._persistent_scores[0] = -1.0
    result = detector.detect_round([
        detector.make_bundle(i, 2, np.array([1.0, 0.01 * i]))
        for i in range(6)
    ])
    assert result[0].persistent_ok is False
    assert result[0].pass_flag is True
    assert "persistent_reference_outlier" in result[0].reason


def test_round_and_cohort_cosines_have_distinct_diagnostics():
    detector = LogicalDualServerSSDF(seed=7, cohort_confirm_rounds=1)
    first = [
        detector.make_bundle(i, 1, np.array([1.0, 0.01 * i]))
        for i in range(6)
    ]
    detector.detect_round(first)
    second = detector.detect_round([
        detector.make_bundle(i, 2, np.array([1.0, 0.02 * i]))
        for i in range(6)
    ])
    assert all(np.isfinite(item.round_reference_cosine) for item in second)
    assert all(np.isfinite(item.round_reference_threshold) for item in second)
    assert all(np.isfinite(item.cohort_cosine) for item in second)
    assert all(item.reference_cosine == item.round_reference_cosine for item in second)


def test_cohort_signal_is_separate_from_current_reference_signal():
    detector = LogicalDualServerSSDF(seed=7)
    bundles = [detector.make_bundle(i, 1, np.array([1.0, 0.01 * i])) for i in range(4)]
    result = detector.detect_round(bundles)
    assert all(hasattr(item, "cohort_ok") for item in result)
    assert all(item.reference_ok for item in result)


def test_aggregation_reference_uses_model_delta_sign_and_returns_copy():
    detector = LogicalDualServerSSDF(seed=7, cohort_confirm_rounds=1)
    updates = [np.array([1.0, 0.01 * i]) for i in range(4)]
    bundles = [detector.make_bundle(i, 1, update) for i, update in enumerate(updates)]
    detector.detect_round(bundles)
    reference = detector.aggregation_reference()
    assert reference is not None
    assert reference[0] < -0.99
    reference[:] = 0.0
    assert np.linalg.norm(detector.aggregation_reference()) > 0.99
