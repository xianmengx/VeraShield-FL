"""State-transition and adversarial invariants for the unified candidate."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import numpy as np
import unittest

from pytorchexample.experiment.unified_evidence import UnifiedEvidenceState, direction_evidence
from pytorchexample.experiment.unified_logical_ssdf import UnifiedLogicalSSDF
from pytorchexample.experiment.ssdf_backends import LogicalDualServerSSDFBackend
from pytorchexample.experiment.dynamic_weight import calculate_mixed_dynamic_weights


def round_updates(detector, round_id, malicious=None):
    updates = [np.array([1.0, 0.01 * i, 0.0]) for i in range(8)]
    if malicious is not None:
        updates[-1] = np.asarray(malicious, dtype=float)
    return detector.detect_round([detector.make_bundle(i, round_id, x) for i, x in enumerate(updates)])


def test_first_round_attack_cannot_bypass_statistics():
    for update in [[-1.0, 0.0, 0.0], [100.0, 0.0, 0.0]]:
        detector = UnifiedLogicalSSDF(seed=7)
        result = round_updates(detector, 1, update)
        assert not result[-1].pass_flag
        assert all(x.pass_flag for x in result[:-1])
        assert 7 not in detector._history_shares
        assert result[-1].evidence_diagnostics["evidence_updated"] == 1


def test_repeated_weak_evidence_rejects_then_recovers_without_blacklist():
    state = UnifiedEvidenceState()
    decisions = [state.observe(1, 0.95, 0) for _ in range(20)]
    assert decisions[0].accepted
    assert not decisions[-1].accepted
    assert decisions[-1].previous < decisions[-1].accumulated
    normal = [state.observe(1, 0, 0) for _ in range(20)]
    assert normal[-1].accepted
    assert normal[-1].accumulated < 0.02


def test_temporal_or_cohort_tolerates_drift_but_cannot_bypass_reference():
    state = UnifiedEvidenceState()
    assert state.observe(1, 0, 0, 1, 0).accepted
    assert not state.observe(2, 0, 1, 0, 0).accepted
    assert not state.observe(3, 0, 0, 1, 1).accepted
    assert state.observe(4, 0, 0, None, None).accepted
    assert not state.observe(5, 0, 0, None, 1).accepted


def test_rejection_updates_evidence_but_freezes_history_then_requalifies():
    detector = UnifiedLogicalSSDF(seed=7)
    for r in range(1, 5):
        round_updates(detector, r)
    old = len(detector._history_shares[7])
    for r in range(5, 10):
        result = round_updates(detector, r, [100, 0, 0])
        assert not result[-1].pass_flag
    assert len(detector._history_shares[7]) == old
    assert detector.evidence.scores[7] > 0.6
    for r in range(10, 30):
        round_updates(detector, r)
    assert detector.evidence.scores[7] < 0.1
    assert len(detector._history_shares[7]) > old
    assert 7 in detector._cohort_members
    assert not hasattr(detector, "_anchor_excluded")
    assert not hasattr(detector, "_anomaly_streak")


def test_absence_freezes_score_and_requires_anchor_requalification():
    detector = UnifiedLogicalSSDF(seed=7)
    for r in range(1, 5):
        round_updates(detector, r)
    old = detector.evidence.scores[7]
    detector.detect_round([detector.make_bundle(i, 5, np.array([1., .01 * i, 0])) for i in range(7)])
    assert detector.evidence.scores[7] == old
    assert 7 not in detector._cohort_members
    result = round_updates(detector, 6)
    assert not result[7].evidence_diagnostics["anchor_admitted"]


def test_recovery_round_preserves_unqualified_anchor_and_history():
    detector = UnifiedLogicalSSDF(seed=7)
    for r in range(1, 5):
        round_updates(detector, r)
    anchor = detector.aggregation_reference()
    history = {cid: len(h) for cid, h in detector._history_shares.items()}
    # A formerly suspicious cohort returns to clean behavior. Its statistical
    # scores recover, but it remains unqualified for trusted state updates.
    detector.evidence.scores = {cid: 0.99 for cid in range(8)}
    result = detector.detect_round([
        detector.make_bundle(i, 5, np.array([1., .01 * i, 0])) for i in range(8)
    ])
    # A clean round lets the score decay below the threshold; explicit recovery
    # is expected, rather than an unconditional permanent quarantine.
    assert all(x.pass_flag for x in result)
    assert {cid: len(h) for cid, h in detector._history_shares.items()} == history
    assert np.array_equal(detector.aggregation_reference(), anchor)


def test_invalid_and_duplicate_observations_do_not_change_evidence():
    detector = UnifiedLogicalSSDF(seed=7)
    bundle = detector.make_bundle(0, 1, np.array([1., 0]))
    from dataclasses import replace
    bad = replace(bundle, vector_digest="corrupt")
    result = detector.detect_round([bad])
    assert not result[0].pass_flag
    assert not detector.evidence.scores
    with unittest.TestCase().assertRaises(ValueError):
        detector.detect_round([bundle])
    good = detector.make_bundle(0, 2, np.array([1., 0]))
    with unittest.TestCase().assertRaises(ValueError):
        detector.detect_round([good, good])


def test_confidence_reduces_entire_weight_and_weights_remain_bounded():
    weights = calculate_mixed_dynamic_weights([1, 1, 1, 1], confidence=[1, 1, 1, .05])
    assert weights[-1] < .03
    assert abs(sum(weights) - 1) < 1e-12
    assert max(weights) <= .5
    fallback = calculate_mixed_dynamic_weights([-1, -1], confidence=[1, .1])
    assert fallback[0] > fallback[1]
    with unittest.TestCase().assertRaises(ValueError):
        calculate_mixed_dynamic_weights([1], confidence=[0])


def test_candidate_backend_selects_one_state_machine():
    backend = LogicalDualServerSSDFBackend(evidence_policy="unified_v1_candidate")
    assert isinstance(backend.detector, UnifiedLogicalSSDF)
    assert not hasattr(backend.detector, "_persistent_scores")
    assert not isinstance(LogicalDualServerSSDFBackend().detector, UnifiedLogicalSSDF)


def test_absolute_floor_still_rejects_when_reference_center_falls_below_it():
    assert direction_evidence(-.5, -.4, -.25) == 1.0


def test_empty_accepted_set_cannot_advance_trusted_state():
    detector = UnifiedLogicalSSDF(seed=7, persistent_threshold=.7)
    for r in range(1, 5):
        round_updates(detector, r)
    anchor = detector.aggregation_reference()
    history = {cid: len(h) for cid, h in detector._history_shares.items()}
    detector.evidence.scores = {cid: .99 for cid in range(8)}
    results = round_updates(detector, 5)
    assert not any(x.pass_flag for x in results)
    assert all(x.evidence_diagnostics["evidence_updated"] for x in results)
    assert np.array_equal(anchor, detector.aggregation_reference())
    assert history == {cid: len(h) for cid, h in detector._history_shares.items()}


def test_new_diagnostics_replay_exact_decisions_and_active_period_fpr():
    import csv
    import importlib.util
    from pathlib import Path
    import tempfile
    script = Path(__file__).resolve().parents[1] / "scripts/replay_unified_evidence.py"
    spec = importlib.util.spec_from_file_location("unified_replay", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    detector = UnifiedLogicalSSDF(seed=7, evidence_policy="unified_v2_candidate")
    rows = []
    for r in range(1, 15):
        results = round_updates(detector, r, [-5, 0, 0] if r >= 4 else None)
        for result in results:
            rows.append(dict(round=r, client_id=result.client_id,
                             is_malicious_gt=int(result.client_id == 7),
                             attack_active=int(result.client_id == 7 and r >= 4),
                             detection_eligible=1, **{"pass": int(result.pass_flag)},
                             **result.evidence_diagnostics))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "detection_statistics.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = module.replay(path)
    assert report["mode"] == "full_frozen_feature_replay"
    assert report["decision_mismatches"] == 0
    assert report["counts"]["TN"] == 11 * 7
    assert report["counts"]["TP"] == 11
    assert report["recall"] == 1
    assert report["fpr"] == 0
    assert report["gate_pass"]


def test_v2_direction_accumulates_without_immediate_rejection_or_zero_weight():
    state = UnifiedEvidenceState(immediate_policy="norm_only")
    decisions = [state.observe(1, 0, 1, 0, 0) for _ in range(8)]
    assert decisions[0].accepted and decisions[0].confidence > 0
    assert all(x.accepted for x in decisions[:7])
    assert not decisions[7].accepted
    assert not state.observe(2, 1, 0).accepted
    assert state.observe(1, 0, 0).accepted


def test_v2_stale_temporal_baseline_expires_and_reboots():
    detector = UnifiedLogicalSSDF(seed=7, evidence_policy="unified_v2_candidate")
    for r in range(1, 6):
        round_updates(detector, r)
    for r in range(6, 10):
        detector.detect_round([detector.make_bundle(i, r, np.array([1., .01*i, 0])) for i in range(7)])
    result = round_updates(detector, 10)[7]
    assert result.evidence_diagnostics["temporal_reset_reason"] == "expired"
    assert not result.evidence_diagnostics["temporal_available"]
    assert result.history_updated
    assert len(detector._history_shares[7]) == 1
    assert not detector._temporal_observations.get(7)


def test_v2_temporal_refresh_independent_of_anchor_admission():
    detector = UnifiedLogicalSSDF(seed=7, evidence_policy="unified_v2_candidate")
    for r in range(1, 6):
        round_updates(detector, r)
    detector.evidence.scores = {cid: .9 for cid in range(8)}
    results = round_updates(detector, 6)
    assert all(x.pass_flag and x.history_updated for x in results)
    assert not any(x.evidence_diagnostics["anchor_admitted"] for x in results)
    assert detector._cohort_reference is None
    for r in range(7, 16):
        results = round_updates(detector, r)
    assert detector._cohort_reference is not None
    assert all(x.evidence_diagnostics["temporal_history_age"] == 1 for x in results)


def test_v2_shrinking_cohort_is_unavailable_and_does_not_freeze_anchor():
    detector = UnifiedLogicalSSDF(seed=7, evidence_policy="unified_v2_candidate")
    for r in range(1, 6):
        round_updates(detector, r)
    detector._cohort_members = {0, 1}
    results = round_updates(detector, 6)
    assert all(not x.evidence_diagnostics["cohort_available"] for x in results)
    assert all(x.evidence_diagnostics["cohort_required"] == 4 for x in results)
    assert detector._anchor_round == 6
    results = round_updates(detector, 7)
    assert all(x.evidence_diagnostics["cohort_available"] for x in results)


def test_v2_backend_selection_and_equal_norm_attack_delay():
    backend = LogicalDualServerSSDFBackend(evidence_policy="unified_v2_candidate")
    detector = backend.detector
    results = [round_updates(detector, r, [-1., 0, 0]) for r in range(1, 10)]
    assert results[0][-1].pass_flag
    assert not results[-1][-1].pass_flag
    assert 7 not in detector._history_shares
    assert 7 not in detector._cohort_members


def test_v2_gradual_population_drift_refreshes_references():
    detector = UnifiedLogicalSSDF(seed=7, evidence_policy="unified_v2_candidate")
    for r in range(1, 61):
        angle = .02 * r
        vectors = [np.array([np.cos(angle + .002*i), np.sin(angle + .002*i), 0]) for i in range(8)]
        results = detector.detect_round([detector.make_bundle(i, r, v) for i, v in enumerate(vectors)])
        assert all(x.pass_flag and x.history_updated for x in results)
        assert all(x.evidence_diagnostics["temporal_history_age"] == 1 for x in results) if r > 1 else True
    assert detector._anchor_round == 60
    assert all(x.evidence_diagnostics["cohort_available"] for x in results)


def test_v2_rejected_observations_cannot_refresh_expired_history():
    detector = UnifiedLogicalSSDF(seed=7, evidence_policy="unified_v2_candidate")
    for r in range(1, 6):
        round_updates(detector, r)
    for r in range(6, 11):
        result = round_updates(detector, r, [100., 0, 0])[7]
        assert not result.pass_flag and not result.history_updated
    assert 7 not in detector._history_shares
    assert 7 not in detector._history_round
