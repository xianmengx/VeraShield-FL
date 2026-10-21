"""Short, real-crypto round sequences for tentative model/state commits."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord, RecordDict, MetricRecord, ConfigRecord

from pytorchexample.experiment.client_verifier import state_digest
from pytorchexample.experiment.crypto_vl import encode_vector, make_client_record
from pytorchexample.experiment.crypto_vl_keys import initialize_crypto_vl_setup
from pytorchexample.experiment.participation import candidate_digest
from pytorchexample.experiment.recorder import ExperimentRecorder
from pytorchexample.experiment.ssdf_recovery_strategy import SSDFRecoveryStrategy


@pytest.fixture
def strategy(tmp_path, monkeypatch):
    keys = tmp_path / "keys"
    initialize_crypto_vl_setup(keys, client_ids=range(20))
    s = SSDFRecoveryStrategy(
        share_n=20, share_t=16, normal_threshold=16, emergency_threshold=12,
        threshold_controller_enabled=True, ssdf_backend="logical_dual_server",
        evidence_policy="unified_v2_candidate", vl_backend="crypto", vl_key_dir=str(keys),
        vl_verification=True, vl_enforce=True, vl_commit_mode="precommit",
        vl_tamper_rounds=[2, 3, 5], seed=7,
        dynamic_weight=True,
    )
    s._current_arrays = ArrayRecord({"w": torch.zeros(8)})
    s.total_nodes = 20
    s.train_metrics_aggr_fn = lambda *args: None
    monkeypatch.setattr(s, "_check_and_log_replies", lambda replies, **kwargs: (list(replies), []))
    return s


def replies(s, r):
    state = s._current_arrays.to_torch_state_dict()
    s.candidate_sets[r] = list(range(20))
    s.records[r] = {"participating_clients": 20}
    result = []
    for cid in range(20):
        local = state["w"] + torch.linspace(.01, .08, 8) * (1 + cid / 100)
        delta = (local - state["w"]).numpy().astype(np.float64)
        record = make_client_record(
            setup=s.crypto_vl_setup, client_id=cid, task_id="vl-ssdf-uavfl",
            round_id=r, previous_model_version=state_digest(state),
            candidate_set_digest=candidate_digest(range(20)),
            encoded_update=encode_vector(delta, s.vl_update_scale),
        )
        result.append(SimpleNamespace(content=RecordDict({
            "arrays": ArrayRecord({"w": local}),
            "metrics": MetricRecord({"client_partition_id": cid, "num-examples": 1}),
            "participation": ConfigRecord({"crypto_vl_record": json.dumps(record.as_dict())}),
        })))
    return result


def mutable_state(s):
    return pickle.dumps([
        s.detector.__dict__, s.logical_backend.detector.__dict__, s.controller.__dict__
    ])


def test_normal_consecutive_tamper_recovery_and_final_round(strategy, tmp_path):
    from pytorchexample.client_app import _verify_crypto_vl_payload
    s = strategy
    for r in range(1, 6):
        inputs = replies(s, r)
        before_model = state_digest(s._current_arrays.to_torch_state_dict())
        before_state = mutable_state(s)
        arrays, _ = s.aggregate_train(r, inputs)
        row = s.vl_verification_records[r]
        if r in {2, 3, 5}:
            assert row["tamper_applied"] == 1
            assert row["precommit_verification_success"] == 0
            assert row["tamper_detection_location"] == "precommit"
            assert row["candidate_model_digest"] != before_model
            assert state_digest(arrays.to_torch_state_dict()) == before_model
            assert mutable_state(s) == before_state
            assert row["model_committed"] == row["state_committed"] == 0
        else:
            assert row["precommit_verification_success"] == 1, row
            assert row["model_committed"] == row["state_committed"] == 1
            assert state_digest(arrays.to_torch_state_dict()) != before_model
            assert mutable_state(s) != before_state
            payload = json.loads(row["proof"])
            result = _verify_crypto_vl_payload(
                proof_payload=payload, setup=s.crypto_vl_setup,
                candidate_set_digest=candidate_digest(range(20)),
                received_model_digest=state_digest(arrays.to_torch_state_dict()),
                previous_state=s._current_arrays.to_torch_state_dict(),
                received_state=arrays.to_torch_state_dict(),
                update_scale=s.vl_update_scale,
                expected_task_id="vl-ssdf-uavfl", expected_round_id=r,
            )
            assert result["verification_success"], result
            assert result["aggregate_vector_check"] == "match"
        # This is the state Flower would provide to configure_train next round.
        s._current_arrays = arrays
    recorder = ExperimentRecorder(str(tmp_path / "audit"))
    recorder._write_vl_verification(s.vl_verification_records)
    import csv
    with (tmp_path / "audit" / "vl_verification_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["model_committed"] for row in rows] == ["1", "0", "0", "1", "0"]


def test_missing_authenticated_input_discards_history(strategy):
    inputs = replies(strategy, 1)
    for msg in inputs:
        msg.content["participation"] = ConfigRecord({})
    before = mutable_state(strategy)
    old = state_digest(strategy._current_arrays.to_torch_state_dict())
    arrays, _ = strategy.aggregate_train(1, inputs)
    assert mutable_state(strategy) == before
    assert state_digest(arrays.to_torch_state_dict()) == old
    assert strategy.vl_verification_records[1]["candidate_rejected"] == 1


def test_exception_restores_tentative_state(strategy, monkeypatch):
    before = mutable_state(strategy)
    def fail(*args):
        strategy.controller._prev_mode = "test-mutated"
        strategy.logical_backend.detector._last_round = 99
        raise RuntimeError("test exception")
    monkeypatch.setattr(strategy, "_aggregate_train_candidate", fail)
    with pytest.raises(RuntimeError, match="test exception"):
        strategy.aggregate_train(1, [])
    assert mutable_state(strategy) == before


def test_legacy_default_and_precommit_validation():
    assert SSDFRecoveryStrategy().vl_commit_mode == "next_round"
    with pytest.raises(ValueError, match="enforced crypto"):
        SSDFRecoveryStrategy(vl_commit_mode="precommit")


@pytest.mark.parametrize("dropouts", [4, 8])
def test_dropout_threshold_transition_is_discarded_on_rejection(strategy, dropouts):
    s = strategy
    s.unavailable_by_round = {2: list(range(dropouts))}
    arrays, _ = s.aggregate_train(1, replies(s, 1))
    s._current_arrays = arrays
    before = mutable_state(s)
    arrays, _ = s.aggregate_train(2, replies(s, 2))
    assert s.vl_verification_records[2]["tamper_applied"] == 1
    assert s.vl_verification_records[2]["candidate_rejected"] == 1
    assert mutable_state(s) == before
    assert state_digest(arrays.to_torch_state_dict()) == state_digest(s._current_arrays.to_torch_state_dict())


def test_no_input_does_not_commit(strategy):
    before = mutable_state(strategy)
    strategy.candidate_sets[1] = list(range(20))
    arrays, _ = strategy.aggregate_train(1, [])
    assert mutable_state(strategy) == before
    assert state_digest(arrays.to_torch_state_dict()) == state_digest(strategy._current_arrays.to_torch_state_dict())
    assert strategy.vl_verification_records[1]["candidate_rejected"] == 1


def test_rejected_candidate_is_not_delivered_for_delayed_verification(strategy, monkeypatch):
    from pytorchexample.experiment.ssdf_strategy import SSDFStrategy
    s = strategy
    arrays, _ = s.aggregate_train(1, replies(s, 1))
    s._current_arrays = arrays
    s.aggregate_train(2, replies(s, 2))
    message = SimpleNamespace(content=RecordDict({"config": ConfigRecord({})}))
    monkeypatch.setattr(SSDFStrategy, "configure_train", lambda *args: [message])
    s.configure_train(3, arrays, None, None)
    assert message.content["config"]["vl-proof-skip"] == 1
    assert "vl-proof" not in message.content["config"]


def test_explicit_legacy_mode_delivers_tampered_candidate(strategy):
    s = strategy
    s.vl_commit_mode = "next_round"
    s.vl_tamper_rounds = {1}
    arrays, _ = s.aggregate_train(1, replies(s, 1))
    row = s.vl_verification_records[1]
    assert row["tamper_applied"] == 1
    assert row["verification_success"] == 1
    assert row["tamper_detection_location"] == "client_next_round"
    proof = json.loads(row["proof"])
    assert state_digest(arrays.to_torch_state_dict()) != proof["returned_model_digest"]
