"""Client checks the exact committed numerator, not a rounded model delta."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import copy
import json

import pytest
import torch

from pytorchexample.client_app import _verify_crypto_vl_payload
from pytorchexample.experiment.client_verifier import state_digest
from pytorchexample.experiment.crypto_vl import (
    decode_weighted_aggregate, encode_vector, make_aggregation_proof,
    make_client_record, weighted_aggregate,
)
from pytorchexample.experiment.crypto_vl_keys import make_test_crypto_vl_setup


@pytest.fixture
def case():
    setup = make_test_crypto_vl_setup(42, (1, 2))
    previous = {"w": torch.tensor([0.3, -0.7, 0.25], dtype=torch.float32)}
    updates = [encode_vector(x, 65536) for x in (
        [0.123456, -0.234567, 0.000009], [-0.087654, 0.098765, -0.000012],
    )]
    weights = {1: 21845, 2: 43691}
    numerator = weighted_aggregate(updates, list(weights.values()))
    decoded = decode_weighted_aggregate(numerator, update_scale=65536, weight_scale=65536)
    received = {"w": previous["w"] + torch.tensor(decoded, dtype=torch.float32)}
    records = [make_client_record(
        setup=setup, client_id=i, task_id="test", round_id=1,
        previous_model_version=state_digest(previous),
        candidate_set_digest="candidates", encoded_update=z,
    ) for i, z in zip((1, 2), updates)]

    def proof_for(model):
        return make_aggregation_proof(
            setup=setup, task_id="test", round_id=1,
            previous_model_version=state_digest(previous), records=records,
            quantized_weights=weights, aggregate_vector=numerator,
            recovery_digest="recovery", returned_model_digest=state_digest(model),
        )

    return setup, previous, received, proof_for


def verify(case, payload=None, received=None, previous=None):
    setup, old, new, proof_for = case
    model = new if received is None else received
    return _verify_crypto_vl_payload(
        proof_payload=proof_for(new).as_dict() if payload is None else payload,
        setup=setup, candidate_set_digest="candidates",
        received_model_digest=state_digest(model),
        previous_state=old if previous is None else previous,
        received_state=model, update_scale=65536,
        expected_task_id="test", expected_round_id=1,
    )


def test_exact_numerator_survives_json_and_float32_model(case):
    _, old, new, proof_for = case
    proof = proof_for(new)
    # This is the old, incorrect client reconstruction: it cannot equal Y.
    encoded_delta = encode_vector((new["w"] - old["w"]).numpy(), 65536)
    assert encoded_delta != proof.aggregate_numerator
    result = verify(case, payload=json.loads(json.dumps(proof.as_dict())))
    assert result["verification_success"], result
    assert result["reasons"] == []
    assert result["aggregate_vector_check"] == "match"


def test_missing_numerator_fails_closed(case):
    payload = case[3](case[2]).as_dict()
    payload.pop("aggregate_numerator")
    result = verify(case, payload=payload)
    assert not result["verification_success"]
    assert "missing_aggregate_numerator" in result["reasons"]


def test_changed_numerator_is_not_downgraded_to_diagnostic(case):
    payload = case[3](case[2]).as_dict()
    payload["aggregate_numerator"][0] = str(int(payload["aggregate_numerator"][0]) + 1)
    result = verify(case, payload=payload)
    assert not result["verification_success"]
    assert "aggregate_vector_mismatch" in result["reasons"]
    assert result["aggregate_vector_check"] == "mismatch"


def test_model_replacement_is_rejected(case):
    model = {"w": case[2]["w"] + 0.1}
    result = verify(case, received=model)
    assert not result["verification_success"]
    assert "returned_model_digest_mismatch" in result["reasons"]


def test_resigned_wrong_model_still_fails_decode_relation(case):
    model = {"w": case[2]["w"] + 0.1}
    result = verify(case, payload=case[3](model).as_dict(), received=model)
    assert "returned_model_digest_mismatch" not in result["reasons"]
    assert "aggregate_model_mismatch" in result["reasons"]
    assert not result["verification_success"]


def test_incorrect_cached_parent_is_rejected(case):
    result = verify(case, previous={"w": case[1]["w"] + 0.5})
    assert "previous_model_digest_mismatch" in result["reasons"]
    assert not result["verification_success"]


@pytest.mark.parametrize("value", [1.5, True, "not-an-integer"])
def test_malformed_numerator_is_rejected(case, value):
    payload = copy.deepcopy(case[3](case[2]).as_dict())
    payload["aggregate_numerator"][0] = value
    with pytest.raises(ValueError):
        verify(case, payload=payload)


@pytest.mark.parametrize("task,round_id,reason", [
    ("other-task", 1, "task_id_mismatch"),
    ("test", 2, "round_id_mismatch"),
])
def test_valid_proof_cannot_choose_its_expected_context(case, task, round_id, reason):
    setup, old, new, proof_for = case
    result = _verify_crypto_vl_payload(
        proof_payload=proof_for(new).as_dict(), setup=setup,
        candidate_set_digest="candidates", received_model_digest=state_digest(new),
        previous_state=old, received_state=new,
        expected_task_id=task, expected_round_id=round_id,
    )
    assert not result["verification_success"]
    assert reason in result["reasons"]


def client_call(monkeypatch, case, *, payload=None, skip=False, changed=False, round_id=2, accepted=False):
    from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
    from pytorchexample import client_app as client
    setup, old, new, proof_for = case
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", old["w"].clone())
    monkeypatch.setattr(client, "Net", lambda *args: Model())
    monkeypatch.setattr(client, "load_crypto_vl_setup", lambda *args: setup)
    monkeypatch.setattr(client, "candidate_digest", lambda ids: "candidates")
    context = Context(
        run_id=1, node_id=1, node_config={"partition-id": 0, "num-partitions": 1},
        state=RecordDict({
            client._VL_GLOBAL_STATE_KEY: ArrayRecord(old),
            client._VL_ROUND_STATE_KEY: ConfigRecord({"round": 1}),
        }),
        run_config={"vl-enforce": 1, "vl-key-dir": "test", "control-run-id": "test", "batch-size": 1},
    )
    model = new if not skip or changed else old
    msg = Message(content=RecordDict({
        "arrays": ArrayRecord(model),
        "config": ConfigRecord({
            "server-round": round_id, "vl-proof-skip": int(skip),
            "vl-candidate-set": "[1,2]",
            "vl-proof": json.dumps(proof_for(new).as_dict() if payload is None else payload),
        }),
    }), dst_node_id=1, message_type="train")
    if accepted:
        class ReachedTraining(Exception):
            pass
        def stop_at_data(*args, **kwargs):
            raise ReachedTraining
        monkeypatch.setattr(client, "load_data", stop_at_data)
        with pytest.raises(ReachedTraining):
            client.train(msg, context)
        assert state_digest(context.state.array_records[client._VL_GLOBAL_STATE_KEY].to_torch_state_dict()) == state_digest(model)
        assert context.state.config_records[client._VL_ROUND_STATE_KEY]["round"] == round_id
        return
    reply = client.train(msg, context)
    receipt = json.loads(reply.content["participation"]["vl_receipt"])
    assert not receipt["verification_success"]
    # Rejection must leave both cached model and local round untouched.
    assert state_digest(context.state.array_records[client._VL_GLOBAL_STATE_KEY].to_torch_state_dict()) == state_digest(old)
    assert context.state.config_records[client._VL_ROUND_STATE_KEY]["round"] == 1
    return receipt


def test_skip_flag_cannot_authorize_changed_model(monkeypatch, case):
    receipt = client_call(monkeypatch, case, skip=True, changed=True)
    assert "unproved_model_change" in receipt["reasons"]


@pytest.mark.parametrize("skip", [False, True])
def test_normal_proof_and_unchanged_rollback_reach_training(monkeypatch, case, skip):
    client_call(monkeypatch, case, skip=skip, accepted=True)


def test_rollback_resend_requires_exact_local_model(case):
    from pytorchexample.client_app import _validate_unproved_model
    old = case[1]
    assert _validate_unproved_model(previous_state=old, received_state=copy.deepcopy(old))["verification_success"]
    assert not _validate_unproved_model(previous_state=None, received_state=old)["verification_success"]


@pytest.mark.parametrize("round_id", [0, 1])
def test_round_reset_cannot_bypass_verification(monkeypatch, case, round_id):
    receipt = client_call(monkeypatch, case, skip=True, changed=True, round_id=round_id)
    assert "non_monotonic_round" in receipt["reasons"]


@pytest.mark.parametrize("payload", [[], {"backend": "crypto_homomorphic_hash"}])
def test_malformed_proof_is_structured_rejection(monkeypatch, case, payload):
    if isinstance(payload, dict):
        payload = {**payload, "previous_model_version": state_digest(case[1])}
    receipt = client_call(monkeypatch, case, payload=payload)
    assert any(reason.startswith("malformed_proof:") for reason in receipt["reasons"])


def test_invalid_round_field_does_not_break_rejection(monkeypatch, case):
    payload = case[3](case[2]).as_dict()
    payload["round_id"] = "invalid"
    receipt = client_call(monkeypatch, case, payload=payload)
    assert any(reason.startswith("malformed_proof:") for reason in receipt["reasons"])
