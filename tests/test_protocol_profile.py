
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
from pytorchexample.experiment.protocol_profile import (
    STAGES,
    build_protocol_profile,
    validate_protocol_profile,
)


def test_profile_records_end_to_end_stage_order_and_excludes_ta():
    profile = build_protocol_profile(
        {
            "ssdf-enabled": 1,
            "dropout-timing": "post_share",
            "dynamic-weight": 1,
            "vl-verification": 1,
        }
    )
    assert tuple(profile["stages"]) == STAGES
    assert profile["trust_model"]["ta_in_experiment"] is False
    assert profile["trust_model"]["ta_per_round_cost_measured"] is False
    assert profile["enabled"]["ssdf_screening"] is True
    assert profile["enabled"]["recovery_and_aggregation"] is True
    assert validate_protocol_profile(profile) == []


def test_profile_does_not_claim_cryptographic_security():
    profile = build_protocol_profile({})
    assert profile["implementation_boundary"]["cryptographic_proof"] is False
    assert profile["implementation_boundary"]["malicious_server_security"] is False
    assert "cryptographic_proof_overclaim" not in validate_protocol_profile(profile)


def test_profile_declares_logical_two_server_boundary():
    profile = build_protocol_profile({})
    trust = profile["trust_model"]
    assert trust["server_topology"] == "logical_two_server"
    assert trust["server_names"] == ["C1", "C2"]
    assert trust["independent_server_processes"] is False
    assert trust["server_to_server_wire_measurement"] is False


def test_profile_rejects_wrong_stage_order():
    profile = build_protocol_profile({})
    profile["stages"] = list(reversed(STAGES))
    assert "stage_order_mismatch" in validate_protocol_profile(profile)


def test_crypto_vl_profile_records_qualified_security_boundary():
    profile = build_protocol_profile(
        {
            "ssdf-enabled": 1,
            "dropout-timing": "post_share",
            "dynamic-weight": 1,
            "vl-verification": 1,
            "vl-backend": "crypto",
        }
    )
    boundary = profile["implementation_boundary"]
    assert boundary["verification_backend"] == "crypto"
    assert boundary["cryptographic_proof"] is True
    assert boundary["cryptographic_key_lifecycle"] == (
        "ta_generated_persistent_setup_id_bound"
    )
    assert profile["trust_model"]["ta"] == "offline_persistent_key_initialization"
    assert boundary["malicious_server_security"].startswith("requires_non_collusion")
    assert validate_protocol_profile(profile) == []
