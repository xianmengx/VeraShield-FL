"""Executable protocol profile for the VL-SSDF-UAVFL prototype.

The profile keeps the implementation aligned with the end-to-end design note
without pretending that the prototype deploys TA or two independent server
processes.  It is intentionally metadata and validation only.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

from typing import Any, Mapping


PROTOCOL_NAME = "VL-SSDF-UAVFL"
PROTOCOL_VERSION = "prototype-flow-v2"

STAGES = (
    "model_broadcast",
    "local_training",
    "update_preprocessing",
    "share_provisioning",
    "ssdf_screening",
    "dynamic_weighting",
    "recovery_and_aggregation",
    "transcript_diagnostic",
    "global_model_update",
)


def build_protocol_profile(run_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the implemented protocol boundary for a recorded run."""

    ssdf = bool(int(run_config.get("ssdf-enabled", 0)))
    recovery = str(run_config.get("dropout-timing", "pre_share")) == "post_share"
    weighting = bool(int(run_config.get("dynamic-weight", 0)))
    vl = bool(int(run_config.get("vl-verification", 0)))
    vl_backend = str(run_config.get("vl-backend", "linear")).lower()
    ssdf_backend = str(run_config.get("ssdf-backend", "plaintext")).lower()
    crypto_vl = vl and vl_backend == "crypto"
    return {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "stages": list(STAGES),
        "enabled": {
            "update_preprocessing": ssdf,
            "ssdf_screening": ssdf,
            "share_provisioning": recovery,
            "recovery_and_aggregation": recovery,
            "dynamic_weighting": weighting,
            "transcript_diagnostic": vl,
        },
        "trust_model": {
            "ta": (
                "offline_persistent_key_initialization"
                if crypto_vl else "offline_logical_initialization_only"
            ),
            "ta_in_experiment": False,
            "ta_per_round_cost_measured": False,
            "aggregation_servers": "logical_roles_in_single_process",
            "server_topology": "logical_two_server",
            "server_names": ["C1", "C2"],
            "independent_server_processes": False,
            "server_to_server_wire_measurement": False,
            "independent_server_deployment": False,
        },
        "implementation_boundary": {
            "ssdf": (
                str(run_config.get("ssdf-detector-mode", "v2_full"))
                + ("-logical-dual-server-additive-share" if ssdf_backend == "logical_dual_server"
                   else "-plaintext-core")
            ),
            "ssdf_backend": ssdf_backend,
            "ssdf_backend_boundary": (
                "logical_two_server_sequential_simulation; no secure MPC"
                if ssdf_backend == "logical_dual_server"
                else "single_process_plaintext_detector"
            ),
            "recovery": "logical_fixed_point_shamir_update_reconstruction",
            "verification": (
                "authenticated_homomorphic_hash_client_verification"
                if crypto_vl
                else "client_visible_scalar_transcript_consistency_diagnostic"
            ),
            "verification_backend": vl_backend,
            "cryptographic_proof": crypto_vl,
            "cryptographic_key_lifecycle": (
                "ta_generated_persistent_setup_id_bound"
                if crypto_vl else False
            ),
            "malicious_server_security": (
                "requires_non_collusion_and_at_least_one_honest_logical_server"
                if crypto_vl else False
            ),
            "candidate_set_completeness": (
                "AS1_AS2_attested_expected_manifest" if crypto_vl else False
            ),
        },
    }


def validate_protocol_profile(profile: Mapping[str, Any]) -> list[str]:
    """Validate safety-critical declarations before they reach result files."""

    errors: list[str] = []
    if tuple(profile.get("stages", ())) != STAGES:
        errors.append("stage_order_mismatch")
    trust = profile.get("trust_model", {})
    if trust.get("ta_in_experiment") is not False:
        errors.append("ta_must_be_excluded_from_experiment")
    boundary = profile.get("implementation_boundary", {})
    crypto_declared = boundary.get("cryptographic_proof") is True
    if crypto_declared and boundary.get("verification_backend") != "crypto":
        errors.append("cryptographic_backend_mismatch")
    if boundary.get("malicious_server_security") is True:
        errors.append("malicious_server_overclaim")
    return errors
