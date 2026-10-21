"""pytorchexample: A Flower / PyTorch app."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import time
import json
import gc

import numpy as np
import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Error,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

from pytorchexample.experiment.attack import (
    ATTACK_TYPE_CODE,
    apply_update_attack,
    select_malicious_client_ids,
    update_cosine,
    update_l2_norm,
    update_ratio,
)
from pytorchexample.experiment.utils import (
    compute_class_for_partition,
    run_simulated_workload,
    set_seed,
    simulated_workload_units,
)
from pytorchexample.experiment.client_verifier import (
    AggregationProof,
    ClientAggregationVerifier,
    state_digest,
)
from pytorchexample.experiment.crypto_vl import (
    CryptoAggregationProof,
    CryptoVLVerifier,
    decode_weighted_aggregate,
    encode_vector,
    homomorphic_hash,
    make_client_record,
)
from pytorchexample.experiment.crypto_vl_keys import (
    CryptoVLSetup,
    load_crypto_vl_setup,
)
from pytorchexample.experiment.recovery import flatten_state
from pytorchexample.experiment.participation import (
    candidate_digest,
    sign_receipt,
    sign_record,
)
from pytorchexample.experiment.participation import records_digest, verify_record
from pytorchexample.task import Net, load_data, release_federated_cache
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn
from pytorchexample.task import train_backdoor as train_backdoor_fn

# Flower ClientApp
app = ClientApp()

_VL_GLOBAL_STATE_KEY = "vl-previous-global-model"
_VL_ROUND_STATE_KEY = "vl-last-observed-round"


def _validate_unproved_model(*, previous_state, received_state) -> dict:
    """A rollback/no-op may only resend the locally retained global model."""
    reasons = []
    if previous_state is None:
        reasons.append("missing_previous_model_cache")
    elif state_digest(previous_state) != state_digest(received_state):
        reasons.append("unproved_model_change")
    return {"verification_success": not reasons, "reasons": reasons}


def _rejected_verification_reply(
    *, msg, received_state, client_id: int, server_round: int, seed: int,
    reasons: list[str], proof_payload: dict, diagnostics: dict | None = None,
) -> Message:
    observed_digest = state_digest(received_state)
    receipt_payload = {
        "client_id": int(client_id),
        # Malformed proof fields must not break the rejection path itself.
        "round_id": int(server_round - 1),
        "client_round": int(server_round),
        "verification_success": False,
        "reasons": list(reasons),
        "previous_model_digest": str(
            proof_payload.get(
                "previous_model_version", proof_payload.get("model_digest", "")
            )
        ),
        "returned_model_digest": str(
            proof_payload.get("returned_model_digest", "")
        ),
        "observed_model_digest": observed_digest,
        "proof_binding": str(proof_payload.get("binding", "")),
    }
    if diagnostics:
        receipt_payload.update(
            {k: v for k, v in diagnostics.items() if k not in receipt_payload}
        )
    receipt = sign_receipt(seed=seed, receipt=receipt_payload)
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(received_state),
                "metrics": MetricRecord(
                    {
                        "client_partition_id": int(client_id),
                        "num-examples": 0,
                        "train_loss": 0.0,
                        "vl-rejected": 1,
                    }
                ),
                "participation": ConfigRecord(
                    {"vl_receipt": json.dumps(receipt, sort_keys=True)}
                ),
            }
        ),
        reply_to=msg,
    )


def _verify_linear_vl_payload(
    *, proof_payload: dict, seed: int, previous_state: dict, received_state: dict,
    expected_task_id: str, expected_round_id: int,
) -> dict:
    participation_records = [
        dict(item) for item in proof_payload.get("participation_records", [])
    ]
    participation_ids = sorted(
        int(item.get("client_id", -1)) for item in participation_records
    )
    expected_ids = sorted(int(item) for item in proof_payload["accepted_set"])
    if (
        participation_ids != expected_ids
        or any(not verify_record(item) for item in participation_records)
        or any(
            str(item.get("candidate_set_digest", ""))
            != str(proof_payload.get("candidate_set_digest", ""))
            for item in participation_records
        )
        or records_digest(participation_records)
        != str(proof_payload.get("participation_digest", ""))
    ):
        return {"verification_success": False, "reasons": ["participation_record_mismatch"]}

    proof = AggregationProof(
        task_id=str(proof_payload["task_id"]),
        round_id=int(proof_payload["round_id"]),
        model_digest=str(proof_payload["model_digest"]),
        accepted_set=tuple(int(x) for x in proof_payload["accepted_set"]),
        quantized_weights={
            int(k): int(v) for k, v in proof_payload["quantized_weights"].items()
        },
        commitments=tuple(float(x) for x in proof_payload["commitments"]),
        aggregate_commitment=float(proof_payload["aggregate_commitment"]),
        binding=str(proof_payload["binding"]),
        participation_digest=str(proof_payload.get("participation_digest", "")),
        candidate_set_digest=str(proof_payload.get("candidate_set_digest", "")),
    )
    previous_delta = {
        key: received_state[key] - previous_state[key] for key in received_state
    }
    aggregate_delta, _ = flatten_state(previous_delta)
    verifier = ClientAggregationVerifier(
        task_id=expected_task_id,
        expected_round_id=expected_round_id,
        expected_model_digest=state_digest(previous_state),
        hash_seed=seed,
    )
    return verifier.verify(
        proof=proof,
        aggregate_update=np.asarray(aggregate_delta, dtype=np.float64),
        expected_client_ids=list(proof.accepted_set),
        participation_records=participation_records,
        expected_candidate_set_digest=str(
            proof_payload.get("candidate_set_digest", "")
        ),
    )


def _verify_crypto_vl_payload(
    *, proof_payload: dict, setup: CryptoVLSetup, candidate_set_digest: str,
    received_model_digest: str, previous_state: dict | None = None,
    received_state: dict | None = None, update_scale: int = 1 << 16,
    expected_task_id: str, expected_round_id: int,
) -> dict:
    proof = CryptoAggregationProof.from_dict(proof_payload)
    verifier = CryptoVLVerifier(
        setup=setup,
        task_id=expected_task_id,
        expected_round_id=expected_round_id,
        expected_previous_model_version=(
            state_digest(previous_state) if previous_state is not None else ""
        ),
    )
    aggregate_vector = proof.aggregate_numerator
    result = verifier.verify(
        proof=proof,
        aggregate_vector=aggregate_vector,
        expected_records=proof.client_records,
        expected_recovery_digest=proof.recovery_digest,
        received_model_digest=received_model_digest,
        expected_candidate_set_digest=candidate_set_digest,
    )
    if aggregate_vector is None:
        result["reasons"].append("missing_aggregate_numerator")
    if sum(proof.weights_dict().values()) != 1 << 16:
        result["reasons"].append("weight_scale_mismatch")
    if previous_state is None or received_state is None:
        result["reasons"].append("missing_model_state")
    else:
        if state_digest(previous_state) != proof.previous_model_version:
            result["reasons"].append("previous_model_digest_mismatch")
        if state_digest(received_state) != received_model_digest:
            result["reasons"].append("received_model_digest_mismatch")
        if aggregate_vector is not None:
            if previous_state.keys() != received_state.keys() or any(
                previous_state[key].shape != received_state[key].shape
                for key in previous_state
            ):
                result["reasons"].append("model_structure_mismatch")
            else:
                actual, _ = flatten_state({
                    key: received_state[key] - previous_state[key]
                    for key in previous_state
                })
                expected = decode_weighted_aggregate(
                    aggregate_vector, update_scale=int(update_scale),
                    weight_scale=1 << 16,
                )
                # Same fixed-point/float tolerance as the server's existing
                # model-to-numerator check. This is separate from the exact
                # integer hash relation; no hash mismatch is downgraded.
                if actual.shape != expected.shape or not np.allclose(
                    actual, expected, rtol=2e-4, atol=3.0 / int(update_scale),
                ):
                    result["reasons"].append("aggregate_model_mismatch")
    result["verification_success"] = not result["reasons"]
    result["aggregate_vector_check"] = (
        "missing" if aggregate_vector is None else
        "mismatch" if any(reason in result["reasons"] for reason in (
            "aggregate_vector_mismatch", "aggregate_dimension_mismatch",
            "malformed_aggregate_vector",
        )) else "match"
    )
    return result


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Simulated client unavailability after selection: a sampled client may return a
    # Flower Error reply instead of a training update. The decision is deterministic
    # and independent per (global_seed, server_round, client) -- this is NOT a real
    # network dropout claim, it models "client unavailability after selection".
    seed: int = int(context.run_config.get("seed", 42))
    partition_id: int = context.node_config["partition-id"]
    dropout_rate: float = float(context.run_config.get("dropout-rate", 0.0))
    dropout_timing: str = str(context.run_config.get("dropout-timing", "pre_share"))
    server_round: int = int(msg.content["config"].get("server-round", 0))
    candidate_ids = [int(x) for x in json.loads(str(msg.content["config"].get("vl-candidate-set", "[]")))]
    candidate_set_digest = candidate_digest(candidate_ids) if candidate_ids else ""
    if dropout_timing == "pre_share" and dropout_rate > 0.0:
        decision_rng = np.random.default_rng([seed, server_round, partition_id])
        if float(decision_rng.random()) < dropout_rate:
            return Message(
                error=Error(
                    code=1000,
                    reason="simulated client dropout (unavailability after selection)",
                ),
                reply_to=msg,
            )

    # Client-specific seed = global_seed + partition_id (deterministic, distinct).
    client_seed = seed + partition_id
    set_seed(client_seed)

    # Load the model and initialize it with the received weights
    model = Net(context.run_config.get("dataset", "cifar10"))
    received_state = msg.content["arrays"].to_torch_state_dict()
    proof_text = msg.content["config"].get("vl-proof", "")
    skip_proof = bool(int(msg.content["config"].get("vl-proof-skip", 0)))
    if skip_proof:
        proof_text = ""
    vl_enforce = bool(int(context.run_config.get("vl-enforce", 0)))
    previous_record = context.state.array_records.get(_VL_GLOBAL_STATE_KEY)
    previous_state = previous_record.to_torch_state_dict() if previous_record is not None else None
    round_record = context.state.config_records.get(_VL_ROUND_STATE_KEY)
    last_round = int(round_record["round"]) if round_record is not None else 0
    # Neither a skip flag nor resetting server-round to 1 authorizes a new model.
    guard = None
    if vl_enforce and (server_round < 1 or server_round <= last_round):
        guard = {"verification_success": False, "reasons": ["non_monotonic_round"]}
    elif vl_enforce and skip_proof:
        guard = _validate_unproved_model(previous_state=previous_state, received_state=received_state)
    elif vl_enforce and server_round == 1 and previous_state is not None:
        guard = {"verification_success": False, "reasons": ["unexpected_initial_round"]}
    if guard is not None and not guard["verification_success"]:
        return _rejected_verification_reply(
            msg=msg, received_state=received_state, client_id=partition_id,
            server_round=server_round, seed=seed, reasons=guard["reasons"], proof_payload={},
        )
    receipt = {
        "client_id": partition_id,
        "round": server_round,
        "verification_success": True,
        "reasons": ["not_applicable_initial_round"],
    }
    if guard is not None:
        receipt.update(guard)
        receipt["reasons"] = ["unchanged_cached_model"]
    proof_payload = {}
    if (
        vl_enforce
        and server_round > 1
        and not proof_text
        and not int(msg.content["config"].get("vl-proof-skip", 0))
    ):
        return _rejected_verification_reply(
            msg=msg,
            received_state=received_state,
            client_id=partition_id,
            server_round=server_round,
            seed=seed,
            reasons=["missing_aggregation_proof"],
            proof_payload={},
        )
    # Round 1 establishes the initial update and has no previous aggregate to
    # verify.  Proof validation starts strictly from round 2.
    if proof_text and server_round > 1:
        previous_digest = ""
        returned_digest = ""
        proof_payload = {}
        try:
            proof_payload = json.loads(str(proof_text))
            if not isinstance(proof_payload, dict):
                proof_payload = {}
                raise ValueError("proof must be an object")
            previous_record = context.state.array_records.get(_VL_GLOBAL_STATE_KEY)
            previous_state = (
                previous_record.to_torch_state_dict()
                if previous_record is not None
                else None
            )
            received_digest = state_digest(received_state)
            previous_digest = str(
                proof_payload.get(
                    "previous_model_version", proof_payload.get("model_digest", "")
                )
            )
            returned_digest = str(
                proof_payload.get("returned_model_digest", "")
            )
            is_crypto_proof = proof_payload.get("backend") == "crypto_homomorphic_hash"
            if received_digest != returned_digest and not is_crypto_proof:
                receipt = {
                    "client_id": partition_id,
                    "round": server_round,
                    "verification_success": False,
                    "reasons": ["received_model_digest_mismatch"],
                }
            elif (
                previous_state is None
                or state_digest(previous_state) != previous_digest
            ):
                receipt = {
                    "client_id": partition_id,
                    "round": server_round,
                    "verification_success": not vl_enforce,
                    "reasons": [
                        "missing_previous_model_cache"
                        if vl_enforce
                        else "client_vector_check_skipped_missing_previous_model"
                    ],
                }
            else:
                if proof_payload.get("backend") == "crypto_homomorphic_hash":
                    key_dir = str(context.run_config.get("vl-key-dir", "") or "")
                    if not key_dir:
                        raise ValueError("crypto VL requires vl-key-dir")
                    receipt = _verify_crypto_vl_payload(
                        proof_payload=proof_payload,
                        setup=load_crypto_vl_setup(key_dir),
                        candidate_set_digest=candidate_set_digest,
                        received_model_digest=received_digest,
                        previous_state=previous_state,
                        received_state=received_state,
                        update_scale=int(context.run_config.get("vl-update-scale", 1 << 16)),
                        expected_task_id=str(context.run_config.get("control-run-id", "") or "vl-ssdf-uavfl"),
                        expected_round_id=server_round - 1,
                    )
                else:
                    receipt = _verify_linear_vl_payload(
                        proof_payload=proof_payload,
                        seed=seed,
                        previous_state=previous_state,
                        received_state=received_state,
                        expected_task_id=str(context.run_config.get("control-run-id", "") or "vl-ssdf-uavfl"),
                        expected_round_id=server_round - 1,
                    )
                receipt.update({"client_id": partition_id, "round": server_round})
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
            receipt = {
                "client_id": partition_id,
                "round": server_round,
                "verification_success": False,
                "reasons": [f"malformed_proof:{exc}"],
            }
        receipt.update(
            {
                "previous_model_digest": previous_digest,
                "returned_model_digest": returned_digest,
                "proof_binding": str(proof_payload.get("binding", "")),
            }
        )
        if not receipt["verification_success"]:
            return _rejected_verification_reply(
                msg=msg,
                received_state=received_state,
                client_id=partition_id,
                server_round=server_round,
                seed=seed,
                reasons=list(receipt["reasons"]),
                proof_payload=proof_payload,
                diagnostics=receipt,
            )

    # Cache the accepted global model for the next round's aggregate-delta
    # verification.  Clone tensors so later model mutations cannot alter the
    # cached reference.
    context.state.array_records[_VL_GLOBAL_STATE_KEY] = ArrayRecord(
        {
            key: value.detach().cpu().clone()
            for key, value in received_state.items()
        }
    )
    context.state.config_records[_VL_ROUND_STATE_KEY] = ConfigRecord({"round": server_round})
    model.load_state_dict(received_state)
    global_state = {
        k: v.detach().cpu().clone() for k, v in received_state.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    partitioner = context.run_config.get("partitioner", "iid")
    alpha = float(context.run_config.get("alpha", 1.0))
    trainloader, _ = load_data(
        partition_id, num_partitions, batch_size,
        partitioner=partitioner, alpha=alpha, seed=seed,
        dataset=context.run_config.get("dataset", "cifar10"),
    )

    # Simulated compute heterogeneity: fixed UAV group (0-2 fast, 3-6 medium,
    # 7-9 slow) and a per-group simulated_compute_factor / workload size. This
    # only affects wall-clock, never model params / optimizer / gradients / data.
    compute_factors = {
        "fast": float(context.run_config.get("compute-fast-factor", 1.0)),
        "medium": float(context.run_config.get("compute-medium-factor", 1.0)),
        "slow": float(context.run_config.get("compute-slow-factor", 1.0)),
    }
    work_units = int(context.run_config.get("work-units", 0))
    compute_class = compute_class_for_partition(partition_id)
    sim_factor = compute_factors[compute_class]
    workload_units = simulated_workload_units(sim_factor, work_units)

    # Byzantine attack framework: deterministic malicious selection (fixed across
    # rounds), attack is applied at UPDATE level (never touches normal clients' or
    # unfettered models; for attack_type "none" the model is returned unchanged).
    attack_enabled = bool(int(context.run_config.get("attack-enabled", 0)))
    attack_ratio = float(context.run_config.get("attack-ratio", 0.0))
    attack_type = str(context.run_config.get("attack-type", "none"))
    selection_seed = int(context.run_config.get("malicious-selection-seed", seed))
    attack_lambda = float(context.run_config.get("attack-strength", 1.0))
    attack_sigma = float(context.run_config.get("attack-sigma", 0.05))
    attack_mode = str(context.run_config.get("attack-mode", "persistent"))
    attack_start_round = int(context.run_config.get("attack-start-round", 1))
    burst_round = int(context.run_config.get("burst-round", 0))
    population_size = int(context.run_config.get("num-clients", num_partitions))
    malicious_ids = select_malicious_client_ids(
        seed, attack_ratio, population_size, selection_seed
    )
    is_malicious = partition_id in malicious_ids
    should_attack = (
        attack_enabled
        and attack_type != "none"
        and is_malicious
        and (
            (attack_mode == "persistent" and server_round >= attack_start_round)
            or (attack_mode == "burst" and server_round == burst_round)
        )
    )
    backdoor_poison_rate = float(
        context.run_config.get("backdoor-poison-rate", 0.3)
    )
    backdoor_target_label = int(
        context.run_config.get("backdoor-target-label", 0)
    )
    backdoor_trigger_size = int(
        context.run_config.get("backdoor-trigger-size", 3)
    )

    # Call the training function (real training wall-clock; excludes workload)
    t0 = time.perf_counter()
    poisoned_examples = 0
    poison_eligible_examples = 0
    if should_attack and attack_type == "backdoor":
        train_loss, poisoned_examples, poison_eligible_examples = train_backdoor_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
            poison_rate=backdoor_poison_rate,
            target_label=backdoor_target_label,
            trigger_size=backdoor_trigger_size,
        )
    else:
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )
    client_train_time = time.perf_counter() - t0

    # Independent CPU-bound workload executed AFTER training (does not change the
    # returned model). client_compute_time = real training + simulated compute.
    t1 = time.perf_counter()
    run_simulated_workload(workload_units)
    simulation_time = time.perf_counter() - t1
    client_compute_time = client_train_time + simulation_time

    # Compute the local model update and apply attack if this client is malicious.
    local_state = {
        k: v.detach().cpu() for k, v in model.state_dict().items()
    }
    delta = {k: local_state[k] - global_state[k] for k in global_state}
    update_norm_before = update_l2_norm(delta)
    if should_attack:
        strength = attack_sigma if attack_type == "gaussian_noise" else attack_lambda
        if attack_type == "gaussian_noise":
            noise_rng = np.random.default_rng([seed, server_round, partition_id, 7])
        elif attack_type in {"random_gradient", "coordinated_gaussian"}:
            # Coordinated attackers share one random direction per round, so
            # their malicious contributions do not cancel under FedAvg.
            noise_rng = np.random.default_rng([seed, server_round, 17])
        else:
            noise_rng = None
        attacked_delta = apply_update_attack(
            attack_type, delta, strength, noise_rng=noise_rng
        )
        new_state = {k: global_state[k] + attacked_delta[k] for k in global_state}
        model.load_state_dict(new_state)
        update_norm_after = update_l2_norm(attacked_delta)
        attack_active = 1
    else:
        # No attack: the model remains exactly the local model (unchanged).
        attacked_delta = delta
        update_norm_after = update_norm_before
        attack_active = 0
    update_cosine_to_original = update_cosine(attacked_delta, delta)
    update_norm_ratio = update_ratio(update_norm_after, update_norm_before)

    # Noise-specific statistics (noise = attacked_delta - delta; zero for normal).
    noise = {k: attacked_delta[k] - delta[k] for k in delta}
    noise_flat = torch.cat([noise[k].reshape(-1) for k in noise])
    noise_l2_norm = update_l2_norm(noise)
    noise_mean = float(noise_flat.mean())
    noise_std_observed = float(noise_flat.std())
    update_noise_ratio = (
        noise_l2_norm / update_norm_before if update_norm_before > 0.0 else 0.0
    )

    # Source-side guard: invalid local updates never enter the share, SSDF, or
    # aggregation pipeline. The server repeats this check independently.
    finite_update = all(
        bool(torch.isfinite(tensor).all()) for tensor in attacked_delta.values()
    )
    if not finite_update:
        invalid_metrics = MetricRecord({
            "train_loss": 0.0,
            "client_train_time": client_train_time,
            "client_compute_time": client_compute_time,
            "client_partition_id": partition_id,
            "num-examples": 0,
            "is_malicious": int(is_malicious),
            "attack_active": int(attack_active),
            "attack_type_code": ATTACK_TYPE_CODE[attack_type],
            "attack_strength": (
                attack_sigma if attack_type == "gaussian_noise" else attack_lambda
            ),
            "update_norm_before": 0.0,
            "update_norm_after": 0.0,
            "update_cosine_to_original": 0.0,
            "update_norm_ratio": 0.0,
            "noise_l2_norm": 0.0,
            "noise_mean": 0.0,
            "noise_std_observed": 0.0,
            "update_noise_ratio": 0.0,
            "backdoor_poisoned_examples": poisoned_examples,
            "backdoor_poison_rate_observed": 0.0,
            "invalid_update": 1,
        })
        invalid_record = ArrayRecord({k: v.detach().cpu() for k, v in received_state.items()})
        return Message(
            content=RecordDict({"arrays": invalid_record, "metrics": invalid_metrics}),
            reply_to=msg,
        )

    # Construct and return reply Message
    model_record = ArrayRecord(
        {k: v.detach().cpu() for k, v in model.state_dict().items()}
    )
    participation = sign_record(
        seed=seed,
        round_id=server_round,
        client_id=partition_id,
        model_digest=state_digest(local_state),
        candidate_set_digest=candidate_set_digest,
    )
    crypto_vl_record = ""
    if (
        bool(int(context.run_config.get("vl-verification", 0)))
        and str(context.run_config.get("vl-backend", "linear")).lower() == "crypto"
    ):
        key_dir = str(context.run_config.get("vl-key-dir", "") or "")
        if not key_dir:
            raise ValueError("crypto VL requires vl-key-dir")
        crypto_setup = load_crypto_vl_setup(key_dir)
        # Bind the commitment to the exact model payload the server will
        # subtract from its current global model.  Hashing attacked_delta
        # directly can differ after float32(global + delta) - global due to
        # cancellation rounding, causing honest client commitments to fail.
        outbound_state = model_record.to_torch_state_dict()
        outbound_delta = {
            key: outbound_state[key] - global_state[key]
            for key in global_state
        }
        outbound_flat, _ = flatten_state(outbound_delta)
        crypto_record = make_client_record(
            setup=crypto_setup,
            client_id=partition_id,
            task_id=str(context.run_config.get("control-run-id", "") or "vl-ssdf-uavfl"),
            round_id=server_round,
            previous_model_version=state_digest(global_state),
            candidate_set_digest=candidate_set_digest,
            encoded_update=encode_vector(
                outbound_flat,
                int(context.run_config.get("vl-update-scale", 1 << 16)),
            ),
        )
        crypto_vl_record = json.dumps(crypto_record.as_dict(), sort_keys=True)
    metrics = {
        "train_loss": train_loss,
        "client_train_time": client_train_time,
        "client_compute_time": client_compute_time,
        "num-examples": len(trainloader.dataset),
        "client_partition_id": partition_id,
        "is_malicious": int(is_malicious),
        "attack_active": int(attack_active),
        "attack_type_code": ATTACK_TYPE_CODE[attack_type],
        "attack_strength": (
            attack_sigma if attack_type == "gaussian_noise" else attack_lambda
        ),
        "update_norm_before": update_norm_before,
        "update_norm_after": update_norm_after,
        "update_cosine_to_original": update_cosine_to_original,
        "update_norm_ratio": update_norm_ratio,
        "noise_l2_norm": noise_l2_norm,
        "noise_mean": noise_mean,
        "noise_std_observed": noise_std_observed,
        "update_noise_ratio": update_noise_ratio,
        "backdoor_poisoned_examples": poisoned_examples,
        "backdoor_poison_rate_observed": (
            poisoned_examples / poison_eligible_examples
            if poison_eligible_examples > 0 else 0.0
        ),
        "invalid_update": 0,
    }
    participation_record = ConfigRecord(
        {
            "participation_record": json.dumps(participation, sort_keys=True),
            "crypto_vl_record": crypto_vl_record,
            "vl_receipt": json.dumps(
                sign_receipt(
                    seed=seed,
                    receipt={
                        **receipt,
                        "round_id": int(
                            proof_payload.get("round_id", server_round - 1)
                        ) if proof_text else server_round,
                        "client_round": server_round,
                    },
                ),
                sort_keys=True,
            ),
        }
    )
    metric_record = MetricRecord(metrics)
    content = RecordDict(
        {
            "arrays": model_record,
            "metrics": metric_record,
            "participation": participation_record,
        }
    )
    # Client actors persist across rounds. Drop the dataset/model references
    # after serializing the reply so memory does not accumulate in Ray actors.
    del trainloader, model, global_state, local_state, delta, attacked_delta, noise
    gc.collect()
    release_federated_cache()
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net(context.run_config.get("dataset", "cifar10"))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    partitioner = context.run_config.get("partitioner", "iid")
    alpha = float(context.run_config.get("alpha", 1.0))
    seed = int(context.run_config.get("seed", 42))
    _, valloader = load_data(
        partition_id, num_partitions, batch_size,
        partitioner=partitioner, alpha=alpha, seed=seed,
        dataset=context.run_config.get("dataset", "cifar10"),
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    del valloader, model
    gc.collect()
    release_federated_cache()
    return Message(content=content, reply_to=msg)
