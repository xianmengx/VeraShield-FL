"""pytorchexample: A Flower / PyTorch app."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import time
import json
import os
import traceback

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from pytorchexample.experiment.recorded_fedavg import RecordingFedAvg
from pytorchexample.experiment.recorder import ExperimentRecorder
from pytorchexample.experiment.ssdf_strategy import SSDFStrategy
from pytorchexample.experiment.ssdf_recovery_strategy import SSDFRecoveryStrategy
from pytorchexample.experiment.aggregation_strategy import AggregationStrategy
from pytorchexample.experiment.attack import select_malicious_client_ids
from pytorchexample.experiment.utils import env_versions, set_seed
from pytorchexample.task import Net, load_centralized_dataset, test, test_backdoor

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    lr_schedule = str(context.run_config.get("lr-schedule", "cosine")).lower()
    lr_min = float(context.run_config.get("lr-min", 0.001))
    if lr_schedule not in {"constant", "cosine"}:
        raise ValueError("lr-schedule must be 'constant' or 'cosine'")
    if lr <= 0 or lr_min < 0 or lr_min > lr:
        raise ValueError("learning rates must satisfy 0 <= lr-min <= learning-rate")
    fraction_train: float = float(context.run_config.get("fraction-train", 1.0))
    smoke_min_nodes = int(
        os.environ.get("FLOWER_GPU_SMOKE_MIN_NODES", "2")
    )
    seed: int = int(context.run_config.get("seed", 42))
    output_dir: str = context.run_config.get("output-dir", "results/baseline")
    # Flower's Windows subprocesses can lose the WSL UNC share during the
    # final writer phase; resolve the output path once and let the recorder
    # fail loudly if the destination is not reachable.
    ssdf_enabled: bool = bool(int(context.run_config.get("ssdf-enabled", 0)))
    attack_enabled: bool = bool(int(context.run_config.get("attack-enabled", 0)))
    attack_type: str = str(context.run_config.get("attack-type", "none"))
    attack_ratio: float = float(context.run_config.get("attack-ratio", 0.0))
    attack_mode: str = str(context.run_config.get("attack-mode", "persistent"))
    burst_round: int = int(context.run_config.get("burst-round", 0))
    dropout_timing: str = str(context.run_config.get("dropout-timing", "pre_share"))
    dropout_rate: float = float(context.run_config.get("dropout-rate", 0.0))
    share_n: int = int(context.run_config.get("share-n", 10))
    share_t: int = int(context.run_config.get("share-t", 6))
    ssdf_v2 = dict(
        detector_mode=str(
            context.run_config.get("ssdf-detector-mode", "v2_full")
        ),
        reference_k=float(context.run_config.get("ssdf-reference-k", 2.5)),
        reference_floor=float(context.run_config.get("ssdf-reference-floor", -0.25)),
        norm_k=float(context.run_config.get("ssdf-norm-k", 2.70)),
        history_limit=int(context.run_config.get("ssdf-history-limit", 20)),
        ssdf_backend=str(context.run_config.get("ssdf-backend", "plaintext")),
        evidence_policy=str(context.run_config.get("ssdf-evidence-policy", "legacy")),
    )
    # The manifest is the single source of truth for scenario identity and client IDs.
    _manifest_path = str(context.run_config.get("scenario-manifest", "") or "")
    scenario_manifest = None
    if _manifest_path:
        try:
            with open(_manifest_path, "r", encoding="utf-8") as _f:
                scenario_manifest = json.load(_f)
        except Exception as _e:
            print("WARNING: could not load scenario manifest:", _e)
    num_clients = int((scenario_manifest or {}).get("num_clients", context.run_config.get("num-clients", 10)))
    if scenario_manifest and "malicious_ids" in scenario_manifest:
        malicious_ids = [int(x) for x in scenario_manifest["malicious_ids"]]
    else:
        malicious_ids = select_malicious_client_ids(
            seed, attack_ratio, num_clients, int(context.run_config.get("malicious-selection-seed", seed))
        )
    outage: str = str(context.run_config.get("holder-outage", "full"))
    threshold_controller_enabled: bool = bool(
        int(context.run_config.get("threshold-controller-enabled", 0))
    )
    normal_threshold: int = int(context.run_config.get("normal-threshold", 8))
    emergency_threshold: int = int(context.run_config.get("emergency-threshold", 6))
    static_threshold: int | None = (
        int(context.run_config["static-threshold"])
        if context.run_config.get("static-threshold", 0)
        else None
    )
    dynamic_weight: bool = bool(int(context.run_config.get("dynamic-weight", 0)))
    vl_verification: bool = bool(int(context.run_config.get("vl-verification", 0)))
    vl_enforce: bool = bool(int(context.run_config.get("vl-enforce", 0)))
    if vl_enforce and not vl_verification:
        raise ValueError("vl-enforce requires vl-verification=1")
    vl_backend = str(context.run_config.get("vl-backend", "linear")).lower()
    if vl_backend not in {"linear", "crypto"}:
        raise ValueError("vl-backend must be 'linear' or 'crypto'")
    vl_update_scale = int(context.run_config.get("vl-update-scale", 1 << 16))
    if vl_update_scale <= 0:
        raise ValueError("vl-update-scale must be positive")
    vl_key_dir = str(context.run_config.get("vl-key-dir", "") or "")
    if vl_backend == "crypto" and not vl_key_dir:
        raise ValueError("vl-backend=crypto requires vl-key-dir")
    ssdf_backend = str(context.run_config.get("ssdf-backend", "plaintext")).lower()
    if ssdf_backend not in {"plaintext", "logical_dual_server"}:
        raise ValueError("ssdf-backend must be plaintext or logical_dual_server")
    vl_tamper: bool = bool(int(context.run_config.get("vl-tamper", 0)))
    _tamper_rounds = str(context.run_config.get("vl-tamper-rounds", "") or "")
    vl_tamper_rounds = [int(x) for x in _tamper_rounds.split(",") if x.strip()]
    _client_tamper_rounds = str(
        context.run_config.get("vl-client-tamper-rounds", "") or ""
    )
    vl_client_tamper_rounds = [
        int(x) for x in _client_tamper_rounds.split(",") if x.strip()
    ]
    control_run_id = str(context.run_config.get("control-run-id", "") or "")
    paired_run_id = str(context.run_config.get("paired-run-id", "") or "")
    aggregation_method: str = str(context.run_config.get("aggregation-method", "fedavg"))
    krum_f: int = int(context.run_config.get("krum-f", 2))
    trim_ratio: float = float(context.run_config.get("trim-ratio", 0.2))
    # Scenario manifest: per-round unavailable clients drive dropout for ALL schemes.
    unavailable_by_round: dict[int, list[int]] = {}
    if scenario_manifest:
        for r in scenario_manifest.get("per_round", []):
            unavailable_by_round[int(r["round"])] = [int(x) for x in r.get("unavailable_ids", [])]
    _sched = str(context.run_config.get("runtime-available-schedule", "") or "")
    runtime_available_schedule = (
        [int(x) for x in _sched.split(",") if str(x).strip() != ""]
        if _sched.strip()
        else None
    )
    if dropout_timing == "post_share" and dropout_rate > 0.0:
        if scenario_manifest:
            post_share_dropout_ids = sorted(
                unavailable_by_round.get(1, [])
            )
        else:
            n_drop = min(num_clients, int(round(dropout_rate * num_clients)))
            _ids = np.random.default_rng([seed, 99]).choice(
                num_clients, size=n_drop, replace=False
            )
            post_share_dropout_ids = sorted(int(x) for x in _ids)
    else:
        post_share_dropout_ids = []

    # Deterministic seeding for the server-side model init / evaluation.
    set_seed(seed)

    # Load global model
    global_model = Net(context.run_config.get("dataset", "cifar10"))
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy (observational wrapper; aggregation unchanged)
    common = dict(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=smoke_min_nodes,
        min_evaluate_nodes=smoke_min_nodes,
        min_available_nodes=smoke_min_nodes,
        seed=seed,
        base_learning_rate=lr,
        min_learning_rate=lr_min,
        learning_rate_schedule=lr_schedule,
        total_rounds=num_rounds,
    )
    # Crypto-VL needs the recovery strategy even with pre-share dropout: it
    # validates client rejection receipts, excludes those replies from model
    # aggregation, and skips proofs for rolled-back candidates.  SSDF-only
    # pre-share runs retain the lean SSDFStrategy path.
    use_recovery_strategy = (
        (vl_verification and vl_backend == "crypto")
        or (ssdf_enabled and dropout_timing == "post_share")
        or (threshold_controller_enabled and dropout_timing == "post_share")
    )
    if use_recovery_strategy:
        strategy = SSDFRecoveryStrategy(
            **common,
            **ssdf_v2,
            malicious_client_ids=malicious_ids,
            ema_beta=float(context.run_config.get("ema-beta", 0.9)),
            k=float(context.run_config.get("threshold-k", 2.0)),
            norm_tol=float(context.run_config.get("norm-tol", 1e-3)),
            warmup_rounds=int(context.run_config.get("warmup-rounds", 2)),
            attack_mode=attack_mode,
            burst_round=burst_round,
            share_n=share_n,
            share_t=share_t,
            outage=outage,
            dropout_timing=dropout_timing,
            post_share_dropout_ids=set(post_share_dropout_ids),
            normal_threshold=normal_threshold,
            emergency_threshold=emergency_threshold,
            threshold_controller_enabled=threshold_controller_enabled,
            runtime_available_schedule=runtime_available_schedule,
            static_threshold=static_threshold,
            dynamic_weight=dynamic_weight,
            dynamic_weight_blend=float(
                context.run_config.get("dynamic-weight-blend", 0.25)
            ),
            vl_verification=vl_verification,
            vl_enforce=vl_enforce,
            vl_commit_mode=str(context.run_config.get("vl-commit-mode", "next_round")),
            vl_backend=vl_backend,
            vl_update_scale=vl_update_scale,
            vl_key_dir=vl_key_dir,
            vl_tamper=vl_tamper,
            vl_tamper_rounds=vl_tamper_rounds,
            vl_client_tamper_rounds=vl_client_tamper_rounds,
            control_run_id=control_run_id,
            paired_run_id=paired_run_id,
            unavailable_by_round=unavailable_by_round,
            ssdf_filter_enabled=ssdf_enabled,
        )
    elif ssdf_enabled:
        strategy = SSDFStrategy(
            **common,
            **ssdf_v2,
            malicious_client_ids=malicious_ids,
            ema_beta=float(context.run_config.get("ema-beta", 0.9)),
            k=float(context.run_config.get("threshold-k", 2.0)),
            norm_tol=float(context.run_config.get("norm-tol", 1e-3)),
            warmup_rounds=int(context.run_config.get("warmup-rounds", 2)),
            attack_mode=attack_mode,
            burst_round=burst_round,
            unavailable_by_round=unavailable_by_round,
            dynamic_weight=dynamic_weight,
        )
    else:
        strategy = AggregationStrategy(
            **common,
            aggregation_method=aggregation_method,
            krum_f=krum_f,
            trim_ratio=trim_ratio,
            unavailable_by_round=unavailable_by_round,
            vl_tamper=vl_tamper,
            vl_tamper_rounds=vl_tamper_rounds,
        )

    # Start strategy, run FedAvg for `num_rounds`
    t0 = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=lambda server_round, model_arrays: global_evaluate(
            server_round,
            model_arrays,
            dataset=context.run_config.get("dataset", "cifar10"),
            backdoor_enabled=attack_enabled and attack_type == "backdoor",
            backdoor_target_label=int(
                context.run_config.get("backdoor-target-label", 0)
            ),
            backdoor_trigger_size=int(
                context.run_config.get("backdoor-trigger-size", 3)
            ),
        ),
    )
    elapsed = time.perf_counter() - t0
    strategy.finalize(elapsed)

    # Persist config.json / results.csv / summary.json
    recorder = ExperimentRecorder(output_dir, seed=seed)
    detection_records = getattr(strategy, "detection_records", None)
    recovery_records = getattr(strategy, "recovery_records", None)
    try:
        recorded_run_config = dict(context.run_config)
        recorded_run_config.pop("vl-key-dir", None)
        crypto_setup = getattr(strategy, "crypto_vl_setup", None)
        if crypto_setup is not None:
            recorded_run_config["vl-setup-id"] = crypto_setup.setup_id
        recorder.write(
            result, strategy.records, recorded_run_config, env_versions(), elapsed,
            detection_records=detection_records,
            malicious_client_ids=malicious_ids,
            attack_mode=attack_mode,
            burst_round=burst_round,
            attack_enabled=attack_enabled,
            attack_ratio=attack_ratio,
            dropout_timing=dropout_timing,
            share_n=share_n,
            share_t=share_t,
            holder_outage=outage,
            post_share_dropout_ids=post_share_dropout_ids,
            recovery_records=recovery_records,
            runtime_threshold_records=getattr(strategy, "runtime_threshold_records", None),
            dynamic_weight_records=getattr(strategy, "dynamic_weight_records", None),
            dw_timing_records=getattr(strategy, "dw_timing_records", None),
            vl_verification_records=getattr(strategy, "vl_verification_records", None),
            aggregation_records=getattr(strategy, "aggregation_records", None),
            protocol_trace_records=getattr(strategy, "protocol_trace_records", None),
        )
    except Exception:
        traceback.print_exc()
        raise

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    *,
    dataset: str = "cifar10",
    backdoor_enabled: bool = False,
    backdoor_target_label: int = 0,
    backdoor_trigger_size: int = 3,
) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net(dataset)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset(dataset)

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    metrics = {"accuracy": test_acc, "loss": test_loss}
    if backdoor_enabled:
        metrics["backdoor_asr"] = test_backdoor(
            model,
            test_dataloader,
            device,
            target_label=backdoor_target_label,
            trigger_size=backdoor_trigger_size,
        )
    return MetricRecord(metrics)
