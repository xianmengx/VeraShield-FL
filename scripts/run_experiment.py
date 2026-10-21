"""Unified experiment runner for the Flower Quickstart project (Step 1).

This is a thin orchestrator: it ensures the correct environment (virtualenv
executables, offline dataset, no user-site Python), runs ``flwr run`` for a single
experiment, and verifies the result files were produced.

The actual algorithm / recording logic lives in the Flower ServerApp (via the
``experiment`` package); this script only sets up and invokes it.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _app_dir() -> Path:
    return _script_dir().parent


def _repo_root() -> Path:
    return _app_dir().parent


def _venv_scripts() -> Path:
    root = _repo_root()
    env = os.environ.get("FLOWER_VENV", "")
    if not env:
        raise RuntimeError("FLOWER_VENV_NOT_SET")
    scripts = Path(env)
    if scripts.name.lower() not in ("scripts", "bin"):
        scripts = scripts / ("Scripts" if os.name == "nt" else "bin")
    return scripts.resolve()


def build_flower_argv(
    flwr: Path,
    run_config: str,
    num_supernodes: int,
    app_path: Path | None = None,
    allow_explicit_three_rounds: bool = False,
) -> list[str]:
    """Build argv without shell interpolation; config strings stay one token."""
    if num_supernodes < 1:
        raise ValueError("num_supernodes must be positive")
    # Reject only the exact default-round token; substring matching would
    # incorrectly reject valid values such as ``num-server-rounds=30``.
    if (
        "num-server-rounds=3" in run_config.split()
        and not allow_explicit_three_rounds
    ):
        raise ValueError("run_config contains forbidden default rounds=3")
    client_gpus = os.environ.get("FLOWER_CLIENT_RESOURCES_NUM_GPUS", "0.0")
    init_gpus = os.environ.get("FLOWER_INIT_ARGS_NUM_GPUS", "0")
    client_cpus = os.environ.get("FLOWER_CLIENT_RESOURCES_NUM_CPUS", "2")
    return [
        str(flwr), "run", str(app_path or Path(".")), "--stream",
        "--run-config", run_config,
        "--federation-config",
        f"num-supernodes={num_supernodes} client-resources-num-cpus={client_cpus} "
        f"client-resources-num-gpus={client_gpus} init-args-num-gpus={init_gpus}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Flower experiment (Step 1).")
    parser.add_argument("--preset", default="baseline", help="Experiment preset name.")
    parser.add_argument("--dataset", choices=("cifar10", "mnist"), default="cifar10")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--rounds", type=int, default=60, help="Number of FL rounds.")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs per client round.")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Initial client SGD learning rate.")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine",
                        help="Per-round client learning-rate schedule.")
    parser.add_argument("--lr-min", type=float, default=0.001,
                        help="Final learning rate for cosine scheduling.")
    parser.add_argument(
        "--clients", type=int, default=10, help="Expected number of clients (informational)."
    )
    parser.add_argument(
        "--fraction-train",
        type=float,
        default=1.0,
        help="Fraction of clients sampled for training each round.",
    )
    parser.add_argument(
        "--experiment",
        default="baseline",
        help="Experiment name recorded in config.json.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.0,
        help="Per-sampled-client dropout probability (simulated client unavailability).",
    )
    parser.add_argument(
        "--partitioner",
        default="iid",
        help="Data partitioner: 'iid' or 'dirichlet'.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Dirichlet concentration alpha (used when partitioner='dirichlet').",
    )
    parser.add_argument(
        "--compute-fast-factor", type=float, default=1.0,
        help="Simulated compute factor for fast UAVs.",
    )
    parser.add_argument(
        "--compute-medium-factor", type=float, default=1.0,
        help="Simulated compute factor for medium UAVs.",
    )
    parser.add_argument(
        "--compute-slow-factor", type=float, default=1.0,
        help="Simulated compute factor for slow UAVs.",
    )
    parser.add_argument(
        "--work-units", type=int, default=0,
        help="Dummy CPU workload units per (1/factor - 1).",
    )
    parser.add_argument(
        "--attack-enabled", type=int, default=0,
        help="1 to enable the Byzantine attack; 0 to run attack_type=none / no-op.",
    )
    parser.add_argument("--attack-type", default="none",
                        help="'none', 'sign_flip', 'scaling', 'gaussian_noise', 'random_gradient', 'coordinated_gaussian', or 'backdoor'.")
    parser.add_argument("--attack-ratio", type=float, default=0.0,
                        help="Fraction of clients that are malicious.")
    parser.add_argument("--attack-strength", type=float, default=1.0,
                        help="Attack strength (lambda for sign_flip/scaling).")
    parser.add_argument("--attack-sigma", type=float, default=0.05,
                        help="Noise std for the 'noise' attack.")
    parser.add_argument("--backdoor-poison-rate", type=float, default=0.3,
                        help="Fraction of eligible malicious-client samples receiving the trigger.")
    parser.add_argument("--backdoor-target-label", type=int, default=0,
                        help="Target class for the source-agnostic backdoor attack.")
    parser.add_argument("--backdoor-trigger-size", type=int, default=3,
                        help="Bottom-right square trigger size in pixels.")
    parser.add_argument("--malicious-selection-seed", type=int, default=42,
                        help="Seed for deterministic malicious client selection.")
    parser.add_argument("--attack-mode", default="persistent",
                        help="'persistent' (attack every round) or 'burst' (attack only at burst-round).")
    parser.add_argument("--attack-start-round", type=int, default=1,
                        help="First round in which a persistent attack is active (1-based).")
    parser.add_argument("--burst-round", type=int, default=3,
                        help="Round at which the 'burst' attack is applied.")
    parser.add_argument("--ssdf-enabled", type=int, default=0,
                        help="1 to enable SSDF detection; 0 to run plain FedAvg.")
    parser.add_argument(
        "--ssdf-backend", choices=("plaintext", "logical_dual_server"),
        default="plaintext",
        help="SSDF backend; logical_dual_server is an isolated validation prototype.",
    )
    parser.add_argument(
        "--ssdf-detector-mode", default="v2_full",
        choices=("v1_temporal", "v2_reference_only", "v2_norm_only", "v2_full"),
        help="SSDF baseline/ablation mode.",
    )
    parser.add_argument("--ssdf-reference-k", type=float, default=2.5)
    parser.add_argument("--ssdf-reference-floor", type=float, default=-0.25)
    parser.add_argument("--ssdf-norm-k", type=float, default=2.70)
    parser.add_argument(
        "--ssdf-evidence-policy", default="legacy",
        choices=("legacy", "unified_v1_candidate", "unified_v2_candidate"),
        help="Unified attack-independent evidence candidate (logical backend only).",
    )
    parser.add_argument("--ssdf-history-limit", type=int, default=20)
    parser.add_argument("--ema-beta", type=float, default=0.9, help="SSDF EMA beta.")
    parser.add_argument("--threshold-k", type=float, default=2.0, help="SSDF threshold k.")
    parser.add_argument("--warmup-rounds", type=int, default=2, help="SSDF warm-up rounds.")
    parser.add_argument("--norm-tol", type=float, default=1e-3, help="SecJudge norm tolerance.")
    parser.add_argument("--dropout-timing", default="pre_share",
                        help="'pre_share' (Error reply before share gen) or 'post_share' (logical server recovery).")
    parser.add_argument("--share-n", type=int, default=10, help="Shamir n.")
    parser.add_argument("--share-t", type=int, default=6, help="Shamir threshold t.")
    parser.add_argument("--holder-outage", default="full",
                        help="Holder outage scenario e.g. 'full','AS1_down','AS2_down','relay_down','AS1_AS2_down'.")
    parser.add_argument("--threshold-controller-enabled", type=int, default=0,
                        help="1 to enable the runtime Dual-Shamir threshold controller.")
    parser.add_argument(
        "--secret-recovery-enabled", type=int, default=0,
        help=("Enable the complete logical dual-server Shamir recovery profile. "
              "This selects SSDF, post-share recovery, and dynamic A/B thresholds."),
    )
    parser.add_argument("--normal-threshold", type=int, default=8,
                        help="Normal (A-share set) reconstruction threshold.")
    parser.add_argument("--emergency-threshold", type=int, default=6,
                        help="Emergency (B-share set) reconstruction threshold.")
    parser.add_argument("--runtime-available-schedule", default="",
                        help="Optional comma-separated per-round available-share counts "
                             "(e.g. '10,7,10') to move between normal/emergency in one run.")
    parser.add_argument("--static-threshold", type=int, default=0,
                        help="If >0, force the runtime controller into a fixed threshold "
                             "(Static t=8 baseline); 0 = Dynamic D2.")
    parser.add_argument("--dynamic-weight", type=int, default=0,
                        help="1 to use SSDF + Dynamic Weight aggregation; 0 = FedAvg.")
    parser.add_argument("--dynamic-weight-blend", type=float, default=0.25,
                        help="Adaptive anchor-weight fraction; the remainder is uniform.")
    parser.add_argument("--vl-verification", type=int, default=0,
                        help="1 to enable the VL lightweight aggregation verification.")
    parser.add_argument("--vl-enforce", type=int, default=0,
                        help="1 to fail closed and rollback the candidate model when VL verification fails.")
    parser.add_argument("--vl-commit-mode", choices=("next_round", "precommit"), default="next_round",
                        help="Legacy next-round detection or verify candidate before committing model and detector state.")
    parser.add_argument("--vl-backend", choices=("linear", "crypto"), default="linear",
                        help="VL proof backend. 'linear' preserves historical runs; 'crypto' enables the authenticated homomorphic hash.")
    parser.add_argument("--vl-update-scale", type=int, default=1 << 16,
                        help="Fixed-point update scale used by the cryptographic VL backend.")
    parser.add_argument("--vl-key-dir", default="",
                        help="TA-generated persistent key directory (required for crypto VL).")
    parser.add_argument("--vl-tamper", type=int, default=0,
                        help="1 to simulate server aggregation tampering (detection demo).")
    parser.add_argument("--vl-tamper-rounds", default="",
                        help="Comma-separated verification-only tamper rounds; empty preserves vl-tamper behavior.")
    parser.add_argument("--vl-client-tamper-rounds", default="",
                        help="Comma-separated train rounds whose outbound model is tampered after server verification.")
    parser.add_argument("--control-run-id", default="",
                        help="Stable identifier for an isolated verification control run.")
    parser.add_argument("--paired-run-id", default="",
                        help="Identifier linking paired normal/tampered control runs.")
    parser.add_argument("--aggregation-method", default="fedavg",
                        help="Baseline aggregation: 'fedavg'|'krum'|'trimmed_mean'|'feddmc'.")
    parser.add_argument("--krum-f", type=int, default=2,
                        help="Krum assumed Byzantine count f (from config, never ground truth).")
    parser.add_argument("--trim-ratio", type=float, default=0.2,
                        help="Trimmed Mean trim ratio (fraction of per-coordinate values removed).")
    parser.add_argument("--scenario-id", default="",
                        help="Scenario id recorded for cross-run reproducibility (from manifest).")
    parser.add_argument("--scenario-manifest", default="",
                        help="Path to a scenario manifest JSON whose per_round.unavailable_ids "
                             "drive dropout for ALL schemes (apples-to-apples).")
    parser.add_argument(
        "--output-dir",
        default="results/baseline",
        help="Output directory (relative to the app dir unless absolute).",
    )
    parser.add_argument("--output-root", default="", help="Contract root; resolves root/scenario_id/method.")
    parser.add_argument("--method", default="", help="Contract method directory name.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved Flower argv and exit.")
    args = parser.parse_args()
    if args.secret_recovery_enabled:
        # Normalize the protocol components required by the complete recovery
        # profile, while preserving explicitly selected threshold parameters.
        # Older runs used the 8/6 defaults; dropout studies may explicitly use
        # the 16/12 profile and must not be silently rewritten here.
        args.ssdf_enabled = 1
        args.ssdf_backend = "logical_dual_server"
        args.dropout_timing = "post_share"
        args.threshold_controller_enabled = 1
        provided = set()
        for token in sys.argv[1:]:
            if token == "--normal-threshold" or token.startswith("--normal-threshold="):
                provided.add("normal")
            if token == "--emergency-threshold" or token.startswith("--emergency-threshold="):
                provided.add("emergency")
        if "normal" not in provided:
            args.normal_threshold = 8
        if "emergency" not in provided:
            args.emergency_threshold = 6
        if args.emergency_threshold > args.normal_threshold:
            parser.error("emergency-threshold must be <= normal-threshold")
        if args.normal_threshold > args.share_n:
            parser.error("normal-threshold must be <= share-n")
    rounds_were_explicit = any(
        arg == "--rounds" or arg.startswith("--rounds=")
        for arg in sys.argv[1:]
    )
    if args.vl_enforce and not args.vl_verification:
        parser.error("--vl-enforce requires --vl-verification 1")
    if args.vl_commit_mode == "precommit" and not (
        args.vl_enforce and args.vl_verification and args.vl_backend == "crypto"
    ):
        parser.error("--vl-commit-mode precommit requires enforced crypto VL")
    if not 0.0 <= args.dynamic_weight_blend <= 1.0:
        parser.error("--dynamic-weight-blend must be in [0, 1]")
    if args.vl_update_scale <= 0:
        parser.error("--vl-update-scale must be positive")
    if args.learning_rate <= 0 or args.lr_min < 0 or args.lr_min > args.learning_rate:
        parser.error("learning rates must satisfy 0 <= --lr-min <= --learning-rate")
    if not 0.0 <= args.backdoor_poison_rate <= 1.0:
        parser.error("--backdoor-poison-rate must be in [0, 1]")
    if not 0 <= args.backdoor_target_label < 10:
        parser.error("--backdoor-target-label must be in [0, 9]")
    if args.backdoor_trigger_size <= 0:
        parser.error("--backdoor-trigger-size must be positive")
    if args.vl_backend == "crypto" and not args.vl_key_dir:
        parser.error("--vl-backend crypto requires --vl-key-dir")
    if args.vl_client_tamper_rounds and not (
        args.vl_enforce and args.vl_verification
    ):
        parser.error(
            "--vl-client-tamper-rounds requires --vl-enforce 1 and --vl-verification 1"
        )

    app_dir = _app_dir()
    try:
        venv_scripts = _venv_scripts()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    py_name = "python.exe" if os.name == "nt" else "python"
    flwr_name = "flwr.exe" if os.name == "nt" else "flwr"
    venv_python = venv_scripts / py_name
    flwr = venv_scripts / flwr_name

    # --- Environment: virtualenv first on PATH, no user-site, offline dataset ---
    os.environ["PATH"] = str(venv_scripts) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # --- Pre-flight environment check ------------------------------------------
    print("== env_check ==", flush=True)
    check = subprocess.run([str(venv_python), str(_script_dir() / "env_check.py")])
    if check.returncode != 0:
        print("env_check FAILED: expected Flower executables in", venv_scripts, file=sys.stderr)
        return check.returncode

    # --- Resolve output dir (absolute) and run/federation config ----------------
    from output_contract import OutputContractError, resolve_output_path, validate_output_dir
    try:
        if args.output_root:
            if not args.method:
                raise OutputContractError("INVALID_OUTPUT_ROOT: --method is required with --output-root")
            out_dir = resolve_output_path(args.output_root, args.scenario_id, args.method)
        else:
            out_dir = Path(args.output_dir)
            if not out_dir.is_absolute():
                out_dir = app_dir / out_dir
            out_dir = validate_output_dir(out_dir, args.scenario_id, args.method)
    except OutputContractError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # Values must follow TOML: ints/bools bare, strings double-quoted.
    run_config = (
        f"num-server-rounds={args.rounds} num-clients={args.clients} "
        f'dataset="{args.dataset}" '
        f"local-epochs={args.local_epochs} "
        f"learning-rate={args.learning_rate} "
        f'lr-schedule="{args.lr_schedule}" '
        f"lr-min={args.lr_min} "
        f"seed={args.seed} "
        f"fraction-train={args.fraction_train} "
        f"dropout-rate={args.dropout_rate} "
        f'partitioner="{args.partitioner}" '
        f"alpha={args.alpha} "
        f"compute-fast-factor={args.compute_fast_factor} "
        f"compute-medium-factor={args.compute_medium_factor} "
        f"compute-slow-factor={args.compute_slow_factor} "
        f"work-units={args.work_units} "
        f"attack-enabled={args.attack_enabled} "
        f'attack-type="{args.attack_type}" '
        f"attack-ratio={args.attack_ratio} "
        f"attack-strength={args.attack_strength} "
        f"attack-sigma={args.attack_sigma} "
        f"backdoor-poison-rate={args.backdoor_poison_rate} "
        f"backdoor-target-label={args.backdoor_target_label} "
        f"backdoor-trigger-size={args.backdoor_trigger_size} "
        f"malicious-selection-seed={args.malicious_selection_seed} "
        f'attack-mode="{args.attack_mode}" '
        f"attack-start-round={args.attack_start_round} "
        f"burst-round={args.burst_round} "
        f"ssdf-enabled={args.ssdf_enabled} "
        f'ssdf-backend="{args.ssdf_backend}" '
        f'ssdf-detector-mode="{args.ssdf_detector_mode}" '
        f"ssdf-reference-k={args.ssdf_reference_k} "
        f"ssdf-reference-floor={args.ssdf_reference_floor} "
        f"ssdf-norm-k={args.ssdf_norm_k} "
        f'ssdf-evidence-policy="{args.ssdf_evidence_policy}" '
        f"ssdf-history-limit={args.ssdf_history_limit} "
        f"ema-beta={args.ema_beta} "
        f"threshold-k={args.threshold_k} "
        f"warmup-rounds={args.warmup_rounds} "
        f"norm-tol={args.norm_tol} "
        f'dropout-timing="{args.dropout_timing}" '
        f"share-n={args.share_n} "
        f"share-t={args.share_t} "
        f'holder-outage="{args.holder_outage}" '
        f"threshold-controller-enabled={args.threshold_controller_enabled} "
        f"secret-recovery-enabled={args.secret_recovery_enabled} "
        f"normal-threshold={args.normal_threshold} "
        f"emergency-threshold={args.emergency_threshold} "
        f'runtime-available-schedule="{args.runtime_available_schedule}" '
        f"static-threshold={args.static_threshold} "
        f"dynamic-weight={args.dynamic_weight} "
        f"dynamic-weight-blend={args.dynamic_weight_blend} "
        f"vl-verification={args.vl_verification} "
        f"vl-enforce={args.vl_enforce} "
        f'vl-commit-mode="{args.vl_commit_mode}" '
        f'vl-backend="{args.vl_backend}" '
        f"vl-update-scale={args.vl_update_scale} "
        f"vl-tamper={args.vl_tamper} "
        f'vl-tamper-rounds="{args.vl_tamper_rounds}" '
        f'vl-client-tamper-rounds="{args.vl_client_tamper_rounds}" '
        f'control-run-id="{args.control_run_id}" '
        f'paired-run-id="{args.paired_run_id}" '
        f'aggregation-method="{args.aggregation_method}" '
        f"krum-f={args.krum_f} "
        f"trim-ratio={args.trim_ratio} "
        f'scenario-id="{args.scenario_id}" '
        f'output-dir="{out_dir.as_posix()}"' 
        f' experiment="{args.experiment}"'
    )
    if args.vl_key_dir:
        key_dir = Path(args.vl_key_dir).expanduser().resolve()
        if not key_dir.is_dir():
            parser.error(f"--vl-key-dir does not exist: {key_dir}")
        run_config += f' vl-key-dir="{key_dir.as_posix()}"'
    if args.scenario_manifest:
        mp = Path(args.scenario_manifest)
        if not mp.is_absolute():
            mp = app_dir / mp
        mp = mp.resolve()
        run_config += f' scenario-manifest="{mp.as_posix()}"'
        print(f"== scenario manifest == {mp}", flush=True)
    print(f"== output dir == {out_dir}", flush=True)

    # Build the FAB from a small staging tree. The workspace also contains large
    # cached wheels and historical result files; Flower's FAB collector traverses
    # the app tree before applying ignore rules, so handing it the workspace root
    # can block for minutes before StartRun is sent.
    stage_dir = Path(tempfile.mkdtemp(prefix="quickstart-flower-app-"))
    try:
        shutil.copy2(app_dir / "pyproject.toml", stage_dir / "pyproject.toml")
        shutil.copy2(app_dir / "LICENSE", stage_dir / "LICENSE")
        shutil.copytree(app_dir / "pytorchexample", stage_dir / "pytorchexample")
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    # num-supernodes controls the number of simulated clients (virtual SuperNodes).
    # This changes the client count WITHOUT touching any algorithm code.
    cmd = build_flower_argv(
        flwr,
        run_config,
        args.clients,
        stage_dir,
        allow_explicit_three_rounds=(args.rounds == 3 and rounds_were_explicit),
    )
    print("== flwr command ==")
    print("  " + " ".join(cmd), flush=True)
    if args.dry_run:
        print(json.dumps({
            "command_construction": "PASS",
            "argv": cmd,
            "resolved_rounds": args.rounds,
            "resolved_num_supernodes": args.clients,
            "run_config_is_single_argv": cmd[cmd.index("--run-config") + 1].startswith("num-server-rounds="),
            "training_started": False,
        }, indent=2))
        return 0
    try:
        proc = subprocess.run(cmd, cwd=str(app_dir))
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    # Persist the exact scenario input beside successful run outputs. This makes
    # the output contract self-contained and prevents a completed directory from
    # being admitted again merely because its external manifest was not copied.
    core_outputs = ("config.json", "results.csv", "summary.json")
    if proc.returncode == 0 and args.scenario_manifest and all(
        (out_dir / name).is_file() for name in core_outputs
    ):
        scenario_copy = out_dir / "scenario.json"
        shutil.copy2(mp, scenario_copy)
        scenario_hash = hashlib.sha256(scenario_copy.read_bytes()).hexdigest()
        (out_dir / "scenario.sha256").write_text(
            f"{scenario_hash}  scenario.json\n", encoding="ascii"
        )

    # --- Verify outputs ----------------------------------------------------------
    required = ["config.json", "results.csv", "summary.json"]
    if args.scenario_manifest:
        required.extend(["scenario.json", "scenario.sha256"])
    ok = True
    for name in required:
        p = out_dir / name
        if not p.exists():
            print(f"MISSING result file: {p}", file=sys.stderr)
            ok = False
    if ok:
        print("== result files ==")
        for name in required:
            print(f"  {out_dir / name}")
        try:
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            print(
                "final_accuracy=%.6f final_loss=%.6f total_time=%.2fs"
                % (summary["final_accuracy"], summary["final_loss"], summary["total_time"])
            )
        except Exception as exc:  # pragma: no cover
            print("could not read summary.json:", exc, file=sys.stderr)

    return proc.returncode if ok else (proc.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
