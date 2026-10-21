"""Audit and plot the standalone CIFAR-10 server-tampering experiment."""
from __future__ import annotations

import os
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = _ReleasePath(_release_sys.modules["os"].environ.get("VERASHIELD_FIGURE_ROOT", str(_RELEASE_ROOT / "figures"))) / "server_tamper_events"
RUNS = {
    7: ROOT / "results/server_tamper_precommit_seed7_20261015/verashield_fl",
    21: ROOT / "results/server_tamper_precommit_multiseed_20261015/seed21/verashield_fl",
    42: ROOT / "results/server_tamper_precommit_multiseed_20261015/seed42/verashield_fl",
}
TAMPER_ROUNDS = (20, 40, 59)
SEED_STYLE = {
    7: ("#3373A5", "o", "-"),
    21: ("#C78317", "s", "--"),
    42: ("#26856B", "^", "-."),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def audit() -> tuple[dict[int, list[float]], list[dict], list[dict]]:
    curves: dict[int, list[float]] = {}
    events: list[dict] = []
    manifest: list[dict] = []
    normalized_configs = []

    for seed, folder in RUNS.items():
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        curve_rows = read_csv(folder / "results.csv")
        verification_rows = read_csv(folder / "vl_verification_results.csv")
        assert config["seed"] == seed
        assert config["dataset"].lower().startswith("cifar-10") or config["dataset"].lower() == "cifar10"
        assert config["partitioner"] == "iid"
        assert config["vl_commit_mode"] == "precommit"
        assert config["num_clients"] == 20 and config["num_rounds"] == 60
        assert config["attack_enabled"] == 0 and config["dropout_rate"] == 0
        assert [int(row["round"]) for row in curve_rows] == list(range(61))
        assert [int(row["round"]) for row in verification_rows] == list(range(1, 61))

        run_config = dict(config["run_config"])
        for key in ("seed", "experiment", "output-dir"):
            run_config.pop(key, None)
        normalized_configs.append(run_config)

        accuracy = [100 * float(row["accuracy"]) for row in curve_rows]
        assert all(math.isfinite(value) and 0 <= value <= 100 for value in accuracy[1:])
        assert math.isclose(accuracy[-1] / 100, summary["final_accuracy"], abs_tol=1e-12)
        curves[seed] = accuracy

        vr = {int(row["round"]): row for row in verification_rows}
        assert [r for r, row in vr.items() if int(row["tamper_applied"])] == list(TAMPER_ROUNDS)
        assert [r for r, row in vr.items() if int(row["candidate_rejected"])] == list(TAMPER_ROUNDS)
        for round_id, record in vr.items():
            event = round_id in TAMPER_ROUNDS
            assert record["commit_mode"] == "precommit"
            committed = 0 if event else 1
            assert int(record["model_committed"]) == committed
            assert int(record["state_committed"]) == committed
            assert int(record["precommit_verification_success"]) == committed
            assert int(record["candidate_rejected"]) == int(event)
            if event:
                assert record["tamper_detection_location"] == "precommit"
                assert record["candidate_model_digest"] != record["committed_model_digest"]
                assert record["committed_model_digest"] == vr[round_id - 1]["committed_model_digest"]
                assert math.isclose(accuracy[round_id - 1], accuracy[round_id], abs_tol=1e-10)
            if round_id not in TAMPER_ROUNDS:
                assert int(record["tamper_applied"]) == 0

        assert sum(int(row["candidate_rejected"]) for row in verification_rows) == 3
        assert sum(int(row["model_committed"]) for row in verification_rows) == 57
        assert sum(int(row["state_committed"]) for row in verification_rows) == 57
        assert sum(int(row["tamper_applied"]) for row in verification_rows) == 3

        for round_id in TAMPER_ROUNDS:
            row = vr[round_id]
            events.append({
                "seed": seed,
                "tamper_round": round_id,
                "candidate_rejected_precommit": int(row["candidate_rejected"]),
                "precommit_verification_success": int(row["precommit_verification_success"]),
                "model_committed": int(row["model_committed"]),
                "state_committed": int(row["state_committed"]),
                "candidate_digest_matches_committed": int(row["candidate_model_digest"] == row["committed_model_digest"]),
                "accuracy_previous_round_pct": accuracy[round_id - 1],
                "accuracy_rejection_round_pct": accuracy[round_id],
                "accuracy_retained": int(math.isclose(accuracy[round_id - 1], accuracy[round_id], abs_tol=1e-10)),
                "detection_location": row["tamper_detection_location"],
            })

        for name in ("config.json", "results.csv", "vl_verification_results.csv", "summary.json"):
            manifest.append({
                "seed": seed,
                "source": (folder / name).relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest(),
            })

    assert all(config == normalized_configs[0] for config in normalized_configs[1:])
    assert len(events) == 9
    return curves, events, manifest


def draw(curves: dict[int, list[float]], events: list[dict]) -> list:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.16, 3.55))
    grid = fig.add_gridspec(2, 1, height_ratios=(2.25, 1), hspace=0.42)
    ax = fig.add_subplot(grid[0])
    outcome_ax = fig.add_subplot(grid[1])

    for seed, accuracy in curves.items():
        color, marker, linestyle = SEED_STYLE[seed]
        ax.plot(
            range(61), accuracy,
            label=f"Seed {seed}", color=color, linestyle=linestyle,
            linewidth=1.05, marker=marker, markersize=2.8,
            markerfacecolor="white", markeredgewidth=0.7, markevery=6,
        )
        ax.scatter(
            TAMPER_ROUNDS, [accuracy[r] for r in TAMPER_ROUNDS],
            marker=marker, s=30, facecolor="white", edgecolor=color,
            linewidth=1.0, zorder=5,
        )

    for round_id in TAMPER_ROUNDS:
        ax.axvline(round_id, color="#9B3A42", linewidth=0.8,
                   linestyle=(0, (3, 2)), alpha=0.8, zorder=0)
    ax.set(xlim=(0, 60), ylim=(0, 60), ylabel="Test accuracy (%)")
    ax.set_xticks(range(0, 61, 10))
    ax.set_title("(a) Committed-model accuracy and tampering rounds", pad=4)
    ax.grid(color="#DEDEDE", linewidth=0.4, linestyle=":")
    ax.tick_params(direction="in", length=2.5, width=0.65, top=True, right=True)
    ax.legend(ncol=3, loc="upper left", frameon=False,
              handlelength=2.1, columnspacing=1.2, handletextpad=0.5)
    ax.text(20, 57, "Tamper", ha="center", va="top", fontsize=6.5, color="#7F2730")
    ax.text(40, 57, "Tamper", ha="center", va="top", fontsize=6.5, color="#7F2730")
    ax.text(59, 57, "Tamper", ha="right", va="top", fontsize=6.5, color="#7F2730")

    outcome_ax.set_xlim(-0.5, 3.5)
    outcome_ax.set_ylim(-0.65, 2.6)
    outcome_ax.set_xticks([0.7, 1.7, 2.7], ["Round 20", "Round 40", "Round 59"])
    outcome_ax.set_yticks([2, 1, 0], ["Seed 7", "Seed 21", "Seed 42"])
    outcome_ax.tick_params(axis="both", length=0, labelsize=7)
    outcome_ax.set_title("(b) Authenticated precommit outcomes", loc="left", pad=3)
    outcome_ax.set_frame_on(False)
    outcome_ax.grid(False)
    for row_index, seed in enumerate((7, 21, 42)):
        y = 2 - row_index
        for col_index, round_id in enumerate(TAMPER_ROUNDS):
            record = next(item for item in events if item["seed"] == seed and item["tamper_round"] == round_id)
            assert record["candidate_rejected_precommit"] == 1
            outcome_ax.add_patch(plt.Rectangle((col_index + 0.16, y - 0.32), 1.08, 0.64,
                                               facecolor="#E5F1EB", edgecolor="#9ABBA8", linewidth=0.55))
            outcome_ax.text(col_index + 0.7, y + 0.07, "Rejected", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="#235B43")
            outcome_ax.text(col_index + 0.7, y - 0.13, "no commit", ha="center", va="center",
                            fontsize=6.3, color="#36443C")

    fig.subplots_adjust(left=0.10, right=0.985, top=0.94, bottom=0.10)
    from release_visual_qa import audit_layout
    issues = audit_layout(fig)
    assert not any(level in ("FAIL", "WARN") for level, _ in issues), issues

    stem = OUT / "server_tampering_precommit_events"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    with Image.open(stem.with_suffix(".png")) as image:
        image.convert("L").save(str(stem) + "_grayscale.png", dpi=(600, 600))
    plt.close(fig)
    return issues


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves, events, manifest = audit()
    write_csv(OUT / "tamper_event_audit.csv", events)
    write_csv(OUT / "source_manifest.csv", manifest)
    issues = draw(curves, events)
    mean_final = statistics.mean(curves[seed][-1] for seed in RUNS)
    sd_final = statistics.stdev(curves[seed][-1] for seed in RUNS)
    lines = [
        "# 服务器篡改聚合结果实验图审计",
        "",
        "审计对象：CIFAR-10 IID、VeraShield-FL、seed 7/21/42 的独立服务器篡改提交前验证实验。",
        "核对通过：3/3 runs，180/180 轮验证记录，9/9 预设篡改事件，171/171 非篡改轮正常提交，0 次额外拒绝。",
        "三次篡改轮为 20、40、59；每个 seed 中候选摘要与已提交摘要不同，候选均在 precommit 被拒绝，模型和协议状态均未提交。拒绝轮的集中测试准确率与前一轮相同，说明评估的是保留的已提交模型。",
        "",
        f"最终准确率（3 seeds）：{mean_final:.2f}% ± {sd_final:.2f}%（样本标准差，n=3）。",
        "",
        "图 (a) 展示三个 seed 的逐轮准确率及篡改轮，图 (b) 展示各篡改事件的拒绝/提交结果。该流程在逻辑双服务器原型内本地执行；结果不证明物理双服务器隔离、远程客户端提交证书或普遍恶意服务器检测能力。",
        "",
        "原始实验数据未修改。数据文件与 SHA-256 见 source_manifest.csv；逐事件证据见 tamper_event_audit.csv。",
        "",
        "版式审计：" + ("PASS" if not issues else json.dumps(issues, ensure_ascii=False)),
        "",
        "图注：CIFAR-10 IID 条件下服务器篡改聚合结果的提交前验证实验。上图为 seed 7、21 和 42 的集中测试准确率；虚线标出第 20、40、59 轮篡改尝试。下图给出各 seed 在相应轮次的验证结果：篡改候选均在提交前被拒绝，未提交模型或协议状态。拒绝轮继续使用前一已提交模型进行评估。实验基于逻辑双服务器原型。",
    ]
    (OUT / "AUDIT_AND_CAPTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "layout_audit.json").write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: 3 seeds, 9 tamper events, {OUT}")


if __name__ == "__main__":
    import sys
    main()
