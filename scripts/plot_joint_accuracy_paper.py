"""Current 60-round joint-threat figures; no smoothing or seed filtering."""
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
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = _ReleasePath(_release_sys.modules["os"].environ.get("VERASHIELD_FIGURE_ROOT", str(_RELEASE_ROOT / "figures"))) / "joint"
SEEDS = (7, 21, 42)
METHODS = {
    "verashield_fl": ("VeraShield-FL", "#B83340", "o", "-"),
    "fedavg": ("FedAvg", "#3373A5", "s", "--"),
    "krum": ("Krum", "#26856B", "^", "-."),
    "trimmed_mean": ("Trimmed Mean", "#8366A5", "D", ":"),
    "feddmc": ("FedDMC", "#C78317", "v", (0, (5, 2, 1, 2))),
}

def source_roots(dataset):
    if dataset == "CIFAR-10":
        return {7: ROOT / "final_v7/joint_main_seed7",
                21: ROOT / "final_v7/joint_main_multiseed_extension/seed21",
                42: ROOT / "final_v7/joint_main_multiseed_extension/seed42"}
    return {7: ROOT / "mnist_joint_formal_seed7_20261014",
            21: ROOT / "mnist_joint_multiseed_extension/seed21",
            42: ROOT / "mnist_joint_multiseed_extension/seed42"}

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def latest_verashield(dataset, seed):
    if dataset == "MNIST":
        return ROOT / f"mnist_precommit_joint_multiseed_20261016/seed{seed}/verashield_fl"
    if seed == 7:
        return ROOT / "joint_precommit_cifar10_seed7_20261015/verashield_fl"
    return ROOT / f"joint_precommit_cifar10_multiseed_20261016/seed{seed}/verashield_fl"

def load(dataset):
    raw, sources, series = [], [], {}
    for seed, root in source_roots(dataset).items():
        for method in METHODS:
            path = root / method / "results.csv"
            if method == "verashield_fl":
                path = latest_verashield(dataset, seed) / "results.csv"
            config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            assert dataset in config["dataset"], path
            assert config["seed"] == seed and config["num_rounds"] == 60, path
            assert config["partitioner"] == "dirichlet" and config["alpha"] == 0.5, path
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 61, (path, len(rows))
            assert [int(row["round"]) for row in rows] == list(range(61)), path
            values = [float(row["accuracy"]) * 100 for row in rows]
            assert all(math.isfinite(v) and 0 <= v <= 100 for v in values), path
            series[method, seed] = values
            sources.append({"dataset": dataset, "seed": seed, "method": METHODS[method][0],
                            "source": path.relative_to(ROOT).as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            raw.extend({"dataset": dataset, "method": METHODS[method][0], "seed": seed,
                        "round": r, "accuracy_pct": v} for r, v in enumerate(values))
    stats = []
    for method in METHODS:
        for r in range(61):
            values = [series[method, seed][r] for seed in SEEDS]
            stats.append({"dataset": dataset, "method": METHODS[method][0], "round": r,
                          "n": len(values), "mean_pct": statistics.mean(values),
                          "sample_sd_pct": statistics.stdev(values)})
    return series, raw, stats, sources

def plot(dataset, series):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 7.3,
        "axes.linewidth": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.unicode_minus": False,
    })
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    for idx, (method, (label, color, marker, linestyle)) in enumerate(METHODS.items()):
        values = [statistics.mean(series[method, seed][r] for seed in SEEDS) for r in range(61)]
        ax.plot(range(61), values, label=label, color=color, linestyle=linestyle,
                linewidth=1.4 if idx == 0 else 1.05, marker=marker,
                markevery=(idx + 1, 9), markersize=3.3,
                markerfacecolor="white", markeredgewidth=0.8,
                zorder=5 if idx == 0 else 3)
    ax.set(xlim=(0, 60), ylim=(0, 55 if dataset == "CIFAR-10" else 105),
           xlabel="Communication round", ylabel="Test accuracy (%)")
    ax.set_xticks(range(0, 61, 10))
    ax.set_yticks(range(0, 51, 10) if dataset == "CIFAR-10" else range(0, 101, 20))
    ax.tick_params(direction="in", length=3, width=0.65, top=True, right=True)
    ax.grid(color="#DEDEDE", linewidth=0.4, linestyle=":", zorder=0)
    legend = ax.legend(loc="center right",
              bbox_to_anchor=(0.985, 0.49) if dataset == "CIFAR-10" else (0.985, 0.64),
              frameon=True, fancybox=False, edgecolor="#B5B5B5", facecolor="white",
              framealpha=1, borderpad=0.45, handlelength=2.2, fontsize=7,
              labelspacing=0.25, handletextpad=0.6)
    legend.get_frame().set_linewidth(0.4)
    fig.subplots_adjust(left=0.15, bottom=0.18, right=0.975, top=0.975)
    stem = OUT / f"{dataset.lower().replace('-', '')}_joint_accuracy_3seed_mean"
    skills = Path("<LOCAL_USER>/.codex/skills/scipilot-figure-skill/scripts") if sys.platform == "win32" else Path("<LOCAL_USER>/.codex/skills/scipilot-figure-skill/scripts")
    sys.path.insert(0, str(skills))
    from release_visual_qa import audit_layout
    issues = audit_layout(fig)
    assert not any(level == "FAIL" for level, _ in issues), issues
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer())
    # Sample each segment, rather than just its endpoints: steep transients
    # must not disappear underneath an opaque legend.
    for line in ax.lines:
        points = ax.transData.transform(line.get_xydata())
        for a, b in zip(points[:-1], points[1:]):
            for j in range(31):
                t = j / 30
                x, y = a * (1 - t) + b * t
                assert not box.contains(x, y), (dataset, line.get_label(), "legend occludes data", ax.transData.inverted().transform([[box.x0,box.y0],[box.x1,box.y1]]).tolist(), ax.transData.inverted().transform([x,y]).tolist())
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    from PIL import Image
    with Image.open(stem.with_suffix(".png")) as im:
        im.convert("L").save(str(stem) + "_grayscale.png", dpi=(600, 600))
    plt.close(fig)
    return issues

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_all, stats_all, sources_all, final_all, qa = [], [], [], [], {}
    for dataset in ("CIFAR-10", "MNIST"):
        series, raw, stats, sources = load(dataset)
        qa[dataset] = plot(dataset, series)
        raw_all.extend(raw)
        stats_all.extend(stats)
        sources_all.extend(sources)
        for method in METHODS:
            final_values = [series[method, seed][-1] for seed in SEEDS]
            final_all.append({"dataset": dataset, "method": METHODS[method][0],
                              **{f"seed_{s}_pct": v for s, v in zip(SEEDS, final_values)},
                              "mean_pct": statistics.mean(final_values),
                              "sample_sd_pp": statistics.stdev(final_values), "n": 3})
        print(dataset, [(METHODS[m][0], round(statistics.mean(series[m,s][-1] for s in SEEDS), 4)) for m in METHODS])
    write_csv(OUT / "curve_source_data.csv", raw_all)
    write_csv(OUT / "curve_mean_sd.csv", stats_all)
    write_csv(OUT / "source_manifest.csv", sources_all)
    write_csv(OUT / "final_accuracy_summary.csv", final_all)
    (OUT / "layout_audit.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    caption = (
        "CIFAR-10 / MNIST: Test accuracy over communication rounds under the joint-threat scenario. "
        "Each curve is the arithmetic mean of three runs (seeds 7, 21, and 42); round 0 is initialization. "
        "No smoothing or uncertainty shading is applied. VeraShield-FL uses the latest precommit-verification runs; "
        "the four baselines use the existing joint-scenario runs. "
        "Server tampering is scheduled at rounds 20, 40, and 59. For VeraShield-FL, rejected candidates are not installed, "
        "and accuracy is evaluated on the retained committed model. "
        "Final-round per-seed accuracy and mean/sample SD are reported in final_accuracy_summary.csv.\n\n"
        "These are logical-prototype utility results; the curves do not establish physical two-server privacy isolation. "
        "Baseline instability and differences in actual tamper execution must be discussed separately in the experiment text.\n"
    )
    (OUT / "README.txt").write_text(caption, encoding="utf-8")
    table = ["# 主实验图与配套统计", "", "曲线为 seed 7、21、42 的逐轮算术均值；无平滑、无阴影。VeraShield-FL 使用最新提交前验证结果，四种基线沿用既有联合场景结果。", "",
             "| 数据集 | 方法 | seed 7 (%) | seed 21 (%) | seed 42 (%) | 均值 ± 样本标准差 (%) |",
             "|---|---|---:|---:|---:|---:|"]
    for row in final_all:
        table.append(f"| {row['dataset']} | {row['method']} | {row['seed_7_pct']:.2f} | {row['seed_21_pct']:.2f} | {row['seed_42_pct']:.2f} | {row['mean_pct']:.2f} ± {row['sample_sd_pp']:.2f} |")
    table.extend(["", "图注：联合威胁场景下五种方法的测试准确率随通信轮次的变化。每条曲线为三个独立 seed（7、21、42）的均值，第 0 轮表示初始化。服务器篡改安排于第 20、40、59 轮；VeraShield-FL 在提交前拒绝篡改候选，并使用保留的已提交模型进行评估。图中不绘制标准差阴影；最终准确率的逐 seed 数值与样本标准差见表。", "", "说明：这些曲线展示逻辑原型的模型效用，不证明物理双服务器隐私隔离。基线训练不稳定及实际篡改执行差异应在实验正文中另行说明；MNIST Krum 的跨 seed 差异不能从均值曲线单独判断。", ""])
    (OUT / "captions_and_statistics.md").write_text("\n".join(table), encoding="utf-8")

if __name__ == "__main__":
    main()
