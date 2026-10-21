"""Five-method utility comparison, latest precommit VeraShield-FL, audited sources."""

import os
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import plot_server_tamper_events_paper as latest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/final_v7/server_tamper_5methods_multiseed'
OUT = _ReleasePath(_release_sys.modules["os"].environ.get("VERASHIELD_FIGURE_ROOT", str(_RELEASE_ROOT / "figures"))) / "server_tamper"
SEEDS = (7, 21, 42)
METHODS = {
    'verashield_fl': ('VeraShield-FL', '#B83340', 'o', '-'),
    'fedavg': ('FedAvg', '#3373A5', 's', '--'),
    'krum': ('Krum', '#26856B', '^', '-.'),
    'trimmed_mean': ('Trimmed Mean', '#8366A5', 'D', ':'),
    'feddmc': ('FedDMC', '#C78317', 'v', (0, (5, 2, 1, 2))),
}
from release_visual_qa import audit_layout


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    latest_curves, latest_events, _ = latest.audit()
    curves, raw, stats, manifests, events, finals, configs = {}, [], [], [], [], [], []
    for method, (label, *_style) in METHODS.items():
        for seed in SEEDS:
            folder = latest.RUNS[seed] if method == 'verashield_fl' else BASE / f'seed{seed}' / method
            config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
            assert config['seed'] == seed and config['num_rounds'] == 60
            assert config['num_clients'] == 20 and config['partitioner'] == 'iid'
            assert config['attack_enabled'] == 0 and config['dropout_rate'] == 0
            assert 'CIFAR-10' in config['dataset']
            assert config['vl_tamper_rounds'] == '20,40,59'
            # Check training settings across methods, not method-specific defense switches.
            keys = ('learning-rate', 'batch-size', 'local-epochs', 'fraction-train', 'fraction-evaluate')
            configs.append({key: config['run_config'][key] for key in keys})
            rows = latest.read_csv(folder / 'results.csv')
            assert [int(row['round']) for row in rows] == list(range(61))
            values = [100 * float(row['accuracy']) for row in rows]
            assert all(math.isfinite(v) and 0 <= v <= 100 for v in values)
            summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            assert math.isclose(values[-1], 100 * summary['final_accuracy'], abs_tol=1e-10)
            curves[method, seed] = values
            raw.extend(dict(method=label, seed=seed, round=r, accuracy_pct=v) for r, v in enumerate(values))
            diagnostic_name = 'vl_verification_results.csv' if method == 'verashield_fl' else 'aggregation_diagnostics.csv'
            diagnostic = {int(row['round']): row for row in latest.read_csv(folder / diagnostic_name)}
            for r in latest.TAMPER_ROUNDS:
                record = diagnostic[r]
                applied = int(record['tamper_applied'] or 0)
                outcome = 'precommit rejection' if method == 'verashield_fl' else 'verification not enabled'
                if not applied:
                    assert (method, seed, r) == ('krum', 7, 40)
                    outcome = 'tamper skipped; see aggregation status'
                events.append(dict(method=label, seed=seed, round=r, tamper_applied=applied,
                                   verification_outcome=outcome,
                                   aggregation_status=record.get('aggregation_status', ''),
                                   candidate_rejected=record.get('candidate_rejected', 'not measured')))
            for filename in ('config.json', 'results.csv', 'summary.json', diagnostic_name):
                path = folder / filename
                manifests.append(dict(method=label, seed=seed, source=path.relative_to(ROOT).as_posix(),
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for r in range(61):
            vals = [curves[method, seed][r] for seed in SEEDS]
            stats.append(dict(method=label, round=r, n=3, mean_pct=statistics.mean(vals),
                              sample_sd_pp=statistics.stdev(vals)))
        vals = [curves[method, seed][-1] for seed in SEEDS]
        finals.append(dict(method=label, **{f'seed_{seed}_pct': v for seed, v in zip(SEEDS, vals)},
                           mean_pct=statistics.mean(vals), sample_sd_pp=statistics.stdev(vals)))
    assert all(config == configs[0] for config in configs)
    assert len(raw) == 915 and len(events) == 45
    assert sum(e['tamper_applied'] for e in events) == 44
    for seed in SEEDS:
        assert curves['verashield_fl', seed] == latest_curves[seed]
    for filename, rows in [('curve_source_data.csv', raw), ('curve_mean_sd.csv', stats),
                           ('source_manifest.csv', manifests), ('tamper_event_audit.csv', events),
                           ('final_accuracy_summary.csv', finals)]:
        latest.write_csv(OUT / filename, rows)

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'legend.fontsize': 7, 'axes.linewidth': 0.65,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(3.55, 2.95))
    for idx, (method, (label, color, marker, linestyle)) in enumerate(METHODS.items()):
        means = [statistics.mean(curves[method, seed][r] for seed in SEEDS) for r in range(61)]
        ax.plot(range(61), means, label=label, color=color, marker=marker, linestyle=linestyle,
                linewidth=1.4 if idx == 0 else 1.05, markersize=3.3, markevery=(idx+1, 9),
                markerfacecolor='white', markeredgewidth=0.8, zorder=5 if idx == 0 else 3)
    for r in latest.TAMPER_ROUNDS:
        ax.axvline(r, color='#888888', linestyle=(0, (3, 3)), linewidth=0.7, zorder=0)
        ax.text(r, 58.5, str(r), color='#555555', ha='center', va='top', fontsize=7)
    ax.set(xlim=(0, 60), ylim=(0, 60), xlabel='Communication round', ylabel='Test accuracy (%)')
    ax.set_xticks(range(0, 61, 10))
    ax.set_yticks(range(0, 61, 10))
    ax.grid(color='#DEDEDE', linewidth=0.4, linestyle=':', zorder=0)
    ax.tick_params(direction='in', length=3, width=0.65, top=True, right=True)
    # Outside data area: no opaque legend can conceal attack-induced transients.
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
              handlelength=2.2, columnspacing=0.9, handletextpad=0.5,
              borderaxespad=0, labelspacing=0.4)
    fig.subplots_adjust(left=0.15, right=0.975, bottom=0.16, top=0.81)
    issues = audit_layout(fig)
    assert not any(level in ('FAIL', 'WARN') for level, _ in issues), issues
    stem = OUT / 'cifar10_server_tamper_5methods_3seed_mean'
    for fmt in ('png', 'pdf', 'svg'):
        fig.savefig(stem.with_suffix('.' + fmt), dpi=600, facecolor='white')
    with Image.open(stem.with_suffix('.png')) as im:
        im.convert('L').save(str(stem) + '_grayscale.png', dpi=(600, 600))
        im.resize((1065, 885)).save(OUT / 'preview.png')
    plt.close(fig)
    caption = '''# 服务器篡改实验：五方法对照

图注：CIFAR-10 IID 服务器篡改场景下五种方法的测试准确率。曲线为 seed 7、21、42 的逐轮算术均值，未经平滑，不绘制阴影。竖虚线标记第20、40、59轮预设篡改。VeraShield-FL 使用最新提交前验证实验：拒绝候选后，继续评估上一已提交模型；四种基线使用既有同场景结果。

VeraShield-FL 9/9 次实际篡改均在提交前被拒绝。四种基线未启用同等验证机制，其检测指标为未测量，不得将其标为检测失败。Krum seed 7 第40轮因聚合输入不足未实际施加篡改；其余44/45个预设事件实际执行。因此本图是模型效用对照，不是完全相同实际篡改暴露下的检测率对照。实验为逻辑原型，不证明物理双服务器隐私隔离。

原始数据未修改。source_manifest.csv 记录全部输入及 SHA-256；curve_source_data.csv 保存15条运行的915个逐轮值；curve_mean_sd.csv 和 final_accuracy_summary.csv 提供均值及样本标准差；tamper_event_audit.csv 逐事件记录实际篡改与验证状态。

| 方法 | seed 7 (%) | seed 21 (%) | seed 42 (%) | 均值 ± 样本标准差 (%) |
|---|---:|---:|---:|---:|
'''
    for row in finals:
        caption += f"| {row['method']} | {row['seed_7_pct']:.2f} | {row['seed_21_pct']:.2f} | {row['seed_42_pct']:.2f} | {row['mean_pct']:.2f} ± {row['sample_sd_pp']:.2f} |\n"
    (OUT / 'AUDIT_AND_CAPTION.md').write_text(caption, encoding='utf-8')
    (OUT / 'layout_audit.json').write_text(json.dumps(issues), encoding='utf-8')
    print(json.dumps(finals, indent=2))
    print('PASS: 15 runs, 915 points, 45 scheduled / 44 applied tamper events; ' + str(OUT))


if __name__ == '__main__':
    main()
