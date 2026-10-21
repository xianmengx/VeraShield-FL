"""Audit 135 client-attack runs and render reproducible manuscript figures.

Never changes training results. Non-IID uses all nine unified-v2 runs.
"""
from __future__ import annotations

import os
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import argparse
import csv
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from plot_attack_comparison_three_scenarios import source_dir as historical_dir
from audit_noniid20_multiseed import run_dir

ROOT = Path(__file__).resolve().parents[1]
OUT = _ReleasePath(_release_sys.modules["os"].environ.get("VERASHIELD_FIGURE_ROOT", str(_RELEASE_ROOT / "figures"))) / "client_attacks"
SEEDS = (7, 21, 42)
ATTACKS = (('sign_flip', 'Sign-flip', 5), ('random_gradient', 'Random-gradient', 10), ('scaling', 'Scaling', 10))
SCENARIOS = {'iid20': 'IID, 20% malicious clients', 'noniid20': 'Non-IID, 20% malicious clients', 'iid40': 'IID, 40% malicious clients'}
METHODS = {
    'verashield_fl': ('VeraShield-FL', '#B83340', 'o', '-'),
    'fedavg': ('FedAvg', '#3373A5', 's', '--'),
    'krum': ('Krum', '#26856B', '^', '-.'),
    'trimmed_mean': ('Trimmed Mean', '#8366A5', 'D', ':'),
    'feddmc': ('FedDMC', '#C78317', 'v', (0, (5, 2, 1, 2))),
}

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def source(scenario, attack, method, seed):
    if scenario != 'noniid20':
        return historical_dir(scenario, attack, method, seed)
    if method != 'verashield_fl':
        return run_dir(seed, attack, method)
    if (attack, seed) == ('sign_flip', 21):
        return ROOT / 'results/final_v6/unified_noniid20_signflip_s21/verashield_fl'
    return ROOT / f'results/final_v6/noniid20_unified_v2_gapfill/{attack}_s{seed}/verashield_fl'

def flag(value):
    return str(value).lower() in ('true', '1', '1.0')

def collect():
    curves, manifest, checks, detection, configs = {}, [], [], [], {}
    for scenario in SCENARIOS:
        for attack, _, strength in ATTACKS:
            for seed in SEEDS:
                paired_ids = None
                paired_training = None
                for method in METHODS:
                    path = source(scenario, attack, method, seed)
                    config = read_json(path / 'config.json')
                    summary = read_json(path / 'summary.json')
                    rows = read_csv(path / 'results.csv')
                    scenario_record = read_json(path / 'scenario.json')
                    assert [int(r['round']) for r in rows] == list(range(61)), path
                    vals = [float(r['accuracy']) * 100 for r in rows]
                    assert all(math.isfinite(v) and 0 <= v <= 100 for v in vals), path
                    expected = {'seed': seed, 'num_rounds': 60, 'num_clients': 20,
                                'attack_type': attack, 'attack_strength': strength,
                                'attack_ratio': .4 if scenario == 'iid40' else .2,
                                'attack_start_round': 4, 'attack_mode': 'persistent',
                                'partitioner': 'dirichlet' if scenario == 'noniid20' else 'iid',
                                'fraction_train': 1.0, 'dropout_rate': 0.0,
                                'vl_tamper_enabled': 0}
                    if scenario == 'noniid20':
                        expected['alpha'] = .5
                    if method == 'krum':
                        expected['krum_f'] = 4 if scenario == 'iid40' else 2
                    if method == 'trimmed_mean':
                        expected['trim_ratio'] = .2
                    if method == 'verashield_fl' and scenario == 'noniid20':
                        expected['ssdf_evidence_policy'] = 'unified_v2_candidate'
                    for key, value in expected.items():
                        assert config.get(key) == value, (path, key, config.get(key), value)
                    assert 'CIFAR-10' in config['dataset'], path
                    assert summary['number_of_rounds'] == 60 and summary['number_of_clients'] == 20, path
                    assert abs(vals[-1] / 100 - summary['final_accuracy']) < 1e-10, path
                    ids = sorted(config['malicious_client_ids'])
                    assert len(ids) == (8 if scenario == 'iid40' else 4), path
                    training = config['run_config']
                    expected_training = {'learning-rate': .1, 'local-epochs': 1,
                        'batch-size': 32, 'lr-schedule': 'cosine', 'lr-min': .001,
                        'num-server-rounds': 60, 'num-clients': 20, 'fraction-train': 1.0}
                    training_fields = {k: training[k] for k in expected_training}
                    assert training_fields == expected_training, (path, training_fields)
                    if paired_ids is None:
                        paired_ids, paired_training = ids, training_fields
                    assert ids == paired_ids, path
                    assert training_fields == paired_training, (path, training_fields, paired_training)
                    for r in scenario_record.get('per_round', []):
                        assert not r.get('dropped_clients', []) and not r.get('unavailable_ids', []), path
                    curves[scenario, attack, method, seed] = vals
                    configs[scenario, attack, method, seed] = config
                    version = config.get('ssdf_evidence_policy') or 'historical_fixed_budget'
                    entry = dict(scenario=scenario, attack=attack, method=method, seed=seed,
                                 policy=version, source=str(path.relative_to(ROOT)))
                    for name in ('results.csv', 'config.json', 'scenario.json', 'summary.json'):
                        entry[name.replace('.', '_') + '_sha256'] = hashlib.sha256((path/name).read_bytes()).hexdigest()
                    manifest.append(entry)
                    recorded_partition = scenario_record.get('partitioner', scenario_record.get('training_config', {}).get('partitioner'))
                    checks.append(dict(scenario=scenario, attack=attack, method=method, seed=seed,
                                       rows=len(rows), config_pass=True, matched_ids=True,
                                       matched_training=True,
                                       training_settings=json.dumps(training_fields, sort_keys=True),
                                       scenario_partitioner=recorded_partition,
                                       actual_partitioner=config['partitioner'],
                                       manifest_partition_warning=recorded_partition not in (None, config['partitioner'])))
                    if method == 'verashield_fl':
                        det = [r for r in read_csv(path/'detection_statistics.csv') if 4 <= int(r['round']) <= 60]
                        assert len(det) == 1140, (path, len(det))
                        assert len({(int(r['round']), r['client_id']) for r in det}) == 1140, path
                        count = dict(TP=0, TN=0, FP=0, FN=0)
                        for r in det:
                            assert flag(r.get('detection_eligible', 1)), path
                            truth = flag(r['is_malicious_gt']) and flag(r['attack_active'])
                            pred = flag(r['is_malicious_pred'])
                            count['TP' if truth and pred else 'FN' if truth else 'FP' if pred else 'TN'] += 1
                        assert count['TP'] + count['FN'] == (456 if scenario == 'iid40' else 228), path
                        recorded = read_json(path/'detection_metrics_active.json')
                        vl = read_csv(path/'vl_verification_results.csv')
                        detection.append(dict(scenario=scenario, attack=attack, seed=seed, **count,
                            recall=count['TP']/(count['TP']+count['FN']),
                            fpr=count['FP']/(count['FP']+count['TN']),
                            precision=count['TP']/(count['TP']+count['FP']) if count['TP']+count['FP'] else 0,
                            observed_client_rounds=len(det), expected_client_rounds=1140,
                            recorded_metric_instances=recorded['n_detection_instances'],
                            vl_record_count=len(vl), vl_failures=sum(not flag(r['verification_success']) for r in vl)))
    assert len(manifest) == 135
    write_csv('source_manifest.csv', manifest)
    write_csv('configuration_audit.csv', checks)
    write_csv('detection_per_seed.csv', detection)
    raw, means, final = [], [], []
    for scenario in SCENARIOS:
        for attack, _, _ in ATTACKS:
            for method in METHODS:
                last = [curves[scenario, attack, method, s][-1] for s in SEEDS]
                final.append(dict(scenario=scenario, attack=attack, method=METHODS[method][0],
                                  seed7_pct=last[0], seed21_pct=last[1], seed42_pct=last[2],
                                  mean_pct=st.mean(last), sample_sd_pp=st.stdev(last)))
                for r in range(61):
                    y = [curves[scenario, attack, method, s][r] for s in SEEDS]
                    means.append(dict(scenario=scenario, attack=attack, method=method, round=r,
                                      mean_pct=st.mean(y), sample_sd_pp=st.stdev(y), n=3))
                    raw.extend(dict(scenario=scenario, attack=attack, method=method, seed=s, round=r,
                                    accuracy_pct=v) for s, v in zip(SEEDS, y))
    write_csv('curves_per_seed.csv', raw)
    write_csv('curves_mean_sd.csv', means)
    write_csv('final_accuracy.csv', final)
    report(final, detection, checks)
    return means

def report(final, detection, checks):
    lines = ['# 三类客户端攻击：核对结果与图表说明', '',
             '核对通过：135/135 项，三种场景 × 三种攻击 × 五种方法 × 三个 seed；每项 60 轮，另含初始化第 0 轮。',
             '数据集 CIFAR-10，20 客户端，全部参与，攻击从第 4 轮开始且持续；符号翻转强度 5，随机梯度替换和幅值缩放强度 10。无掉线、无服务器篡改。', '',
             '## 版本与排除规则', '',
             '- Non-IID 20% 主方法采用全部九项 unified_v2_candidate 结果，替代旧的单项替换混合版本。四种基线复用对应 Non-IID 结果。',
             '- IID 20% 与 IID 40% 保留各自完整历史固定配置矩阵。它们与 Non-IID 使用的检测策略不同，因此三张图分别呈现，不能把跨图差异全部归因于数据划分。',
             '- Non-IID 40% 仅有已知失败的 seed 7 诊断批次，不作为正式压力对照，不补数、不合并。',
             '- 修正版策略曾参考已观察到的失败案例；这些结果不属于未见数据上的独立验证。',
             '- IID 40% Krum f=4，20% 场景 f=2；Trimmed Mean 的总裁剪比例均为 0.2。压力测试的基线参数变化需在正文说明。',
             f'- 场景清单划分字段与实际 config 不一致条目：{sum(r["manifest_partition_warning"] for r in checks)}；具体条目见 configuration_audit.csv。实际执行划分以 config 及已存审计为依据。', '',
             '## 最终准确率（均值 ± 样本标准差，%）', '',
             '| 场景 | 攻击 | 方法 | seed 7 | seed 21 | seed 42 | 均值 ± SD |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in final:
        lines.append(f"| {r['scenario']} | {r['attack']} | {r['method']} | {r['seed7_pct']:.2f} | {r['seed21_pct']:.2f} | {r['seed42_pct']:.2f} | {r['mean_pct']:.2f} ± {r['sample_sd_pp']:.2f} |")
    lines.extend(['', '## PoisonGuard 检测指标（统一第 4–60 轮）', '',
                  '从 detection_statistics.csv 重新计算，保证每 seed 分母为 1140 个客户端轮次。历史 detection_metrics_active.json 部分包含前 3 轮，不能直接与新版指标混合。本表先逐 seed 计算比率，再报告三 seed 均值 ± 样本标准差。', '',
                  '| 场景 | 攻击 | Recall (%) | FPR (%) | Precision (%) | 验证失败记录数 |',
                  '|---|---|---:|---:|---:|---:|'])
    for scenario in SCENARIOS:
        for attack, _, _ in ATTACKS:
            rows = [r for r in detection if r['scenario']==scenario and r['attack']==attack]
            values = [f"{st.mean(r[k] for r in rows)*100:.2f} ± {st.stdev(r[k] for r in rows)*100:.2f}" for k in ('recall','fpr','precision')]
            lines.append('| '+scenario+' | '+attack+' | '+' | '.join(values)+f" | {sum(r['vl_failures'] for r in rows)} |")
    lines.extend(['', '## 论文图注', '',
                  'CIFAR-10 在三种客户端投毒攻击下的测试准确率。子图 (a)–(c) 分别为符号翻转、随机梯度替换和幅值缩放。每条曲线表示 seed 7、21、42 的逐轮均值，无平滑、无标准差阴影；最终准确率的逐 seed 数值与样本标准差见配套表。各图分别对应 IID 20%、Non-IID 20%（Dirichlet α=0.5）与 IID 40% 恶意客户端。攻击始于第 4 轮，无掉线与服务器篡改。', '',
                  '三张图反映完整方法的抗投毒效用，检测能力由上述 Recall/FPR/Precision 表支撑；未做模块消融时，不能将准确率优势全部归因于 PoisonGuard。IID scaling 下 Trimmed Mean 的最终均值可能略高，禁止声称所有攻击均绝对最优。', '',
                  '## 文件', '', 'PDF/SVG 为论文用矢量文件；PNG 为 600 dpi 预览；灰度图用于检查线型区分。source_manifest.csv 保存所有输入路径与 SHA-256。原始结果保持不变。', ''])
    (OUT/'AUDIT_AND_CAPTIONS.md').write_text('\n'.join(lines), encoding='utf-8')

def render(data, scenario, export):
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':8,
                         'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.65,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,3, figsize=(7.16,2.65), sharey=True)
    fig.subplots_adjust(left=.07,right=.99,bottom=.22,top=.78,wspace=.13)
    for i,(attack,title,_) in enumerate(ATTACKS):
        ax=axes[i]
        for j,(method,(label,color,marker,style)) in enumerate(METHODS.items()):
            rows=[r for r in data if (r['scenario'],r['attack'],r['method'])==(scenario,attack,method)]
            ax.plot([r['round'] for r in rows],[r['mean_pct'] for r in rows], label=label,
                    color=color,linestyle=style,linewidth=1.4 if j==0 else 1.05,
                    marker=marker,markevery=(j+1,9),markersize=2.7,
                    markerfacecolor='white',markeredgewidth=.7,zorder=5 if j==0 else 3)
        ax.set(xlim=(0,60),ylim=(0,60),xlabel='Communication round')
        ax.set_title(f'({chr(97+i)}) {title}',pad=5)
        ax.set_xticks([0,20,40,60]);ax.set_yticks(range(0,61,10))
        ax.tick_params(direction='in',top=True,right=True,length=2.5,width=.65)
        ax.grid(color='#DEDEDE',linestyle=':',linewidth=.4)
    axes[0].set_ylabel('Test accuracy (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    legend=fig.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.52,.99),
                      frameon=False,handlelength=2.3,columnspacing=1.4,handletextpad=.5)
    from release_visual_qa import audit_layout
    issues=audit_layout(fig)
    assert not any(level in ('FAIL','WARN') for level,_ in issues),issues
    fig.canvas.draw()
    box=legend.get_window_extent(fig.canvas.get_renderer())
    assert all(not box.overlaps(ax.get_window_extent()) for ax in axes)
    stem=OUT/f'{scenario}_three_attacks_accuracy'
    fig.savefig(stem.with_suffix('.png'),dpi=600,facecolor='white')
    with Image.open(stem.with_suffix('.png')) as im:
        im.convert('L').save(str(stem)+'_grayscale.png',dpi=(600,600))
    if export:
        for suffix in ('.pdf','.svg'):
            fig.savefig(stem.with_suffix(suffix),facecolor='white')
    plt.close(fig)
    return issues

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    data=collect()
    qa={scenario:render(data,scenario,args.export) for scenario in SCENARIOS}
    (OUT/'layout_audit.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print('PASS: 135 runs; 27 detector logs; three figures. Output:',OUT)

if __name__=='__main__':
    main()
