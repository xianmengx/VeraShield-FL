"""Read-only audit and aggregate corrected precommit VeraShield-FL runs."""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/precommit_multiseed_audit_20261016'
RUNS = {
    'CIFAR-10': [(7, ROOT/'results/joint_precommit_cifar10_seed7_20261015/verashield_fl'),
                 *[(s, ROOT/f'results/joint_precommit_cifar10_multiseed_20261016/seed{s}/verashield_fl') for s in (21, 42)]],
    'MNIST': [(s, ROOT/f'results/mnist_precommit_joint_multiseed_20261016/seed{s}/verashield_fl') for s in (7, 21, 42)],
}


def rows(path):
    with path.open(encoding='utf-8', newline='') as h:
        return list(csv.DictReader(h))


def audit(dataset, seed, p):
    cfg = json.loads((p/'config.json').read_text())
    summary = json.loads((p/'summary.json').read_text())
    curve = rows(p/'results.csv')
    vl = rows(p/'vl_verification_results.csv')
    rec = rows(p/'recovery_statistics.csv')
    threshold = rows(p/'runtime_threshold.csv')
    weights = rows(p/'dynamic_weight_results.csv')
    metrics = json.loads((p/'detection_metrics_active.json').read_text())
    assert len(curve) == 61 and [int(x['round']) for x in curve] == list(range(61))
    assert len(vl) == len(threshold) == 60
    assert [int(x['round']) for x in vl] == list(range(1, 61))
    tamper = [int(x['round']) for x in vl if int(x['tamper_applied'])]
    rejected = [int(x['round']) for x in vl if int(x['candidate_rejected'])]
    assert tamper == rejected == [20, 40, 59]
    assert all(int(x['model_committed']) == int(x['state_committed']) == (int(x['round']) not in (20, 40, 59)) for x in vl)
    assert all(math.isfinite(float(x['loss'])) for x in curve)
    attempts = [x for x in rec if x['recovery_attempted'] == '1']
    recovery_successes = sum(x['recovery_success'] == '1' for x in attempts)
    threshold_counts = {'normal_16': sum(x['threshold_mode'] == 'normal' and int(x['active_threshold']) == 16 for x in threshold),
                        'emergency_12': sum(x['threshold_mode'] == 'emergency' and int(x['active_threshold']) == 12 for x in threshold)}
    weight_sums = [sum(float(x['lambda_weight']) for x in weights if int(x['round']) == r) for r in range(1, 61)]
    return {
        'dataset': dataset, 'seed': seed, 'final_accuracy': summary['final_accuracy'],
        'final_loss': summary['final_loss'], 'best_accuracy': summary['best_accuracy'],
        'best_round': summary['best_round'], 'rounds': 60, 'tamper_rounds': tamper,
        'rejected_rounds': rejected, 'precommit_pass': True,
        'recovery_attempts': len(attempts), 'recovery_successes': recovery_successes,
        'recovery_rate': recovery_successes / len(attempts) if attempts else None,
        'threshold_normal_rounds': threshold_counts['normal_16'],
        'threshold_emergency_rounds': threshold_counts['emergency_12'],
        'detection_instances': metrics['n_detection_instances'], 'TP': metrics['TP'],
        'TN': metrics['TN'], 'FP': metrics['FP'], 'FN': metrics['FN'],
        'recall': metrics['recall'], 'fpr': metrics['fpr'], 'precision': metrics['precision'],
        'weight_sum_min': min(weight_sums), 'weight_sum_max': max(weight_sums),
        'finite_loss': True, 'config_sha256': hashlib.sha256((p/'config.json').read_bytes()).hexdigest(),
        'scenario_sha256': hashlib.sha256((p/'scenario.json').read_bytes()).hexdigest(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = [audit(dataset, seed, p) for dataset, runs in RUNS.items() for seed, p in runs]
    with (OUT/'per_seed.csv').open('w', encoding='utf-8', newline='') as h:
        writer = csv.DictWriter(h, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    aggregates = []
    for dataset in RUNS:
        rs = [x for x in records if x['dataset'] == dataset]
        for metric in ('final_accuracy', 'best_accuracy', 'recall', 'fpr', 'precision', 'recovery_rate'):
            values = [x[metric] for x in rs]
            aggregates.append({'dataset': dataset, 'metric': metric, 'n_seeds': len(values),
                               'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)})
    with (OUT/'aggregate.csv').open('w', encoding='utf-8', newline='') as h:
        writer = csv.DictWriter(h, fieldnames=list(aggregates[0])); writer.writeheader(); writer.writerows(aggregates)
    lines = ['# Corrected precommit VeraShield-FL multi-seed audit', '',
             'Status: PASS WITH LIMITATIONS. Read-only audit; raw training outputs were not modified.', '',
             '|Dataset|Seed|Final accuracy|Tamper rejection|Recovery|Recall|FPR|Precision|',
             '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"|{r['dataset']}|{r['seed']}|{r['final_accuracy']*100:.2f}%|{r['rejected_rounds']}|{r['recovery_successes']}/{r['recovery_attempts']}|{r['recall']*100:.2f}%|{r['fpr']*100:.2f}%|{r['precision']*100:.2f}%|")
    lines += ['', '## Three-seed summary', '', '|Dataset|Metric|Mean|Sample SD|', '|---|---|---:|---:|']
    for x in aggregates:
        lines.append(f"|{x['dataset']}|{x['metric']}|{x['mean']:.6f}|{x['sample_sd']:.6f}|")
    lines += ['', '## Interpretation boundaries', '',
              '- All six runs use the corrected precommit logic, logical dual-server sequential simulation, and post-share logical dropout.',
              '- Each dataset has three seeds, but no statistical significance test is claimed.',
              '- The four baseline methods are not rerun in this corrected batch; baseline comparisons must use their matching existing records and protocol audit.',
              '- These results support complete-system utility and the tested rejection/recovery behavior, not physical two-server privacy or detector statistical correctness.',
              '- Seed 7 and seed 21/42 may have different recovery-record counts in older runs; report each run from its own raw files before pooling mechanism totals.']
    (OUT/'AUDIT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps({'status':'PASS_WITH_LIMITATIONS','records':records,'aggregates':aggregates}, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(OUT/'AUDIT.md')


if __name__ == '__main__':
    main()
