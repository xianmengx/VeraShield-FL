"""Reproducible five-method mean curves for three existing attack matrices."""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from audit_noniid20_multiseed import run_dir as noniid_run_dir

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/final_v6/attack_comparison_figures'
SEEDS = (7, 21, 42)
ATTACKS = (('sign_flip', 'Sign-flip'), ('random_gradient', 'Random-gradient'), ('scaling', 'Scaling'))
METHODS = (
    ('verashield_fl', 'VeraShield-FL', '#0072B2', '-', 'o'),
    ('fedavg', 'FedAvg', '#D55E00', '--', 's'),
    ('krum', 'Krum', '#009E73', '-.', '^'),
    ('trimmed_mean', 'Trimmed Mean', '#CC79A7', ':', 'D'),
    ('feddmc', 'FedDMC', '#6A51A3', (0, (5, 1.5, 1, 1.5)), 'v'),
)
SCENARIOS = ('noniid20_mixed', 'iid20', 'iid40')


def source_dir(scenario, attack, method, seed):
    if scenario == 'noniid20_mixed':
        if (attack, method, seed) == ('sign_flip', 'verashield_fl', 21):
            return ROOT / 'results/final_v6/unified_noniid20_signflip_s21/verashield_fl'
        return noniid_run_dir(seed, attack, method)
    slug = 'signflip' if attack == 'sign_flip' else attack
    if scenario == 'iid40':
        return ROOT / f'results/final_v2/pressure40_runs_s{seed}/{slug}40_s{seed}/{method}'
    if seed == 7:
        group = 'runs' if method == 'verashield_fl' else 'baseline_runs_s7'
        return ROOT / f'results/final_v2/{group}/{slug}_s7/{method}'
    return ROOT / f'results/final_v2/attack20_iid_runs_s{seed}/{slug}20_iid_s{seed}/{method}'


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    raw, manifest, final = [], [], []
    for scenario in SCENARIOS:
        for attack, _ in ATTACKS:
            paired_ids = None
            for method, label, *_ in METHODS:
                for seed in SEEDS:
                    path = source_dir(scenario, attack, method, seed)
                    config = json.loads((path / 'config.json').read_text())
                    summary = json.loads((path / 'summary.json').read_text())
                    with (path / 'results.csv').open(newline='', encoding='utf-8') as handle:
                        rows = list(csv.DictReader(handle))
                    rounds = [int(r['round']) for r in rows]
                    assert len(rounds) == 61 and set(rounds) == set(range(61)), path
                    assert summary['number_of_rounds'] == 60 and summary['number_of_clients'] == 20, path
                    assert config['seed'] == seed and config['attack_type'] == attack, path
                    assert config['attack_ratio'] == (.4 if scenario == 'iid40' else .2), path
                    assert config['attack_start_round'] == 4, path
                    assert config['attack_strength'] == (5 if attack == 'sign_flip' else 10), path
                    assert config['partitioner'] == ('dirichlet' if scenario == 'noniid20_mixed' else 'iid'), path
                    if scenario == 'noniid20_mixed':
                        assert config['alpha'] == .5, path
                    if method == 'krum':
                        assert config['krum_f'] == (4 if scenario == 'iid40' else 2), path
                    if method == 'trimmed_mean':
                        assert config['trim_ratio'] == .2, path
                    ids = config['malicious_client_ids']
                    if paired_ids is None:
                        paired_ids = ids
                    assert ids == paired_ids, path
                    version = ('unified_v2_candidate' if scenario == 'noniid20_mixed' and
                               (attack, method, seed) == ('sign_flip', 'verashield_fl', 21)
                               else 'historical_final_v2')
                    values = {int(r['round']): float(r['accuracy']) for r in rows}
                    assert all(np.isfinite(v) and 0 <= v <= 1 for v in values.values()), path
                    assert abs(values[60] - summary['final_accuracy']) < 1e-10, path
                    for r in range(61):
                        raw.append(dict(scenario=scenario, attack=attack, method=method,
                                        seed=seed, round=r, accuracy=values[r], version=version))
                    final.append(dict(scenario=scenario, attack=attack, method=method,
                                      seed=seed, accuracy=values[60], version=version))
                    manifest.append(dict(scenario=scenario, attack=attack, method=method, seed=seed,
                                         version=version, source=str(path.relative_to(ROOT)),
                                         results_sha256=hashlib.sha256((path/'results.csv').read_bytes()).hexdigest()))
    assert len(manifest) == 135
    OUT.mkdir(parents=True, exist_ok=True)
    csv_write(OUT/'per_seed_curves.csv', raw)
    csv_write(OUT/'per_seed_final_accuracy.csv', final)
    (OUT/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    aggregate = []
    for scenario in SCENARIOS:
        for attack, _ in ATTACKS:
            for method, label, *_ in METHODS:
                for r in range(61):
                    rr = [x for x in raw if (x['scenario'], x['attack'], x['method'], x['round']) == (scenario, attack, method, r)]
                    assert len(rr) == 3
                    vals = [x['accuracy'] for x in rr]
                    aggregate.append(dict(scenario=scenario, attack=attack, method=method,
                                          round=r, mean_accuracy=float(np.mean(vals)),
                                          sample_sd=float(np.std(vals, ddof=1)), n_seeds=3))
    csv_write(OUT/'mean_curves.csv', aggregate)
    return aggregate


def render(data, scenario, export):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
                         'axes.labelsize':8, 'axes.titlesize':8, 'legend.fontsize':8,
                         'xtick.labelsize':8, 'ytick.labelsize':8,
                         'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.linewidth':.8})
    fig = plt.figure(figsize=(7.16, 2.80), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(.18, 1), hspace=.015, wspace=.08)
    legend = fig.add_subplot(grid[0, :]); legend.axis('off')
    axes = []
    for i, (attack, title) in enumerate(ATTACKS):
        ax = fig.add_subplot(grid[1, i], sharey=axes[0] if axes else None)
        axes.append(ax)
        for j, (method, label, color, style, marker) in enumerate(METHODS):
            rows = [x for x in data if (x['scenario'], x['attack'], x['method']) == (scenario, attack, method)]
            x = np.array([v['round'] for v in rows]); y = np.array([v['mean_accuracy']*100 for v in rows])
            sd = np.array([v['sample_sd']*100 for v in rows])
            assert min(y) >= 0 and max(y) <= 60, (scenario, attack, method)
            # Offset marker sampling only; coordinates are never jittered.
            ax.plot(x, y, label=label, color=color, linestyle=style, linewidth=1.35,
                    marker=marker, markersize=2.8, markeredgewidth=.4,
                    markerfacecolor='white', markevery=(j*2,10),
                    zorder=6 if method=='verashield_fl' else 3)
        ax.axhline(10, color='#555555', linewidth=.7, linestyle=(0,(2,2)), zorder=0)
        ax.axvline(4, color='#777777', linewidth=.7, linestyle='--', zorder=0)
        ax.set_title(f'({chr(97+i)}) {title}', pad=4)
        ax.set_xlim(0,60); ax.set_ylim(0,60)
        ax.set_xticks([0,20,40,60]); ax.set_yticks([0,10,20,30,40,50,60])
        ax.set_xlabel('Federated round', labelpad=2)
        ax.grid(axis='y', color='#DDDDDD', linewidth=.5)
        ax.spines[['top','right']].set_visible(False)
        if i:
            ax.tick_params(axis='y', labelleft=False)
        else:
            ax.set_ylabel('Test accuracy (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    legend.legend(handles, labels, ncol=5, loc='center', frameon=False,
                  handlelength=2.1, columnspacing=1.0, handletextpad=.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    text_issues = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < bounds.x0-1 or box.y0 < bounds.y0-1 or box.x1 > bounds.x1+1 or box.y1 > bounds.y1+1:
                text_issues.append(text.get_text())
    assert not text_issues, text_issues
    stem = OUT / f'fig_{scenario}_accuracy'
    fig.savefig(stem.with_suffix('.png'), dpi=600, facecolor='white')
    # Raster preview and grayscale derive from the same Python-rendered canvas.
    with Image.open(stem.with_suffix('.png')) as image:
        image.convert('L').save(OUT/f'{stem.name}_grayscale.png', dpi=(600,600))
    if export:
        for suffix in ('.pdf','.svg'):
            fig.savefig(stem.with_suffix(suffix), facecolor='white')
    plt.close(fig)


def captions():
    common = ('Global CIFAR-10 test accuracy over federated rounds for five methods using a small '
              'CNN (two convolutional and three fully connected layers), 20 clients and 60 rounds. '
              'Lines show means across seeds 7, 21 and 42; no variability bands are displayed. '
              'Attacks begin at round 4 (vertical '
              'dashed line). Sign-flip strength is 5; random-gradient and scaling strengths are 10. '
              'The horizontal dashed line indicates the 10% random-guessing reference for ten '
              'classes; accuracy alone does not establish constant prediction. Near-chance curves '
              'of FedAvg, Trimmed Mean and FedDMC overlap in some panels. No data jitter or smoothing '
              'is applied. '
              'No client dropout or server tampering is injected. No significance test is displayed.')
    lines = ['# Figure captions and provenance', '',
             'Three independent figures, each 7.16 × 2.80 in; font size 8 pt; PNG 600 dpi; PDF/SVG vector.', '',
             '## Non-IID 20%', '',
             common+' Data use Dirichlet alpha=0.5 with 20% malicious clients.', '',
             '## IID 20% — historical main attack setting', '',
             common+' Data are IID with 20% malicious clients. All curves use the '
             'previously completed historical matrix; no v2 repair result is inserted. This is '
             'an attacked main setting, not an attack-free clean control.', '',
             '## IID 40% — historical stress setting', '',
             common+' Data are IID with 40% malicious clients. '
             'All curves use the historical stress-test matrix; no v2 repair '
             'result is inserted.', '',
             '## Figure contract', '',
             'Purpose: compare model utility and convergence under each attack and scenario; '
             'do not claim universal superiority or equivalent policy versions across scenarios.',
             'Archetype: quantitative grid; panels a–c map to three attacks. Separate figures '
             'prevent mixing IID/main, IID/stress and non-IID/version semantics.',
             'Source manifest lists all 135 inputs and CSV hashes; per_seed_curves.csv retains '
             'all trajectories. mean_curves.csv contains exactly 3 observations per plotted round.',
             'Existing input/result/figure files are preserved. No new training was performed.']
    (OUT/'FIGURE_CAPTIONS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (OUT/'PROVENANCE_NOTES.md').write_text(
        '# Data provenance (separate from figure captions)\n\n'
        'Non-IID 20% is a mixed-version exploratory matrix: only VeraShield-FL '
        '/ sign-flip / seed 21 is unified_v2_candidate; other entries remain historical. '
        'The complete trajectory is replaced. This does not establish revised-policy '
        'performance across three seeds.\n\n'
        'Krum uses f=4 for IID 40%, f=2 for both 20% matrices. Trimmed Mean '
        'uses trim_ratio=0.2, representing total trimming of 20%.\n\n'
        'Mean curves have no displayed variability bands by user request. Sample SD '
        'is retained in mean_curves.csv (ddof=1). Input paths and hashes are in '
        'source_manifest.json.\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--export', action='store_true', help='Export vectors after reviewing raster previews.')
    args = parser.parse_args()
    data = collect()
    captions()
    if not args.prepare_only:
        for scenario in SCENARIOS:
            render(data, scenario, args.export)
    print(OUT)


if __name__ == '__main__':
    main()
