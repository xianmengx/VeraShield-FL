"""Run corrected MNIST joint precommit VeraShield-FL only, seeds 7/21/42."""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

from run_joint_main_seed7 import ROOT, cmd, now, save
from run_joint_precommit_multiseed import scenario_for

csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))

SEEDS = (7, 21, 42)
BATCH = ROOT / 'reproductions/mnist_precommit_joint_multiseed_20261016'


def prepare():
    jobs = []
    for seed in SEEDS:
        out = BATCH / f'seed{seed}'
        scenario = scenario_for(seed)
        scenario.update(dataset='MNIST', formal_experiment=True,
                        scenario_id=f'mnist_precommit_joint_s{seed}_noniid20_signflip_tamper_drop')
        manifest = out / 'scenario.json'
        save(manifest, scenario)
        argv = cmd('verashield_fl', out / 'verashield_fl', manifest)
        for flag, value in [('--seed', seed), ('--scenario-id', scenario['scenario_id']),
                            ('--experiment', f'mnist_precommit_joint_verashield_fl_s{seed}')]:
            argv[argv.index(flag) + 1] = str(value)
        argv += ['--dataset', 'mnist', '--vl-commit-mode', 'precommit']
        save(out / 'registration.json', {
            'created_utc': now(), 'seed': seed, 'method': 'verashield_fl', 'argv': argv,
            'scenario_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'scope': 'corrected VeraShield-FL only; all existing baseline results preserved',
        })
        jobs.append((seed, out, argv))
    return jobs


def validate(out):
    run = out / 'verashield_fl'
    curve = list(csv.DictReader((run / 'results.csv').open(newline='')))
    assert [int(x['round']) for x in curve] == list(range(61))
    vl = list(csv.DictReader((run / 'vl_verification_results.csv').open(newline='')))
    assert [int(x['round']) for x in vl] == list(range(1, 61))
    assert [int(x['round']) for x in vl if int(x['tamper_applied'])] == [20, 40, 59]
    assert [int(x['round']) for x in vl if int(x['candidate_rejected'])] == [20, 40, 59]
    assert all(int(x['model_committed']) == int(x['state_committed']) ==
               (int(x['round']) not in (20, 40, 59)) for x in vl)
    summary = json.loads((run / 'summary.json').read_text())
    return {'status': 'PASS', 'final_accuracy': summary['final_accuracy'],
            'rejected_rounds': [20, 40, 59],
            'scope': 'automatic completion checks; full mechanism audit required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if (BATCH / 'status.json').exists():
        raise RuntimeError(f'Existing batch preserved: {BATCH}')
    jobs = prepare()
    save(BATCH / 'plan.json', {'created_utc': now(), 'dataset': 'MNIST',
         'seeds': list(SEEDS), 'methods': ['verashield_fl'],
         'jobs': [{'seed': s, 'argv': a} for s, _, a in jobs]})
    env = dict(os.environ, FLOWER_VENV=_release_sys.prefix,
               PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               FLOWER_CLIENT_RESOURCES_NUM_CPUS='2', FLOWER_CLIENT_RESOURCES_NUM_GPUS='0.0',
               FLOWER_INIT_ARGS_NUM_GPUS='0')
    for seed, out, argv in jobs:
        with (out / 'preflight.log').open('w') as log:
            subprocess.run(argv + ['--dry-run'], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.check_only:
        print('Preflight PASS: MNIST VeraShield-FL only, seeds 7, 21, 42')
        return
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'total': len(jobs),
             'completed': 0, 'current': None, 'started_utc': now(), 'results': []}
    save(BATCH / 'status.json', state)
    try:
        for seed, out, argv in jobs:
            state['current'] = seed
            save(BATCH / 'status.json', state)
            save(out / 'status.json', {'status': 'RUNNING', 'seed': seed, 'started_utc': now()})
            with (out / 'run.log').open('w') as log:
                subprocess.run(argv, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            report = dict(validate(out), seed=seed)
            save(out / 'verashield_fl/validation_report.json', report)
            save(out / 'status.json', {'status': 'COMPLETED', 'seed': seed,
                                      'completed_utc': now(), 'report': report})
            state['results'].append(report)
            state['completed'] += 1
            save(BATCH / 'status.json', state)
        state.update(status='COMPLETED', current=None, completed_utc=now())
        save(BATCH / 'status.json', state)
    except Exception as exc:
        state.update(status='FAILED', error=str(exc), failed_utc=now())
        save(BATCH / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
