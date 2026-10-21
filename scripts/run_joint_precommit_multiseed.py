"""Run corrected CIFAR-10 joint precommit VeraShield-FL for seeds 21 and 42."""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import csv, hashlib, json, os, subprocess, sys
from pathlib import Path
from run_joint_main_seed7 import ROOT, PY, cmd, now, save

csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))

SEEDS = (21, 42)
ROOT_BATCH = ROOT / 'reproductions/joint_precommit_cifar10_multiseed_20261016'
REFERENCE = ROOT / 'results/joint_precommit_cifar10_seed7_20261015/scenario.json'

def scenario_for(seed):
    s = json.loads(REFERENCE.read_text())
    s['scenario_id'] = f'joint_precommit_cifar10_noniid20_signflip_tamper_drop_s{seed}'
    s['seed'] = seed
    s['partition_seed'] = seed
    s['model_seed'] = seed
    return s

def make_cmd(seed, out, manifest):
    c = cmd('verashield_fl', out, manifest)
    def replace(flag, value): c[c.index(flag)+1] = str(value)
    replace('--seed', seed)
    replace('--scenario-id', json.loads(manifest.read_text())['scenario_id'])
    replace('--experiment', f'joint_precommit_cifar10_verashield_fl_s{seed}')
    return c + ['--dataset', 'cifar10', '--vl-commit-mode', 'precommit']

def run_one(seed):
    out = ROOT_BATCH / f'seed{seed}'
    if (out/'verashield_fl/summary.json').exists():
        report = validate(seed, out)
        state = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        state.update(status='COMPLETED', seed=seed, completed_utc=now(), report=report,
                     reused_completed_training=True)
        save(out/'status.json', state)
        return report
    if (out/'run.log').exists():
        raise RuntimeError(f'Incomplete existing training preserved: {out}')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'scenario.json'
    save(manifest, scenario_for(seed))
    argv = make_cmd(seed, out / 'verashield_fl', manifest)
    save(out / 'registration.json', {'created_utc': now(), 'argv': argv,
        'scenario_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ('pytorchexample', 'scripts')
            for p in sorted((ROOT/directory).rglob('*.py'))},
        'scope': 'corrected VeraShield-FL only; seed 7 preserved separately'})
    env = dict(os.environ, FLOWER_VENV=_release_sys.prefix,
               PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               FLOWER_CLIENT_RESOURCES_NUM_CPUS='2', FLOWER_CLIENT_RESOURCES_NUM_GPUS='0.0',
               FLOWER_INIT_ARGS_NUM_GPUS='0')
    state = {'status':'RUNNING','seed':seed,'started_utc':now()}
    save(out/'status.json', state)
    with (out/'run.log').open('w') as log:
        subprocess.run(argv + ['--dry-run'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = validate(seed, out)
    state.update(status='COMPLETED', completed_utc=now(), report=report)
    save(out/'status.json', state)
    return report

def validate(seed, out):
    vl = list(csv.DictReader((out/'verashield_fl/vl_verification_results.csv').open()))
    assert [int(x['round']) for x in vl] == list(range(1,61))
    assert [int(x['round']) for x in vl if int(x['candidate_rejected'])] == [20,40,59]
    assert [int(x['round']) for x in vl if int(x['tamper_applied'])] == [20,40,59]
    assert all(int(x['model_committed']) == int(x['state_committed']) == (int(x['round']) not in (20,40,59)) for x in vl)
    summary = json.loads((out/'verashield_fl/summary.json').read_text())
    report = {'status':'PASS','seed':seed,'final_accuracy':summary['final_accuracy'],
              'rejected_rounds':[20,40,59],
              'scope':'automatic completion checks; full mechanism audit required'}
    save(out/'verashield_fl/validation_report.json', report)
    return report

def main():
    ROOT_BATCH.mkdir(parents=True, exist_ok=True)
    if '--check-only' in sys.argv:
        print('Preflight PASS:', SEEDS)
        return
    state={'status':'RUNNING','total':len(SEEDS),'completed':0,'current':None,'started_utc':now(),'results':[]}
    save(ROOT_BATCH/'status.json', state)
    try:
        for seed in SEEDS:
            state['current']=seed; save(ROOT_BATCH/'status.json', state)
            report=run_one(seed); state['results'].append(report); state['completed']+=1
            save(ROOT_BATCH/'status.json', state)
        state.update(status='COMPLETED', current=None, completed_utc=now()); save(ROOT_BATCH/'status.json', state)
    except Exception as exc:
        state.update(status='FAILED', error=str(exc), failed_utc=now()); save(ROOT_BATCH/'status.json', state); raise

if __name__ == '__main__': main()
