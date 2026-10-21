"""Run eight missing v2 main-method experiments, then reuse 36 fixed baselines."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from audit_noniid20_multiseed import run_dir, scenario_path, METHODS, LABELS
from replay_unified_evidence import replay

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'reproductions/final_v6/noniid20_unified_v2_gapfill'
REUSED = ROOT / 'results/final_v6/unified_noniid20_signflip_s21'
ATTACKS = {'sign_flip': 5, 'random_gradient': 10, 'scaling': 10}
JOBS = [(a, s) for a in ATTACKS for s in (7, 21, 42) if (a, s) != ('sign_flip', 21)]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main_dir(attack, seed):
    return (REUSED if (attack, seed) == ('sign_flip', 21)
            else BATCH / f'{attack}_s{seed}') / 'verashield_fl'


def audit(path):
    summary = read(path/'summary.json')
    config = read(path/'config.json')
    assert summary['number_of_rounds'] == 60 and summary['number_of_clients'] == 20
    assert config['ssdf_evidence_policy'] == 'unified_v2_candidate'
    assert set(int(x['round']) for x in rows(path/'results.csv')) == set(range(61))
    metrics = read(path/'detection_metrics_active.json')
    detection = rows(path/'detection_statistics.csv')
    trace = [json.loads(x) for x in (path/'protocol_trace.jsonl').read_text().splitlines()]
    vl = rows(path/'vl_verification_results.csv')
    contamination = sum(x['anchor_admitted']=='1' and x['is_malicious_gt']=='1'
                        and int(x['round']) >= 4 for x in detection)
    empty = sum(not x['accepted_set'] for x in trace)
    failures = sum(float(x['verification_success']) != 1 for x in vl)
    playback = replay(path/'detection_statistics.csv')
    gate = (metrics['recall'] >= .9 and metrics['fpr'] <= .05 and not contamination
            and not empty and not failures and len(trace)==60 and len(vl)==60
            and playback['gate_pass'])
    result = dict(status='COMPLETED', gate_pass=gate, metrics=metrics,
                  final_accuracy=summary['final_accuracy'], rounds=len(trace),
                  crypto_failures=failures, empty_accepted_rounds=empty,
                  contaminated_anchor_rows=contamination, replay=playback)
    return result


def preflight():
    template = read(REUSED/'validation_registration.json')
    # Preserve the exact detector/training code used by the reused ninth run.
    for relative, digest in template['source_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
    assert read(REUSED/'validation_report.json')['gate_pass']
    for attack, strength in ATTACKS.items():
        for seed in (7, 21, 42):
            scenario = read(scenario_path(seed, attack))
            assert (scenario['seed'], scenario['attack_type'], scenario['attack_strength'],
                    scenario['partitioner'], scenario['alpha'], scenario['attack_ratio'],
                    scenario['attack_start_round']) == (seed, attack, strength, 'dirichlet', .5, .2, 4)
            for method in METHODS[1:]:
                path = run_dir(seed, attack, method)
                config = read(path/'config.json')
                assert config['seed'] == seed and config['attack_type'] == attack
                assert config['partitioner'] == 'dirichlet' and config['alpha'] == .5
                assert config['attack_strength'] == strength and config['attack_ratio'] == .2
                assert config['malicious_client_ids'] == scenario['malicious_ids']
                assert read(path/'summary.json')['number_of_rounds'] == 60
    return template


def summarize(reports):
    raw = []
    for attack in ATTACKS:
        for method in METHODS:
            for seed in (7, 21, 42):
                path = main_dir(attack,seed) if method=='verashield_fl' else run_dir(seed,attack,method)
                summary = read(path/'summary.json')
                raw.append(dict(attack=attack, method=method, seed=seed,
                                accuracy=summary['final_accuracy'], source=str(path),
                                version='unified_v2_candidate' if method=='verashield_fl' else 'reused_baseline'))
    out = BATCH/'summary'; out.mkdir(exist_ok=True)
    with (out/'per_seed_results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(raw[0])); writer.writeheader(); writer.writerows(raw)
    text = ['# Non-IID 20% unified v2 five-method comparison', '',
            'Seeds 7/21/42, 20 clients, 60 rounds, alpha=0.5, attacks from round 4.',
            'All nine VeraShield-FL runs use unified_v2_candidate; four baseline methods are reused.',
            'Sign-flip strength 5; random-gradient/scaling strength 10. Krum f=2, total trim ratio=0.2.',
            'Mean ± sample SD. No dropout/server tampering; no claim of held-out validation.', '',
            '|Attack|'+'|'.join(LABELS[m] for m in METHODS)+'|', '|---|'+'---:|'*5]
    for attack in ATTACKS:
        vals=[]
        for method in METHODS:
            x=[r['accuracy']*100 for r in raw if r['attack']==attack and r['method']==method]
            vals.append(f'{statistics.mean(x):.2f} ± {statistics.stdev(x):.2f}%')
        text.append('|'+attack+'|'+'|'.join(vals)+'|')
    text.extend(['', '|Attack|Seed|Recall|FPR|Gate|', '|---|---:|---:|---:|---|'])
    for (attack,seed), report in reports.items():
        m=report['metrics']
        text.append(f'|{attack}|{seed}|{m["recall"]:.4f}|{m["fpr"]:.4f}|{report["gate_pass"]}|')
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    template=preflight()
    print('Preflight PASS: eight jobs, one reused v2 run, 36 baseline runs.',flush=True)
    if args.check_only:
        return
    BATCH.mkdir(parents=True,exist_ok=True)
    lock=BATCH/'batch.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.close(fd)
    reports={('sign_flip',21):audit(main_dir('sign_flip',21))}
    state=dict(status='RUNNING',pid=os.getpid(),started_utc=now(),new_jobs_total=8,
               new_jobs_completed=0, reused_jobs=1, current=None)
    try:
        env=dict(os.environ,FLOWER_VENV=_release_sys.prefix,
                 PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                 FLOWER_CLIENT_RESOURCES_NUM_CPUS='2',FLOWER_CLIENT_RESOURCES_NUM_GPUS='0.0',
                 FLOWER_INIT_ARGS_NUM_GPUS='0')
        for attack,seed in JOBS:
            preflight()  # Abort if detector code changes between runs.
            directory=main_dir(attack,seed); job=directory.parent
            state['current']=dict(attack=attack,seed=seed);save(BATCH/'status.json',state)
            if (directory/'summary.json').exists():
                report=audit(directory)
            else:
                if directory.exists():
                    raise RuntimeError(f'Incomplete output exists; preserve and audit before retry: {directory}')
                cmd=list(template['argv']);cmd[0]=sys.executable
                substitutions={'--seed':str(seed),'--attack-type':attack,'--attack-strength':str(ATTACKS[attack]),
                               '--scenario-id':read(scenario_path(seed,attack))['scenario_id'],
                               '--scenario-manifest':str(scenario_path(seed,attack)),
                               '--experiment':f'unified_v2_noniid20_{attack}_s{seed}',
                               '--output-dir':str(directory)}
                for key,value in substitutions.items():
                    cmd[cmd.index(key)+1]=value
                save(job/'registration.json',dict(policy='unified_v2_candidate',argv=cmd,
                     source_sha256=template['source_sha256'],parameters=template['parameters'],
                     gates=template['gates'],started_utc=now()))
                print(f'START {attack} seed={seed}',flush=True)
                with (job/'validation.log').open('ab',buffering=0) as log:
                    result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'{attack} seed={seed}: exit={result.returncode}')
                report=audit(directory)
            save(job/'validation_report.json',report)
            reports[(attack,seed)]=report
            state['new_jobs_completed']+=1
            save(BATCH/'status.json',state)
            print(f'DONE {attack} seed={seed} gate={report["gate_pass"]}',flush=True)
        summarize(reports)
        state.update(status='COMPLETED',current=None,completed_utc=now(),
                     all_main_gates_pass=all(x['gate_pass'] for x in reports.values()))
        save(BATCH/'status.json',state)
    except Exception as exc:
        state.update(status='FAILED',error=str(exc),failed_utc=now())
        save(BATCH/'status.json',state)
        raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
