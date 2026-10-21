"""Run the corrected MNIST VeraShield-FL seed-7 joint main experiment."""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import csv, json, math, os, subprocess
from pathlib import Path
from run_joint_main_seed7 import ROOT, METHODS, cmd, now, save

BATCH = ROOT / 'reproductions/mnist_precommit_joint_formal_seed7_20261016_retry'
REFERENCE = ROOT / 'results/joint_precommit_cifar10_seed7_20261015/scenario.json'

def prepare():
    ref = json.loads(REFERENCE.read_text())
    scenario = dict(ref, dataset='MNIST', formal_experiment=True,
                    scenario_id='mnist_precommit_joint_seed7_noniid20_signflip_tamper_drop')
    save(BATCH / 'scenario.json', scenario)
    jobs=[]
    for method in ('verashield_fl',):
        out=BATCH/method
        argv=cmd(method,out,BATCH/'scenario.json') + ['--dataset','mnist','--vl-commit-mode','precommit']
        for flag,val in [('--scenario-id',scenario['scenario_id']),('--experiment',f'mnist_precommit_joint_{method}_s7')]:
            argv[argv.index(flag)+1]=val
        jobs.append((method,out,argv))
    save(BATCH/'plan.json', {'created_utc':now(),'dataset':'MNIST','seed':7,'jobs':[{'method':m,'argv':a} for m,_,a in jobs]})
    return jobs

def validate(method,out):
    curve=list(csv.DictReader((out/'results.csv').open(newline='')))
    assert [int(x['round']) for x in curve]==list(range(61))
    summary=json.loads((out/'summary.json').read_text())
    report={'method':method,'final_accuracy':summary['final_accuracy'],
            'nonfinite_loss_rounds':[int(x['round']) for x in curve if not math.isfinite(float(x['loss']))]}
    if method=='verashield_fl':
        vl=list(csv.DictReader((out/'vl_verification_results.csv').open(newline='')))
        report.update(tamper_applied_rounds=[int(x['round']) for x in vl if int(x['tamper_applied'])],
                      rejection_rounds=[int(x['round']) for x in vl if int(x['candidate_rejected'])],
                      precommit_pass=all(int(x['model_committed'])==int(x['state_committed'])==(int(x['round']) not in (20,40,59)) for x in vl))
    return report

def main():
    if BATCH.exists() and (BATCH/'status.json').exists():
        raise RuntimeError(f'Existing batch preserved: {BATCH}')
    jobs=prepare(); state={'status':'RUNNING','total':len(jobs),'completed':0,'current':None,'started_utc':now(),'results':[]}; save(BATCH/'status.json',state)
    env=dict(os.environ,FLOWER_VENV=_release_sys.prefix,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',FLOWER_CLIENT_RESOURCES_NUM_CPUS='2',FLOWER_CLIENT_RESOURCES_NUM_GPUS='0.0',FLOWER_INIT_ARGS_NUM_GPUS='0')
    try:
        for method,out,argv in jobs:
            state['current']=method; save(BATCH/'status.json',state); out.mkdir(parents=True,exist_ok=True)
            with (BATCH/f'{method}.log').open('w') as log:
                subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=validate(method,out); save(out/'completion_report.json',report); state['results'].append(report); state['completed']+=1; save(BATCH/'status.json',state)
        state.update(status='COMPLETED',current=None,completed_utc=now()); save(BATCH/'status.json',state)
    except Exception as exc:
        state.update(status='FAILED',error=str(exc),failed_utc=now()); save(BATCH/'status.json',state); raise

if __name__=='__main__': main()
