"""Pilot five-method joint threat experiment (seed 7)."""
from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import csv, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PY=_release_sys.executable
BATCH=ROOT/'reproductions/final_v7/joint_main_seed7'
METHODS=('verashield_fl','fedavg','krum','trimmed_mean','feddmc')

def now(): return datetime.now(timezone.utc).isoformat()
def save(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2)+'\n')

def make_manifest():
 p=BATCH/'scenario.json'
 if p.exists(): return p
 rng=np.random.default_rng([42]); malicious=sorted(int(x) for x in rng.choice(20,4,replace=False))
 drop_first=[16,17,18,19]; drop_final=drop_first+[12,13,14,15]
 rounds=[]
 for r in range(1,61):
  unavailable=[] if r<=20 else drop_first if r<=40 else drop_final
  rounds.append({'round':r,'selected_clients':list(range(20)),'unavailable_ids':unavailable,
                 'dropped_clients':unavailable,'malicious_clients':malicious})
 save(p,{'scenario_id':'joint_main_seed7_noniid20_signflip_tamper_drop','seed':7,
  'num_clients':20,'num_rounds':60,'dataset':'CIFAR-10','partitioner':'dirichlet','alpha':0.5,
  'partition_seed':7,'model_seed':7,'attack_enabled':1,'attack_type':'sign_flip',
  'attack_strength':5.0,'attack_start_round':4,'attack_ratio':0.2,
  'malicious_selection_seed':42,'malicious_ids':malicious,'dropout_timing':'post_share',
  'dropout_schedule':'0% rounds 1-20; 20% rounds 21-40; 40% rounds 41-60',
  'per_round':rounds,'training_config':{'local_epochs':1,'learning_rate':0.1,'batch_size':32,'lr_schedule':'cosine','lr_min':0.001}})
 return p

def cmd(method,out,scen):
 c=[PY,'scripts/run_experiment.py','--seed','7','--rounds','60','--clients','20','--fraction-train','1.0',
 '--local-epochs','1','--learning-rate','0.1','--lr-schedule','cosine','--lr-min','0.001',
 '--partitioner','dirichlet','--alpha','0.5','--attack-enabled','1','--attack-type','sign_flip',
 '--attack-ratio','0.2','--attack-strength','5','--attack-mode','persistent','--attack-start-round','4',
 '--malicious-selection-seed','42','--dropout-rate','0','--dropout-timing','post_share',
 '--scenario-id','joint_main_seed7_noniid20_signflip_tamper_drop','--scenario-manifest',str(scen),
 '--vl-tamper','0','--vl-tamper-rounds','20,40,59','--experiment',f'joint_main_{method}_s7','--output-dir',str(out)]
 if method=='verashield_fl': c += ['--ssdf-enabled','1','--ssdf-backend','logical_dual_server','--ssdf-detector-mode','v2_full',
  '--ssdf-evidence-policy','unified_v2_candidate','--ssdf-norm-k','2.70','--secret-recovery-enabled','1',
  '--share-n','20','--share-t','16','--normal-threshold','16','--emergency-threshold','12','--dynamic-weight','1',
  '--threshold-controller-enabled','1','--vl-verification','1','--vl-enforce','1','--vl-backend','crypto',
  '--vl-key-dir',str(_RELEASE_ROOT / "local_keys"),'--aggregation-method','fedavg']
 else: c += ['--ssdf-enabled','0','--secret-recovery-enabled','0','--threshold-controller-enabled','0','--dynamic-weight','0',
  '--vl-verification','0','--vl-enforce','0','--vl-backend','linear','--aggregation-method',method,'--krum-f','2','--trim-ratio','0.2']
 return c

def audit(out,method):
 s=json.loads((out/'summary.json').read_text()); assert s['number_of_rounds']==60
 result={'method':method,'final_accuracy':s['final_accuracy']}
 if method=='verashield_fl':
  rows=list(csv.DictReader((out/'vl_verification_results.csv').open()))
  applied=[int(r['round']) for r in rows if int(float(r.get('tamper_applied',0) or 0))]
  checks=[r for r in rows if int(r['round']) in (21,41,60)]
  result.update({'tamper_applied':applied,'tamper_detected':all(float(r['verification_success'])==0 and float(r.get('rollback_applied',0))==1 for r in checks),
   'verification_failure_rounds':[int(r['round']) for r in rows if float(r['verification_success'])==0]})
  rec=list(csv.DictReader((out/'recovery_statistics.csv').open()))
  result['recovery_attempts']=sum(int(r['recovery_attempted']) for r in rec); result['recovery_successes']=sum(int(r['recovery_attempted']) and int(r['recovery_success']) for r in rec)
  result['threshold_modes']= {m:sum(r['threshold_mode']==m for r in rec)//20 for m in ('normal','emergency','failure')}
 return result

def main():
 BATCH.mkdir(parents=True,exist_ok=True); scen=make_manifest(); jobs=[]
 for m in METHODS: jobs.append((m,BATCH/m,cmd(m,BATCH/m,scen)))
 save(BATCH/'plan.json',{'created_utc':now(),'settings':'Non-IID alpha=.5; sign_flip strength=5; tamper 20,40,59; staged dropout; thresholds 16/12','jobs':[{'method':m,'argv':c} for m,_,c in jobs]})
 if '--check-only' in sys.argv:
  print('CHECK PASS: 5 joint-threat seed-7 jobs'); return
 if (BATCH/'batch.lock').exists(): raise RuntimeError('batch already running')
 (BATCH/'batch.lock').write_text(str(os.getpid())); state={'status':'RUNNING','total':5,'completed':0,'results':[],'current':None,'started_utc':now()}; save(BATCH/'status.json',state)
 env=dict(os.environ,FLOWER_VENV=_release_sys.prefix,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',FLOWER_CLIENT_RESOURCES_NUM_CPUS='2',FLOWER_CLIENT_RESOURCES_NUM_GPUS='0.0',FLOWER_INIT_ARGS_NUM_GPUS='0')
 try:
  for m,out,c in jobs:
   state['current']=m; save(BATCH/'status.json',state); out.mkdir(parents=True,exist_ok=True)
   with (BATCH/f'{m}.log').open('ab',buffering=0) as log: rc=subprocess.run(c,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
   if rc: raise RuntimeError(f'{m} failed rc={rc}')
   r=audit(out,m); save(out/'validation_report.json',r); state['results'].append(r); state['completed']+=1; save(BATCH/'status.json',state)
  state.update(status='COMPLETED',current=None,completed_utc=now()); save(BATCH/'status.json',state); save(BATCH/'REPORT.json',state)
 except Exception as e:
  state.update(status='FAILED',error=str(e),failed_utc=now()); save(BATCH/'status.json',state); raise
 finally: (BATCH/'batch.lock').unlink(missing_ok=True)
if __name__=='__main__': main()
