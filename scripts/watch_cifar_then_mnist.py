"""Wait for CIFAR corrected multiseed completion, then start MNIST main batch."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import json, subprocess, time
from pathlib import Path

root=Path(__file__).resolve().parents[1]
cifar=root/'reproductions/joint_precommit_cifar10_multiseed_20261016/status.json'
log=root/'reproductions/mnist_precommit_joint_multiseed_20261016/auto_start.log'
while True:
    if cifar.exists():
        try:
            state=json.loads(cifar.read_text())
            if state.get('status')=='COMPLETED':
                log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('a') as h:
                    subprocess.run([_release_sys.executable,str(root/'scripts/run_mnist_precommit_multiseed.py')],cwd=root,stdout=h,stderr=subprocess.STDOUT,check=True)
                break
            if state.get('status')=='FAILED':
                raise RuntimeError('CIFAR multiseed batch failed; MNIST was not started')
        except json.JSONDecodeError:
            pass
    time.sleep(15)
