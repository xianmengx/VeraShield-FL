"""Plan or execute a recorded configuration into a fresh reproduction directory."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Recorded run directory containing config.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--key-dir', type=Path, default=ROOT/'local_keys')
    parser.add_argument('--execute', action='store_true', help='Start training; default only prints the plan')
    args = parser.parse_args()
    source = (ROOT/args.run).resolve()
    if not source.is_relative_to(ROOT/'results'):
        parser.error('--run must be beneath this release results/')
    config = json.loads((source/'config.json').read_text(encoding='utf-8'))
    defaults = tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))['tool']['flwr']['app']['config']
    rc = {key: value for key, value in config['run_config'].items() if key in defaults}
    output = (ROOT/(args.output or Path('reproductions')/source.relative_to(ROOT/'results'))).resolve()
    if not output.is_relative_to(ROOT/'reproductions') or output.exists():
        parser.error('Use a fresh --output beneath reproductions/; recorded evidence cannot be overwritten')
    scenario = source/'scenario.json'
    if not scenario.is_file():
        scenario = source.parent/'scenario.json'
    if rc.get('scenario-manifest') and not scenario.is_file():
        parser.error('Recorded scenario is missing')
    rc['output-dir'] = output.as_posix()
    if scenario.is_file():
        rc['scenario-manifest'] = scenario.as_posix()
    if rc.get('vl-backend') == 'crypto':
        rc['vl-key-dir'] = args.key_dir.resolve().as_posix()
        if args.execute and not args.key_dir.is_dir():
            parser.error('Initialize fresh local keys with scripts/init_crypto_vl_keys.py first')
    venv = Path(os.environ.get('FLOWER_VENV', sys.prefix))
    scripts = venv if venv.name.lower() in {'bin', 'scripts'} else venv/('Scripts' if os.name == 'nt' else 'bin')
    executable = scripts/('flwr.exe' if os.name == 'nt' else 'flwr')
    def value(v):
        return json.dumps(v, ensure_ascii=False) if isinstance(v, str) else str(v).lower() if isinstance(v, bool) else str(v)
    run_config = ' '.join(f'{key}={value(v)}' for key, v in rc.items())
    print(json.dumps({'mode': 'execute' if args.execute else 'plan-only', 'source': str(args.run),
                      'num_clients': rc.get('num-clients'), 'rounds': rc.get('num-server-rounds'),
                      'dataset': rc.get('dataset'), 'output': str(output.relative_to(ROOT)),
                      'fresh_keys_required': rc.get('vl-backend') == 'crypto',
                      'runner': str(executable), 'run_config': rc}, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if not executable.is_file():
        parser.error('Flower executable not found in the selected environment')
    environment = dict(os.environ)
    environment.setdefault('HF_HUB_OFFLINE', '1')
    environment.setdefault('HF_DATASETS_OFFLINE', '1')
    environment.setdefault('FLOWER_CLIENT_RESOURCES_NUM_GPUS', '0.0')
    with tempfile.TemporaryDirectory(prefix='verashield-app-') as directory:
        staged = Path(directory)
        shutil.copy2(ROOT/'pyproject.toml', staged/'pyproject.toml')
        shutil.copy2(ROOT/'LICENSE', staged/'LICENSE')
        shutil.copytree(ROOT/'pytorchexample', staged/'pytorchexample', ignore=shutil.ignore_patterns('__pycache__'))
        argv = [str(executable), 'run', str(staged), '--stream', '--run-config', run_config,
                '--federation-config', f'num-supernodes={rc.get("num-clients", 20)} client-resources-num-cpus=2 client-resources-num-gpus=0.0 init-args-num-gpus=0']
        subprocess.run(argv, cwd=ROOT, env=environment, check=True)
    if scenario.is_file() and output.is_dir():
        shutil.copy2(scenario, output/'scenario.json')


if __name__ == '__main__':
    main()
