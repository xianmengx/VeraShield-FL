"""Validate the local publication tree without training or external services."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    failures = []
    latest = rows(ROOT/'manifests/latest_180_runs.csv')
    assert len(latest) == 180
    for run in latest:
        path = ROOT/run['path']
        data = rows(path)
        assert len(data) == 61 and [int(row['round']) for row in data] == list(range(61)), path
        assert all(math.isfinite(float(row['accuracy'])) and 0 <= float(row['accuracy']) <= 1 for row in data), path
        assert checksum(path) == run['sha256'], path
    figure_inputs = 0
    for manifest in sorted((ROOT/'figures').glob('*/source_manifest.csv')):
        for row in rows(manifest):
            relative = row['source'].replace('\\', '/')
            path = ROOT/relative if relative.startswith('results/') else ROOT/'results'/relative
            targets = [(path, row['sha256'])] if 'sha256' in row else [
                (path/name, row[field]) for name, field in [('results.csv', 'results_csv_sha256'),
                    ('config.json', 'config_json_sha256'), ('scenario.json', 'scenario_json_sha256'),
                    ('summary.json', 'summary_json_sha256')] if row.get(field)]
            for target, expected in targets:
                assert checksum(target) == expected, target
                figure_inputs += 1
    code = list((ROOT/'pytorchexample').rglob('*.py')) + list((ROOT/'scripts').glob('*.py')) + list((ROOT/'tests').glob('*.py'))
    for path in code:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path.relative_to(ROOT)))
    sensitive = re.compile(rb'(?:[A-Za-z]:[/\\]+Users[/\\]+[^/\\\s\"]+|/(?:mnt/[a-z]/Users/[^/\\\s\"]+|root/FlowerStudy|mnt/[a-z]/aaa/flowerstudy))', re.I)
    raw_credentials = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|sk-proj-[A-Za-z0-9_-]{20,}')
    total = 0
    largest = 0
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(p in {'.git', '__pycache__', '.pytest_cache'} for p in path.parts):
            continue
        total += 1
        largest = max(largest, path.stat().st_size)
        if path.suffix.lower() in {'.py', '.json', '.jsonl', '.csv', '.log', '.txt', '.md', '.toml', '.sha256'}:
            content = path.read_bytes()
            if sensitive.search(content):
                failures.append({'path': path.relative_to(ROOT).as_posix(), 'kind': 'machine-path'})
            # Regex literals in the validator itself are not stored credentials.
            if path.name != 'verify_release.py' and raw_credentials.search(content):
                failures.append({'path': path.relative_to(ROOT).as_posix(), 'kind': 'credential-pattern'})
        if path.suffix.lower() == '.json':
            obj = json.loads(path.read_text(encoding='utf-8-sig'))
            def visit(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key in {'private_key', 'private_keys', 'client_private_keys', 'as_private_keys', 'client_prf_keys', 'hash_key'} and child:
                            failures.append({'path': path.relative_to(ROOT).as_posix(), 'kind': 'private-material-field'})
                        visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)
            visit(obj)
    result = {'latest_runs': len(latest), 'figure_input_hash_checks': figure_inputs,
              'python_files_parsed': len(code), 'files_scanned': total,
              'largest_file_bytes': largest, 'findings': failures,
              'training_executed': False, 'privacy_scan_scope': 'configured path, credential and JSON-private-field patterns'}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
