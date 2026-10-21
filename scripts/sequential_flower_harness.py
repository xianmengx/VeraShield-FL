"""Auditable, sequential one-round Flower lifecycle test for Windows."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import json
import os
import subprocess
import sys
import time
import re
from datetime import datetime, timezone
from pathlib import Path


APP = Path(__file__).resolve().parents[1]
PYTHON = APP / ".venv" / "Scripts" / "python.exe"
VENV_PREFIX = str((APP / ".venv").resolve()).lower()
AUDIT = APP / "results" / "final_v2" / "N20_main_pilot" / "process_audit"
SMOKE = AUDIT / "post_run_cleanup_smoke"
MANIFEST = APP / "results" / "final_v2" / "N20_main_pilot" / "scenario_D0" / "scenario.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ps_processes() -> list[dict]:
    script = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(flower-.+|flwr-.+|ray|raylet)\\.exe$' } | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CreationDate | ConvertTo-Json -Compress"
    p = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not p.stdout.strip():
        return []
    try:
        data = json.loads(p.stdout)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return []


def project_processes() -> list[dict]:
    return [p for p in ps_processes() if str(p.get("ExecutablePath") or "").lower().startswith(VENV_PREFIX)]


def ports_for(pids: set[int]) -> list[int]:
    if not pids:
        return []
    lines = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.splitlines()
    ports = set()
    for line in lines:
        parts = line.split()
        if len(parts) >= 5 and parts[-1].isdigit() and int(parts[-1]) in pids:
            try:
                ports.add(int(parts[1].rsplit(":", 1)[-1]))
            except ValueError:
                pass
    return sorted(ports)


def snapshot(label: str) -> dict:
    ps = project_processes()
    pids = {int(p["ProcessId"]) for p in ps if str(p.get("ProcessId", "")).isdigit()}
    return {"label": label, "timestamp": now(), "processes": ps, "ports": ports_for(pids)}


def cleanup(timeout: int = 90) -> dict:
    before = project_processes()
    pids = {int(p["ProcessId"]) for p in before if str(p.get("ProcessId", "")).isdigit()}
    ports_before = ports_for(pids)
    steps = [{"state": "PROCESS_SCAN", "timestamp": now(), "status": "residue" if pids else "clean", "pids": sorted(pids)}]
    grace_deadline = time.monotonic() + min(30, timeout)
    while time.monotonic() < grace_deadline:
        left = project_processes()
        if not left:
            steps.append({"state": "GRACEFUL_WAIT", "timestamp": now(), "status": "clean"})
            return {"before": before, "terminated_pids": [], "remaining": [], "ports_before": ports_before, "ports_after": [], "complete": True, "timeout_s": timeout, "steps": steps}
        time.sleep(2)
    before_cleanup = project_processes()
    pids = {int(p["ProcessId"]) for p in before_cleanup if str(p.get("ProcessId", "")).isdigit()}
    steps.append({"state": "GRACEFUL_WAIT", "timestamp": now(), "status": "residue", "pids": sorted(pids)})
    for pid in sorted(pids):
        subprocess.run(["powershell.exe", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -ErrorAction SilentlyContinue"], capture_output=True)
    steps.append({"state": "PROJECT_PROCESS_CLEANUP", "timestamp": now(), "status": "graceful_requested", "pids": sorted(pids)})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        left = project_processes()
        if not left:
            steps.append({"state": "PROCESS_SCAN", "timestamp": now(), "status": "clean", "elapsed_s": round(timeout - max(0, deadline-time.monotonic()), 2)})
            return {"before": before, "terminated_pids": sorted(pids), "remaining": [], "ports_before": ports_before, "ports_after": [], "complete": True, "timeout_s": timeout, "steps": steps}
        time.sleep(2)
    left = project_processes()
    for p in left:
        pid = int(p["ProcessId"])
        subprocess.run(["powershell.exe", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"], capture_output=True)
    time.sleep(2)
    remaining = project_processes()
    steps.append({"state": "CLEANUP_TIMEOUT", "timestamp": now(), "status": "blocked", "remaining": remaining})
    return {"before": before, "terminated_pids": sorted(pids), "remaining": remaining, "ports_before": ports_before, "ports_after": ports_for({int(p["ProcessId"]) for p in remaining if str(p.get("ProcessId", "")).isdigit()}), "complete": not remaining, "timeout_s": timeout, "steps": steps}


def run_one(name: str, aggregation: str) -> dict:
    out = SMOKE / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), str(APP / "scripts" / "run_experiment.py"), "--rounds", "1", "--clients", "20", "--seed", "42", "--fraction-train", "1.0", "--dropout-rate", "0", "--partitioner", "iid", "--attack-enabled", "1", "--attack-type", "sign_flip", "--attack-ratio", "0.2", "--malicious-selection-seed", "42", "--attack-mode", "persistent", "--ssdf-enabled", "0", "--dropout-timing", "post_share", "--share-n", "20", "--share-t", "16", "--threshold-controller-enabled", "0", "--normal-threshold", "16", "--emergency-threshold", "12", "--aggregation-method", aggregation, "--krum-f", "4", "--trim-ratio", "0.2", "--scenario-id", "scenario_D0", "--scenario-manifest", str(MANIFEST), "--output-dir", str(out), "--experiment", f"lifecycle_{name}"]
    started = now()
    initial = snapshot("START")
    proc = subprocess.Popen(cmd, cwd=APP, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    try:
        output, _ = proc.communicate(timeout=900)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
        timed_out = True
    (out / "flower.log").write_text(output, encoding="utf-8")
    run_ids = re.findall(r"run_id `([^`]+)`|run ([0-9]{8,})", output)
    run_id = next((a or b for a,b in run_ids), "") if run_ids else ""
    observed_ports = sorted({int(x) for x in re.findall(r"(?:127\.0\.0\.1|localhost):([0-9]+)", output) if int(x) > 0})
    completed = proc.returncode == 0 and not timed_out and "Strategy execution finished" in output
    cli_exited = proc.returncode is not None
    cleanup_info = cleanup()
    pids = {int(p["ProcessId"]) for p in project_processes() if str(p.get("ProcessId", "")).isdigit()}
    ports_before = sorted(set(cleanup_info.get("ports_before", [])) | set(observed_ports))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and ports_for(pids):
        time.sleep(2)
        pids = {int(p["ProcessId"]) for p in project_processes() if str(p.get("ProcessId", "")).isdigit()}
    ports_after = ports_for(pids)
    if observed_ports:
        netstat = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        ports_after = [p for p in observed_ports if re.search(rf":{p}\\s+.*LISTENING\\s+\\d+", netstat, re.IGNORECASE)]
    success = completed and cli_exited and all((out / f).exists() for f in ("config.json", "results.csv", "summary.json")) and cleanup_info["complete"] and not ports_after
    return {"run_id": run_id, "name": name, "method": aggregation, "command": cmd, "start_time": started, "end_time": now(), "pid": proc.pid, "executable": str(PYTHON), "parent_pid": os.getpid(), "initial_project_processes": initial, "exit_code": proc.returncode, "timed_out": timed_out, "computation_status": "PASS" if completed else "FAIL", "cli_exit_status": "PASS" if cli_exited else "FAIL", "cleanup": cleanup_info, "cleanup_timeout": cleanup_info.get("timeout_s", 90), "remaining_processes": cleanup_info.get("remaining", []), "port": ports_before, "port_release": not ports_after, "lifecycle_status": "CLEAN" if success else "BLOCKED", "run_success": success}


def main() -> int:
    SMOKE.mkdir(parents=True, exist_ok=True)
    results = []
    first = run_one("test_run_1_fedavg", "fedavg")
    results.append(first)
    (SMOKE / "run1_fedavg.json").write_text(json.dumps(first, indent=2), encoding="utf-8")
    if not first["run_success"]:
        print("LIFECYCLE_BLOCKED: run 1 failed or cleanup incomplete")
        return 1
    second = run_one("test_run_2_krum", "krum")
    results.append(second)
    (SMOKE / "run2_krum.json").write_text(json.dumps(second, indent=2), encoding="utf-8")
    report = ["# Post-Run Cleanup Harness Report", "", f"结论：**{'PASS' if first['run_success'] and second['run_success'] else 'BLOCKED'}**", "", "| Run | Method | Computation | CLI exit | Cleanup | Port release | Lifecycle |", "|---|---|---|---|---|---|---|"]
    for r in (first, second):
        report.append(f"| {r['name']} | {r['method']} | {r['computation_status']} | {r['cli_exit_status']} | {'PASS' if r['cleanup']['complete'] else 'FAIL'} | {'PASS' if r['port_release'] else 'FAIL'} | {r['lifecycle_status']} |")
    report += ["", "两次运行均为 N=20 / D0 / seed=42 / 1 round / CIFAR-10 / IID / sign-flip 0.2；Run 2 使用 Krum f=4。", "每个 run 的详细 PID、父 PID、可执行路径、时间戳、清理步骤和端口记录见 `run1_fedavg.json` 与 `run2_krum.json`。", "", "未启动 N20 Main Pilot 或其他正式实验。"]
    (SMOKE / "POST_RUN_CLEANUP_HARNESS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0 if second["run_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
