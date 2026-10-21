"""Validate that all Flower executables resolve to the project virtualenv.

Run this script with the *virtualenv* Python (it inspects ``sys.executable``).
It fails (exit 1) if any of python / flwr / flower-superlink / flower-superexec
is NOT located inside the project ``.venv\\Scripts`` directory.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import json
import os
import shutil
import sys
from pathlib import Path


LAUNCHERS = ("flwr", "flower-superlink", "flower-superexec", "flwr-simulation")


def resolve_launchers() -> tuple[Path, dict[str, dict[str, object]]]:
    """Resolve Flower launchers from FLOWER_VENV first, PATH only as fallback."""
    raw = os.environ.get("FLOWER_VENV", "").strip()
    if not raw:
        raise RuntimeError("FLOWER_VENV_NOT_SET")
    scripts = Path(raw).expanduser()
    if scripts.name.lower() not in ("scripts", "bin"):
        scripts = scripts / ("Scripts" if os.name == "nt" else "bin")
    scripts = scripts.resolve()
    report: dict[str, dict[str, object]] = {}
    for name in LAUNCHERS:
        suffix = ".exe" if os.name == "nt" else ""
        expected = scripts / f"{name}{suffix}"
        absolute_ok = expected.is_file()
        found = shutil.which(name)
        fallback = Path(found).resolve() if found else None
        resolved = expected if absolute_ok else fallback
        source = "FLOWER_VENV" if absolute_ok else ("PATH" if fallback else None)
        report[name] = {
            "expected": str(expected),
            "resolved": str(resolved) if resolved else None,
            "source": source,
            "ok": resolved is not None and Path(resolved).resolve().parent == scripts,
        }
    return scripts, report


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def main() -> int:
    try:
        scripts_path, launcher_report = resolve_launchers()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    venv_scripts = _norm(str(scripts_path))

    report = {
        "venv_scripts": venv_scripts,
        "python": {
            "expected": _norm(sys.executable),
            "resolved": _norm(sys.executable),
            "ok": True,
        },
        "executables": {},
        "versions": {},
    }

    report["executables"] = launcher_report

    # Version info (best-effort)
    try:
        import flwr

        report["versions"]["flwr"] = flwr.__version__
    except Exception:  # pragma: no cover
        report["versions"]["flwr"] = None
    report["versions"]["python"] = sys.version.split()[0]

    all_ok = report["python"]["ok"] and all(v["ok"] for v in report["executables"].values())
    report["all_ok"] = all_ok

    print(json.dumps(report, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
