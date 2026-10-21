#!/usr/bin/env python3
"""Run one-time logical TA initialization for cryptographic VL."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pytorchexample.experiment.crypto_vl_keys import initialize_crypto_vl_setup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-dir", required=True)
    parser.add_argument("--clients", required=True, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.clients <= 0:
        parser.error("--clients must be positive")
    manifest = initialize_crypto_vl_setup(
        args.key_dir, client_ids=range(args.clients), force=args.force
    )
    print(json.dumps({
        "key_dir": args.key_dir,
        "setup_id": manifest["setup_id"],
        "registered_clients": args.clients,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
