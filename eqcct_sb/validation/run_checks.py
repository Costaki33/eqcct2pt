#!/usr/bin/env python3
"""
Run parity and SeisBench smoke tests in one process (for CI or local sanity).

  PYTHONPATH=... python -m eqcct_sb.validation.run_checks
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root)}
    steps = [
        [sys.executable, "-m", "eqcct_sb.validation.parity_p_model"],
        [sys.executable, "-m", "eqcct_sb.validation.seisbench_p_model", "--dataset", "stead"],
    ]
    for cmd in steps:
        r = subprocess.run(cmd, cwd=str(root), env=env)
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
