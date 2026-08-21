#!/usr/bin/env python3
"""Run all methods on the shared SUMO case set and write outputs under each method/data/sumo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmax",
        type=float,
        default=None,
        help="Optional override for maximum parking cost threshold.",
    )
    args = parser.parse_args()

    cmd = [sys.executable, str(ROOT / "data_logging_pipeline.py"), "run-sumo-cases"]
    if args.cmax is not None:
        cmd.extend(["--cmax", str(args.cmax)])

    subprocess.run(cmd, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
