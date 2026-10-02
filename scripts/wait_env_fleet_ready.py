#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from rl.runtime.envfile import load_runtime_env_file
from rl.runtime.fleet import FleetRunLayout
from rl.runtime.fleet.readiness import readiness_summary


def main() -> int:
    """Poll fleet registry and pool status files until readiness is reached."""
    args = parse_args()
    deadline = time.monotonic() + args.timeout_s

    while True:
        summary = readiness_summary(args)
        if summary["registry_ready"] and summary["ready"] >= summary["min_ready"]:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if time.monotonic() >= deadline:
            print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
            return 1
        if args.verbose:
            print(json.dumps(summary, sort_keys=True), flush=True)
        time.sleep(args.poll_s)


def parse_args() -> argparse.Namespace:
    """Build CLI arguments with Slurm-friendly defaults from the environment."""
    load_runtime_env_file()
    env = os.environ
    layout = FleetRunLayout.from_env(env)
    parser = argparse.ArgumentParser(
        description="Wait for an OSWorld env fleet registry and warm desktop pool readiness."
    )
    parser.add_argument("--run-root", type=Path, default=layout.run_root)
    parser.add_argument("--registry", type=Path, default=layout.registry_path)
    parser.add_argument(
        "--pool-status-dir",
        type=Path,
        default=layout.pool_status_dir,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=(
            Path(env["OSWORLD_DESKTOP_POOL_STATUS_DIR"])
            if "OSWORLD_DESKTOP_POOL_STATUS_DIR" in env
            else None
        ),
    )
    parser.add_argument(
        "--min-ready-sessions",
        type=int,
        default=int(env.get("OSWORLD_DESKTOP_POOL_MIN_READY_TOTAL", "-1")),
        help="Ready sessions required. -1 uses registry metadata.",
    )
    parser.add_argument(
        "--expected-servers",
        type=int,
        default=int(env.get("OSWORLD_EXPECTED_ENV_SERVERS", "0")),
    )
    parser.add_argument(
        "--status-stale-after-s",
        type=float,
        default=float(env.get("OSWORLD_STATUS_STALE_AFTER_S", "120")),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(env.get("OSWORLD_ENV_FLEET_READY_TIMEOUT", "3600")),
    )
    parser.add_argument(
        "--poll-s",
        type=float,
        default=float(env.get("OSWORLD_ENV_FLEET_READY_POLL", "5")),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
