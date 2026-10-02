#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from rl.runtime.envfile import load_runtime_env_file
from rl.runtime.fleet import FleetRunLayout, read_registry
from rl.runtime.paths import prime_rl_root, repo_root


def main() -> int:
    import tomli_w

    args = parse_args()
    registry = read_registry(args.registry)
    if not registry.servers:
        raise ValueError(f"registry has no env servers: {args.registry}")

    layout = resolve_layout(args, registry.metadata)
    output = args.output or layout.prime_rl_config_path
    output_dir = args.output_dir or layout.prime_rl_output_dir
    total_workers = sum(server.num_workers for server in registry.servers)

    config = load_config(args.base_config)
    configure_external_fleet(
        config,
        metadata=registry.metadata,
        output_dir=output_dir,
        max_inflight_rollouts=max(
            total_workers * args.inflight_per_worker,
            args.min_inflight_rollouts,
        ),
        rollout_timeout=args.rollout_timeout,
        max_retries=args.max_retries,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        tomli_w.dump(config, file)
    validate_prime_rl_config(output)
    print(output)
    return 0


def parse_args() -> argparse.Namespace:
    load_runtime_env_file()
    layout = FleetRunLayout.from_env(os.environ)
    parser = argparse.ArgumentParser(
        description="Render and validate a PrimeRL config for an OSWorld fleet."
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--run-id", default=layout.run_id)
    parser.add_argument("--run-base", type=Path, default=layout.run_base)
    parser.add_argument("--registry", type=Path, default=layout.registry_path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rollout-timeout", type=float, default=3600.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--inflight-per-worker", type=int, default=1)
    parser.add_argument("--min-inflight-rollouts", type=int, default=1)
    return parser.parse_args()


def resolve_layout(
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
) -> FleetRunLayout:
    fallback = FleetRunLayout.for_run(
        run_id=args.run_id,
        run_base=args.run_base,
        registry_path=args.registry,
    )
    return FleetRunLayout.from_metadata(metadata, fallback=fallback) or fallback


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def configure_external_fleet(
    config: dict[str, Any],
    *,
    metadata: Mapping[str, Any],
    output_dir: Path,
    max_inflight_rollouts: int,
    rollout_timeout: float,
    max_retries: int,
) -> None:
    orchestrator = require_mapping(config, "orchestrator")
    train = orchestrator.setdefault("train", {})
    if not isinstance(train, dict):
        raise ValueError("base config orchestrator.train must be a table")
    train["env"] = [
        external_env_config(
            metadata,
            rollout_timeout=rollout_timeout,
            max_retries=max_retries,
        )
    ]
    group_size = orchestrator.get("group_size", 1)
    if not isinstance(group_size, int) or group_size < 1:
        raise ValueError(
            "base config orchestrator.group_size must be a positive integer"
        )
    orchestrator["max_inflight_rollouts"] = max(max_inflight_rollouts, group_size)
    config["output_dir"] = str(output_dir)
    absolutize_slurm_template_path(config)


def absolutize_slurm_template_path(config: dict[str, Any]) -> None:
    slurm = config.get("slurm")
    if not isinstance(slurm, dict) or "template_path" not in slurm:
        return
    path = Path(slurm["template_path"])
    if not path.is_absolute():
        path = repo_root() / path
    slurm["template_path"] = str(path.resolve())


def external_env_config(
    metadata: Mapping[str, Any],
    *,
    rollout_timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    env_id = required_string(metadata, "env_id")
    task_base_path = required_string(metadata, "task_base_path")
    harness_raw = metadata.get("harness")
    if not isinstance(harness_raw, Mapping):
        raise ValueError("registry metadata is missing harness configuration")

    harness = deepcopy(dict(harness_raw))
    harness["id"] = env_id
    desktop = harness.setdefault("desktop", {})
    if not isinstance(desktop, dict):
        raise ValueError("registry harness.desktop must be a table")
    pool = desktop.setdefault("desktop_pool_config", {})
    if not isinstance(pool, dict):
        raise ValueError("registry desktop_pool_config must be a table")
    pool["min_ready_sessions"] = 0

    max_steps = harness.get("max_steps")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("registry harness.max_steps must be a positive integer")

    return {
        "name": str(metadata.get("env_name_prefix") or env_id),
        "address": gateway_public_address(metadata),
        "taskset": {
            "id": env_id,
            "base_path": task_base_path,
            "max_tasks": int(metadata.get("max_tasks", 0)),
            "shuffle_seed": int(metadata.get("shuffle_seed", 0)),
        },
        "harness": harness,
        "timeout": {"rollout": rollout_timeout},
        "retries": {"rollout": {"max_retries": max_retries}},
        "max_turns": max_steps,
    }


def gateway_public_address(metadata: Mapping[str, Any]) -> str:
    gateway = metadata.get("gateway")
    if not isinstance(gateway, Mapping):
        raise ValueError("registry metadata is missing gateway.public_address")
    address = gateway.get("public_address")
    if not isinstance(address, str) or not address.strip():
        raise ValueError("registry metadata is missing gateway.public_address")
    return address.strip()


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"base config {key} must be a table")
    return value


def required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"registry metadata is missing {key}")
    return value.strip()


def validate_prime_rl_config(path: Path) -> None:
    root = repo_root()
    prime_rl_dir = prime_rl_root()
    python = prime_rl_dir / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(
            "PrimeRL environment is not installed; run "
            'UV_PROJECT_ENVIRONMENT="$PWD/deps/prime-rl/.venv" '
            "uv sync --project deps/prime-rl --locked --extra all"
        )
    code = (
        "import sys, tomllib; "
        "from prime_rl.configs.rl import RLConfig; "
        "RLConfig.model_validate(tomllib.load(open(sys.argv[1], 'rb')))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [str(python), "-c", code, str(path)],
        cwd=prime_rl_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"generated PrimeRL config is invalid:\n{detail}")


if __name__ == "__main__":
    raise SystemExit(main())
