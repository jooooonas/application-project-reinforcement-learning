#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from rl.runtime.envfile import load_runtime_env_file
from rl.runtime.paths import prime_rl_root, repo_root

CLUSTER_PROFILE_DEFAULTS = (
    ("PRIME_RL_SLURM_ACCOUNT", "--slurm.account"),
    ("PRIME_RL_SLURM_PARTITION", "--slurm.partition"),
    ("PRIME_RL_SLURM_TIME", "--slurm.time"),
    (
        "PRIME_RL_NUM_TRAIN_NODES",
        "--deployment.num-train-nodes",
    ),
    (
        "PRIME_RL_NUM_INFER_NODES",
        "--deployment.num-infer-nodes",
    ),
    (
        "PRIME_RL_GPUS_PER_NODE",
        "--deployment.gpus-per-node",
    ),
)
RUNTIME_ENV_DEFAULTS = (("OSWORLD_PRIME_RL_OUTPUT_DIR", "--output-dir"),)


def main(argv: Sequence[str] | None = None) -> int:
    load_runtime_env_file()
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["run"]:
        return run_in_allocation(parse_run_args(args[1:]))

    return launch(args)


def launch(args: Sequence[str]) -> int:
    root = repo_root()
    prime_rl_dir = prime_rl_root()
    rl_bin = prime_rl_dir / ".venv" / "bin" / "rl"

    error = prime_rl_setup_error(prime_rl_dir, rl_bin)
    if error is not None:
        print(error, file=sys.stderr)
        return 127

    resolved_args = absolutize_config_args(args, Path.cwd())
    resolved_args = with_runtime_env_defaults(resolved_args, os.environ)
    resolved_args = with_cluster_profile_defaults(resolved_args, os.environ)
    resolved_args = with_forced_option(
        resolved_args,
        "--slurm.project-dir",
        str(prime_rl_dir),
    )

    result = subprocess.run(
        [str(rl_bin), *resolved_args],
        cwd=prime_rl_dir,
        env=subprocess_env(root, os.environ),
        check=False,
    )
    return result.returncode


def parse_run_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="prime_rl.py run",
        description="Run a generated PrimeRL batch payload in this allocation.",
    )
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--het-group", type=nonnegative_int)
    parser.add_argument("--expected-nodes", type=positive_int)
    return parser.parse_args(argv)


def run_in_allocation(args: argparse.Namespace) -> int:
    if not os.environ.get("SLURM_JOB_ID"):
        print("prime_rl.py run requires an active Slurm allocation.", file=sys.stderr)
        return 2

    script = args.script.resolve()
    if not script.is_file():
        print(f"Generated PrimeRL script does not exist: {script}", file=sys.stderr)
        return 2

    try:
        env = allocation_runtime_env(
            os.environ,
            het_group=args.het_group,
            expected_nodes=args.expected_nodes,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    result = subprocess.run(
        ["bash", str(script)],
        cwd=script.parent,
        env=env,
        check=False,
    )
    return result.returncode


def allocation_runtime_env(
    base_env: Mapping[str, str],
    *,
    het_group: int | None,
    expected_nodes: int | None,
) -> dict[str, str]:
    env = dict(base_env)
    if het_group is None:
        nodelist = env.get("SLURM_JOB_NODELIST")
        num_nodes = env.get("SLURM_JOB_NUM_NODES") or env.get("SLURM_NNODES")
    else:
        nodelist = heterogeneous_env_value(
            env,
            "SLURM_JOB_NODELIST",
            het_group,
        )
        num_nodes = heterogeneous_env_value(
            env,
            "SLURM_JOB_NUM_NODES",
            het_group,
        )
        env["PRIME_RL_SLURM_HET_GROUP"] = str(het_group)

    if not nodelist:
        raise ValueError("Could not determine the PrimeRL Slurm node list.")
    if not num_nodes:
        raise ValueError("Could not determine the number of PrimeRL Slurm nodes.")
    try:
        resolved_num_nodes = int(num_nodes)
    except ValueError as exc:
        raise ValueError(f"Invalid PrimeRL Slurm node count: {num_nodes}") from exc
    if expected_nodes is not None and resolved_num_nodes != expected_nodes:
        raise ValueError(
            "PrimeRL allocation has "
            f"{resolved_num_nodes} nodes; expected {expected_nodes}."
        )

    env["PRIME_RL_SLURM_NODELIST"] = nodelist
    env["PRIME_RL_SLURM_NUM_NODES"] = str(resolved_num_nodes)
    env["SLURM_JOB_NODELIST"] = nodelist
    env["SLURM_JOB_NUM_NODES"] = str(resolved_num_nodes)
    env["SLURM_NNODES"] = str(resolved_num_nodes)
    return env


def heterogeneous_env_value(
    env: Mapping[str, str],
    name: str,
    group: int,
) -> str | None:
    return env.get(f"{name}_HET_GROUP_{group}")


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def prime_rl_setup_error(prime_rl_dir: Path, rl_bin: Path) -> str | None:
    if not (prime_rl_dir / "pyproject.toml").is_file():
        return (
            f"ERROR: PrimeRL submodule is not initialized: {prime_rl_dir}\n"
            "Run: git submodule update --init --recursive"
        )
    if not rl_bin.is_file():
        return (
            f"ERROR: PrimeRL launcher not found: {rl_bin}\n"
            'Run: UV_PROJECT_ENVIRONMENT="$PWD/deps/prime-rl/.venv" '
            "uv sync --project deps/prime-rl --locked --extra all"
        )
    return None


def prepend_path(path: Path, value: str | None) -> str:
    if not value:
        return str(path)
    return f"{path}{os.pathsep}{value}"


def subprocess_env(root: Path, base_env: Mapping[str, str]) -> dict[str, str]:
    env = dict(base_env)
    env["PYTHONPATH"] = prepend_path(root, env.get("PYTHONPATH"))
    for name in tuple(env):
        if name.startswith("SBATCH_"):
            env.pop(name)
    return env


def absolutize_config_args(args: Sequence[str], base_dir: Path) -> list[str]:
    result: list[str] = []
    absolutize_next = False
    for item in args:
        if absolutize_next:
            result.append(absolutize_path_arg(item, base_dir))
            absolutize_next = False
            continue
        if item == "@":
            result.append(item)
            absolutize_next = True
            continue
        if item.startswith("@") and len(item) > 1:
            result.append(f"@{absolutize_path_arg(item[1:], base_dir)}")
            continue
        result.append(item)
    return result


def absolutize_path_arg(value: str, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return value
    return str((base_dir / path).resolve())


def with_cluster_profile_defaults(
    args: Sequence[str],
    env: Mapping[str, str],
) -> list[str]:
    result = list(args)
    for env_name, option in CLUSTER_PROFILE_DEFAULTS:
        value = env.get(env_name)
        if value:
            result = with_default_option(result, option, value)
    return result


def with_runtime_env_defaults(
    args: Sequence[str],
    env: Mapping[str, str],
) -> list[str]:
    result = list(args)
    for env_name, option in RUNTIME_ENV_DEFAULTS:
        value = env.get(env_name)
        if value:
            result = with_default_option(result, option, value)
    return result


def with_default_option(
    args: Sequence[str],
    option: str,
    value: str,
) -> list[str]:
    if has_option(args, option):
        return list(args)
    return [*args, option, value]


def has_option(args: Sequence[str], option: str) -> bool:
    option_prefix = f"{option}="
    return any(item == option or item.startswith(option_prefix) for item in args)


def with_forced_option(args: Sequence[str], option: str, value: str) -> list[str]:
    option_prefix = f"{option}="
    result: list[str] = []
    skip_next = False
    for item in args:
        if skip_next:
            skip_next = False
            continue
        if item == option:
            skip_next = True
            continue
        if item.startswith(option_prefix):
            continue
        result.append(item)
    result.extend([option, value])
    return result


if __name__ == "__main__":
    raise SystemExit(main())
