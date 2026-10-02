#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl.runtime.envfile import load_runtime_env_file
from rl.runtime.fleet import FleetRunLayout, read_registry
from rl.runtime.fleet.slurm import query_squeue, slurm_job_id_from_registry
from rl.runtime.paths import osworld_asset_cache_dir, osworld_task_base_path, repo_root

DEFAULT_BATCH_SCRIPT = Path("sbatch/run_osworld.sbatch")
DEFAULT_PRIME_RL_CONFIG = Path("configs/prime_rl/multi_node.toml")
DEFAULT_JOB_NAME = "osworld_run"
UV_PYTHON_COMMAND = ("uv", "run", "--no-sync", "python")


@dataclass(frozen=True)
class PrimeResources:
    num_train_nodes: int
    num_infer_nodes: int
    num_infer_replicas: int
    gpus_per_node: int
    partition: str | None
    account: str | None
    time: str

    @property
    def total_infer_nodes(self) -> int:
        return self.num_infer_nodes * self.num_infer_replicas

    @property
    def total_nodes(self) -> int:
        return self.num_train_nodes + self.total_infer_nodes


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "submit":
        return submit(args)
    if args.command == "status":
        return delegate_fleet_command("status", args)
    if args.command == "cancel":
        return cancel(args)
    raise ValueError(f"unknown command: {args.command}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_runtime_env_file()
    env = os.environ
    layout = FleetRunLayout.from_env(env)

    parser = argparse.ArgumentParser(
        description=("Atomically allocate and run an OSWorld fleet with a PrimeRL job.")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit the co-scheduled heterogeneous Slurm job.",
    )
    submit_parser.add_argument("--script", type=Path, default=DEFAULT_BATCH_SCRIPT)
    submit_parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_PRIME_RL_CONFIG,
    )
    submit_parser.add_argument("--run-id", default=env.get("OSWORLD_RUN_ID"))
    submit_parser.add_argument("--run-base", type=Path, default=layout.run_base)
    submit_parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    submit_parser.add_argument(
        "--account",
        default=env.get("PRIME_RL_SLURM_ACCOUNT") or env.get("SBATCH_ACCOUNT"),
        help="Slurm account used by both heterogeneous components.",
    )
    submit_parser.add_argument(
        "--time",
        default=env.get("PRIME_RL_SLURM_TIME")
        or env.get("OSWORLD_FLEET_SLURM_TIME")
        or env.get("SBATCH_TIMELIMIT"),
        help="Shared wall time for the fleet and PrimeRL components.",
    )
    submit_parser.add_argument(
        "--fleet-partition",
        default=env.get("OSWORLD_FLEET_SLURM_PARTITION") or env.get("SBATCH_PARTITION"),
    )
    submit_parser.add_argument(
        "--fleet-nodes",
        type=positive_int,
        default=env_int_default(env, "OSWORLD_FLEET_SLURM_NODES", 1),
    )
    submit_parser.add_argument(
        "--fleet-cpus-per-task",
        type=positive_int,
        default=env_int_default(env, "OSWORLD_FLEET_SLURM_CPUS_PER_TASK", 48),
    )
    submit_parser.add_argument(
        "--fleet-mem",
        default=env.get("OSWORLD_FLEET_SLURM_MEM_PER_NODE") or "256G",
    )
    submit_parser.add_argument(
        "--prime-partition",
        default=env.get("PRIME_RL_SLURM_PARTITION"),
    )
    submit_parser.add_argument(
        "--prime-cpus-per-task",
        type=positive_int,
        default=env_int_default(env, "PRIME_RL_SLURM_CPUS_PER_TASK", 32),
    )
    submit_parser.add_argument(
        "--prime-mem",
        default=env.get("PRIME_RL_SLURM_MEM_PER_NODE") or "256G",
    )
    submit_parser.add_argument(
        "--num-train-nodes",
        type=positive_int,
        default=env_int(env, "PRIME_RL_NUM_TRAIN_NODES"),
    )
    submit_parser.add_argument(
        "--num-infer-nodes",
        type=nonnegative_int,
        default=env_int(env, "PRIME_RL_NUM_INFER_NODES"),
        help="Inference nodes per replica.",
    )
    submit_parser.add_argument(
        "--gpus-per-node",
        type=positive_int,
        default=env_int(env, "PRIME_RL_GPUS_PER_NODE"),
    )
    submit_parser.add_argument(
        "--ready-timeout-s",
        type=positive_float,
        default=env_float_default(env, "OSWORLD_ENV_FLEET_READY_TIMEOUT", 3600.0),
    )
    submit_parser.add_argument(
        "--inflight-per-worker",
        type=positive_int,
        default=env_int_default(env, "OSWORLD_PRIME_INFLIGHT_PER_WORKER", 1),
    )
    submit_parser.add_argument(
        "--clean-output-dir",
        action=argparse.BooleanOptionalAction,
        default=env_bool(env, "OSWORLD_RUN_CLEAN_OUTPUT_DIR", True),
    )
    submit_parser.add_argument(
        "--prefetch-assets",
        action=argparse.BooleanOptionalAction,
        default=env_bool(env, "OSWORLD_PREFETCH_ASSETS", True),
    )
    submit_parser.add_argument(
        "--asset-source-root",
        type=Path,
        default=env.get("OSWORLD_ASSET_SOURCE_ROOT"),
    )
    submit_parser.add_argument("--dry-run", action="store_true")

    for command in ("status", "cancel"):
        command_parser = subparsers.add_parser(
            command,
            help=f"{command.title()} a combined OSWorld run.",
        )
        command_parser.add_argument("--run-id", default=layout.run_id)
        command_parser.add_argument("--run-base", type=Path, default=layout.run_base)
        command_parser.add_argument("--registry", type=Path)
        command_parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
        if command == "cancel":
            command_parser.add_argument("--job-id")
            command_parser.add_argument("--yes", action="store_true")

    return parser.parse_args(argv)


def submit(args: argparse.Namespace) -> int:
    try:
        resources = load_prime_resources(args, os.environ)
        command = build_sbatch_command(args, resources)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        if args.prefetch_assets:
            print(format_shell_command(build_prefetch_command(args)))
        print(format_shell_command(command))
        return 0

    if args.prefetch_assets:
        prefetch_result = subprocess.run(build_prefetch_command(args), check=False)
        if prefetch_result.returncode != 0:
            return prefetch_result.returncode

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode

    job_id = parse_sbatch_job_id(result.stdout)
    run_id = args.run_id or job_id
    layout = FleetRunLayout.for_run(run_id=run_id, run_base=args.run_base)
    print(format_submit_report(job_id, layout, args.job_name))
    return 0


def load_prime_resources(
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> PrimeResources:
    config_path = resolve_repo_path(args.base_config)
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    deployment = require_table(config, "deployment")
    if deployment.get("type") != "multi_node":
        raise ValueError("Combined OSWorld runs require deployment.type='multi_node'.")

    num_train_nodes = args.num_train_nodes or required_positive_int(
        deployment,
        "num_train_nodes",
    )
    configured_infer_nodes = deployment.get("num_infer_nodes")
    if args.num_infer_nodes is not None:
        num_infer_nodes = args.num_infer_nodes
    elif isinstance(configured_infer_nodes, int) and configured_infer_nodes >= 0:
        num_infer_nodes = configured_infer_nodes
    else:
        raise ValueError(
            "Set deployment.num_infer_nodes in the PrimeRL config so Slurm can "
            "determine the atomic resource request."
        )
    num_infer_replicas = optional_positive_int(
        deployment,
        "num_infer_replicas",
        default=1,
    )
    gpus_per_node = args.gpus_per_node or required_positive_int(
        deployment,
        "gpus_per_node",
    )

    slurm = require_table(config, "slurm")
    partition = args.prime_partition or optional_string(slurm.get("partition"))
    account = args.account or optional_string(slurm.get("account"))
    time = args.time or optional_string(slurm.get("time")) or "24:00:00"
    if not partition:
        partition = env.get("SBATCH_PARTITION")

    resources = PrimeResources(
        num_train_nodes=num_train_nodes,
        num_infer_nodes=num_infer_nodes,
        num_infer_replicas=num_infer_replicas,
        gpus_per_node=gpus_per_node,
        partition=partition,
        account=account,
        time=time,
    )
    if resources.total_nodes < 1:
        raise ValueError("PrimeRL must request at least one node.")
    return resources


def build_sbatch_command(
    args: argparse.Namespace,
    resources: PrimeResources,
) -> list[str]:
    root = repo_root()
    script = resolve_repo_path(args.script)
    base_config = resolve_repo_path(args.base_config)
    if not script.is_file():
        raise ValueError(f"Combined run batch script does not exist: {script}")
    if not args.run_base.is_absolute():
        raise ValueError("--run-base must be an absolute path")

    exports = {
        "ROOT_DIR": str(root),
        "OSWORLD_RUN_BASE": str(args.run_base),
        "OSWORLD_PRIME_BASE_CONFIG": str(base_config),
        "OSWORLD_PRIME_INFLIGHT_PER_WORKER": str(args.inflight_per_worker),
        "OSWORLD_ENV_FLEET_READY_TIMEOUT": str(args.ready_timeout_s),
        "OSWORLD_FLEET_ALLOCATED_NODES": str(args.fleet_nodes),
        "OSWORLD_RUN_PRIME_NODES": str(resources.total_nodes),
        "OSWORLD_RUN_CLEAN_OUTPUT_DIR": "1" if args.clean_output_dir else "0",
        "PRIME_RL_NUM_TRAIN_NODES": str(resources.num_train_nodes),
        "PRIME_RL_NUM_INFER_NODES": str(resources.num_infer_nodes),
        "PRIME_RL_GPUS_PER_NODE": str(resources.gpus_per_node),
    }
    if args.run_id:
        exports["OSWORLD_RUN_ID"] = str(args.run_id)
        exports["OSWORLD_FLEET_RUN_ID"] = str(args.run_id)

    command = ["sbatch", "--parsable"]
    add_option(command, "--job-name", args.job_name)
    add_option(command, "--account", resources.account)
    add_option(command, "--partition", args.fleet_partition or resources.partition)
    add_option(command, "--time", resources.time)
    add_option(command, "--nodes", args.fleet_nodes)
    add_option(command, "--ntasks-per-node", 1)
    add_option(command, "--cpus-per-task", args.fleet_cpus_per_task)
    add_option(command, "--mem", args.fleet_mem)
    command.append(
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    )

    command.append(":")
    add_option(command, "--account", resources.account)
    add_option(command, "--partition", resources.partition or args.fleet_partition)
    add_option(command, "--time", resources.time)
    add_option(command, "--nodes", resources.total_nodes)
    add_option(command, "--ntasks-per-node", 1)
    add_option(command, "--cpus-per-task", args.prime_cpus_per_task)
    add_option(command, "--mem", args.prime_mem)
    add_option(command, "--gpus-per-node", resources.gpus_per_node)
    command.append(str(script))
    return command


def build_prefetch_command(args: argparse.Namespace) -> list[str]:
    command = [
        *UV_PYTHON_COMMAND,
        str(repo_root() / "scripts" / "prefetch_osworld_assets.py"),
        "--tasks",
        str(osworld_task_base_path()),
        "--cache-dir",
        str(osworld_asset_cache_dir(env=os.environ)),
    ]
    if args.asset_source_root is not None:
        command.extend(["--source-root", str(args.asset_source_root)])
    return command


def delegate_fleet_command(command: str, args: argparse.Namespace) -> int:
    delegated = [
        sys.executable,
        str(repo_root() / "scripts" / "osworld_fleet.py"),
        command,
        "--run-id",
        str(args.run_id),
        "--run-base",
        str(args.run_base),
        "--job-name",
        args.job_name,
    ]
    if args.registry is not None:
        delegated.extend(["--registry", str(args.registry)])
    result = subprocess.run(delegated, check=False)
    return result.returncode


def cancel(args: argparse.Namespace) -> int:
    registry_path = args.registry
    if registry_path is None:
        registry_path = FleetRunLayout.for_run(
            run_id=args.run_id,
            run_base=args.run_base,
        ).registry_path

    job_id = args.job_id
    if job_id is None and registry_path.exists():
        job_id = slurm_job_id_from_registry(read_registry(registry_path))
    if job_id is None and str(args.run_id).split("+", 1)[0].isdigit():
        job_id = str(args.run_id)
    if job_id is None:
        print("Could not determine the combined Slurm job id.", file=sys.stderr)
        return 2

    root_job_id = str(job_id).split("+", 1)[0]
    jobs = query_squeue(job_id=root_job_id, job_name=args.job_name)
    if not jobs:
        print(f"No matching running or pending combined job found for {root_job_id}.")
        return 0
    user = os.environ.get("USER", "")
    if any(job.user != user for job in jobs):
        print("Refusing to cancel a job owned by another user.", file=sys.stderr)
        return 2
    if any(job.name != args.job_name for job in jobs):
        names = ", ".join(sorted({job.name for job in jobs}))
        print(
            f"Refusing to cancel jobs with unexpected names: {names}", file=sys.stderr
        )
        return 2
    if not args.yes:
        answer = input(
            f"Cancel combined job {root_job_id} and all its components? [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancel aborted.")
            return 0

    result = subprocess.run(["scancel", root_job_id], check=False)
    if result.returncode == 0:
        print(f"Cancelled combined job {root_job_id}.")
    return result.returncode


def format_submit_report(
    job_id: str,
    layout: FleetRunLayout,
    job_name: str,
) -> str:
    status = format_shell_command(
        [
            *UV_PYTHON_COMMAND,
            "scripts/osworld_run.py",
            "status",
            "--run-id",
            layout.run_id,
            "--run-base",
            str(layout.run_base),
            "--job-name",
            job_name,
        ]
    )
    cancel_command = format_shell_command(
        [
            *UV_PYTHON_COMMAND,
            "scripts/osworld_run.py",
            "cancel",
            "--run-id",
            layout.run_id,
            "--run-base",
            str(layout.run_base),
            "--job-name",
            job_name,
        ]
    )
    return "\n".join(
        [
            f"Submitted combined OSWorld run job {job_id}",
            f"Run id: {layout.run_id}",
            f"Registry: {layout.registry_path}",
            f"Fleet logs: {layout.logs_dir}",
            f"PrimeRL output: {layout.prime_rl_output_dir}",
            "",
            f"Status: {status}",
            f"Cancel: {cancel_command}",
        ]
    )


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def require_table(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"PrimeRL config {key} must be a table.")
    return value


def required_positive_int(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"PrimeRL config {key} must be a positive integer.")
    return value


def optional_positive_int(
    values: Mapping[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = values.get(key, default)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"PrimeRL config {key} must be a positive integer.")
    return value


def optional_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def add_option(command: list[str], option: str, value: object | None) -> None:
    if value is not None:
        command.extend([option, str(value)])


def parse_sbatch_job_id(output: str) -> str:
    first = output.strip().splitlines()[0]
    return first.split(";", 1)[0].split("+", 1)[0]


def env_int(env: Mapping[str, str], key: str) -> int | None:
    value = env.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def env_int_default(env: Mapping[str, str], key: str, default: int) -> int:
    value = env_int(env, key)
    return default if value is None else value


def env_float_default(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def format_shell_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
