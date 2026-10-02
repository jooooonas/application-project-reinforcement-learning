#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from rl.runtime.envfile import load_runtime_env_file
from rl.runtime.fleet import EnvFleetRegistry, FleetRunLayout, read_registry
from rl.runtime.fleet.readiness import ReadinessSummary, readiness_summary
from rl.runtime.fleet.slurm import (
    SlurmJob,
    confirm_cancel,
    query_squeue,
    select_cancel_job,
    slurm_job_id_from_registry,
)
from rl.runtime.paths import (
    osworld_asset_cache_dir,
    osworld_task_base_path,
    repo_root,
)

DEFAULT_FLEET_SCRIPT = Path("sbatch/run_osworld_env_fleet.sbatch")
DEFAULT_JOB_NAME = "osworld_env_fleet"
DEFAULT_SLURM_LOG_NAME = "slurm-%x.%j.out"
DEFAULT_PRIME_RL_CONFIG = Path("configs/prime_rl/multi_node.toml")
UV_PYTHON_COMMAND = ("uv", "run", "--no-sync", "python")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "submit":
        return submit(args)
    if args.command == "run":
        return run_in_allocation(args)
    if args.command == "status":
        return status(args)
    if args.command == "cancel":
        return cancel(args)
    raise ValueError(f"unknown command: {args.command}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_runtime_env_file()
    env = os.environ
    defaults = FleetRunLayout.from_env(env)
    parser = argparse.ArgumentParser(
        description="Manage a long-lived OSWorld env fleet Slurm service."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit the fleet Slurm job.")
    submit_parser.add_argument("--script", type=Path, default=DEFAULT_FLEET_SCRIPT)
    submit_parser.add_argument("--run-id", default=env.get("OSWORLD_FLEET_RUN_ID"))
    submit_parser.add_argument("--run-base", type=Path, default=defaults.run_base)
    submit_parser.set_defaults(task_base_path=default_task_base_path())
    submit_parser.set_defaults(asset_cache_dir=default_asset_cache_dir(env))
    submit_parser.add_argument(
        "--asset-source-root",
        type=Path,
        default=env.get("OSWORLD_ASSET_SOURCE_ROOT"),
    )
    submit_parser.add_argument(
        "--prefetch-assets",
        action=argparse.BooleanOptionalAction,
        default=env_bool(env, "OSWORLD_PREFETCH_ASSETS", True),
        help=(
            "Populate the OSWorld task-asset cache before submitting the fleet, "
            "so compute nodes do not need external Hugging Face access."
        ),
    )
    submit_parser.add_argument(
        "--account",
        default=env.get("SBATCH_ACCOUNT"),
    )
    submit_parser.add_argument(
        "--partition",
        default=env.get("OSWORLD_FLEET_SLURM_PARTITION") or env.get("SBATCH_PARTITION"),
    )
    submit_parser.add_argument(
        "--time",
        default=env.get("OSWORLD_FLEET_SLURM_TIME")
        or env.get("SBATCH_TIMELIMIT")
        or "24:00:00",
    )
    submit_parser.add_argument(
        "--nodes",
        type=int,
        default=first_env_int(
            env,
            "OSWORLD_FLEET_SLURM_NODES",
            "SBATCH_NODES",
        ),
    )
    submit_parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=first_env_int(
            env,
            "OSWORLD_FLEET_SLURM_CPUS_PER_TASK",
            "SBATCH_CPUS_PER_TASK",
            default=48,
        ),
    )
    submit_parser.add_argument(
        "--mem",
        default=env.get("OSWORLD_FLEET_SLURM_MEM_PER_NODE")
        or env.get("SBATCH_MEM_PER_NODE")
        or "256G",
    )
    submit_parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    submit_parser.add_argument(
        "--slurm-output",
        type=Path,
        default=Path(env["SBATCH_OUTPUT"]) if env.get("SBATCH_OUTPUT") else None,
        help="Path pattern for the Slurm job's standard output.",
    )
    submit_parser.add_argument(
        "--slurm-error",
        type=Path,
        default=Path(env["SBATCH_ERROR"]) if env.get("SBATCH_ERROR") else None,
        help="Path pattern for the Slurm job's standard error.",
    )
    submit_parser.add_argument(
        "--servers-per-node",
        type=int,
        default=env_int_default(env, "OSWORLD_ENV_SERVERS_PER_NODE", 8),
    )
    submit_parser.add_argument(
        "--workers-per-server",
        type=int,
        default=env_int_default(env, "OSWORLD_ENV_WORKERS_PER_SERVER", 4),
    )
    submit_parser.add_argument("--base-port", type=int)
    submit_parser.add_argument(
        "--max-tasks",
        type=int,
        default=env_int_default(env, "OSWORLD_MAX_TASKS", 0),
        help="Maximum number of OSWorld task JSONs exposed by each env replica.",
    )
    submit_parser.add_argument(
        "--artifact-output-dir",
        type=Path,
        default=(
            Path(env["OSWORLD_ARTIFACT_DIR"]) if "OSWORLD_ARTIFACT_DIR" in env else None
        ),
        help="Artifact root baked into env-worker harness configs.",
    )
    submit_parser.add_argument(
        "--desktop-pool-min-ready-sessions",
        type=int,
        default=env_int_default(env, "OSWORLD_DESKTOP_POOL_MIN_READY_SESSIONS", 1),
        help="Ready desktop sessions kept warm per env worker.",
    )
    submit_parser.add_argument(
        "--desktop-pool-max-sessions",
        type=int,
        default=env_int_default(env, "OSWORLD_DESKTOP_POOL_MAX_SESSIONS", 64),
        help="Maximum desktop sessions per env worker.",
    )
    submit_parser.add_argument(
        "--desktop-pool-max-rollouts-per-session",
        type=int,
        default=env_int_default(
            env,
            "OSWORLD_DESKTOP_POOL_MAX_ROLLOUTS_PER_SESSION",
            32,
        ),
        help="Retire a desktop session after this many rollouts.",
    )
    submit_parser.add_argument(
        "--desktop-pool-checkout-timeout",
        type=float,
        default=env_float_default(
            env,
            "OSWORLD_DESKTOP_POOL_CHECKOUT_TIMEOUT",
            500.0,
        ),
        help="Seconds to wait for a ready desktop session.",
    )
    submit_parser.add_argument(
        "--desktop-pool-lease-timeout",
        type=float,
        default=env_float_default(env, "OSWORLD_DESKTOP_POOL_LEASE_TIMEOUT", 300.0),
        help="Seconds a checked-out desktop session may stay idle before reset.",
    )
    submit_parser.add_argument(
        "--desktop-pool-startup-timeout",
        type=float,
        default=env_float_default(env, "OSWORLD_DESKTOP_POOL_STARTUP_TIMEOUT", 840.0),
        help="Seconds to allow one desktop session startup before failing it.",
    )
    submit_parser.add_argument(
        "--desktop-pool-startup-retry-backoff",
        type=float,
        default=env_float_default(
            env, "OSWORLD_DESKTOP_POOL_STARTUP_RETRY_BACKOFF", 30.0
        ),
        help="Seconds to wait before retrying a failed desktop startup.",
    )
    submit_parser.add_argument(
        "--desktop-pool-startup-retry-backoff-max",
        type=float,
        default=env_float_default(
            env, "OSWORLD_DESKTOP_POOL_STARTUP_RETRY_BACKOFF_MAX", 300.0
        ),
        help="Maximum seconds for exponential desktop startup retry backoff.",
    )
    submit_parser.add_argument(
        "--desktop-pool-status-heartbeat-interval",
        type=float,
        default=env_float_default(
            env, "OSWORLD_DESKTOP_POOL_STATUS_HEARTBEAT_INTERVAL", 10.0
        ),
        help="Seconds between desktop-pool status heartbeat writes.",
    )
    submit_parser.add_argument(
        "--desktop-pool-root",
        type=Path,
        default=(
            Path(env["OSWORLD_DESKTOP_POOL_ROOT"])
            if "OSWORLD_DESKTOP_POOL_ROOT" in env
            else None
        ),
        help="Shared desktop pool root for status, logs, and port locks.",
    )
    submit_parser.add_argument(
        "--desktop-pool-runtime-dir",
        type=Path,
        default=(
            Path(env["OSWORLD_DESKTOP_POOL_RUNTIME_DIR"])
            if "OSWORLD_DESKTOP_POOL_RUNTIME_DIR" in env
            else None
        ),
        help="Node-local desktop runtime root for QEMU workdirs and sockets.",
    )
    submit_parser.add_argument(
        "--desktop-pool-log-runtime-dir",
        type=Path,
        default=(
            Path(env["OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR"])
            if "OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR" in env
            else None
        ),
        help="Short node-local symlink path used when writing persistent desktop logs.",
    )
    submit_parser.add_argument(
        "--rollout-timeout",
        type=float,
        default=env_float_default(env, "OSWORLD_ROLLOUT_TIMEOUT", 900.0),
    )
    submit_parser.add_argument(
        "--env-max-retries",
        type=int,
        default=env_int_default(env, "OSWORLD_ENV_MAX_RETRIES", 2),
    )
    submit_parser.add_argument(
        "--replica-unhealthy-s",
        type=float,
        default=env_float_default(
            env,
            "OSWORLD_SUPERVISOR_REPLICA_UNHEALTHY_S",
            120.0,
        ),
    )
    submit_parser.add_argument(
        "--fleet-unhealthy-s",
        type=float,
        default=env_float_default(env, "OSWORLD_SUPERVISOR_FLEET_UNHEALTHY_S", 300.0),
    )
    submit_parser.add_argument(
        "--max-fleet-restarts",
        type=int,
        default=env_int_default(env, "OSWORLD_SUPERVISOR_MAX_FLEET_RESTARTS", 3),
    )
    submit_parser.add_argument(
        "--gateway-request-timeout-s",
        type=float,
        default=env_float_default(env, "OSWORLD_GATEWAY_REQUEST_TIMEOUT_S", 900.0),
    )
    submit_parser.add_argument(
        "--status-stale-after-s",
        type=float,
        default=env_float_default(env, "OSWORLD_STATUS_STALE_AFTER_S", 120.0),
    )
    submit_parser.add_argument("--dry-run", action="store_true")

    run_parser = subparsers.add_parser(
        "run",
        help="Run the fleet inside the current Slurm allocation.",
    )
    run_parser.add_argument("--script", type=Path, default=DEFAULT_FLEET_SCRIPT)
    run_parser.add_argument(
        "--het-group",
        type=nonnegative_int,
        help="Heterogeneous allocation component used by fleet srun steps.",
    )

    status_parser = subparsers.add_parser("status", help="Summarize fleet health.")
    add_layout_args(status_parser, defaults)
    status_parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    status_parser.add_argument("--expected-servers", type=int, default=0)
    status_parser.add_argument("--min-ready-sessions", type=int, default=-1)
    status_parser.add_argument(
        "--status-stale-after-s",
        type=float,
        default=env_float_default(env, "OSWORLD_STATUS_STALE_AFTER_S", 120.0),
    )

    cancel_parser = subparsers.add_parser(
        "cancel", help="Cancel this user's fleet job."
    )
    add_layout_args(cancel_parser, defaults)
    cancel_parser.add_argument("--job-id")
    cancel_parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    cancel_parser.add_argument("--yes", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "submit":
        if args.slurm_output is None:
            args.slurm_output = args.run_base / "logs" / DEFAULT_SLURM_LOG_NAME
        if args.slurm_error is None:
            args.slurm_error = args.slurm_output
    return args


def add_layout_args(parser: argparse.ArgumentParser, defaults: FleetRunLayout) -> None:
    parser.add_argument("--run-id", default=defaults.run_id)
    parser.add_argument("--run-base", type=Path, default=defaults.run_base)
    parser.add_argument("--registry", type=Path)


def submit(args: argparse.Namespace) -> int:
    command = build_sbatch_command(args)
    if args.dry_run:
        if args.prefetch_assets:
            print(format_shell_command(build_prefetch_command(args)))
        print(format_shell_command(command))
        print()
        print(
            "PrimeRL trainer command preview "
            "(replace <run-id> with the submitted fleet job id if unset):"
        )
        print(format_trainer_command(dry_run_layout(args)))
        return 0

    prepare_slurm_log_directories(args.slurm_output, args.slurm_error)
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
    print(format_submit_report(job_id, layout, args))
    return 0


def run_in_allocation(args: argparse.Namespace) -> int:
    """Execute the fleet batch payload without submitting another Slurm job."""
    if not os.environ.get("SLURM_JOB_ID"):
        print(
            "osworld_fleet.py run requires an active Slurm allocation.",
            file=sys.stderr,
        )
        return 2

    root = repo_root()
    script = args.script
    if not script.is_absolute():
        script = root / script
    if not script.is_file():
        print(f"Fleet runtime script does not exist: {script}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env["ROOT_DIR"] = str(root)
    if args.het_group is not None:
        env["OSWORLD_SLURM_HET_GROUP"] = str(args.het_group)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env=env,
        check=False,
    )
    return result.returncode


def build_sbatch_command(args: argparse.Namespace) -> list[str]:
    command = ["sbatch", "--parsable"]
    for option, value in (
        ("--account", args.account),
        ("--partition", args.partition),
        ("--time", args.time),
        ("--nodes", args.nodes),
        ("--cpus-per-task", args.cpus_per_task),
        ("--mem", args.mem),
        ("--job-name", args.job_name),
        ("--output", args.slurm_output),
        ("--error", args.slurm_error),
    ):
        if value is not None:
            command.extend([option, str(value)])

    exports = {
        "OSWORLD_RUN_BASE": str(args.run_base),
    }
    arg_values = vars(args)
    if args.run_id:
        exports["OSWORLD_FLEET_RUN_ID"] = str(args.run_id)
    if args.servers_per_node is not None:
        exports["OSWORLD_ENV_SERVERS_PER_NODE"] = str(args.servers_per_node)
    if args.workers_per_server is not None:
        exports["OSWORLD_ENV_WORKERS_PER_SERVER"] = str(args.workers_per_server)
    if args.base_port is not None:
        exports["OSWORLD_FLEET_BASE_PORT"] = str(args.base_port)
    max_tasks = arg_values.get("max_tasks")
    if max_tasks is not None:
        exports["OSWORLD_MAX_TASKS"] = str(max_tasks)
    artifact_output_dir = arg_values.get("artifact_output_dir")
    if artifact_output_dir is not None:
        exports["OSWORLD_ARTIFACT_DIR"] = str(artifact_output_dir)
    for attr, env_name in (
        ("desktop_pool_min_ready_sessions", "OSWORLD_DESKTOP_POOL_MIN_READY_SESSIONS"),
        ("desktop_pool_max_sessions", "OSWORLD_DESKTOP_POOL_MAX_SESSIONS"),
        (
            "desktop_pool_max_rollouts_per_session",
            "OSWORLD_DESKTOP_POOL_MAX_ROLLOUTS_PER_SESSION",
        ),
        ("desktop_pool_checkout_timeout", "OSWORLD_DESKTOP_POOL_CHECKOUT_TIMEOUT"),
        ("desktop_pool_lease_timeout", "OSWORLD_DESKTOP_POOL_LEASE_TIMEOUT"),
        ("desktop_pool_startup_timeout", "OSWORLD_DESKTOP_POOL_STARTUP_TIMEOUT"),
        (
            "desktop_pool_startup_retry_backoff",
            "OSWORLD_DESKTOP_POOL_STARTUP_RETRY_BACKOFF",
        ),
        (
            "desktop_pool_startup_retry_backoff_max",
            "OSWORLD_DESKTOP_POOL_STARTUP_RETRY_BACKOFF_MAX",
        ),
        (
            "desktop_pool_status_heartbeat_interval",
            "OSWORLD_DESKTOP_POOL_STATUS_HEARTBEAT_INTERVAL",
        ),
        ("desktop_pool_root", "OSWORLD_DESKTOP_POOL_ROOT"),
        ("desktop_pool_runtime_dir", "OSWORLD_DESKTOP_POOL_RUNTIME_DIR"),
        ("desktop_pool_log_runtime_dir", "OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR"),
    ):
        value = arg_values.get(attr)
        if value is not None:
            exports[env_name] = str(value)
    for attr, env_name in (
        ("rollout_timeout", "OSWORLD_ROLLOUT_TIMEOUT"),
        ("env_max_retries", "OSWORLD_ENV_MAX_RETRIES"),
        ("replica_unhealthy_s", "OSWORLD_SUPERVISOR_REPLICA_UNHEALTHY_S"),
        ("fleet_unhealthy_s", "OSWORLD_SUPERVISOR_FLEET_UNHEALTHY_S"),
        ("max_fleet_restarts", "OSWORLD_SUPERVISOR_MAX_FLEET_RESTARTS"),
        ("gateway_request_timeout_s", "OSWORLD_GATEWAY_REQUEST_TIMEOUT_S"),
        ("status_stale_after_s", "OSWORLD_STATUS_STALE_AFTER_S"),
    ):
        value = arg_values.get(attr)
        if value is not None:
            exports[env_name] = str(value)
    if exports:
        rendered_exports = ",".join(f"{key}={value}" for key, value in exports.items())
        command.append(f"--export=ALL,{rendered_exports}")
    command.append(str(args.script))
    return command


def prepare_slurm_log_directories(*log_paths: Path) -> None:
    """Create log parents before Slurm attempts to open the output files."""
    for parent in {path.parent for path in log_paths}:
        parent.mkdir(parents=True, exist_ok=True)


def build_prefetch_command(args: argparse.Namespace) -> list[str]:
    command = [
        *UV_PYTHON_COMMAND,
        script_path("prefetch_osworld_assets.py"),
        "--tasks",
        str(args.task_base_path),
        "--cache-dir",
        str(args.asset_cache_dir),
    ]
    if args.asset_source_root is not None:
        command.extend(["--source-root", str(args.asset_source_root)])
    return command


def parse_sbatch_job_id(output: str) -> str:
    first = output.strip().splitlines()[0]
    return first.split(";", 1)[0]


def format_submit_report(
    job_id: str,
    layout: FleetRunLayout,
    args: argparse.Namespace,
) -> str:
    status_command = format_shell_command(
        [
            *UV_PYTHON_COMMAND,
            "scripts/osworld_fleet.py",
            "status",
            "--run-id",
            layout.run_id,
        ]
    )
    readiness_command = format_shell_command(
        [
            *UV_PYTHON_COMMAND,
            "scripts/wait_env_fleet_ready.py",
            "--registry",
            str(layout.registry_path),
        ]
    )
    trainer_command = format_trainer_command(layout)
    cancel_command = format_shell_command(
        [
            *UV_PYTHON_COMMAND,
            "scripts/osworld_fleet.py",
            "cancel",
            "--run-id",
            layout.run_id,
        ]
    )
    return "\n".join(
        [
            f"Submitted OSWorld env fleet job {job_id}",
            f"Run id: {layout.run_id}",
            f"Registry: {layout.registry_path}",
            f"Status dir: {layout.pool_status_dir}",
            f"Logs: {layout.logs_dir}",
            "",
            "Commands:",
            "",
            "Status:",
            f"  {status_command}",
            "",
            "Readiness:",
            f"  {readiness_command}",
            "",
            "Render config and launch PrimeRL trainer:",
            trainer_command,
            "",
            "Cancel:",
            f"  {cancel_command}",
        ]
    )


def format_trainer_command(layout: FleetRunLayout) -> str:
    """Build the copy-pastable PrimeRL launch command for this fleet."""
    rendered_config = shlex.quote(str(layout.prime_rl_config_path))
    launch_command = format_shell_command(
        [
            *UV_PYTHON_COMMAND,
            "scripts/prime_rl.py",
            "@",
            str(layout.prime_rl_config_path),
            "--clean-output-dir",
        ]
    )
    render_command = format_shell_command(
        [
            *UV_PYTHON_COMMAND,
            "scripts/render_prime_rl_fleet_config.py",
            "--base-config",
            str(DEFAULT_PRIME_RL_CONFIG),
            "--registry",
            str(layout.registry_path),
            "--output",
            str(layout.prime_rl_config_path),
            "--output-dir",
            str(layout.prime_rl_output_dir),
        ]
    )
    return "\n".join(
        [
            f"  cd {shlex.quote(str(repo_root()))}",
            f"  {render_command}",
            f"  # generated config: {rendered_config}",
            f"  {launch_command}",
        ]
    )


def dry_run_layout(args: argparse.Namespace) -> FleetRunLayout:
    return FleetRunLayout.for_run(
        run_id=args.run_id or "<run-id>",
        run_base=args.run_base,
    )


def status(args: argparse.Namespace) -> int:
    registry_path = registry_path_for_args(args)
    registry, registry_error = read_registry_for_status(registry_path)
    layout = layout_for_args(args, registry, registry_path=registry_path)
    summary = readiness_summary(
        argparse.Namespace(
            registry=layout.registry_path,
            status_dir=None,
            pool_status_dir=layout.pool_status_dir,
            run_root=layout.run_root,
            min_ready_sessions=args.min_ready_sessions,
            expected_servers=args.expected_servers,
            status_stale_after_s=args.status_stale_after_s,
        )
    )
    job_id = args.run_id
    if registry is not None:
        job_id = slurm_job_id_from_registry(registry) or job_id
    jobs = query_squeue(job_id=job_id, job_name=args.job_name)
    print(format_status_report(layout, registry, registry_error, summary, jobs))
    return 0


def registry_path_for_args(args: argparse.Namespace) -> Path:
    registry_path = args.registry
    if registry_path is None:
        registry_path = FleetRunLayout.for_run(
            run_id=args.run_id,
            run_base=args.run_base,
        ).registry_path
    return Path(registry_path)


def read_registry_optional(path: Path) -> EnvFleetRegistry | None:
    if not path.exists():
        return None
    return read_registry(path)


def read_registry_for_status(
    registry_path: Path,
) -> tuple[EnvFleetRegistry | None, str | None]:
    if not registry_path.exists():
        return None, f"missing: {registry_path}"
    try:
        return read_registry(registry_path), None
    except Exception as exc:
        return None, f"{registry_path}: {exc!r}"


def layout_for_args(
    args: argparse.Namespace,
    registry: EnvFleetRegistry | None,
    *,
    registry_path: Path | None = None,
) -> FleetRunLayout:
    run_id = registry.run_id if registry is not None else args.run_id
    fallback = FleetRunLayout.for_run(
        run_id=run_id,
        run_base=args.run_base,
        registry_path=registry_path or args.registry,
    )
    if registry is None:
        return fallback
    return (
        FleetRunLayout.from_metadata(registry.metadata, fallback=fallback) or fallback
    )


def format_status_report(
    layout: FleetRunLayout,
    registry: EnvFleetRegistry | None,
    registry_error: str | None,
    summary: ReadinessSummary,
    jobs: Sequence[SlurmJob],
) -> str:
    lines = [
        f"Run id: {layout.run_id}",
        f"Registry: {layout.registry_path}",
        f"Status dir: {summary['status_dir']}",
        f"Logs: {layout.logs_dir}",
    ]
    if jobs:
        job = jobs[0]
        lines.append(
            "Slurm: "
            f"{job.job_id} {job.state} nodes={job.nodes or '?'} "
            f"cpus={job.cpus or '?'} reason={job.reason or '-'}"
        )
    else:
        lines.append("Slurm: no matching running or pending job found")

    lines.extend(
        [
            "Registry ready: "
            f"{summary['registry_ready']} "
            f"({summary['registered_servers']}/{summary['expected_servers']} servers)",
            "Pool: "
            f"ready={summary['ready']}/{summary['min_ready']} "
            f"starting={summary['starting']} "
            f"leased={summary['leased']} "
            f"stale_status_files={summary['stale_status_files']} "
            f"total_failed={summary['total_failed']} "
            f"stale_leases_retired={summary.get('stale_leases_retired', 0)}",
        ]
    )
    unhealthy_servers = int(summary.get("unhealthy_servers", 0))
    if unhealthy_servers:
        lines.append(f"Unhealthy replicas: {unhealthy_servers}")
    retry_scheduled_workers = summary["retry_scheduled_workers"]
    cooling_down_workers = summary["cooling_down_workers"]
    consecutive_start_failures = summary["consecutive_start_failures"]
    startup_cooldown_remaining_s = summary["startup_cooldown_remaining_s"]
    if (
        retry_scheduled_workers
        or cooling_down_workers
        or consecutive_start_failures
        or startup_cooldown_remaining_s > 0.0
    ):
        lines.append(
            "Startup retry: "
            f"scheduled_workers={retry_scheduled_workers} "
            f"cooling_down_workers={cooling_down_workers} "
            f"consecutive_failures={consecutive_start_failures} "
            f"cooldown_remaining_s={startup_cooldown_remaining_s:.1f}"
        )
    if registry_error:
        lines.append(f"Registry error: {registry_error}")
    unrecoverable_path = layout.run_root / "fleet_unrecoverable.json"
    if unrecoverable_path.exists():
        lines.append(f"Fleet unrecoverable marker: {unrecoverable_path}")

    if registry is not None and registry.servers:
        server_summaries = {
            item["name"]: item
            for item in summary.get("server_summaries", [])
            if isinstance(item, Mapping) and item.get("name")
        }
        lines.append("Env servers:")
        for server in registry.servers:
            server_summary = server_summaries.get(server.name)
            pool_status = ""
            if server_summary is not None:
                pool_status = (
                    f" ready={server_summary['ready']}"
                    f" starting={server_summary['starting']}"
                    f" leased={server_summary['leased']}"
                    f" stale_status={server_summary['stale_status_files']}"
                    f" failed={server_summary['total_failed']}"
                )
            lines.append(
                f"  {server.name} {server.public_address} "
                f"workers={server.num_workers}{pool_status} log={server.log_path}"
            )
    errors = summary["last_errors"]
    if errors:
        lines.append(f"Last errors: {errors}")
    return "\n".join(lines)


def cancel(args: argparse.Namespace) -> int:
    registry_path = registry_path_for_args(args)
    registry = read_registry_optional(registry_path)
    job_id = args.job_id
    if job_id is None and registry is not None:
        job_id = slurm_job_id_from_registry(registry)
    if job_id is None and str(args.run_id).isdigit():
        job_id = str(args.run_id)
    if job_id is None:
        print("Could not determine a Slurm job id for this fleet.", file=sys.stderr)
        return 2

    jobs = query_squeue(job_id=job_id, job_name=args.job_name)
    try:
        job = select_cancel_job(
            jobs, user=os.environ.get("USER", ""), job_name=args.job_name
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if job is None:
        print(f"No matching running or pending fleet job found for {job_id}.")
        return 0

    if not confirm_cancel(job, yes=args.yes):
        print("Cancel aborted.")
        return 0

    result = subprocess.run(["scancel", job.job_id], check=False)
    if result.returncode == 0:
        print(f"Cancelled fleet job {job.job_id}.")
    return result.returncode


def script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def env_int(env: Mapping[str, str], key: str) -> int | None:
    value = env.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def first_env_int(
    env: Mapping[str, str],
    *keys: str,
    default: int | None = None,
) -> int | None:
    for key in keys:
        value = env_int(env, key)
        if value is not None:
            return value
    return default


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


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def default_task_base_path() -> Path:
    return osworld_task_base_path()


def default_asset_cache_dir(env: Mapping[str, str]) -> Path:
    return osworld_asset_cache_dir(env=env)


def format_shell_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
