#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rl.runtime.envfile import load_runtime_env_file
from rl.runtime.fleet import (
    FleetRunLayout,
    default_public_host,
    make_server_specs,
    upsert_registry,
)
from rl.runtime.fleet.config_rendering import write_env_server_config
from rl.runtime.fleet.slurm import (
    slurm_metadata,
    slurm_node_addrs,
)
from rl.runtime.paths import (
    osworld_asset_cache_dir,
    osworld_qcow_path,
    osworld_task_base_path,
)


def main() -> int:
    args = parse_args()
    layout = resolve_layout(args)
    args.run_root = layout.run_root
    args.registry = layout.registry_path
    args.desktop_pool_root = layout.pool_root

    config_dir = layout.node_configs_dir(args.node_rank)
    log_dir = layout.node_logs_dir(args.node_rank)
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    replica_count = args.replica_count or args.servers_per_node
    replica_offset = args.replica_offset
    if replica_offset < 0:
        replica_offset = args.node_rank * args.servers_per_node

    specs = make_server_specs(
        host=args.host,
        bind_host=args.bind_host,
        base_port=args.base_port,
        node_rank=args.node_rank,
        servers_per_node=args.servers_per_node,
        workers_per_server=args.workers_per_server,
        replica_count=replica_count,
        replica_offset=replica_offset,
        name_prefix=args.env_name_prefix,
        config_dir=config_dir,
        log_dir=log_dir,
        pool_status_root=layout.pool_status_dir,
    )

    metadata = registry_metadata(args, layout, replica_count)
    for spec in specs:
        write_env_server_config(spec, args, metadata)

    registry = upsert_registry(
        path=layout.registry_path,
        run_id=args.run_id,
        metadata=metadata,
        servers=specs,
    )
    print(json.dumps(registry.as_dict(), indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    load_runtime_env_file()
    env = os.environ
    layout = FleetRunLayout.from_env(env)
    task_base_path = osworld_task_base_path()
    qcow_path = osworld_qcow_path(env=env)
    cache_dir = osworld_asset_cache_dir(env=env)
    artifact_output_dir = (
        Path(env["OSWORLD_ARTIFACT_DIR"]) if env.get("OSWORLD_ARTIFACT_DIR") else None
    )

    parser = argparse.ArgumentParser(
        description="Prepare PrimeRL env-server configs for a CPU OSWorld fleet."
    )
    parser.add_argument("--run-id", default=layout.run_id)
    parser.add_argument("--run-base", type=Path, default=layout.run_base)
    parser.add_argument("--run-root", type=Path, default=layout.run_root)
    parser.add_argument("--registry", type=Path, default=layout.registry_path)
    parser.add_argument("--configs-dir", type=Path, default=layout.configs_dir)
    parser.add_argument("--logs-dir", type=Path, default=layout.logs_dir)
    parser.add_argument(
        "--prime-rl-config-path",
        type=Path,
        default=layout.prime_rl_config_path,
    )
    parser.add_argument(
        "--prime-rl-output-dir",
        type=Path,
        default=layout.prime_rl_output_dir,
    )
    parser.add_argument(
        "--host", default=env.get("OSWORLD_FLEET_HOST") or default_public_host()
    )
    parser.add_argument(
        "--bind-host", default=env.get("OSWORLD_FLEET_BIND_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--base-port", type=int, default=int(env.get("OSWORLD_FLEET_BASE_PORT", "5200"))
    )
    parser.add_argument(
        "--node-rank", type=int, default=int(env.get("SLURM_PROCID", "0"))
    )
    parser.add_argument(
        "--servers-per-node",
        type=int,
        default=int(env.get("OSWORLD_ENV_SERVERS_PER_NODE", "1")),
    )
    parser.add_argument(
        "--workers-per-server",
        type=int,
        default=int(env.get("OSWORLD_ENV_WORKERS_PER_SERVER", "1")),
    )
    parser.add_argument(
        "--replica-count",
        type=int,
        default=int(env.get("OSWORLD_ENV_REPLICA_COUNT", "0")),
    )
    parser.add_argument(
        "--replica-offset",
        type=int,
        default=int(env.get("OSWORLD_ENV_REPLICA_OFFSET", "-1")),
    )
    parser.add_argument(
        "--replica-hosts", default=env.get("OSWORLD_ENV_REPLICA_HOSTS", "")
    )
    parser.add_argument("--gateway-host", default=env.get("OSWORLD_GATEWAY_HOST"))
    parser.add_argument(
        "--gateway-bind-host", default=env.get("OSWORLD_GATEWAY_BIND_HOST")
    )
    parser.add_argument(
        "--gateway-port", type=int, default=int(env.get("OSWORLD_GATEWAY_PORT", "0"))
    )
    parser.add_argument("--env-id", default=env.get("OSWORLD_ENV_ID", "rl"))
    parser.add_argument(
        "--env-name-prefix",
        default=env.get("OSWORLD_ENV_NAME_PREFIX", "osworld-target-box"),
    )
    parser.set_defaults(task_base_path=task_base_path)
    parser.add_argument(
        "--max-tasks", type=int, default=int(env.get("OSWORLD_MAX_TASKS", "0"))
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=int(env.get("OSWORLD_TASK_SHUFFLE_SEED", "0")),
    )
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument(
        "--screen-width", type=int, default=int(env.get("OSWORLD_SCREEN_WIDTH", "1920"))
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=int(env.get("OSWORLD_SCREEN_HEIGHT", "1080")),
    )
    parser.add_argument(
        "--screenshot-timeout",
        type=float,
        default=float(env.get("OSWORLD_SCREENSHOT_TIMEOUT", "60.0")),
    )
    parser.add_argument(
        "--artifact-output-dir",
        type=Path,
        default=artifact_output_dir,
        help="Artifact root used by env workers. Defaults to <run-root>/artifacts.",
    )
    parser.add_argument(
        "--desktop-pool-min-ready-sessions",
        type=int,
        default=int(env.get("OSWORLD_DESKTOP_POOL_MIN_READY_SESSIONS", "1")),
    )
    parser.add_argument(
        "--desktop-pool-max-sessions",
        type=int,
        default=int(env.get("OSWORLD_DESKTOP_POOL_MAX_SESSIONS", "1")),
    )
    parser.add_argument(
        "--desktop-pool-max-rollouts-per-session",
        type=int,
        default=int(env.get("OSWORLD_DESKTOP_POOL_MAX_ROLLOUTS_PER_SESSION", "1")),
    )
    parser.add_argument(
        "--desktop-pool-checkout-timeout",
        type=float,
        default=float(env.get("OSWORLD_DESKTOP_POOL_CHECKOUT_TIMEOUT", "900")),
    )
    parser.add_argument(
        "--desktop-pool-lease-timeout",
        type=float,
        default=float(env.get("OSWORLD_DESKTOP_POOL_LEASE_TIMEOUT", "300")),
    )
    parser.add_argument(
        "--desktop-pool-startup-timeout",
        type=float,
        default=float(env.get("OSWORLD_DESKTOP_POOL_STARTUP_TIMEOUT", "840")),
    )
    parser.add_argument(
        "--desktop-pool-startup-retry-backoff",
        type=float,
        default=float(env.get("OSWORLD_DESKTOP_POOL_STARTUP_RETRY_BACKOFF", "30")),
    )
    parser.add_argument(
        "--desktop-pool-startup-retry-backoff-max",
        type=float,
        default=float(env.get("OSWORLD_DESKTOP_POOL_STARTUP_RETRY_BACKOFF_MAX", "300")),
    )
    parser.add_argument(
        "--desktop-pool-status-heartbeat-interval",
        type=float,
        default=float(env.get("OSWORLD_DESKTOP_POOL_STATUS_HEARTBEAT_INTERVAL", "10")),
    )
    parser.add_argument(
        "--desktop-pool-root",
        type=Path,
        default=layout.pool_root,
    )
    parser.add_argument(
        "--desktop-pool-runtime-dir",
        type=Path,
        default=(
            Path(env["OSWORLD_DESKTOP_POOL_RUNTIME_DIR"])
            if env.get("OSWORLD_DESKTOP_POOL_RUNTIME_DIR")
            else None
        ),
    )
    parser.add_argument(
        "--desktop-pool-log-runtime-dir",
        type=Path,
        default=(
            Path(env["OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR"])
            if env.get("OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR")
            else None
        ),
    )
    parser.add_argument(
        "--rollout-timeout",
        type=float,
        default=float(env.get("OSWORLD_ROLLOUT_TIMEOUT", "900")),
    )
    parser.add_argument(
        "--env-max-retries",
        type=int,
        default=int(env.get("OSWORLD_ENV_MAX_RETRIES", "2")),
    )
    parser.add_argument(
        "--qcow-path",
        type=Path,
        default=qcow_path,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=cache_dir,
    )
    return parser.parse_args()


def resolve_layout(args: argparse.Namespace) -> FleetRunLayout:
    """Build the effective layout after CLI overrides have been parsed."""
    return FleetRunLayout.for_run(
        run_id=args.run_id,
        run_base=args.run_base,
        run_root=args.run_root,
        registry_path=args.registry,
        pool_root=args.desktop_pool_root,
        logs_dir=args.logs_dir,
        configs_dir=args.configs_dir,
        prime_rl_config_path=args.prime_rl_config_path,
        prime_rl_output_dir=args.prime_rl_output_dir,
    )


def registry_metadata(
    args: argparse.Namespace,
    layout: FleetRunLayout,
    replica_count: int,
) -> dict[str, Any]:
    """Build the static service contract written to the fleet registry."""
    metadata = {
        "run_id": args.run_id,
        "env_id": args.env_id,
        "env_name_prefix": args.env_name_prefix,
        "task_base_path": str(args.task_base_path),
        "max_tasks": args.max_tasks,
        "shuffle_seed": args.shuffle_seed,
        "harness": harness_config(args),
        "gateway": gateway_config(args, replica_count),
        "layout": layout.as_metadata(),
        "expected_env_servers": replica_count,
        "expected_env_workers": replica_count * args.workers_per_server,
        "expected_ready_sessions": expected_ready_sessions(args, replica_count),
        "slurm": slurm_metadata(os.environ),
    }
    return metadata


def harness_config(args: argparse.Namespace) -> dict[str, Any]:
    desktop_pool_config = {
        "min_ready_sessions": args.desktop_pool_min_ready_sessions,
        "max_sessions": args.desktop_pool_max_sessions,
        "max_rollouts_per_session": args.desktop_pool_max_rollouts_per_session,
        "checkout_timeout_s": args.desktop_pool_checkout_timeout,
        "lease_timeout_s": args.desktop_pool_lease_timeout,
        "startup_timeout_s": args.desktop_pool_startup_timeout,
        "startup_retry_backoff_s": args.desktop_pool_startup_retry_backoff,
        "startup_retry_backoff_max_s": (args.desktop_pool_startup_retry_backoff_max),
        "status_heartbeat_interval_s": (args.desktop_pool_status_heartbeat_interval),
        "root_dir": str(args.desktop_pool_root),
    }
    desktop_pool_runtime_dir = getattr(args, "desktop_pool_runtime_dir", None)
    if desktop_pool_runtime_dir is not None:
        desktop_pool_config["runtime_dir"] = str(desktop_pool_runtime_dir)
    desktop_pool_log_runtime_dir = getattr(args, "desktop_pool_log_runtime_dir", None)
    if desktop_pool_log_runtime_dir is not None:
        desktop_pool_config["log_runtime_dir"] = str(desktop_pool_log_runtime_dir)
    return {
        "max_steps": args.max_steps,
        "desktop": {
            "screen_width": args.screen_width,
            "screen_height": args.screen_height,
            "screenshot_timeout": args.screenshot_timeout,
            "cache_dir": str(args.cache_dir),
            "output_dir": str(resolve_artifact_output_dir(args)),
            "qcow_path": str(args.qcow_path),
            "desktop_pool_config": desktop_pool_config,
        },
    }


def resolve_artifact_output_dir(args: argparse.Namespace) -> Path:
    output_dir = vars(args).get("artifact_output_dir")
    if output_dir is not None:
        return Path(output_dir)
    return args.run_root / "artifacts"


def gateway_config(args: argparse.Namespace, replica_count: int) -> dict[str, Any]:
    """Build the logical rollout gateway endpoint and backend list."""
    gateway_port = args.gateway_port or args.base_port + replica_count
    replica_hosts = resolve_replica_hosts(args)
    gateway_host = args.gateway_host or (
        replica_hosts[0] if replica_hosts else args.host
    )
    bind_host = args.gateway_bind_host or args.bind_host
    return {
        "bind_address": f"tcp://{bind_host}:{gateway_port}",
        "public_address": f"tcp://{gateway_host}:{gateway_port}",
        "backend_addresses": backend_addresses(args, replica_count, replica_hosts),
    }


def backend_addresses(
    args: argparse.Namespace,
    replica_count: int,
    replica_hosts: list[str],
) -> list[str]:
    """Return env-server public addresses in replica-index order."""
    addresses: list[str] = []
    for replica_index in range(replica_count):
        node_rank = replica_index // args.servers_per_node
        local_index = replica_index % args.servers_per_node
        host = replica_hosts[node_rank] if node_rank < len(replica_hosts) else args.host
        port = args.base_port + node_rank * args.servers_per_node + local_index
        addresses.append(f"tcp://{host}:{port}")
    return addresses


def resolve_replica_hosts(args: argparse.Namespace) -> list[str]:
    """Resolve one routable host per fleet node."""
    explicit_hosts = split_csv(vars(args).get("replica_hosts", ""))
    if explicit_hosts:
        return explicit_hosts
    slurm_hosts = slurm_node_addrs(os.environ)
    if slurm_hosts:
        return slurm_hosts
    return [args.host]


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def expected_ready_sessions(args: argparse.Namespace, replica_count: int) -> int:
    """Compute the mandatory warm sessions required before trainer launch."""
    return (
        replica_count * args.workers_per_server * args.desktop_pool_min_ready_sessions
    )


if __name__ == "__main__":
    raise SystemExit(main())
