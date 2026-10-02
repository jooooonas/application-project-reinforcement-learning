#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rl.runtime.envfile import load_runtime_env_file
from rl.runtime.fleet import EnvServerSpec, FleetRunLayout, read_registry
from rl.runtime.zmq_gateway import run_gateway

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=LOG_LEVELS[args.log_level],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = resolve_gateway_config(args)
    logging.info(
        "Starting rollout gateway on %s for %d backend(s)",
        config["bind_address"],
        len(config["backend_addresses"]),
    )
    run_gateway(
        bind_address=config["bind_address"],
        backend_addresses=config["backend_addresses"],
        backend_status_dirs=config.get("backend_status_dirs"),
        health_check_interval=args.health_check_interval,
        health_check_timeout=args.health_check_timeout,
        request_timeout_s=args.request_timeout_s,
        backend_quarantine_s=args.backend_quarantine_s,
        capacity_check_interval=args.capacity_check_interval,
        capacity_startup_grace_s=args.capacity_startup_grace_s,
        capacity_wait_timeout_s=args.capacity_wait_timeout_s,
        max_pending_requests=args.max_pending_requests,
        status_stale_after_s=args.status_stale_after_s,
    )
    return 0


def parse_args() -> argparse.Namespace:
    load_runtime_env_file()
    layout = FleetRunLayout.from_env(os.environ)
    parser = argparse.ArgumentParser(
        description="Run a ZMQ rollout gateway for OSWorld env-server replicas."
    )
    parser.add_argument("--registry", type=Path, default=layout.registry_path)
    parser.add_argument("--bind-address")
    parser.add_argument("--backend-address", action="append", default=[])
    parser.add_argument("--backend-status-dir", action="append", default=[])
    parser.add_argument("--wait-timeout-s", type=float, default=600.0)
    parser.add_argument("--poll-s", type=float, default=1.0)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    parser.add_argument("--health-check-timeout", type=float, default=5.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--backend-quarantine-s", type=float, default=30.0)
    parser.add_argument("--capacity-check-interval", type=float, default=5.0)
    parser.add_argument("--capacity-startup-grace-s", type=float, default=900.0)
    parser.add_argument("--capacity-wait-timeout-s", type=float)
    parser.add_argument("--max-pending-requests", type=int, default=0)
    parser.add_argument("--status-stale-after-s", type=float, default=120.0)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def resolve_gateway_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.registry:
        return wait_for_registry_gateway(args)
    if not args.bind_address:
        raise ValueError("--bind-address is required without --registry")
    if not args.backend_address:
        raise ValueError("--backend-address is required without --registry")
    return {
        "bind_address": args.bind_address,
        "backend_addresses": list(args.backend_address),
        "backend_status_dirs": list(args.backend_status_dir) or None,
    }


def wait_for_registry_gateway(args: argparse.Namespace) -> dict[str, Any]:
    deadline = time.monotonic() + args.wait_timeout_s
    last_error = "registry not read yet"
    while time.monotonic() <= deadline:
        try:
            registry = read_registry(args.registry)
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(args.poll_s)
            continue
        metadata = registry.metadata
        expected = int(metadata.get("expected_env_servers", 0) or 0)
        backend_addresses = list(args.backend_address) or gateway_backend_addresses(
            metadata,
            registry.servers,
        )
        backend_status_dirs = list(
            args.backend_status_dir
        ) or gateway_backend_status_dirs(
            registry.servers,
            backend_addresses,
        )
        bind_address = args.bind_address or gateway_bind_address(metadata)
        if bind_address and backend_addresses and len(registry.servers) >= expected:
            return {
                "bind_address": bind_address,
                "backend_addresses": backend_addresses,
                "backend_status_dirs": backend_status_dirs,
            }
        last_error = (
            f"bind={bool(bind_address)} backends={len(backend_addresses)} "
            f"registered={len(registry.servers)} expected={expected}"
        )
        time.sleep(args.poll_s)
    raise TimeoutError(f"gateway registry metadata was not ready: {last_error}")


def gateway_bind_address(metadata: Mapping[str, Any]) -> str | None:
    gateway = metadata.get("gateway")
    if not isinstance(gateway, Mapping):
        return None
    address = gateway.get("bind_address")
    return str(address) if address else None


def gateway_backend_addresses(
    metadata: Mapping[str, Any],
    servers: list[EnvServerSpec],
) -> list[str]:
    gateway = metadata.get("gateway")
    if isinstance(gateway, Mapping):
        addresses = gateway.get("backend_addresses")
        if isinstance(addresses, list):
            return [str(address) for address in addresses if address]
    return [server.public_address for server in servers]


def gateway_backend_status_dirs(
    servers: list[EnvServerSpec],
    backend_addresses: list[str],
) -> list[str | None]:
    """Return pool status dirs ordered like gateway backend addresses."""
    by_address = {server.public_address: server.pool_status_dir for server in servers}
    return [by_address.get(address) for address in backend_addresses]


if __name__ == "__main__":
    raise SystemExit(main())
