from __future__ import annotations

import os
import socket
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvServerSpec:
    name: str
    host: str
    port: int
    bind_address: str
    public_address: str
    num_workers: int
    node_rank: int
    local_index: int
    replica_index: int
    replica_count: int
    config_path: str
    log_path: str
    pool_status_dir: str | None = None
    status: str = "starting"


def default_public_host(
    env: Mapping[str, str] = os.environ,
    *,
    run_command: Callable[[list[str]], str] | None = None,
    fqdn_func: Callable[[], str] = socket.getfqdn,
    hostname_func: Callable[[], str] = socket.gethostname,
) -> str:
    if explicit_host := env.get("OSWORLD_FLEET_HOST"):
        return explicit_host

    if slurm_host := _slurm_node_address(
        env,
        run_command=run_command,
        hostname_func=hostname_func,
    ):
        return slurm_host

    return fqdn_func() or hostname_func()


def make_server_specs(
    *,
    host: str,
    bind_host: str,
    base_port: int,
    node_rank: int,
    servers_per_node: int,
    workers_per_server: int,
    replica_count: int,
    replica_offset: int,
    name_prefix: str,
    config_dir: Path,
    log_dir: Path,
    pool_status_root: Path | None = None,
) -> list[EnvServerSpec]:
    specs: list[EnvServerSpec] = []
    for local_index in range(servers_per_node):
        replica_index = replica_offset + local_index
        port = base_port + node_rank * servers_per_node + local_index
        name = f"{name_prefix}-{replica_index:04d}"
        pool_status_dir = (
            str(Path(pool_status_root) / name) if pool_status_root is not None else None
        )
        specs.append(
            EnvServerSpec(
                name=name,
                host=host,
                port=port,
                bind_address=f"tcp://{bind_host}:{port}",
                public_address=f"tcp://{host}:{port}",
                num_workers=workers_per_server,
                node_rank=node_rank,
                local_index=local_index,
                replica_index=replica_index,
                replica_count=replica_count,
                config_path=str(config_dir / f"{name}.toml"),
                log_path=str(log_dir / f"{name}.log"),
                pool_status_dir=pool_status_dir,
            )
        )
    return specs


def _slurm_node_address(
    env: Mapping[str, str],
    *,
    run_command: Callable[[list[str]], str] | None,
    hostname_func: Callable[[], str],
) -> str | None:
    node_name = env.get("SLURMD_NODENAME") or hostname_func()
    if not node_name:
        return None

    candidates = [node_name]
    if "." in node_name:
        candidates.append(node_name.split(".", 1)[0])

    for candidate in dict.fromkeys(candidates):
        try:
            output = (
                run_command(["scontrol", "show", "node", candidate])
                if run_command is not None
                else _run_scontrol_show_node(candidate)
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if node_addr := _parse_scontrol_node_addr(output):
            return node_addr
    return None


def _run_scontrol_show_node(node_name: str) -> str:
    result = subprocess.run(
        ["scontrol", "show", "node", node_name],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _parse_scontrol_node_addr(output: str) -> str | None:
    for node_field in output.split():
        if not node_field.startswith("NodeAddr="):
            continue
        node_addr = node_field.split("=", 1)[1]
        if node_addr and node_addr != "(null)":
            return node_addr
    return None
