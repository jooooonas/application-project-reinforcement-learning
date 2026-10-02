#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from rl.runtime.fleet import read_registry
from rl.runtime.fleet.readiness import (
    active_worker_statuses,
    read_statuses,
    stale_worker_statuses,
    sum_int_field,
)


@dataclass(frozen=True)
class PoolHealth:
    status_files: int
    active_status_files: int
    stale_status_files: int
    ready: int
    starting: int
    fresh_starting: int
    stale_starting: int
    oldest_starting_age_s: float | None
    leased: int
    total_failed: int
    last_errors: list[str]

    @property
    def usable_capacity(self) -> int:
        return self.ready + self.leased

    @property
    def startup_capacity(self) -> int:
        return self.ready + self.fresh_starting + self.leased


@dataclass(frozen=True)
class SupervisorPolicy:
    poll_s: float
    startup_grace_s: float
    replica_unhealthy_s: float
    failure_window_s: float
    max_failures_per_window: int
    restart_backoff_s: float
    fleet_unhealthy_s: float
    max_fleet_restarts: int
    terminate_timeout_s: float
    status_stale_after_s: float


@dataclass
class ReplicaRuntime:
    name: str
    config_path: Path
    log_path: Path
    status_dir: Path | None
    command: list[str]
    process: subprocess.Popen[Any] | None = None
    started_at: float = 0.0
    restart_count: int = 0
    next_restart_at: float = 0.0
    unhealthy_since: float | None = None
    last_total_failed: int = 0
    failure_window_started_at: float = 0.0
    failures_in_window: int = 0
    last_restart_reason: str | None = None


LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


class StartingSessionSummary(TypedDict):
    fresh: int
    stale: int
    oldest_age_s: float | None


def gateway_script_path() -> Path:
    return Path(__file__).resolve().with_name("zmq_rollout_gateway.py")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=LOG_LEVELS[args.log_level],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    policy = SupervisorPolicy(
        poll_s=args.poll_s,
        startup_grace_s=args.startup_grace_s,
        replica_unhealthy_s=args.replica_unhealthy_s,
        failure_window_s=args.failure_window_s,
        max_failures_per_window=args.max_failures_per_window,
        restart_backoff_s=args.restart_backoff_s,
        fleet_unhealthy_s=args.fleet_unhealthy_s,
        max_fleet_restarts=args.max_fleet_restarts,
        terminate_timeout_s=args.terminate_timeout_s,
        status_stale_after_s=args.status_stale_after_s,
    )
    supervisor = FleetSupervisor(args=args, policy=policy)
    return supervisor.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervise OSWorld env-server replicas inside one Slurm allocation."
    )
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--env-server-bin", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--start-gateway", action="store_true")
    parser.add_argument("--gateway-log", type=Path)
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument("--startup-grace-s", type=float, default=900.0)
    parser.add_argument("--replica-unhealthy-s", type=float, default=120.0)
    parser.add_argument("--failure-window-s", type=float, default=300.0)
    parser.add_argument("--max-failures-per-window", type=int, default=8)
    parser.add_argument("--restart-backoff-s", type=float, default=10.0)
    parser.add_argument("--fleet-unhealthy-s", type=float, default=300.0)
    parser.add_argument("--max-fleet-restarts", type=int, default=3)
    parser.add_argument("--terminate-timeout-s", type=float, default=30.0)
    parser.add_argument("--status-stale-after-s", type=float, default=120.0)
    parser.add_argument("--gateway-request-timeout-s", type=float, default=900.0)
    parser.add_argument("--gateway-backend-quarantine-s", type=float, default=30.0)
    parser.add_argument("--gateway-capacity-check-interval", type=float, default=5.0)
    parser.add_argument("--gateway-health-check-interval", type=float, default=2.0)
    parser.add_argument("--gateway-health-check-timeout", type=float, default=5.0)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


class FleetSupervisor:
    def __init__(self, *, args: argparse.Namespace, policy: SupervisorPolicy):
        self.args = args
        self.policy = policy
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stop_requested = False
        self.fleet_unhealthy_since: float | None = None
        self.fleet_restart_count = 0
        self.gateway_process: subprocess.Popen[Any] | None = None
        self.gateway_started_at = 0.0
        self.run_root = args.run_root or args.registry.parent
        self.status_path = args.logs_dir / "supervisor_status.json"
        self.unrecoverable_path = self.run_root / "fleet_unrecoverable.json"
        self.replicas = self.load_replicas()

    def run(self) -> int:
        self.args.logs_dir.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.install_signal_handlers()

        try:
            self.start_replicas(self.replicas, reason="initial start")
            if self.args.start_gateway:
                self.start_gateway(reason="initial start")

            while not self.stop_requested:
                now = time.monotonic()
                self.monitor_replicas(now)
                self.monitor_gateway()
                if self.maybe_restart_fleet(now):
                    return 42
                self.write_status(now)
                time.sleep(self.policy.poll_s)
        finally:
            self.stop_gateway()
            self.stop_replicas(self.replicas)
            self.write_status(time.monotonic())
        return 0

    def install_signal_handlers(self) -> None:
        def request_stop(_signum: int, _frame: Any) -> None:
            self.stop_requested = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

    def load_replicas(self) -> list[ReplicaRuntime]:
        registry = read_registry(self.args.registry)
        by_name = {server.name: server for server in registry.servers}
        replicas: list[ReplicaRuntime] = []
        for config_path in sorted(self.args.config_dir.glob("*.toml")):
            name = config_path.stem
            server = by_name.get(name)
            status_dir = (
                Path(server.pool_status_dir)
                if server is not None and server.pool_status_dir
                else None
            )
            replicas.append(
                ReplicaRuntime(
                    name=name,
                    config_path=config_path,
                    log_path=self.args.logs_dir / f"{name}.stdout.log",
                    status_dir=status_dir,
                    command=[self.args.env_server_bin, "@", str(config_path)],
                )
            )
        if not replicas:
            raise RuntimeError(f"no env-server configs found in {self.args.config_dir}")
        return replicas

    def start_replicas(self, replicas: list[ReplicaRuntime], *, reason: str) -> None:
        for replica in replicas:
            self.start_replica(replica, reason=reason)

    def start_replica(self, replica: ReplicaRuntime, *, reason: str) -> None:
        replica.log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = replica.log_path.open("ab")
        try:
            replica.process = subprocess.Popen(
                replica.command,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                cwd=Path.cwd(),
                env=os.environ.copy(),
                start_new_session=True,
            )
        finally:
            stdout.close()
        now = time.monotonic()
        replica.started_at = now
        replica.next_restart_at = now + self.policy.restart_backoff_s
        replica.unhealthy_since = None
        replica.failure_window_started_at = now
        replica.failures_in_window = 0
        replica.last_total_failed = 0
        replica.last_restart_reason = reason
        self.logger.info(
            "Started env-server %s pid=%s reason=%s",
            replica.name,
            replica.process.pid if replica.process else "?",
            reason,
        )

    def restart_replica(self, replica: ReplicaRuntime, *, reason: str) -> None:
        now = time.monotonic()
        if now < replica.next_restart_at:
            return
        self.logger.warning("Restarting env-server %s: %s", replica.name, reason)
        self.stop_replica(replica)
        archive_status_files(replica.status_dir, replica.name)
        replica.restart_count += 1
        self.start_replica(replica, reason=reason)

    def stop_replicas(self, replicas: list[ReplicaRuntime]) -> None:
        for replica in replicas:
            self.stop_replica(replica)

    def stop_replica(self, replica: ReplicaRuntime) -> None:
        terminate_process(replica.process, timeout_s=self.policy.terminate_timeout_s)
        cleanup_owned_process_groups(
            replica.status_dir,
            timeout_s=self.policy.terminate_timeout_s,
            logger=self.logger,
        )
        replica.process = None

    def monitor_replicas(self, now: float) -> None:
        for replica in self.replicas:
            health = read_pool_health(
                replica.status_dir,
                status_stale_after_s=self.policy.status_stale_after_s,
            )
            observe_failure_window(replica, health, now=now, policy=self.policy)
            reason = restart_reason(replica, health, now=now, policy=self.policy)
            if reason is not None:
                self.restart_replica(replica, reason=reason)

    def monitor_gateway(self) -> None:
        if not self.args.start_gateway:
            return
        if self.gateway_process is None:
            self.start_gateway(reason="missing process")
            return
        return_code = self.gateway_process.poll()
        if return_code is None:
            return
        self.logger.warning("Rollout gateway exited with code %s", return_code)
        self.start_gateway(reason=f"process exited with code {return_code}")

    def start_gateway(self, *, reason: str) -> None:
        self.stop_gateway()
        log_path = (
            self.args.gateway_log or self.args.logs_dir / "rollout-gateway.stdout.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        script_path = gateway_script_path()
        command = [
            self.args.python,
            str(script_path),
            "--registry",
            str(self.args.registry),
            "--request-timeout-s",
            str(self.args.gateway_request_timeout_s),
            "--backend-quarantine-s",
            str(self.args.gateway_backend_quarantine_s),
            "--capacity-check-interval",
            str(self.args.gateway_capacity_check_interval),
            "--status-stale-after-s",
            str(self.args.status_stale_after_s),
            "--health-check-interval",
            str(self.args.gateway_health_check_interval),
            "--health-check-timeout",
            str(self.args.gateway_health_check_timeout),
        ]
        stdout = log_path.open("ab")
        try:
            self.gateway_process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                cwd=script_path.parent.parent,
                env=os.environ.copy(),
                start_new_session=True,
            )
        finally:
            stdout.close()
        self.gateway_started_at = time.monotonic()
        self.logger.info(
            "Started rollout gateway pid=%s reason=%s",
            self.gateway_process.pid if self.gateway_process else "?",
            reason,
        )

    def stop_gateway(self) -> None:
        terminate_process(
            self.gateway_process, timeout_s=self.policy.terminate_timeout_s
        )
        self.gateway_process = None

    def maybe_restart_fleet(self, now: float) -> bool:
        if any(
            replica_healthy(replica, now=now, policy=self.policy)
            for replica in self.replicas
        ):
            self.fleet_unhealthy_since = None
            return False
        if self.fleet_unhealthy_since is None:
            self.fleet_unhealthy_since = now
            return False
        if now - self.fleet_unhealthy_since < self.policy.fleet_unhealthy_s:
            return False

        self.fleet_restart_count += 1
        if (
            self.policy.max_fleet_restarts >= 0
            and self.fleet_restart_count > self.policy.max_fleet_restarts
        ):
            self.write_unrecoverable_marker(now)
            return True

        self.logger.warning(
            "Restarting whole env fleet after %.1fs without a healthy replica",
            now - self.fleet_unhealthy_since,
        )
        self.stop_gateway()
        self.stop_replicas(self.replicas)
        for replica in self.replicas:
            archive_status_files(replica.status_dir, replica.name)
        self.start_replicas(self.replicas, reason="fleet unhealthy")
        if self.args.start_gateway:
            self.start_gateway(reason="fleet unhealthy")
        self.fleet_unhealthy_since = None
        return False

    def write_unrecoverable_marker(self, now: float) -> None:
        payload = {
            "status": "unrecoverable",
            "updated_at": time.time(),
            "fleet_restart_count": self.fleet_restart_count,
            "max_fleet_restarts": self.policy.max_fleet_restarts,
            "unhealthy_for_s": (
                0.0
                if self.fleet_unhealthy_since is None
                else now - self.fleet_unhealthy_since
            ),
            "replicas": [
                replica_status(replica, now=now, policy=self.policy)
                for replica in self.replicas
            ],
        }
        write_json_atomic(self.unrecoverable_path, payload)
        self.logger.error("Fleet marked unrecoverable: %s", self.unrecoverable_path)

    def write_status(self, now: float) -> None:
        payload = {
            "updated_at": time.time(),
            "registry": str(self.args.registry),
            "fleet_restart_count": self.fleet_restart_count,
            "fleet_unhealthy_since": self.fleet_unhealthy_since,
            "gateway": {
                "enabled": self.args.start_gateway,
                "pid": self.gateway_process.pid if self.gateway_process else None,
                "return_code": (
                    self.gateway_process.poll() if self.gateway_process else None
                ),
            },
            "replicas": [
                replica_status(replica, now=now, policy=self.policy)
                for replica in self.replicas
            ],
        }
        write_json_atomic(self.status_path, payload)


def read_pool_health(
    status_dir: Path | None,
    *,
    status_stale_after_s: float | None = None,
) -> PoolHealth:
    if status_dir is None:
        return PoolHealth(
            status_files=0,
            active_status_files=0,
            stale_status_files=0,
            ready=0,
            starting=0,
            fresh_starting=0,
            stale_starting=0,
            oldest_starting_age_s=None,
            leased=0,
            total_failed=0,
            last_errors=[],
        )
    statuses = read_statuses(status_dir, recursive=False)
    now = time.time()
    active_statuses = active_worker_statuses(
        statuses,
        now=now,
        stale_after_s=status_stale_after_s,
    )
    stale_statuses = stale_worker_statuses(
        statuses,
        now=now,
        stale_after_s=status_stale_after_s,
    )
    last_errors = [
        str(status["last_error"])
        for status in active_statuses
        if status.get("last_error")
    ]
    starting_details = summarize_starting_sessions(active_statuses, now=now)
    return PoolHealth(
        status_files=len(statuses),
        active_status_files=len(active_statuses),
        stale_status_files=len(stale_statuses),
        ready=sum_int_field(active_statuses, "ready"),
        starting=sum_int_field(active_statuses, "starting"),
        fresh_starting=starting_details["fresh"],
        stale_starting=starting_details["stale"],
        oldest_starting_age_s=starting_details["oldest_age_s"],
        leased=sum_int_field(active_statuses, "leased"),
        total_failed=sum_int_field(active_statuses, "total_failed"),
        last_errors=last_errors[-5:],
    )


def summarize_starting_sessions(
    statuses: Sequence[Mapping[str, Any]],
    *,
    now: float,
) -> StartingSessionSummary:
    fresh = 0
    stale = 0
    oldest_age_s: float | None = None
    for status in statuses:
        startup_timeout_s = _positive_float_or_none(status.get("startup_timeout_s"))
        sessions = status.get("starting_sessions")
        if not isinstance(sessions, list):
            count = _nonnegative_int(status.get("starting"))
            fresh += count
            continue
        for session in sessions:
            if not isinstance(session, dict):
                fresh += 1
                continue
            created_at = _positive_float_or_none(session.get("created_at"))
            age_s = (
                max(0.0, now - created_at)
                if created_at is not None
                else _positive_float_or_none(session.get("age_s"))
            )
            if age_s is not None:
                oldest_age_s = (
                    age_s if oldest_age_s is None else max(oldest_age_s, age_s)
                )
            if (
                startup_timeout_s is not None
                and age_s is not None
                and age_s >= startup_timeout_s
            ):
                stale += 1
            else:
                fresh += 1
    return {"fresh": fresh, "stale": stale, "oldest_age_s": oldest_age_s}


def _positive_float_or_none(value: object) -> float | None:
    if isinstance(value, int | float) and value >= 0:
        return float(value)
    return None


def _nonnegative_int(value: object) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def observe_failure_window(
    replica: ReplicaRuntime,
    health: PoolHealth,
    *,
    now: float,
    policy: SupervisorPolicy,
) -> None:
    if now - replica.failure_window_started_at > policy.failure_window_s:
        replica.failure_window_started_at = now
        replica.failures_in_window = 0
    delta = max(0, health.total_failed - replica.last_total_failed)
    replica.last_total_failed = health.total_failed
    replica.failures_in_window += delta


def restart_reason(
    replica: ReplicaRuntime,
    health: PoolHealth,
    *,
    now: float,
    policy: SupervisorPolicy,
) -> str | None:
    process = replica.process
    if process is None:
        return "process missing"
    return_code = process.poll()
    if return_code is not None:
        return f"process exited with code {return_code}"

    if (
        policy.max_failures_per_window > 0
        and replica.failures_in_window >= policy.max_failures_per_window
    ):
        return (
            f"{replica.failures_in_window} desktop failures within "
            f"{policy.failure_window_s:.1f}s"
        )

    uptime = now - replica.started_at
    if uptime < policy.startup_grace_s:
        replica.unhealthy_since = None
        return None

    if health.active_status_files <= 0:
        reason = "no active desktop-pool worker status files"
    elif health.usable_capacity <= 0:
        if health.stale_starting > 0:
            reason = f"{health.stale_starting} desktop sessions stuck starting" + (
                ""
                if health.oldest_starting_age_s is None
                else f" for up to {health.oldest_starting_age_s:.1f}s"
            )
        elif health.fresh_starting > 0:
            replica.unhealthy_since = None
            return None
        else:
            reason = "no ready or leased desktop sessions"
    else:
        replica.unhealthy_since = None
        return None

    if replica.unhealthy_since is None:
        replica.unhealthy_since = now
        return None
    if now - replica.unhealthy_since >= policy.replica_unhealthy_s:
        return reason
    return None


def replica_healthy(
    replica: ReplicaRuntime,
    *,
    now: float,
    policy: SupervisorPolicy,
) -> bool:
    process = replica.process
    if process is None or process.poll() is not None:
        return False
    if now - replica.started_at < policy.startup_grace_s:
        return True
    health = read_pool_health(
        replica.status_dir,
        status_stale_after_s=policy.status_stale_after_s,
    )
    return health.usable_capacity > 0 or health.fresh_starting > 0


def replica_status(
    replica: ReplicaRuntime,
    *,
    now: float,
    policy: SupervisorPolicy,
) -> dict[str, Any]:
    health = read_pool_health(
        replica.status_dir,
        status_stale_after_s=policy.status_stale_after_s,
    )
    process = replica.process
    return {
        "name": replica.name,
        "pid": process.pid if process else None,
        "return_code": process.poll() if process else None,
        "healthy": replica_healthy(replica, now=now, policy=policy),
        "restart_count": replica.restart_count,
        "last_restart_reason": replica.last_restart_reason,
        "config_path": str(replica.config_path),
        "log_path": str(replica.log_path),
        "status_dir": str(replica.status_dir) if replica.status_dir else None,
        "pool": {
            "status_files": health.status_files,
            "active_status_files": health.active_status_files,
            "stale_status_files": health.stale_status_files,
            "ready": health.ready,
            "starting": health.starting,
            "fresh_starting": health.fresh_starting,
            "stale_starting": health.stale_starting,
            "oldest_starting_age_s": health.oldest_starting_age_s,
            "leased": health.leased,
            "total_failed": health.total_failed,
            "last_errors": health.last_errors,
        },
    }


def terminate_process(
    process: subprocess.Popen[Any] | None,
    *,
    timeout_s: float,
) -> None:
    if process is None or process.poll() is not None:
        return
    process_group_id = process_group_id_for_pid(process.pid)
    if not safe_process_group_id(process_group_id):
        process_group_id = None
    if process_group_id is not None:
        with suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if process_group_id is not None:
            with suppress(ProcessLookupError):
                os.killpg(process_group_id, signal.SIGKILL)
        else:
            process.kill()
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass


def cleanup_owned_process_groups(
    status_dir: Path | None,
    *,
    timeout_s: float,
    logger: logging.Logger,
) -> None:
    group_ids = owned_process_group_ids(status_dir)
    if not group_ids:
        return
    logger.info("Cleaning %d owned desktop process groups", len(group_ids))
    terminate_process_groups(group_ids, timeout_s=timeout_s)


def owned_process_group_ids(status_dir: Path | None) -> tuple[int, ...]:
    if status_dir is None or not status_dir.exists():
        return ()
    group_ids: list[int] = []
    for status in read_statuses(status_dir, recursive=False):
        for session in _mapping_list(status.get("sessions")):
            health = session.get("health")
            if isinstance(health, dict):
                _append_unique_positive_int(group_ids, health.get("vm_pgid"))
        for session in _mapping_list(status.get("starting_sessions")):
            _append_unique_positive_int(group_ids, session.get("apptainer_pgid"))
            pidfile = session.get("apptainer_pidfile")
            if isinstance(pidfile, str):
                _append_unique_positive_int(
                    group_ids,
                    process_group_id_from_pidfile(Path(pidfile)),
                )
    return tuple(group_ids)


def terminate_process_groups(group_ids: tuple[int, ...], *, timeout_s: float) -> None:
    group_ids = safe_process_group_ids(group_ids)
    if not group_ids:
        return
    for process_group_id in group_ids:
        with suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGTERM)
    if wait_for_process_groups_exit(group_ids, timeout_s=timeout_s):
        return
    for process_group_id in group_ids:
        with suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
    wait_for_process_groups_exit(group_ids, timeout_s=min(timeout_s, 5.0))


def safe_process_group_ids(group_ids: tuple[int, ...]) -> tuple[int, ...]:
    safe_ids: list[int] = []
    for process_group_id in group_ids:
        if safe_process_group_id(process_group_id) and process_group_id not in safe_ids:
            safe_ids.append(process_group_id)
    return tuple(safe_ids)


def safe_process_group_id(process_group_id: int | None) -> bool:
    return (
        process_group_id is not None
        and process_group_id > 0
        and process_group_id != os.getpgrp()
    )


def wait_for_process_groups_exit(
    group_ids: tuple[int, ...],
    *,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if not any(
            process_group_alive(process_group_id) for process_group_id in group_ids
        ):
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            return False
        time.sleep(min(0.05, remaining_s))


def process_group_id_from_pidfile(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    process_group_id = _positive_int_or_none(payload.get("pgid"))
    if process_group_id is not None:
        return process_group_id
    process_id = _positive_int_or_none(payload.get("pid"))
    if process_id is None:
        return None
    return process_group_id_for_pid(process_id)


def process_group_id_for_pid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def process_group_alive(process_group_id: int) -> bool:
    proc_status = linux_process_group_has_live_members(process_group_id)
    if proc_status is not None:
        return proc_status
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def linux_process_group_has_live_members(process_group_id: int) -> bool | None:
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        return None
    try:
        entries = list(proc_dir.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        stat = _linux_process_state_and_group(entry)
        if stat is None:
            continue
        state, member_process_group_id = stat
        if member_process_group_id == process_group_id and state != "Z":
            return True
    return False


def _linux_process_state_and_group(path: Path) -> tuple[str, int] | None:
    try:
        stat_text = (path / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    fields_start = stat_text.rfind(")")
    fields = stat_text[fields_start + 2 :].split() if fields_start >= 0 else []
    if len(fields) < 3:
        return None
    try:
        return fields[0], int(fields[2])
    except ValueError:
        return None


def _mapping_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _append_unique_positive_int(values: list[int], value: object) -> None:
    item = _positive_int_or_none(value)
    if item is not None and item not in values:
        values.append(item)


def _positive_int_or_none(value: object) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    return None


def archive_status_files(status_dir: Path | None, replica_name: str) -> Path | None:
    if status_dir is None or not status_dir.exists():
        return None
    paths = sorted(status_dir.glob("*.json"))
    if not paths:
        return None
    archive_dir = status_dir / "archive" / time.strftime("%Y%m%d-%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = archive_dir / path.name
        if target.exists():
            target = archive_dir / f"{path.stem}-{replica_name}{path.suffix}"
        path.replace(target)
    return archive_dir


def write_json_atomic(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


MappingLike = dict[str, Any]


if __name__ == "__main__":
    raise SystemExit(main())
