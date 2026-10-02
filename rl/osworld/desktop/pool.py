from __future__ import annotations

import json
import os
import socket
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
    cast,
)

from rl.runtime.ports import PortLease, allocate_worker_ports

SessionStatus = Literal["ready", "leased"]
RetireReason = Literal["retired", "failed"]


class DesktopSessionEnv(Protocol):
    def close(self) -> None: ...


class PortAllocator(Protocol):
    def __call__(
        self,
        *,
        lock_dir: str | Path,
        work_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
    ) -> PortLease: ...


@dataclass(frozen=True)
class DesktopPoolConfig:
    min_ready_sessions: int = 1
    max_sessions: int = 5
    max_rollouts_per_session: int = 50
    checkout_timeout_s: float = 900.0
    lease_timeout_s: float = 300.0
    startup_timeout_s: float = 840.0
    startup_retry_backoff_s: float = 30.0
    startup_retry_backoff_max_s: float = 300.0
    status_heartbeat_interval_s: float = 10.0
    root_dir: Path | None = None
    status_dir: Path | None = None
    runtime_dir: Path | None = None
    log_runtime_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate pool sizing, timeout, and path fields after construction."""
        if self.root_dir is not None and not isinstance(self.root_dir, Path):
            object.__setattr__(self, "root_dir", Path(self.root_dir))
        if self.status_dir is not None and not isinstance(self.status_dir, Path):
            object.__setattr__(self, "status_dir", Path(self.status_dir))
        if self.runtime_dir is not None and not isinstance(self.runtime_dir, Path):
            object.__setattr__(self, "runtime_dir", Path(self.runtime_dir))
        if self.log_runtime_dir is not None and not isinstance(
            self.log_runtime_dir,
            Path,
        ):
            object.__setattr__(
                self,
                "log_runtime_dir",
                Path(self.log_runtime_dir),
            )
        if self.min_ready_sessions < 0:
            raise ValueError("min_ready_sessions must be non-negative")
        if self.max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if self.min_ready_sessions > self.max_sessions:
            raise ValueError("min_ready_sessions cannot exceed max_sessions")
        if self.max_rollouts_per_session < 1:
            raise ValueError("max_rollouts_per_session must be at least 1")
        if self.checkout_timeout_s <= 0:
            raise ValueError("checkout_timeout_s must be positive")
        if self.lease_timeout_s <= 0:
            raise ValueError("lease_timeout_s must be positive")
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if self.startup_retry_backoff_s < 0:
            raise ValueError("startup_retry_backoff_s must be non-negative")
        if self.startup_retry_backoff_max_s <= 0:
            raise ValueError("startup_retry_backoff_max_s must be positive")
        if self.status_heartbeat_interval_s < 0:
            raise ValueError("status_heartbeat_interval_s must be non-negative")


@dataclass
class DesktopPoolSession[DesktopEnvT: DesktopSessionEnv]:
    session_id: str
    env: DesktopEnvT
    lease: PortLease
    status: SessionStatus
    rollouts_completed: int
    created_at: float
    updated_at: float
    leased_at: float | None = None
    last_activity_at: float | None = None
    last_error: str | None = None
    closed: bool = False


@dataclass
class StartingDesktopSession:
    session_id: str
    created_at: float
    updated_at: float
    lease: PortLease | None = None
    last_error: str | None = None


class CheckedOutDesktopSession[DesktopEnvT: DesktopSessionEnv]:
    def __init__(
        self,
        pool: DesktopSessionPool[DesktopEnvT],
        session: DesktopPoolSession[DesktopEnvT],
    ):
        self._pool = pool
        self._session = session
        self._released = False

    @property
    def env(self) -> DesktopEnvT:
        return self._session.env

    @property
    def session_id(self) -> str:
        return self._session.session_id

    def tracked_env(self) -> DesktopEnvT:
        """Return an env proxy that refreshes lease activity on method calls."""
        return cast(
            DesktopEnvT,
            _ActivityTrackedDesktopEnv(
                env=self._session.env,
                touch=lambda: self._pool.touch(self._session.session_id),
            ),
        )

    def touch(self) -> None:
        """Refresh activity for this checked-out session."""
        self._pool.touch(self._session.session_id)

    def release(self, *, failed: bool = False, error: str | None = None) -> None:
        """Return this leased session to the pool exactly once."""
        if self._released:
            return
        self._released = True
        self._pool.release(self._session.session_id, failed=failed, error=error)


class DesktopSessionPool[DesktopEnvT: DesktopSessionEnv]:
    """Prewarms OSWorld desktop sessions inside one env-worker process."""

    def __init__(
        self,
        *,
        config: DesktopPoolConfig,
        root_dir: Path,
        session_factory: Callable[[PortLease], DesktopEnvT],
        port_allocator: PortAllocator = allocate_worker_ports,
        worker_name: str | None = None,
        clock: Callable[[], float] = time.time,
    ):
        """Create pool bookkeeping and dependency hooks without starting sessions."""
        self.config = config
        self.root_dir = Path(root_dir)
        self.status_dir = (
            Path(config.status_dir)
            if config.status_dir is not None
            else self.root_dir / "status"
        )
        self.port_lock_dir = self.root_dir / "port_locks"
        self.runtime_dir = (
            Path(config.runtime_dir)
            if config.runtime_dir is not None
            else self.root_dir / "runtime"
        )
        self.log_dir = self.root_dir / "logs"
        self.log_write_dir = (
            Path(config.log_runtime_dir)
            if config.log_runtime_dir is not None
            else self.log_dir
        )
        self.artifact_dir = self.root_dir / "artifacts"
        self.status_path = (
            self.status_dir / f"{worker_name or _default_worker_name()}.json"
        )
        self._session_factory = session_factory
        self._port_allocator = port_allocator
        self._clock = clock
        self._condition = threading.Condition()
        self._sessions: dict[str, DesktopPoolSession[DesktopEnvT]] = {}
        self._starting_sessions: dict[str, StartingDesktopSession] = {}
        self._retiring_session_ids: set[str] = set()
        self._session_seq = 0
        self._closed = False
        self._started = False
        self._retry_scheduled = False
        self._total_started = 0
        self._total_failed = 0
        self._total_stale_leases_retired = 0
        self._last_error: str | None = None
        self._lease_watchdog_started = False
        self._status_heartbeat_started = False
        self._consecutive_start_failures = 0
        self._next_start_attempt_at: float | None = None

    def start(self) -> None:
        """Create pool directories and begin prewarming ready desktop sessions."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.port_lock_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.log_write_dir != self.log_dir:
            _ensure_symlink_dir(self.log_write_dir, self.log_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with self._condition:
            if self._started:
                return
            self._started = True
            self._ensure_min_ready_locked()
            self._start_lease_watchdog_locked()
            self._start_status_heartbeat_locked()
            self._write_status_locked()

    def checkout(
        self,
        *,
        timeout_s: float | None = None,
    ) -> CheckedOutDesktopSession[DesktopEnvT]:
        """Block until a ready desktop session is available or checkout times out."""
        if not self._started:
            self.start()
        effective_timeout_s = (
            self.config.checkout_timeout_s if timeout_s is None else timeout_s
        )
        if effective_timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = self._clock() + effective_timeout_s
        with self._condition:
            while True:
                self._raise_if_closed_locked()
                ready = self._ready_sessions_locked()
                if ready:
                    session = ready[0]
                    now = self._clock()
                    session.status = "leased"
                    session.leased_at = now
                    session.last_activity_at = now
                    session.updated_at = now
                    self._write_status_locked()
                    return CheckedOutDesktopSession(self, session)

                self._ensure_min_ready_locked()
                remaining_s = deadline - self._clock()
                if remaining_s <= 0:
                    raise TimeoutError(
                        "timed out waiting for a ready OSWorld desktop session "
                        f"after {effective_timeout_s:.1f}s"
                    )
                self._condition.wait(timeout=min(1.0, remaining_s))

    def touch(self, session_id: str) -> None:
        """Record client activity for a checked-out session."""
        with self._condition:
            session = self._sessions.get(session_id)
            if session is None or session.status != "leased":
                return
            now = self._clock()
            session.last_activity_at = now
            session.updated_at = now
            self._write_status_locked()

    def release(
        self,
        session_id: str,
        *,
        failed: bool = False,
        error: str | None = None,
    ) -> None:
        """Return a leased session and retire or reuse it according to policy."""
        session: DesktopPoolSession[DesktopEnvT] | None = None
        close_reason: RetireReason = "retired"
        with self._condition:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.rollouts_completed += 1
            session.updated_at = self._clock()
            session.last_activity_at = session.updated_at
            session.last_error = error
            should_retire = (
                failed
                or session.rollouts_completed >= self.config.max_rollouts_per_session
            )
            if should_retire:
                self._sessions.pop(session_id, None)
                self._retiring_session_ids.add(session_id)
                close_reason = "failed" if failed else "retired"
                if failed:
                    self._total_failed += 1
                    self._last_error = error
            else:
                session.status = "ready"
                session.leased_at = None
                session.last_error = None
            self._write_status_locked()
            self._condition.notify_all()

        if should_retire and session is not None:
            self._retire_session_async(session, reason=close_reason)
        else:
            with self._condition:
                self._ensure_min_ready_locked()
                self._write_status_locked()

    def close(self) -> None:
        """Stop the pool and close every session that is still tracked."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                self._retiring_session_ids.add(session.session_id)
            self._write_status_locked()
            self._condition.notify_all()

        for session in sessions:
            _close_session_resources(session)

        with self._condition:
            for session in sessions:
                self._retiring_session_ids.discard(session.session_id)
            self._write_status_locked()
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return self._status_payload_locked()

    def reap_stale_leases(self) -> int:
        """Retire sessions leased longer than the configured activity timeout."""
        stale_sessions: list[DesktopPoolSession[DesktopEnvT]] = []
        with self._condition:
            now = self._clock()
            for session in list(self._sessions.values()):
                if not self._session_is_stale_locked(session, now=now):
                    continue
                self._sessions.pop(session.session_id, None)
                self._retiring_session_ids.add(session.session_id)
                session.updated_at = now
                session.last_activity_at = now
                session.last_error = (
                    "lease timed out after "
                    f"{self.config.lease_timeout_s:.1f}s without activity"
                )
                self._total_failed += 1
                self._total_stale_leases_retired += 1
                self._last_error = session.last_error
                stale_sessions.append(session)
            if stale_sessions:
                self._write_status_locked()
                self._condition.notify_all()

        for session in stale_sessions:
            self._retire_session_async(session, reason="failed")
        return len(stale_sessions)

    def _ensure_min_ready_locked(self) -> None:
        """Start background sessions until the ready-session floor is covered."""
        if self._closed or not self._started or self._startup_cooling_down_locked():
            return
        while (
            len(self._ready_sessions_locked()) + len(self._starting_sessions)
            < self.config.min_ready_sessions
            and self._active_session_count_locked() < self.config.max_sessions
        ):
            self._start_session_async_locked()

    def _start_session_async_locked(self) -> None:
        """Reserve a session id and launch its startup thread."""
        self._session_seq += 1
        session_id = f"session-{self._session_seq:06d}"
        now = self._clock()
        self._starting_sessions[session_id] = StartingDesktopSession(
            session_id=session_id,
            created_at=now,
            updated_at=now,
        )
        self._write_status_locked()
        _start_daemon_thread(
            name=f"desktop-pool-start-{session_id}",
            target=self._start_session,
            session_id=session_id,
        )

    def _start_session(self, session_id: str) -> None:
        """Allocate ports and construct one desktop proxy for the pool."""
        lease: PortLease | None = None
        env: DesktopEnvT | None = None
        try:
            lease = self._port_allocator(
                lock_dir=self.port_lock_dir,
                work_dir=self.runtime_dir,
                log_dir=self.log_write_dir,
            )
            with self._condition:
                starting = self._starting_sessions.get(session_id)
                if starting is not None:
                    starting.lease = lease
                    starting.updated_at = self._clock()
                    self._write_status_locked()
                    self._condition.notify_all()
                if self._closed:
                    self._starting_sessions.pop(session_id, None)
                    self._write_status_locked()
                    self._condition.notify_all()
                    lease.release()
                    return
            env = self._session_factory(lease)
        except Exception as exc:
            if env is not None:
                _close_env(env)
            if lease is not None:
                lease.release()
            self._record_start_failure(session_id, exc)
            return

        with self._condition:
            starting = self._starting_sessions.get(session_id)
            created_at = self._clock() if starting is None else starting.created_at
        session = DesktopPoolSession[DesktopEnvT](
            session_id=session_id,
            env=env,
            lease=lease,
            status="ready",
            rollouts_completed=0,
            created_at=created_at,
            updated_at=self._clock(),
        )
        close_immediately = False
        with self._condition:
            self._starting_sessions.pop(session_id, None)
            if self._closed:
                close_immediately = True
            else:
                self._sessions[session_id] = session
                self._total_started += 1
                self._consecutive_start_failures = 0
                self._next_start_attempt_at = None
                self._ensure_min_ready_locked()
            self._write_status_locked()
            self._condition.notify_all()

        if close_immediately:
            _close_session_resources(session)

    def _record_start_failure(self, session_id: str, exc: Exception) -> None:
        """Record a failed startup and schedule the next retry attempt."""
        message = _exception_message(exc)
        with self._condition:
            self._starting_sessions.pop(session_id, None)
            self._total_failed += 1
            self._last_error = message
            self._consecutive_start_failures += 1
            self._next_start_attempt_at = self._next_retry_deadline_locked()
            self._schedule_retry_locked()
            self._write_status_locked()
            self._condition.notify_all()

    def _schedule_retry_locked(self) -> None:
        """Schedule a delayed prewarm retry after a startup failure."""
        if self._closed or self._retry_scheduled:
            return
        if self.config.startup_retry_backoff_s <= 0:
            self._ensure_min_ready_locked()
            return
        self._retry_scheduled = True
        _start_daemon_thread(
            name="desktop-pool-retry",
            target=self._retry_after_backoff,
        )

    def _retry_after_backoff(self) -> None:
        """Wait until the current retry deadline, then try to refill the pool."""
        with self._condition:
            while True:
                if self._closed:
                    self._retry_scheduled = False
                    self._write_status_locked()
                    self._condition.notify_all()
                    return

                remaining = self._startup_cooldown_remaining_locked()
                if remaining <= 0:
                    self._retry_scheduled = False
                    self._ensure_min_ready_locked()
                    self._write_status_locked()
                    self._condition.notify_all()
                    return
                self._condition.wait(timeout=remaining)

    def _startup_cooling_down_locked(self) -> bool:
        return self._startup_cooldown_remaining_locked() > 0

    def _startup_cooldown_remaining_locked(self) -> float:
        if self._next_start_attempt_at is None:
            return 0.0
        return self._next_start_attempt_at - self._clock()

    def _next_retry_deadline_locked(self) -> float | None:
        if self.config.startup_retry_backoff_s <= 0:
            return None
        failures = max(1, self._consecutive_start_failures)
        delay = min(
            self.config.startup_retry_backoff_s, self.config.startup_retry_backoff_max_s
        )
        for _ in range(failures - 1):
            delay = min(delay * 2, self.config.startup_retry_backoff_max_s)
            if delay >= self.config.startup_retry_backoff_max_s:
                break
        return self._clock() + delay

    def _retire_session_async(
        self,
        session: DesktopPoolSession[DesktopEnvT],
        *,
        reason: RetireReason,
    ) -> None:
        """Close a retired session on a background thread."""
        _start_daemon_thread(
            name=f"desktop-pool-retire-{session.session_id}",
            target=self._retire_session,
            session=session,
            reason=reason,
        )

    def _retire_session(
        self,
        session: DesktopPoolSession[DesktopEnvT],
        reason: RetireReason,
    ) -> None:
        """Close one session, release its ports, and trigger replenishment."""
        try:
            _close_session_resources(session)
        except Exception as exc:
            with self._condition:
                self._total_failed += 1
                self._last_error = f"{reason} close failed: {_exception_message(exc)}"
        finally:
            with self._condition:
                self._retiring_session_ids.discard(session.session_id)
                self._ensure_min_ready_locked()
                self._write_status_locked()
                self._condition.notify_all()

    def _start_lease_watchdog_locked(self) -> None:
        """Start the background stale-lease reaper once per pool."""
        if self._lease_watchdog_started:
            return
        self._lease_watchdog_started = True
        _start_daemon_thread(
            name="desktop-pool-lease-watchdog",
            target=self._lease_watchdog_loop,
        )

    def _lease_watchdog_loop(self) -> None:
        """Periodically retire abandoned checked-out desktop sessions."""
        poll_s = min(10.0, max(1.0, self.config.lease_timeout_s / 10.0))
        while True:
            with self._condition:
                if self._closed:
                    return
                self._condition.wait(timeout=poll_s)
                if self._closed:
                    return
            self.reap_stale_leases()

    def _start_status_heartbeat_locked(self) -> None:
        """Start the background status heartbeat once per pool."""
        if (
            self._status_heartbeat_started
            or self.config.status_heartbeat_interval_s <= 0
        ):
            return
        self._status_heartbeat_started = True
        _start_daemon_thread(
            name="desktop-pool-status-heartbeat",
            target=self._status_heartbeat_loop,
        )

    def _status_heartbeat_loop(self) -> None:
        """Refresh top-level status timestamps while the worker is alive."""
        interval_s = self.config.status_heartbeat_interval_s
        while True:
            with self._condition:
                if self._closed:
                    return
                self._condition.wait(timeout=interval_s)
                if self._closed:
                    return
                self._write_status_locked()

    def _session_is_stale_locked(
        self,
        session: DesktopPoolSession[DesktopEnvT],
        *,
        now: float,
    ) -> bool:
        """Return whether a leased session has exceeded the activity timeout."""
        if session.status != "leased":
            return False
        activity_at = session.last_activity_at or session.leased_at
        if activity_at is None:
            return False
        return now - activity_at >= self.config.lease_timeout_s

    def _ready_sessions_locked(self) -> list[DesktopPoolSession[DesktopEnvT]]:
        """Return ready sessions ordered by age while the lock is held."""
        return sorted(
            (
                session
                for session in self._sessions.values()
                if session.status == "ready"
            ),
            key=lambda session: session.created_at,
        )

    def _active_session_count_locked(self) -> int:
        return (
            len(self._sessions)
            + len(self._starting_sessions)
            + len(self._retiring_session_ids)
        )

    def _raise_if_closed_locked(self) -> None:
        if self._closed:
            raise RuntimeError("DesktopSessionPool is closed")

    def _write_status_locked(self) -> None:
        self.status_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(self.status_path, self._status_payload_locked())

    def _status_payload_locked(self) -> dict[str, Any]:
        """Build the JSON-serializable status payload for this worker pool."""
        now = self._clock()
        sessions = [
            _session_payload(session, now=now)
            for session in sorted(
                self._sessions.values(),
                key=lambda item: item.session_id,
            )
        ]
        starting_sessions = [
            _starting_session_payload(session, now=now)
            for session in sorted(
                self._starting_sessions.values(),
                key=lambda item: item.session_id,
            )
        ]
        starting_ages = [
            session["age_s"]
            for session in starting_sessions
            if isinstance(session.get("age_s"), int | float)
        ]
        return {
            "worker_name": self.status_path.stem,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "root_dir": str(self.root_dir),
            "status_dir": str(self.status_dir),
            "runtime_dir": str(self.runtime_dir),
            "log_dir": str(self.log_dir),
            "log_write_dir": str(self.log_write_dir),
            "status_path": str(self.status_path),
            "updated_at": self._clock(),
            "closed": self._closed,
            "min_ready_sessions": self.config.min_ready_sessions,
            "max_sessions": self.config.max_sessions,
            "max_rollouts_per_session": self.config.max_rollouts_per_session,
            "checkout_timeout_s": self.config.checkout_timeout_s,
            "lease_timeout_s": self.config.lease_timeout_s,
            "startup_timeout_s": self.config.startup_timeout_s,
            "startup_retry_backoff_s": self.config.startup_retry_backoff_s,
            "startup_retry_backoff_max_s": self.config.startup_retry_backoff_max_s,
            "status_heartbeat_interval_s": self.config.status_heartbeat_interval_s,
            "ready": sum(1 for session in sessions if session["status"] == "ready"),
            "starting": len(self._starting_sessions),
            "leased": sum(1 for session in sessions if session["status"] == "leased"),
            "retiring": len(self._retiring_session_ids),
            "oldest_starting_age_s": max(starting_ages) if starting_ages else None,
            "total_started": self._total_started,
            "total_failed": self._total_failed,
            "stale_leases_retired": self._total_stale_leases_retired,
            "retry_scheduled": self._retry_scheduled,
            "consecutive_start_failures": self._consecutive_start_failures,
            "next_start_attempt_at": self._next_start_attempt_at,
            "startup_cooldown_remaining_s": max(
                0.0,
                self._startup_cooldown_remaining_locked(),
            ),
            "last_error": self._last_error,
            "starting_sessions": starting_sessions,
            "sessions": sessions,
        }


def _default_worker_name() -> str:
    host = socket.gethostname().split(".")[0]
    return f"{host}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _session_payload[DesktopEnvT: DesktopSessionEnv](
    session: DesktopPoolSession[DesktopEnvT],
    *,
    now: float,
) -> dict[str, Any]:
    """Serialize one session's state, port lease, and health details."""
    activity_at = session.last_activity_at or session.updated_at
    payload = {
        "session_id": session.session_id,
        "status": session.status,
        "rollouts_completed": session.rollouts_completed,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "leased_at": session.leased_at,
        "last_activity_at": session.last_activity_at,
        "lease_age_s": (
            None if session.leased_at is None else max(0.0, now - session.leased_at)
        ),
        "idle_s": max(0.0, now - activity_at),
        "last_error": session.last_error,
        "lease_slot": session.lease.slot,
        "workdir": str(session.lease.workdir),
        "ports": _ports_payload(session.lease),
        "health": _env_health(session.env),
    }
    if session.lease.logdir is not None:
        payload["logdir"] = str(session.lease.logdir)
        payload["persistent_logdir"] = str(session.lease.logdir.resolve())
    return payload


def _starting_session_payload(
    session: StartingDesktopSession,
    *,
    now: float,
) -> dict[str, Any]:
    lease = session.lease
    payload: dict[str, Any] = {
        "session_id": session.session_id,
        "status": "starting",
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "age_s": max(0.0, now - session.created_at),
        "last_error": session.last_error,
    }
    if lease is not None:
        payload.update(
            {
                "lease_slot": lease.slot,
                "workdir": str(lease.workdir),
                "apptainer_pidfile": str(lease.workdir / "apptainer.pid.json"),
                "ports": _ports_payload(lease),
            }
        )
    return payload


class _ActivityTrackedDesktopEnv:
    """Lightweight proxy that touches a pool lease around env method calls."""

    def __init__(self, *, env: DesktopSessionEnv, touch: Callable[[], None]):
        self._env = env
        self._touch = touch

    def __getattr__(self, name: str) -> object:
        attr = getattr(self._env, name)
        if not callable(attr):
            return attr

        def tracked_call(*args: object, **kwargs: object) -> object:
            self._touch()
            try:
                return attr(*args, **kwargs)
            finally:
                self._touch()

        return tracked_call


def _ports_payload(lease: PortLease) -> dict[str, int]:
    """Serialize a port lease's OSWorld port block."""
    ports = lease.ports
    return {
        "server": ports.server,
        "chromium": ports.chromium,
        "vnc": ports.vnc,
        "vlc": ports.vlc,
        "qemu_vnc": ports.qemu_vnc,
    }


def _env_health(env: DesktopSessionEnv) -> Mapping[str, object]:
    """Safely ask a desktop proxy for health details."""
    health = getattr(env, "health", None)
    if not callable(health):
        return {}
    try:
        value = health()
    except Exception as exc:
        return {"health_error": repr(exc)}
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {"value": value}


def _close_session_resources[DesktopEnvT: DesktopSessionEnv](
    session: DesktopPoolSession[DesktopEnvT],
) -> None:
    if session.closed:
        return
    session.closed = True
    try:
        _close_env(session.env)
    finally:
        session.lease.release()


def _close_env(env: DesktopSessionEnv) -> None:
    env.close()


def _exception_message(exc: Exception) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _ensure_symlink_dir(link_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_path.symlink_to(target_dir, target_is_directory=True)
    except FileExistsError:
        if not link_path.is_symlink() or link_path.resolve() != target_dir.resolve():
            raise RuntimeError(
                f"Refusing to replace non-symlink or wrong-target log path: {link_path}"
            )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a sibling temp file before replacing the target."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _start_daemon_thread(
    *,
    name: str,
    target: Callable[..., None],
    **kwargs: object,
) -> None:
    """Start a daemon thread with keyword arguments for background pool work."""
    thread = threading.Thread(
        target=target,
        kwargs=kwargs,
        name=name,
        daemon=True,
    )
    thread.start()
