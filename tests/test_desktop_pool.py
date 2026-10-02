from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from rl.osworld.desktop.pool import DesktopPoolConfig, DesktopSessionPool
from rl.runtime.ports import WorkerPorts


class FakeEnv:
    def __init__(self):
        self.closed = False
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        self.closed = True

    def health(self) -> dict[str, bool]:
        return {"alive": not self.closed}


@dataclass
class FakeLease:
    slot: int
    workdir: Path
    ports: WorkerPorts
    logdir: Path | None = None
    released: bool = False
    release_count: int = 0

    def release(self) -> None:
        self.release_count += 1
        self.released = True


class FakeDesktopRuntime:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.envs: list[FakeEnv] = []
        self.leases: list[FakeLease] = []
        self.factory_started = threading.Event()
        self.factory_gate: threading.Event | None = None
        self.failures_before_success = 0
        self.create_calls = 0

    def allocate(
        self,
        *,
        lock_dir: Path,
        work_dir: Path | None = None,
        log_dir: Path | None = None,
    ) -> FakeLease:
        slot = len(self.leases)
        root = Path(work_dir) if work_dir is not None else Path(lock_dir)
        log_root = Path(log_dir) if log_dir is not None else None
        lease = FakeLease(
            slot=slot,
            workdir=root / f"worker_{slot}",
            ports=WorkerPorts(
                server=20000 + slot * 10,
                chromium=20001 + slot * 10,
                vnc=20002 + slot * 10,
                vlc=20003 + slot * 10,
                qemu_vnc=20004 + slot * 10,
            ),
            logdir=log_root / f"worker_{slot}" if log_root is not None else None,
        )
        lease.workdir.mkdir(parents=True, exist_ok=True)
        if lease.logdir is not None:
            lease.logdir.mkdir(parents=True, exist_ok=True)
        self.leases.append(lease)
        return lease

    def create(self, _lease: FakeLease) -> FakeEnv:
        self.create_calls += 1
        self.factory_started.set()
        if self.factory_gate is not None:
            assert self.factory_gate.wait(timeout=1)
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise RuntimeError("startup failed")
        env = FakeEnv()
        self.envs.append(env)
        return env


def test_pool_prewarms_min_ready_session(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    pool = make_pool(tmp_path, runtime)

    pool.start()

    wait_for(lambda: pool.snapshot()["ready"] == 1)
    status = json.loads(pool.status_path.read_text(encoding="utf-8"))
    assert status["ready"] == 1
    assert status["total_started"] == 1
    assert "failed" not in status

    pool.close()


def test_pool_writes_status_to_configured_status_dir(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    status_dir = tmp_path / "replica-status" / "osworld-0000"
    pool = make_pool(tmp_path, runtime, status_dir=status_dir)

    pool.start()

    wait_for(lambda: pool.snapshot()["ready"] == 1)
    assert pool.status_dir == status_dir
    assert pool.status_path == status_dir / "test-worker.json"
    assert pool.status_path.exists()
    assert not (tmp_path / "pool" / "status" / "test-worker.json").exists()

    pool.close()


def test_pool_uses_configured_runtime_dir_for_worker_workdirs(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    shared_root = tmp_path / "pool"
    runtime_dir = tmp_path / "runtime"
    log_runtime_dir = tmp_path / "log"
    pool = make_pool(
        tmp_path,
        runtime,
        runtime_dir=runtime_dir,
        log_runtime_dir=log_runtime_dir,
    )

    pool.start()

    wait_for(lambda: pool.snapshot()["ready"] == 1)
    assert pool.root_dir == shared_root
    assert pool.runtime_dir == runtime_dir
    assert runtime.leases[0].workdir == runtime_dir / "worker_0"
    assert runtime.leases[0].logdir == log_runtime_dir / "worker_0"
    assert log_runtime_dir.is_symlink()
    assert log_runtime_dir.resolve() == (shared_root / "logs").resolve()
    assert (shared_root / "logs" / "worker_0").is_dir()
    assert pool.port_lock_dir == shared_root / "port_locks"
    assert pool.status_dir == shared_root / "status"
    assert pool.log_dir == shared_root / "logs"
    assert pool.log_write_dir == log_runtime_dir
    assert pool.artifact_dir == shared_root / "artifacts"
    assert pool.snapshot()["runtime_dir"] == str(runtime_dir)
    assert pool.snapshot()["log_dir"] == str(shared_root / "logs")
    assert pool.snapshot()["log_write_dir"] == str(log_runtime_dir)

    pool.close()


def test_pool_status_includes_starting_session_metadata(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    runtime.factory_gate = threading.Event()
    pool = make_pool(tmp_path, runtime, startup_timeout_s=123.0)

    pool.start()

    assert runtime.factory_started.wait(timeout=1)
    snapshot = pool.snapshot()
    assert snapshot["startup_timeout_s"] == 123.0
    assert snapshot["starting"] == 1
    assert snapshot["oldest_starting_age_s"] is not None
    assert len(snapshot["starting_sessions"]) == 1

    starting = snapshot["starting_sessions"][0]
    assert starting["session_id"] == "session-000001"
    assert starting["status"] == "starting"
    assert starting["lease_slot"] == 0
    assert starting["workdir"] == str(runtime.leases[0].workdir)
    assert starting["apptainer_pidfile"] == str(
        runtime.leases[0].workdir / "apptainer.pid.json"
    )
    assert starting["ports"]["server"] == 20000

    runtime.factory_gate.set()
    wait_for(lambda: pool.snapshot()["ready"] == 1)
    pool.close()


def test_pool_rejects_nonpositive_startup_timeout() -> None:
    with pytest.raises(ValueError, match="startup_timeout_s must be positive"):
        DesktopPoolConfig(startup_timeout_s=0)


def test_checkout_waits_until_session_is_ready(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    runtime.factory_gate = threading.Event()
    pool = make_pool(tmp_path, runtime)
    checked_out = []
    thread = threading.Thread(
        target=lambda: checked_out.append(pool.checkout(timeout_s=1)),
        daemon=True,
    )

    pool.start()
    assert runtime.factory_started.wait(timeout=1)
    thread.start()
    time.sleep(0.02)
    assert checked_out == []

    runtime.factory_gate.set()
    thread.join(timeout=1)

    assert checked_out
    assert checked_out[0].env is runtime.envs[0]
    checked_out[0].release()
    pool.close()


def test_release_retires_after_rollout_and_starts_replacement(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    pool = make_pool(tmp_path, runtime)
    pool.start()
    wait_for(lambda: pool.snapshot()["ready"] == 1)

    checkout = pool.checkout(timeout_s=1)
    first_env = runtime.envs[0]
    first_lease = runtime.leases[0]
    checkout.release()

    wait_for(lambda: first_env.closed and first_lease.released)
    wait_for(lambda: pool.snapshot()["ready"] == 1 and len(runtime.envs) == 2)

    assert runtime.envs[1] is not first_env
    pool.close()


def test_failed_release_records_error_and_starts_replacement(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    pool = make_pool(tmp_path, runtime)
    pool.start()
    wait_for(lambda: pool.snapshot()["ready"] == 1)

    checkout = pool.checkout(timeout_s=1)
    first_env = runtime.envs[0]
    checkout.release(failed=True, error="reset failed")

    wait_for(lambda: first_env.closed)
    wait_for(lambda: pool.snapshot()["ready"] == 1 and len(runtime.envs) == 2)
    snapshot = pool.snapshot()
    assert snapshot["total_failed"] == 1
    assert snapshot["last_error"] == "reset failed"
    assert "failed" not in snapshot

    pool.close()


def test_stale_leased_session_is_retired_and_replaced(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    now = 1000.0
    pool = make_pool(
        tmp_path,
        runtime,
        clock=lambda: now,
        lease_timeout_s=300.0,
    )
    pool.start()
    wait_for(lambda: pool.snapshot()["ready"] == 1)

    checkout = pool.checkout(timeout_s=1)
    first_env = runtime.envs[0]
    first_lease = runtime.leases[0]
    now += 301.0

    assert pool.reap_stale_leases() == 1
    wait_for(lambda: first_env.closed and first_lease.released)
    wait_for(lambda: pool.snapshot()["ready"] == 1 and len(runtime.envs) == 2)

    snapshot = pool.snapshot()
    assert snapshot["stale_leases_retired"] == 1
    assert snapshot["total_failed"] == 1
    assert "lease timed out" in snapshot["last_error"]

    checkout.release()
    pool.close()


def test_tracked_env_method_call_refreshes_lease_activity(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    now = 1000.0
    pool = make_pool(
        tmp_path,
        runtime,
        clock=lambda: now,
        lease_timeout_s=300.0,
    )
    pool.start()
    wait_for(lambda: pool.snapshot()["ready"] == 1)

    checkout = pool.checkout(timeout_s=1)
    now += 200.0
    assert checkout.tracked_env().health() == {"alive": True}
    now += 200.0

    assert pool.reap_stale_leases() == 0
    assert pool.snapshot()["leased"] == 1

    now += 101.0
    assert pool.reap_stale_leases() == 1
    wait_for(lambda: runtime.envs[0].closed)
    checkout.release()
    pool.close()


def test_close_closes_ready_sessions_and_releases_leases(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    pool = make_pool(tmp_path, runtime)
    pool.start()
    wait_for(lambda: pool.snapshot()["ready"] == 1)

    pool.close()

    assert runtime.envs[0].closed
    assert runtime.leases[0].released
    snapshot = pool.snapshot()
    assert snapshot["closed"] is True
    assert snapshot["ready"] == 0


def test_close_is_idempotent_for_session_resources(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    pool = make_pool(tmp_path, runtime)
    pool.start()
    wait_for(lambda: pool.snapshot()["ready"] == 1)

    pool.close()
    pool.close()

    assert runtime.envs[0].close_count == 1
    assert runtime.leases[0].release_count == 1


def test_status_heartbeat_refreshes_worker_timestamp(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    pool = make_pool(tmp_path, runtime, status_heartbeat_interval_s=0.02)
    pool.start()
    wait_for(lambda: pool.snapshot()["ready"] == 1)
    first_updated_at = json.loads(pool.status_path.read_text(encoding="utf-8"))[
        "updated_at"
    ]

    wait_for(
        lambda: (
            json.loads(pool.status_path.read_text(encoding="utf-8"))["updated_at"]
            > first_updated_at
        ),
        timeout_s=1.0,
    )

    pool.close()


def test_startup_retry_backoff_is_exponential_and_capped(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    now = 1000.0
    pool = make_pool(
        tmp_path,
        runtime,
        clock=lambda: now,
        startup_retry_backoff_s=10.0,
        startup_retry_backoff_max_s=15.0,
    )

    with pool._condition:
        pool._consecutive_start_failures = 1
        assert pool._next_retry_deadline_locked() == 1010.0
        pool._consecutive_start_failures = 2
        assert pool._next_retry_deadline_locked() == 1015.0
        pool._consecutive_start_failures = 8
        assert pool._next_retry_deadline_locked() == 1015.0


def test_startup_failure_backoff_blocks_checkout_refill(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    runtime.failures_before_success = 1
    pool = make_pool(tmp_path, runtime, startup_retry_backoff_s=0.1)
    checked_out = []
    errors = []

    def checkout() -> None:
        try:
            checked_out.append(pool.checkout(timeout_s=1))
        except Exception as exc:  # pragma: no cover - surfaced by assertion below.
            errors.append(exc)

    pool.start()
    wait_for(lambda: pool.snapshot()["total_failed"] == 1)

    snapshot = pool.snapshot()
    assert runtime.create_calls == 1
    assert snapshot["ready"] == 0
    assert snapshot["retry_scheduled"] is True
    assert snapshot["consecutive_start_failures"] == 1
    assert snapshot["startup_cooldown_remaining_s"] > 0

    thread = threading.Thread(target=checkout, daemon=True)
    thread.start()
    time.sleep(0.03)

    assert runtime.create_calls == 1
    assert checked_out == []

    thread.join(timeout=1)
    assert not errors
    assert checked_out
    assert runtime.create_calls == 2
    assert pool.snapshot()["consecutive_start_failures"] == 0

    checked_out[0].release()
    pool.close()


def test_startup_failure_releases_failed_startup_lease(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    runtime.failures_before_success = 1
    pool = make_pool(tmp_path, runtime, startup_retry_backoff_s=60.0)

    pool.start()
    wait_for(lambda: pool.snapshot()["total_failed"] == 1)

    assert runtime.leases[0].released is True
    assert runtime.leases[0].release_count == 1

    pool.close()
    wait_for(lambda: pool.snapshot()["retry_scheduled"] is False)


def test_overlapping_start_failures_share_one_retry_gate(tmp_path):
    runtime = FakeDesktopRuntime(tmp_path)
    runtime.failures_before_success = 2
    pool = make_pool(
        tmp_path,
        runtime,
        min_ready_sessions=2,
        max_sessions=2,
        startup_retry_backoff_s=0.1,
    )

    pool.start()
    wait_for(lambda: pool.snapshot()["total_failed"] == 2)

    snapshot = pool.snapshot()
    assert runtime.create_calls == 2
    assert snapshot["starting"] == 0
    assert snapshot["ready"] == 0
    assert snapshot["retry_scheduled"] is True
    assert snapshot["consecutive_start_failures"] == 2

    time.sleep(0.03)
    assert runtime.create_calls == 2

    wait_for(lambda: pool.snapshot()["ready"] == 2)
    snapshot = pool.snapshot()
    assert runtime.create_calls == 4
    assert snapshot["retry_scheduled"] is False
    assert snapshot["consecutive_start_failures"] == 0
    assert snapshot["next_start_attempt_at"] is None
    assert snapshot["startup_cooldown_remaining_s"] == 0.0

    pool.close()


def make_pool(
    tmp_path: Path,
    runtime: FakeDesktopRuntime,
    *,
    clock=time.time,
    **config_overrides,
) -> DesktopSessionPool:
    config_values = {
        "min_ready_sessions": 1,
        "max_sessions": 1,
        "max_rollouts_per_session": 1,
        "checkout_timeout_s": 1,
        "startup_retry_backoff_s": 0,
    }
    config_values.update(config_overrides)
    return DesktopSessionPool(
        config=DesktopPoolConfig(**config_values),
        root_dir=tmp_path / "pool",
        session_factory=runtime.create,
        port_allocator=runtime.allocate,
        worker_name="test-worker",
        clock=clock,
    )


def wait_for(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for condition")
