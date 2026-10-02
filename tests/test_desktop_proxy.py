from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rl.osworld.desktop.proxy import DesktopEnvProxy


def test_proxy_timeout_terminates_host_and_vm_process_groups() -> None:
    host_process = start_sleeping_process(with_pipes=True)
    vm_process = start_sleeping_process()
    proxy = make_proxy(host_process, vm_process_group_id=vm_process.pid)

    try:
        health = proxy.health()
        assert health["host_pid"] == host_process.pid
        assert health["host_pgid"] == host_process.pid
        assert health["vm_pgid"] == vm_process.pid
        assert health["alive"] is True

        with pytest.raises(TimeoutError):
            proxy._terminate_after_timeout("get_obs", 0.01, [])

        wait_for(lambda: host_process.poll() is not None)
        wait_for(lambda: vm_process.poll() is not None)

        proxy.close()
        proxy.close()
        assert host_process.stdin is not None and host_process.stdin.closed
        assert host_process.stdout is not None and host_process.stdout.closed
    finally:
        cleanup_process(host_process)
        cleanup_process(vm_process)


def test_proxy_timeout_without_vm_pgid_still_cleans_host() -> None:
    host_process = start_sleeping_process(with_pipes=True)
    proxy = make_proxy(host_process, vm_process_group_id=None)

    try:
        with pytest.raises(TimeoutError):
            proxy._terminate_after_timeout("get_obs", 0.01, [])

        wait_for(lambda: host_process.poll() is not None)
        proxy.close()
    finally:
        cleanup_process(host_process)


def test_proxy_timeout_reads_vm_pgid_from_pidfile(tmp_path: Path) -> None:
    host_process = start_sleeping_process(with_pipes=True)
    vm_process = start_sleeping_process()
    pidfile = tmp_path / "apptainer.pid.json"
    pidfile.write_text(
        json.dumps({"pid": vm_process.pid, "pgid": vm_process.pid}),
        encoding="utf-8",
    )
    proxy = make_proxy(
        host_process,
        vm_process_group_id=None,
        vm_pidfile=str(pidfile),
    )

    try:
        with pytest.raises(TimeoutError):
            proxy._terminate_after_timeout("init", 0.01, [])

        wait_for(lambda: host_process.poll() is not None)
        wait_for(lambda: vm_process.poll() is not None)
        assert proxy.health()["vm_pgid"] == vm_process.pid
    finally:
        cleanup_process(host_process)
        cleanup_process(vm_process)


def test_proxy_known_process_group_ids_skip_own_group() -> None:
    proxy = DesktopEnvProxy.__new__(DesktopEnvProxy)
    proxy._host_process_group_id = os.getpgrp()
    proxy._vm_process_group_id = os.getpgrp()

    assert proxy._known_process_group_ids() == ()


def make_proxy(
    process: subprocess.Popen[bytes],
    *,
    vm_process_group_id: int | None,
    vm_pidfile: str | None = None,
) -> DesktopEnvProxy:
    proxy = DesktopEnvProxy.__new__(DesktopEnvProxy)
    proxy._process = process
    proxy._host_process_group_id = process.pid
    proxy._vm_process_group_id = vm_process_group_id
    proxy._vm_pidfile = vm_pidfile
    proxy._closed = False
    proxy._next_id = 0
    return proxy


def start_sleeping_process(
    *,
    with_pipes: bool = False,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE if with_pipes else subprocess.DEVNULL,
        stdout=subprocess.PIPE if with_pipes else subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def cleanup_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1.0)


def wait_for(predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for condition")
