from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from osworld_desktop_host import server


@pytest.fixture(autouse=True)
def runtime_state(monkeypatch):
    state = server._RuntimeState()
    monkeypatch.setattr(server, "_runtime_state", state)
    return state


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload


def test_entrypoint_does_not_require_project_dependencies():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)

    result = subprocess.run(
        [sys.executable, "-S", "-m", "osworld_desktop_host"],
        cwd=root,
        env=env,
        input="",
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr


def test_server_cursor_position_uses_osworld_endpoint(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return _Response(b"[12, 34]")

    monkeypatch.setattr(server, "urlopen", fake_urlopen)
    env = SimpleNamespace(
        controller=SimpleNamespace(http_server="http://127.0.0.1:5000")
    )

    assert server._server_cursor_position(env) == [12, 34]
    assert calls == [("http://127.0.0.1:5000/cursor_position", 10)]


def test_init_returns_vm_metadata_when_provider_process_exists(monkeypatch):
    vm_process = start_sleeping_process()
    expected_vm_pgid = os.getpgid(vm_process.pid)
    created_envs = []

    class FakeDesktopEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.provider = SimpleNamespace(process=vm_process)
            self.close_count = 0
            created_envs.append(self)

        def close(self):
            self.close_count += 1
            cleanup_process(vm_process)

    install_fake_desktop_env(monkeypatch, FakeDesktopEnv)
    output = io.StringIO()

    try:
        server._run_protocol(
            io.StringIO(json.dumps(init_request(request_id=7)) + "\n"),
            output,
        )
    finally:
        cleanup_process(vm_process)

    response = json.loads(output.getvalue().splitlines()[0])
    assert response["id"] == 7
    assert response["ok"] is True
    assert response["result"] == {
        "vm_pid": vm_process.pid,
        "vm_pgid": expected_vm_pgid,
    }
    assert created_envs[0].close_count == 1


def test_signal_shutdown_closes_live_env_before_exit(monkeypatch, runtime_state):
    env = SimpleNamespace(close_count=0)

    def close():
        env.close_count += 1

    env.close = close
    runtime_state.set_env(env)

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(server.os, "_exit", fake_exit)
    monkeypatch.setattr(server.signal, "signal", lambda *_args: None)

    with pytest.raises(SystemExit) as exc_info:
        server._shutdown_and_exit(signal.SIGTERM, None)

    assert exc_info.value.code == 128 + signal.SIGTERM
    assert env.close_count == 1
    server._runtime_state.close_env_once()
    assert env.close_count == 1


def test_explicit_close_calls_env_close_once(monkeypatch):
    created_envs = []

    class FakeDesktopEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.provider = SimpleNamespace(process=None)
            self.close_count = 0
            created_envs.append(self)

        def close(self):
            self.close_count += 1

    install_fake_desktop_env(monkeypatch, FakeDesktopEnv)
    output = io.StringIO()
    requests = "\n".join(
        [
            json.dumps(init_request(request_id=1)),
            json.dumps({"id": 2, "cmd": "close"}),
        ]
    )

    server._run_protocol(io.StringIO(requests + "\n"), output)

    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[0]["ok"] is True
    assert responses[0]["result"] == {"vm_pid": None, "vm_pgid": None}
    assert responses[1] == {"id": 2, "ok": True, "result": None}
    assert created_envs[0].close_count == 1
    server._runtime_state.close_env_once()
    assert created_envs[0].close_count == 1


def install_fake_desktop_env(monkeypatch, desktop_env_cls):
    package = types.ModuleType("desktop_env")
    module = types.ModuleType("desktop_env.desktop_env")
    module.DesktopEnv = desktop_env_cls
    package.desktop_env = module
    monkeypatch.setitem(sys.modules, "desktop_env", package)
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", module)


def init_request(*, request_id: int) -> dict[str, object]:
    return {
        "id": request_id,
        "cmd": "init",
        "kwargs": {
            "path_to_vm": "/tmp/fake-vm",
            "screen_size": [1280, 720],
            "cache_dir": None,
        },
    }


def start_sleeping_process() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def cleanup_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1.0)
