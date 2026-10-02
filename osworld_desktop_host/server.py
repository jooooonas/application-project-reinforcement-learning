from __future__ import annotations

import base64
import json
import os
import signal
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from typing import Any, TextIO
from urllib.request import urlopen

_PARENT_POLL_S = 1.0


class _RuntimeState:
    def __init__(self) -> None:
        self._env: Any | None = None
        self._close_started = False
        self._lock = threading.RLock()

    def set_env(self, env: Any) -> None:
        with self._lock:
            self._env = env
            self._close_started = False

    def require_env(self) -> Any:
        with self._lock:
            if self._env is None:
                raise RuntimeError("DesktopEnv has not been initialized")
            return self._env

    def close_env_once(self) -> None:
        with self._lock:
            if self._close_started:
                return
            env = self._env
            self._close_started = True

        if env is None:
            return

        try:
            env.close()
        finally:
            with self._lock:
                if self._env is env:
                    self._env = None


_runtime_state = _RuntimeState()


def main() -> None:
    _install_signal_handlers()
    _start_parent_watchdog()
    protocol_stdout = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr
    _run_protocol(sys.stdin, protocol_stdout)


def _run_protocol(input_stream: TextIO, protocol_stdout: TextIO) -> None:
    for line in input_stream:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _write(protocol_stdout, None, False, error="invalid JSON request")
            continue

        request_id = request.get("id")
        command = request.get("cmd")
        try:
            if command == "init":
                metadata = _create_desktop_env(request["kwargs"])
                _write(protocol_stdout, request_id, True, result=metadata)
                continue
            if command == "close":
                _runtime_state.close_env_once()
                _write(protocol_stdout, request_id, True, result=None)
                break

            desktop_env = _runtime_state.require_env()
            if command == "reset":
                obs = desktop_env.reset(task_config=request["task_config"])
                _write(protocol_stdout, request_id, True, result=_encode_obs(obs))
            elif command == "get_obs":
                obs = desktop_env._get_obs()
                _write(protocol_stdout, request_id, True, result=_encode_obs(obs))
            elif command == "step":
                obs, reward, done, info = desktop_env.step(
                    request["action"],
                    request.get("pause", 1.0),
                )
                _write(
                    protocol_stdout,
                    request_id,
                    True,
                    result={
                        "obs": _encode_obs(obs),
                        "reward": reward,
                        "done": done,
                        "info": info,
                    },
                )
            elif command == "evaluate":
                _write(protocol_stdout, request_id, True, result=desktop_env.evaluate())
            elif command == "cursor_position":
                position = _server_cursor_position(desktop_env)
                _write(protocol_stdout, request_id, True, result=position)
            elif command == "move_cursor_to":
                x = int(request["x"])
                y = int(request["y"])
                desktop_env.controller.execute_python_command(
                    f"pyautogui.moveTo({x}, {y})"
                )
                _write(protocol_stdout, request_id, True, result=None)
            else:
                raise ValueError(f"unknown command: {command!r}")
        except Exception as exc:
            _write(
                protocol_stdout,
                request_id,
                False,
                error=repr(exc),
                traceback_text=traceback.format_exc(),
            )

    try:
        _runtime_state.close_env_once()
    except Exception:
        traceback.print_exc(file=sys.stderr)


def _create_desktop_env(kwargs: Mapping[str, Any]) -> dict[str, int | None]:
    from desktop_env.desktop_env import DesktopEnv

    env_kwargs = {
        "provider_name": "apptainer",
        "path_to_vm": kwargs["path_to_vm"],
        "action_space": "pyautogui",
        "screen_size": tuple(kwargs["screen_size"]),
        "headless": True,
        "os_type": "Ubuntu",
        "require_a11y_tree": False,
    }
    if kwargs.get("cache_dir"):
        env_kwargs["cache_dir"] = kwargs["cache_dir"]
    env = DesktopEnv(**env_kwargs)
    _runtime_state.set_env(env)
    return _desktop_env_metadata(env)


def _desktop_env_metadata(env: Any) -> dict[str, int | None]:
    provider = getattr(env, "provider", None)
    process = getattr(provider, "process", None)
    vm_pid = _positive_int_or_none(getattr(process, "pid", None))
    if vm_pid is None:
        return {"vm_pid": None, "vm_pgid": None}
    try:
        vm_pgid = os.getpgid(vm_pid)
    except OSError:
        vm_pgid = None
    return {"vm_pid": vm_pid, "vm_pgid": vm_pgid}


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _shutdown_and_exit)
    signal.signal(signal.SIGINT, _shutdown_and_exit)


def _start_parent_watchdog() -> None:
    raw_parent_pid = os.environ.get("OSWORLD_DESKTOP_HOST_PARENT_PID")
    try:
        expected_parent_pid = int(raw_parent_pid) if raw_parent_pid else 0
    except ValueError:
        expected_parent_pid = 0
    initial_parent_pid = os.getppid()

    def watch_parent() -> None:
        while True:
            time.sleep(_PARENT_POLL_S)
            if os.getppid() != initial_parent_pid:
                _shutdown_and_exit(signal.SIGTERM, None)
            if expected_parent_pid > 0 and not _pid_exists(expected_parent_pid):
                _shutdown_and_exit(signal.SIGTERM, None)

    thread = threading.Thread(
        target=watch_parent,
        name="desktop-host-parent-watchdog",
        daemon=True,
    )
    thread.start()


def _pid_exists(pid: int) -> bool:
    try:
        # this is a check to see if the parent process is still alive
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _shutdown_and_exit(signum: int, _frame: object | None) -> None:
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _runtime_state.close_env_once()
    except Exception:
        traceback.print_exc(file=sys.stderr)
    os._exit(128 + signum)


def _positive_int_or_none(value: object) -> int | None:
    if not isinstance(value, int) or value <= 0:
        return None
    return value


def _encode_obs(obs: dict[str, Any]) -> dict[str, Any]:
    encoded = dict(obs)
    screenshot = encoded.get("screenshot")
    if isinstance(screenshot, bytes):
        encoded["screenshot"] = {
            "__base64__": base64.b64encode(screenshot).decode("ascii")
        }
    return encoded


def _server_cursor_position(env: Any) -> list[int]:
    controller = getattr(env, "controller", None)
    http_server = getattr(controller, "http_server", None)
    if not isinstance(http_server, str) or not http_server:
        raise RuntimeError("DesktopEnv controller has no http_server")
    with urlopen(f"{http_server}/cursor_position", timeout=10) as response:
        position = json.loads(response.read().decode("utf-8"))
    if not isinstance(position, list) or len(position) != 2:
        raise RuntimeError(f"invalid cursor position: {position!r}")
    return [int(position[0]), int(position[1])]


def _write(
    stream: TextIO,
    request_id: int | None,
    ok: bool,
    *,
    result: Any = None,
    error: str | None = None,
    traceback_text: str | None = None,
) -> None:
    payload: dict[str, Any] = {"id": request_id, "ok": ok}
    if ok:
        payload["result"] = result
    else:
        payload["error"] = error
        payload["traceback"] = traceback_text
    stream.write(json.dumps(payload) + "\n")
    stream.flush()
