from __future__ import annotations

import base64
import contextlib
import json
import os
import selectors
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

_REQUEST_TIMEOUT_S = 1800.0
_CLOSE_REQUEST_TIMEOUT_S = 30.0
_TERMINATE_TIMEOUT_S = 30.0
_KILL_TIMEOUT_S = 10.0


class DesktopEnvProxy:
    """Small JSON-line proxy to run OSWorld DesktopEnv in its own Python env."""

    def __init__(
        self,
        *,
        python: str,
        process_env: Mapping[str, str],
        path_to_vm: str,
        screen_size: tuple[int, int],
        cache_dir: str | None = None,
        init_timeout_s: float | None = _REQUEST_TIMEOUT_S,
    ):
        self._next_id = 0
        self._closed = False
        host_env = dict(process_env)
        host_env["OSWORLD_DESKTOP_HOST_PARENT_PID"] = str(os.getpid())
        self._vm_pidfile = host_env.get("OSWORLD_APPTAINER_PIDFILE")
        self._process = subprocess.Popen(
            [python, "-m", "osworld_desktop_host"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
            env=host_env,
            start_new_session=True,
        )
        self._host_process_group_id = (
            self._process_group_id_for_pid(self._process.pid) or self._process.pid
        )
        self._vm_process_group_id: int | None = None
        try:
            init_result = self._request(
                "init",
                request_timeout=init_timeout_s,
                kwargs={
                    "path_to_vm": path_to_vm,
                    "screen_size": list(screen_size),
                    "cache_dir": cache_dir,
                },
            )
        except Exception:
            self._closed = True
            self._refresh_vm_process_group_id_from_pidfile()
            self._terminate_and_wait_for_process_groups(
                terminate_timeout_s=5.0,
                kill_timeout_s=5.0,
            )
            self._close_pipes()
            raise
        self._vm_process_group_id = self._metadata_process_group_id(init_result)
        if self._vm_process_group_id is None:
            self._refresh_vm_process_group_id_from_pidfile()

    def reset(self, *, task_config: Mapping[str, Any]) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._decode_obs(self._request("reset", task_config=task_config)),
        )

    def observe(
        self,
        *,
        request_timeout: float | None = _REQUEST_TIMEOUT_S,
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._decode_obs(self._request("get_obs", request_timeout=request_timeout)),
        )

    def _get_obs(
        self,
        *,
        request_timeout: float | None = _REQUEST_TIMEOUT_S,
    ) -> dict[str, Any]:
        return self.observe(request_timeout=request_timeout)

    def step(
        self,
        action: str,
        pause: float,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        result = cast(
            dict[str, Any],
            self._request("step", action=action, pause=pause),
        )
        return (
            self._decode_obs(cast(dict[str, Any], result["obs"])),
            float(result["reward"]),
            bool(result["done"]),
            cast(dict[str, Any], result.get("info") or {}),
        )

    def evaluate(self) -> float:
        return float(self._request("evaluate"))

    def cursor_position(self) -> tuple[int, int]:
        result = self._request("cursor_position")
        if not isinstance(result, list | tuple) or len(result) != 2:
            raise RuntimeError(f"invalid cursor_position response: {result!r}")
        return int(result[0]), int(result[1])

    def move_cursor_to(self, x: int, y: int) -> None:
        self._request("move_cursor_to", x=int(x), y=int(y))

    def health(self) -> dict[str, Any]:
        returncode = self._process.poll()
        return {
            "closed": self._closed,
            "host_pid": self._process.pid,
            "host_pgid": self._host_process_group_id,
            "vm_pgid": self._vm_process_group_id,
            "vm_pidfile": self._vm_pidfile,
            "pid": self._process.pid,
            "returncode": returncode,
            "alive": not self._closed and returncode is None,
        }

    @staticmethod
    def _metadata_process_group_id(metadata: object) -> int | None:
        if not isinstance(metadata, Mapping):
            return None
        return DesktopEnvProxy._positive_int_or_none(metadata.get("vm_pgid"))

    @staticmethod
    def _positive_int_or_none(value: object) -> int | None:
        if not isinstance(value, int) or value <= 0:
            return None

        return value

    @staticmethod
    def _process_group_id_for_pid(pid: int) -> int | None:
        try:
            return os.getpgid(pid)
        except OSError:
            return None

    def close(self) -> None:
        if self._closed:
            self._close_pipes()
            return
        self._closed = True
        try:
            if self._process.poll() is None:
                try:
                    self._request("close", request_timeout=_CLOSE_REQUEST_TIMEOUT_S)
                except Exception:
                    self._terminate_and_wait_for_process_groups(
                        terminate_timeout_s=_TERMINATE_TIMEOUT_S,
                        kill_timeout_s=_KILL_TIMEOUT_S,
                    )
                else:
                    try:
                        self._process.wait(timeout=_TERMINATE_TIMEOUT_S)
                    except subprocess.TimeoutExpired:
                        self._terminate_and_wait_for_process_groups(
                            terminate_timeout_s=_TERMINATE_TIMEOUT_S,
                            kill_timeout_s=_KILL_TIMEOUT_S,
                        )
                    else:
                        if self._alive_process_group_ids():
                            self._terminate_and_wait_for_process_groups(
                                terminate_timeout_s=_TERMINATE_TIMEOUT_S,
                                kill_timeout_s=_KILL_TIMEOUT_S,
                            )
            else:
                self._terminate_and_wait_for_process_groups(
                    terminate_timeout_s=0.2,
                    kill_timeout_s=_KILL_TIMEOUT_S,
                )
        finally:
            self._close_pipes()

    def _close_pipes(self) -> None:
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _request(
        self,
        command: str,
        *,
        request_timeout: float | None = _REQUEST_TIMEOUT_S,
        **payload: Any,
    ) -> Any:
        if self._closed and command != "close":
            raise RuntimeError("DesktopEnv host is closed")
        if self._process.poll() is not None:
            raise RuntimeError(
                f"DesktopEnv host exited with code {self._process.returncode}"
            )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("DesktopEnv host pipes are not available")

        if request_timeout is not None and request_timeout <= 0:
            raise ValueError("request_timeout must be positive or None")
        deadline = (
            None if request_timeout is None else time.monotonic() + request_timeout
        )
        request_id = self._next_id
        self._next_id += 1
        request = {"id": request_id, "cmd": command, **payload}
        self._process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        self._process.stdin.flush()

        skipped_lines: list[str] = []
        buffer = bytearray()
        selector = selectors.DefaultSelector()
        selector.register(self._process.stdout, selectors.EVENT_READ)
        try:
            while True:
                line = self._read_response_line(
                    command=command,
                    deadline=deadline,
                    request_timeout=request_timeout,
                    skipped_lines=skipped_lines,
                    buffer=buffer,
                    selector=selector,
                )
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    skipped_lines.append(line.rstrip())
                    continue
                if response.get("id") != request_id:
                    skipped_lines.append(line.rstrip())
                    continue
                if response.get("ok"):
                    return response.get("result")
                raise RuntimeError(
                    f"DesktopEnv host failed during {command!r}: {response.get('error')}\n"
                    f"{response.get('traceback') or ''}"
                )
        finally:
            selector.unregister(self._process.stdout)

    def _read_response_line(
        self,
        *,
        command: str,
        deadline: float | None,
        request_timeout: float | None,
        skipped_lines: list[str],
        buffer: bytearray,
        selector: selectors.BaseSelector,
    ) -> str:
        stdout = self._process.stdout
        if stdout is None:
            raise RuntimeError("DesktopEnv host stdout is not available")

        while b"\n" not in buffer:
            wait = None
            if deadline is not None:
                wait = max(0.0, deadline - time.monotonic())
            if not selector.select(wait):
                self._terminate_after_timeout(
                    command,
                    request_timeout,
                    skipped_lines,
                )

            chunk = os.read(stdout.fileno(), 4096)
            if chunk == b"":
                if buffer:
                    break
                raise RuntimeError(
                    f"DesktopEnv host exited while handling {command!r}; "
                    f"skipped_stdout={skipped_lines[-5:]}"
                )
            buffer.extend(chunk)

        if b"\n" in buffer:
            line, _, rest = bytes(buffer).partition(b"\n")
            buffer[:] = rest
        else:
            line = bytes(buffer)
            buffer.clear()
        return line.decode("utf-8", errors="replace")

    def _terminate_after_timeout(
        self,
        command: str,
        request_timeout: float | None,
        skipped_lines: list[str],
    ) -> None:
        if self._process.poll() is None:
            self._closed = True
            self._refresh_vm_process_group_id_from_pidfile()
            self._terminate_and_wait_for_process_groups(
                terminate_timeout_s=5.0,
                kill_timeout_s=5.0,
            )
        timeout_text = (
            "the configured timeout"
            if request_timeout is None
            else f"{request_timeout:.1f}s"
        )
        raise TimeoutError(
            f"DesktopEnv host timed out during {command!r} after {timeout_text}; "
            f"skipped_stdout={skipped_lines[-5:]}"
        )

    def _terminate_process_group(self, sig: int) -> None:
        """Backward-compatible wrapper around all known owned process groups."""
        self._terminate_process_groups(sig)

    def _terminate_and_wait_for_process_groups(
        self,
        *,
        terminate_timeout_s: float,
        kill_timeout_s: float,
    ) -> None:
        self._refresh_vm_process_group_id_from_pidfile()
        remaining = self._alive_process_group_ids()
        if remaining:
            self._terminate_process_groups(signal.SIGTERM, group_ids=remaining)
        if self._wait_for_process_groups_exit(terminate_timeout_s):
            return
        remaining = self._alive_process_group_ids()
        if remaining:
            self._terminate_process_groups(signal.SIGKILL, group_ids=remaining)
        self._wait_for_process_groups_exit(kill_timeout_s)

    def _terminate_process_groups(
        self,
        sig: int,
        *,
        group_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Best-effort signal to the host group and the tracked VM group."""
        for process_group_id in group_ids or self._known_process_group_ids():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process_group_id, sig)

    def _wait_for_process_groups_exit(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            if self._process.poll() is None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self._process.wait(timeout=0)
            if not self._alive_process_group_ids():
                return True
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return False
            wait_s = min(0.05, remaining_s)
            if self._process.poll() is None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self._process.wait(timeout=wait_s)
            else:
                time.sleep(wait_s)

    def _alive_process_group_ids(self) -> tuple[int, ...]:
        return tuple(
            process_group_id
            for process_group_id in self._known_process_group_ids()
            if self._process_group_alive(process_group_id)
        )

    def _known_process_group_ids(self) -> tuple[int, ...]:
        current_process_group_id = os.getpgrp()
        known_process_group_ids: list[int] = []

        for process_group_id in (
            self._host_process_group_id,
            self._vm_process_group_id,
        ):
            if process_group_id is None:
                continue
            if process_group_id <= 0 or process_group_id == current_process_group_id:
                continue
            if process_group_id not in known_process_group_ids:
                known_process_group_ids.append(process_group_id)

        return tuple(known_process_group_ids)

    def _refresh_vm_process_group_id_from_pidfile(self) -> None:
        process_group_id = self._pidfile_process_group_id()
        if process_group_id is not None:
            self._vm_process_group_id = process_group_id

    def _pidfile_process_group_id(self) -> int | None:
        if not self._vm_pidfile:
            return None
        try:
            payload = json.loads(Path(self._vm_pidfile).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        process_group_id = self._positive_int_or_none(payload.get("pgid"))
        if process_group_id is not None:
            return process_group_id
        process_id = self._positive_int_or_none(payload.get("pid"))
        if process_id is None:
            return None
        return self._process_group_id_for_pid(process_id)

    @staticmethod
    def _process_group_alive(process_group_id: int) -> bool:
        proc_status = _linux_process_group_has_live_members(process_group_id)
        if proc_status is not None:
            return proc_status

        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _decode_obs(obs: dict[str, Any]) -> dict[str, Any]:
        screenshot = obs.get("screenshot")
        if isinstance(screenshot, Mapping) and "__base64__" in screenshot:
            obs = dict(obs)
            obs["screenshot"] = base64.b64decode(str(screenshot["__base64__"]))
        return obs


def _linux_process_group_has_live_members(process_group_id: int) -> bool | None:
    proc_dir = "/proc"
    if not os.path.isdir(proc_dir):
        return None

    try:
        entries = os.listdir(proc_dir)
    except OSError:
        return None

    for entry in entries:
        if not entry.isdigit():
            continue
        stat_path = os.path.join(proc_dir, entry, "stat")
        try:
            with open(stat_path, encoding="utf-8") as stat_file:
                stat_text = stat_file.read()
        except OSError:
            continue
        fields_start = stat_text.rfind(")")
        if fields_start < 0:
            continue
        fields = stat_text[fields_start + 2 :].split()
        if len(fields) < 3:
            continue
        state = fields[0]
        try:
            member_process_group_id = int(fields[2])
        except ValueError:
            continue
        if member_process_group_id != process_group_id:
            continue
        if state != "Z":
            return True
    return False
