from __future__ import annotations

import asyncio
import io
import logging
import time
from enum import Enum
from typing import Any, Protocol

from PIL import Image

logger = logging.getLogger(__name__)

_DESKTOP_READY_INITIAL_DELAY_S = 130.0
_DESKTOP_READY_TIMEOUT_S = 900.0
_DESKTOP_READY_POLL_S = 5.0
_DESKTOP_READY_GET_OBS_TIMEOUT_S = 60.0
_DESKTOP_READY_MIN_NON_DARK_RATIO = 0.05
_DESKTOP_READY_MIN_LUMA_STDDEV = 5.0


class DesktopObserver(Protocol):
    def observe(
        self,
        *,
        request_timeout: float | None = ...,
    ) -> dict[str, Any]: ...


class _DesktopScreenshotStatus(Enum):
    READY = "ready"
    NOT_READY = "not_ready"
    MISSING = "missing"
    INVALID = "invalid"
    EMPTY = "empty"


async def wait_for_desktop_ready(
    env: DesktopObserver,
    *,
    initial_obs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    if initial_obs is not None:
        status, detail = desktop_screenshot_ready(initial_obs)
        if status is _DesktopScreenshotStatus.READY:
            _log_desktop_wait(f"ready immediately ({detail})")
            return initial_obs

    _log_desktop_wait(f"waiting {_DESKTOP_READY_INITIAL_DELAY_S:.1f}s before polling")
    await asyncio.sleep(_DESKTOP_READY_INITIAL_DELAY_S)

    obs: dict[str, Any] = {}
    status = _DesktopScreenshotStatus.MISSING
    detail = "missing screenshot bytes"
    while True:
        elapsed_s = time.monotonic() - started_at
        remaining_s = _DESKTOP_READY_TIMEOUT_S - elapsed_s
        if remaining_s <= 0:
            _log_desktop_wait(f"timed out after {elapsed_s:.1f}s ({detail})")
            _raise_for_structural_screenshot_timeout(status, detail, elapsed_s)
            return obs

        obs = await _observe_desktop(
            env,
            request_timeout=min(_DESKTOP_READY_GET_OBS_TIMEOUT_S, remaining_s),
        )
        status, detail = desktop_screenshot_ready(obs)
        elapsed_s = time.monotonic() - started_at

        if status is _DesktopScreenshotStatus.READY:
            _log_desktop_wait(f"ready after {elapsed_s:.1f}s ({detail})")
            return obs

        if elapsed_s >= _DESKTOP_READY_TIMEOUT_S:
            _log_desktop_wait(f"timed out after {elapsed_s:.1f}s ({detail})")
            _raise_for_structural_screenshot_timeout(status, detail, elapsed_s)
            return obs

        sleep_s = min(_DESKTOP_READY_POLL_S, _DESKTOP_READY_TIMEOUT_S - elapsed_s)
        _log_desktop_wait(
            f"not ready after {elapsed_s:.1f}s; "
            f"status={status.value} ({detail}); retrying in {sleep_s:.1f}s"
        )
        await asyncio.sleep(sleep_s)


def desktop_screenshot_ready(
    obs: dict[str, Any],
) -> tuple[_DesktopScreenshotStatus, str]:
    screenshot = obs.get("screenshot")
    if not isinstance(screenshot, bytes):
        return _DesktopScreenshotStatus.MISSING, "missing screenshot bytes"

    try:
        image = Image.open(io.BytesIO(screenshot)).convert("L")
        image.thumbnail((160, 90))
        pixels = list(image.getdata())
    except Exception as exc:
        return _DesktopScreenshotStatus.INVALID, f"invalid screenshot: {exc!r}"

    if not pixels:
        return _DesktopScreenshotStatus.EMPTY, "empty screenshot"

    mean = sum(pixels) / len(pixels)
    non_dark_ratio = sum(value > 12 for value in pixels) / len(pixels)
    variance = sum((value - mean) ** 2 for value in pixels) / len(pixels)
    stddev = variance**0.5
    status = (
        _DesktopScreenshotStatus.READY
        if (
            non_dark_ratio >= _DESKTOP_READY_MIN_NON_DARK_RATIO
            and stddev >= _DESKTOP_READY_MIN_LUMA_STDDEV
        )
        else _DesktopScreenshotStatus.NOT_READY
    )
    detail = (
        f"mean_luma={mean:.2f}, "
        f"non_dark_ratio={non_dark_ratio:.3f}, "
        f"luma_stddev={stddev:.2f}"
    )
    return status, detail


async def _observe_desktop(
    env: DesktopObserver,
    *,
    request_timeout: float,
) -> dict[str, Any]:
    return await asyncio.to_thread(env.observe, request_timeout=request_timeout)


def _raise_for_structural_screenshot_timeout(
    status: _DesktopScreenshotStatus,
    detail: str,
    elapsed_s: float,
) -> None:
    if status is _DesktopScreenshotStatus.MISSING:
        message = "OSWorld desktop readiness timed out without screenshot bytes"
    elif status is _DesktopScreenshotStatus.INVALID:
        message = "OSWorld desktop readiness timed out with invalid screenshot bytes"
    elif status is _DesktopScreenshotStatus.EMPTY:
        message = "OSWorld desktop readiness timed out with empty screenshot"
    else:
        return
    raise TimeoutError(f"{message} after {elapsed_s:.1f}s ({detail})")


def _log_desktop_wait(message: str) -> None:
    full_message = f"OSWorld desktop readiness: {message}"
    logger.info(full_message)
    print(full_message, flush=True)
