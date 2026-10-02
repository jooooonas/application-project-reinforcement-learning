from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast


@dataclass(frozen=True)
class ParsedComputerUseAction:
    action: str
    pyautogui_code: str | None = None
    terminate_status: Literal["success", "failure"] | None = None
    delta: tuple[int, int] | None = None

    @property
    def terminate(self) -> bool:
        return self.action == "terminate"


def computer_use_arguments_to_action(
    arguments: Mapping[str, Any],
) -> ParsedComputerUseAction | None:
    action = str(arguments.get("action") or "")
    match action:
        case "terminate":
            return ParsedComputerUseAction(
                action=action,
                terminate_status=_terminal_status(arguments),
            )
        case (
            "left_click"
            | "right_click"
            | "middle_click"
            | "double_click"
            | "triple_click"
        ):
            code = _click_code(action, arguments)
        case "mouse_move":
            dx, dy = _delta(arguments)
            code = f"pyautogui.moveRel({dx}, {dy})"
            return ParsedComputerUseAction(
                action=action,
                pyautogui_code=code,
                delta=(dx, dy),
            )
        case "left_click_drag":
            dx, dy = _delta(arguments)
            code = f"pyautogui.dragRel({dx}, {dy}, duration=0.5, button='left')"
            return ParsedComputerUseAction(
                action=action,
                pyautogui_code=code,
                delta=(dx, dy),
            )
        case "type":
            text = str(arguments.get("text") or "")
            code = f"pyautogui.write({json.dumps(text)})"
        case "key":
            key_code = _key_code(arguments)
            if key_code is None:
                return None
            code = key_code
        case "scroll" | "hscroll":
            pixels = int(float(arguments.get("pixels") or 0))
            method = "hscroll" if action == "hscroll" else "scroll"
            code = f"pyautogui.{method}({pixels})"
        case "wait":
            seconds = max(0.0, float(arguments.get("time") or 1.0))
            code = f"time.sleep({seconds:.3f})"
        case _:
            return None
    return ParsedComputerUseAction(action=action, pyautogui_code=code)


def _terminal_status(
    arguments: Mapping[str, Any],
) -> Literal["success", "failure"] | None:
    status = arguments.get("status")
    if status in {"success", "failure"}:
        return cast(Literal["success", "failure"], status)
    return None


def _click_code(action: str, arguments: Mapping[str, Any]) -> str:
    if "coordinate" in arguments or "delta" in arguments:
        raise ValueError("computer_use click actions do not accept position fields")
    match action:
        case "right_click":
            return "pyautogui.click(button='right')"
        case "middle_click":
            return "pyautogui.click(button='middle')"
        case "double_click":
            return "pyautogui.doubleClick()"
        case "triple_click":
            return "pyautogui.tripleClick()"
        case _:
            return "pyautogui.click()"


def _key_code(arguments: Mapping[str, Any]) -> str | None:
    keys = arguments.get("keys", arguments.get("value", arguments.get("key")))
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list) or not keys:
        return None
    normalized = [json.dumps(str(key).lower()) for key in keys]
    if len(normalized) == 1:
        return f"pyautogui.press({normalized[0]})"
    return f"pyautogui.hotkey({', '.join(normalized)})"


def _delta(arguments: Mapping[str, Any]) -> tuple[int, int]:
    if "coordinate" in arguments:
        raise ValueError("computer_use mouse actions use delta, not coordinate")
    delta = arguments.get("delta")
    if not isinstance(delta, list | tuple) or len(delta) != 2:
        raise ValueError("computer_use delta must be a two-item array")
    return int(round(float(delta[0]))), int(round(float(delta[1])))
