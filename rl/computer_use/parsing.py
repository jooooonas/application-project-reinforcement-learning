from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from verifiers.v1 import ToolCall

from .actions import ParsedComputerUseAction, computer_use_arguments_to_action

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_computer_use_actions(response_text: str) -> list[ParsedComputerUseAction]:
    return parse_computer_use_tool_calls(
        computer_use_tool_calls_from_text(response_text)
    )


def parse_computer_use_tool_calls(
    tool_calls: Iterable[ToolCall],
) -> list[ParsedComputerUseAction]:
    actions: list[ParsedComputerUseAction] = []
    for tool_call in tool_calls:
        parsed = computer_use_tool_call_to_action(tool_call)
        if parsed is not None:
            actions.append(parsed)
    return actions


def computer_use_tool_call_to_action(
    tool_call: ToolCall,
) -> ParsedComputerUseAction | None:
    if tool_call.name != "computer_use":
        return None
    raw_arguments = tool_call.arguments or "{}"
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw_arguments, Mapping):
        arguments = raw_arguments
    else:
        return None
    if not isinstance(arguments, dict):
        return None
    try:
        return computer_use_arguments_to_action(arguments)
    except (TypeError, ValueError, OverflowError):
        return None


def computer_use_tool_calls_from_text(response_text: str) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    for idx, payload in enumerate(_tool_call_payloads(response_text)):
        tool_call = _tool_call_from_payload(payload, fallback_id=f"content-call-{idx}")
        if tool_call is not None:
            tool_calls.append(tool_call)
    return tool_calls


def _tool_call_payloads(response_text: str) -> Iterable[dict[str, Any]]:
    matched = False
    for match in _TOOL_CALL_RE.finditer(response_text or ""):
        matched = True
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload
    if matched:
        return
    try:
        payload = json.loads(response_text or "")
    except json.JSONDecodeError:
        return
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def _tool_call_from_payload(
    payload: Mapping[str, Any],
    *,
    fallback_id: str,
) -> ToolCall | None:
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return None

    raw_arguments = payload.get("arguments") or {}
    if isinstance(raw_arguments, str):
        arguments = raw_arguments
    elif isinstance(raw_arguments, Mapping):
        try:
            arguments = json.dumps(raw_arguments, separators=(",", ":"))
        except (TypeError, ValueError):
            return None
    else:
        return None

    tool_call_id = payload.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        tool_call_id = fallback_id
    return ToolCall(id=tool_call_id, name=name, arguments=arguments)
