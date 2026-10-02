from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from pathlib import Path
from textwrap import indent
from typing import Any

from PIL import Image
from verifiers.v1 import Message, Messages

from rl.computer_use import COMPUTER_USE_TOOL_DEFS

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def openai_messages(messages: Messages) -> list[dict[str, Any]]:
    return [openai_message(message) for message in messages]


def openai_message(message: Message) -> dict[str, Any]:
    data = message.model_dump(exclude_none=True)
    if data.get("role") != "assistant" or "tool_calls" not in data:
        return data

    data["tool_calls"] = [
        {
            "type": "function",
            "id": tool_call["id"],
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            },
        }
        for tool_call in data["tool_calls"]
    ]
    return data


def openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": tool.model_dump(exclude_none=True),
        }
        for tool in COMPUTER_USE_TOOL_DEFS
    ]


def format_prompt(messages: list[dict[str, Any]]) -> str:
    lines = ["=== Prompt sent to vLLM ==="]
    for index, message in enumerate(messages):
        lines.append("")
        lines.append(f"[{index}] {str(message.get('role', 'unknown')).upper()}")
        lines.extend(format_content(message.get("content")))
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            lines.extend(format_tool_calls(tool_calls))
    return "\n".join(lines)


def format_content(content: Any) -> list[str]:
    if isinstance(content, str):
        return [indent(content.rstrip() or "(empty)", "  ")]
    if not isinstance(content, list):
        return [f"  {content!r}"]

    lines: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            lines.append(f"  {part!r}")
            continue

        if part.get("type") == "text":
            text = part.get("text")
            lines.append(indent(str(text).rstrip() if text else "(empty)", "  "))
            continue

        if part.get("type") == "image_url":
            lines.append(format_image_part(part))
            continue

        lines.append(f"  [{part.get('type', 'unknown part')}] {part!r}")
    return lines


def format_image_part(part: Mapping[str, Any]) -> str:
    image_url = part.get("image_url")
    if not isinstance(image_url, Mapping):
        return "  [image_url] <missing>"

    url = image_url.get("url")
    if not isinstance(url, str):
        return "  [image_url] <missing>"

    if url.startswith("data:image/"):
        mime = url.split(";", 1)[0].removeprefix("data:")
        return f"  [image_url] {mime}; base64 omitted from display ({len(url)} chars)"

    return f"  [image_url] {url}"


def format_tool_calls(tool_calls: list[Any]) -> list[str]:
    lines = ["  tool_calls:"]
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            lines.append(f"    {tool_call!r}")
            continue
        function = tool_call.get("function")
        if isinstance(function, Mapping):
            lines.append(f"    {function.get('name')}: {function.get('arguments')}")
        else:
            lines.append(f"    {tool_call!r}")
    return lines


def format_tools(tools: list[dict[str, Any]]) -> str:
    lines = ["", "=== Tools sent to vLLM ==="]
    for tool in tools:
        function = tool.get("function", {})
        if not isinstance(function, Mapping):
            lines.append(f"- {tool!r}")
            continue
        lines.append(f"- {function.get('name')}: {function.get('description')}")
    return "\n".join(lines)


def print_model_response(resp: Any) -> None:
    print("\n=== Model response ===")
    message = resp.choices[0].message
    if message.content:
        print(message.content)
    if message.tool_calls:
        for tool_call in message.tool_calls:
            print(
                json.dumps(
                    {
                        "tool": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                    sort_keys=True,
                )
            )
