from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from client_common import (
    API_KEY,
    BASE_URL,
    MODEL,
    format_prompt,
    image_data_url,
    image_size,
    print_model_response,
)
from openai import APIConnectionError, OpenAI

SYSTEM_PROMPT = """You are a visual desktop coordinate inspector.

Estimate coordinates in the screenshot coordinate system. Return only JSON.
Do not describe actions, movements, or tool calls.
"""


@dataclass(frozen=True)
class ScreenSize:
    width: int
    height: int


@dataclass(frozen=True)
class BoxBounds:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the screenshot to inspect.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", MODEL))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", BASE_URL),
        help="OpenAI-compatible endpoint. Defaults to OPENAI_BASE_URL or localhost.",
    )
    parser.add_argument(
        "--screen",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Screen size to put in the prompt. Defaults to the image size.",
    )
    parser.add_argument(
        "--box",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional red-box bounds to include as a visual anchor.",
    )
    parser.add_argument(
        "--cursor",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="Known cursor position for local comparison only; not sent to the model.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompt only.")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    screen = resolve_screen_size(image_path, args.screen)
    box = BoxBounds(*args.box) if args.box else None
    messages = build_cursor_messages(image_path, screen, box)

    print(format_prompt(messages))
    print_reference(args.cursor)

    if args.dry_run:
        return

    client = OpenAI(base_url=args.base_url, api_key=API_KEY)
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
    except APIConnectionError as exc:
        raise SystemExit(
            f"Could not connect to {args.base_url}. Start vLLM on this node, "
            "forward the port, or pass --base-url/OPENAI_BASE_URL."
        ) from exc

    print_model_response(resp)
    compare_to_reference(resp.choices[0].message.content, args.cursor)


def resolve_screen_size(image_path: Path, override: list[int] | None) -> ScreenSize:
    if override is not None:
        return ScreenSize(width=override[0], height=override[1])
    width, height = image_size(image_path)
    return ScreenSize(width=width, height=height)


def build_cursor_messages(
    image_path: Path,
    screen: ScreenSize,
    box: BoxBounds | None,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": cursor_prompt_text(screen, box)},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(image_path)},
                },
            ],
        },
    ]


def cursor_prompt_text(screen: ScreenSize, box: BoxBounds | None) -> str:
    lines = [
        "Estimate the current mouse cursor position in the screenshot.",
        f"Screen resolution: {screen.width}x{screen.height}.",
        (
            "Coordinate system: [0, 0] is the top-left pixel; "
            "x increases right; y increases down."
        ),
        "",
        "Coordinate calibration examples for this screen:",
        (
            f"- If the cursor tip is at the exact screen center, return "
            f"[{screen.width // 2}, {screen.height // 2}], not [500, 500]."
        ),
        (
            f"- If the cursor tip is at the top-right pixel, return "
            f"[{screen.width - 1}, 0]."
        ),
        (
            f"- If the cursor tip is in the right quarter of the screen, "
            f"its x coordinate must be greater than {screen.width * 3 // 4}."
        ),
        (
            f"- If the cursor tip is in the bottom-right quarter, "
            f"x must be greater than {screen.width // 2} and "
            f"y must be greater than {screen.height // 2}."
        ),
        "- Do not return normalized 0..1000 coordinates.",
    ]
    if box is not None:
        center_x, center_y = box.center
        lines.extend(
            [
                f"Red box bounds: x={box.x1}..{box.x2}, y={box.y1}..{box.y2}.",
                f"Red box center: [{center_x}, {center_y}].",
            ]
        )
    lines.extend(
        [
            "Look for the visible mouse pointer/cursor, not the red box.",
            "This is only a localization task; do not propose a mouse movement.",
            "If no cursor is visible, set cursor_position to null and confidence to 0.",
            (
                'Return only JSON with keys "cursor_position", '
                '"confidence", and "reason".'
            ),
            "Use [x, y] integer pixels for cursor_position, or null.",
        ]
    )
    return "\n".join(lines)


def print_reference(cursor: list[int] | None) -> None:
    if cursor is None:
        return
    print("\n=== Reference cursor (not sent to model) ===")
    print(f"cursor_position: [{cursor[0]}, {cursor[1]}]")


def compare_to_reference(content: str | None, cursor: list[int] | None) -> None:
    if content is None or cursor is None:
        return

    estimate = parse_cursor_position(content)
    print("\n=== Cursor error ===")
    if estimate is None:
        print("Could not parse a numeric cursor_position from the model response.")
        return

    dx = estimate[0] - cursor[0]
    dy = estimate[1] - cursor[1]
    distance = math.hypot(dx, dy)
    print(f"estimate: [{estimate[0]}, {estimate[1]}]")
    print(f"reference: [{cursor[0]}, {cursor[1]}]")
    print(f"delta: [{dx}, {dy}]")
    print(f"distance: {distance:.1f}px")


def parse_cursor_position(content: str) -> tuple[int, int] | None:
    data = parse_json_object(content)
    if not isinstance(data, dict):
        return None
    position = data.get("cursor_position")
    if (
        not isinstance(position, list)
        or len(position) != 2
        or not all(isinstance(value, int | float) for value in position)
    ):
        return None
    return (round(position[0]), round(position[1]))


def parse_json_object(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    main()
