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

Estimate coordinates in the requested coordinate systems. Return only JSON.
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
    right_edge = screen.width - 1
    bottom_edge = screen.height - 1
    center_x = screen.width // 2
    center_y = screen.height // 2
    right_quarter_x = round(screen.width * 0.75)
    lower_half_y = screen.height // 2
    lines = [
        "Estimate the current mouse cursor tip/hotspot position in the screenshot.",
        f"Screen resolution: {screen.width}x{screen.height}.",
        (
            "Pixel coordinate system: [0, 0] is the top-left pixel; "
            "x increases right; y increases down."
        ),
        (
            "The model may naturally estimate positions on a 0-1000 normalized "
            "coordinate grid. First estimate that normalized position, then "
            "convert it to pixel coordinates."
        ),
        (
            "Pixel conversion: x_px = round(x_norm / 1000 * "
            f"{screen.width}); y_px = round(y_norm / 1000 * {screen.height})."
        ),
        (
            "Return cursor_position in original screenshot pixel coordinates, "
            "not normalized coordinates."
        ),
        (
            f"Calibration: the center is [{center_x}, {center_y}], the right "
            f"edge is x={right_edge}, and the bottom edge is y={bottom_edge}."
        ),
        (
            f"A cursor in the right quarter has x > {right_quarter_x}; "
            f"a cursor in the lower half has y > {lower_half_y}."
        ),
    ]
    if box is not None:
        box_center_x, box_center_y = box.center
        lines.extend(
            [
                f"Red box bounds: x={box.x1}..{box.x2}, y={box.y1}..{box.y2}.",
                f"Red box center: [{box_center_x}, {box_center_y}].",
            ]
        )
    lines.extend(
        [
            "Look for the visible mouse pointer/cursor, not the red box.",
            "This is only a localization task; do not propose a mouse movement.",
            (
                "If no cursor is visible, set normalized_position_1000 and "
                "cursor_position to null and confidence to 0."
            ),
            (
                'Return only JSON with keys "normalized_position_1000", '
                '"cursor_position", "confidence", and "reason".'
            ),
            (
                "Use [x_norm, y_norm] integer values from 0 to 1000 for "
                "normalized_position_1000, or null."
            ),
            "Use [x_px, y_px] integer pixels for cursor_position, or null.",
            'Example shape: {"normalized_position_1000": [900, 880], '
            '"cursor_position": [1728, 950], "confidence": 0.8, '
            '"reason": "short visual cue"}',
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
