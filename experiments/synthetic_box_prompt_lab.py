#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openai import APIConnectionError, BadRequestError, OpenAI
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from client_common import API_KEY, BASE_URL, MODEL, format_prompt, format_tools
from verifiers.v1 import ToolCall

from rl.computer_use import (
    TARGET_BOX_COMPUTER_USE_TOOL_DEFS,
    ParsedComputerUseAction,
    computer_use_tool_calls_from_text,
    parse_computer_use_tool_calls,
)
from rl.osworld.tasks.target_box.geometry import (
    TargetBox,
    TargetBoxConfig,
    distance_to_box,
    point_in_box,
)
from rl.osworld.tasks.target_box.prompting import (
    SYSTEM_PROMPT,
    TARGET_BOX_INSTRUCTION,
    keep_latest_image_url,
)

BOX_COLORS = {
    "green": (0, 255, 0),
    "red": (255, 0, 0),
}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "synthetic_box_prompt_lab_output"
DEFAULT_TARGET_BOX_CONFIG = TargetBoxConfig()


DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT


@dataclass(frozen=True)
class ScreenConfig:
    screen_width: int
    screen_height: int


@dataclass(frozen=True)
class SyntheticState:
    cursor: tuple[int, int]
    box: TargetBox
    screen: ScreenConfig
    color: str

    @property
    def box_center(self) -> tuple[int, int]:
        return ((self.box.x1 + self.box.x2) // 2, (self.box.y1 + self.box.y2) // 2)


@dataclass(frozen=True)
class RenderedObservation:
    screenshot: bytes
    path: Path


@dataclass(frozen=True)
class StepOutcome:
    action: ParsedComputerUseAction
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    distance_before: float
    distance_after: float
    moved_closer: bool
    inside_after: bool
    stop_reason: str | None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_output_dir(args.output_dir)
    reset_output_dir(output_dir)
    steps_dir = output_dir / "steps"
    prompts_dir = output_dir / "prompts"
    steps_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    screen = resolve_screen(args)
    box = resolve_box(args, screen)
    cursor = resolve_cursor(args, screen, box)
    state = SyntheticState(cursor=cursor, box=box, screen=screen, color=args.color)

    client = None if args.dry_run else OpenAI(base_url=args.base_url, api_key=API_KEY)
    messages = initial_messages(state, args)
    tools = openai_tools()
    transcript_path = output_dir / "transcript.jsonl"

    observation = render_observation(state, args, steps_dir, step_idx=0)
    replace_latest_user_image(messages, observation.screenshot)
    messages = apply_screenshot_history(messages, args)

    print_run_header(state, args, output_dir, observation.path)
    write_prompt_snapshot(prompts_dir, step_idx=1, messages=messages)

    if args.dry_run:
        print_prompt_for_turn(args, messages, tools, step_idx=1)
        print("\nDry run complete. No model request was sent.")
        return 0

    final_reason = "max_steps"
    for step_idx in range(1, args.max_steps + 1):
        print_prompt_for_turn(args, messages, tools, step_idx=step_idx)
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=cast(Any, messages),
                tools=tools,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except APIConnectionError as exc:
            raise SystemExit(
                f"Could not connect to {args.base_url}. Start vLLM first or pass "
                "--base-url/OPENAI_BASE_URL."
            ) from exc
        except BadRequestError as exc:
            if "enable-auto-tool-choice" in str(exc):
                raise SystemExit(
                    "vLLM rejected native tools. Restart vLLM with "
                    "--enable-auto-tool-choice and a compatible "
                    "--tool-call-parser, or use a server that supports tools."
                ) from exc
            raise

        message = response.choices[0].message
        messages.append(assistant_history_message(message))
        actions, tool_calls = parse_model_actions(message)
        response_text = str(message.content or "")

        if len(actions) != 1:
            final_reason = (
                "no_actions_parsed" if not actions else "multiple_actions_parsed"
            )
            record_parse_stop(
                transcript_path,
                step_idx=step_idx,
                response_text=response_text,
                tool_calls=tool_calls,
                stop_reason=final_reason,
            )
            print(f"step {step_idx}: {final_reason}")
            break

        outcome = apply_action(state, actions[0])
        state = SyntheticState(
            cursor=outcome.cursor_after,
            box=state.box,
            screen=state.screen,
            color=state.color,
        )

        next_observation: RenderedObservation | None = None
        if outcome.stop_reason is None:
            next_observation = render_observation(state, args, steps_dir, step_idx)

        record_step(
            transcript_path,
            step_idx=step_idx,
            response_text=response_text,
            tool_calls=tool_calls,
            outcome=outcome,
            screenshot=next_observation.path if next_observation else None,
        )
        print_step_summary(step_idx, outcome, response_text)

        if outcome.stop_reason is not None:
            final_reason = outcome.stop_reason
            break

        assert next_observation is not None
        messages.append(
            user_message(state, args, prefix="Newest observation.", step_idx=step_idx)
        )
        replace_latest_user_image(messages, next_observation.screenshot)
        messages = apply_screenshot_history(messages, args)
        write_prompt_snapshot(prompts_dir, step_idx=step_idx + 1, messages=messages)

    write_summary(output_dir, state, final_reason)
    print(f"\nfinished: {final_reason}")
    print(f"artifacts: {output_dir}")
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a synthetic multi-turn target-box prompt lab without OSWorld or a VM."
        )
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", MODEL))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", BASE_URL),
        help="OpenAI-compatible endpoint. Defaults to OPENAI_BASE_URL or localhost.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. Defaults to experiments/synthetic_box_prompt_lab_output "
            "and is overwritten on each run."
        ),
    )
    parser.add_argument("--background-image", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument(
        "--box-width",
        type=int,
        default=DEFAULT_TARGET_BOX_CONFIG.box_width,
    )
    parser.add_argument(
        "--box-height",
        type=int,
        default=DEFAULT_TARGET_BOX_CONFIG.box_height,
    )
    parser.add_argument("--margin", type=int, default=DEFAULT_TARGET_BOX_CONFIG.margin)
    parser.add_argument("--cursor", nargs=2, type=int, metavar=("X", "Y"))
    parser.add_argument(
        "--cursor-margin",
        type=int,
        default=DEFAULT_TARGET_BOX_CONFIG.cursor_margin,
    )
    parser.add_argument(
        "--instruction",
        default=TARGET_BOX_INSTRUCTION,
        help="Instruction text. Defaults to rl.osworld.tasks.target_box.prompting.TARGET_BOX_INSTRUCTION.",
    )
    parser.add_argument(
        "--color",
        choices=sorted(BOX_COLORS),
        default="green",
        help="Target-box color to render and name in the prompt.",
    )
    parser.add_argument("--box-outline-width", type=int, default=6)
    parser.add_argument(
        "--grid", action="store_true", help="Draw a faint coordinate grid."
    )
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--turn-prompt-file", type=Path)
    parser.add_argument(
        "--screenshot-history",
        choices=("all", "latest"),
        default="latest",
        help=(
            "Use all screenshots in the chat history, or keep only the newest "
            "image while retaining previous text."
        ),
    )
    parser.add_argument(
        "--quiet-prompt",
        action="store_true",
        help="Do not print the full prompt before each model request.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Deprecated; prompts are printed by default.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Render and print only.")
    return parser.parse_args(argv)


def resolve_output_dir(path: Path | None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    return DEFAULT_OUTPUT_DIR


def reset_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory_name in ("steps", "prompts"):
        path = output_dir / directory_name
        if path.exists():
            shutil.rmtree(path)
    for file_name in ("transcript.jsonl", "summary.json"):
        path = output_dir / file_name
        if path.exists():
            path.unlink()


def resolve_screen(args: argparse.Namespace) -> ScreenConfig:
    if args.width is None and args.height is None:
        if args.background_image is not None:
            with Image.open(args.background_image.expanduser()) as image:
                return ScreenConfig(
                    screen_width=image.width, screen_height=image.height
                )
        return ScreenConfig(screen_width=1920, screen_height=1080)
    if args.width is None or args.height is None:
        raise SystemExit("--width and --height must be provided together")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    return ScreenConfig(screen_width=args.width, screen_height=args.height)


def resolve_box(args: argparse.Namespace, screen: ScreenConfig) -> TargetBox:
    if args.box is not None:
        box = TargetBox(*args.box)
        validate_box(box, screen)
        return box

    if args.box_width <= 0 or args.box_height <= 0:
        raise SystemExit("--box-width and --box-height must be positive")
    if args.margin < 0:
        raise SystemExit("--margin must be non-negative")
    if args.box_width + 2 * args.margin > screen.screen_width:
        raise SystemExit("box width plus margins does not fit the screen")
    if args.box_height + 2 * args.margin > screen.screen_height:
        raise SystemExit("box height plus margins does not fit the screen")

    rng = random.Random(f"{args.seed}:box")
    x1 = rng.randint(args.margin, screen.screen_width - args.margin - args.box_width)
    y1 = rng.randint(args.margin, screen.screen_height - args.margin - args.box_height)
    return TargetBox(
        x1=x1, y1=y1, x2=x1 + args.box_width - 1, y2=y1 + args.box_height - 1
    )


def resolve_cursor(
    args: argparse.Namespace,
    screen: ScreenConfig,
    box: TargetBox,
) -> tuple[int, int]:
    if args.cursor is not None:
        cursor = (args.cursor[0], args.cursor[1])
        validate_cursor(cursor, screen)
        return cursor

    margin = args.cursor_margin
    if 2 * margin >= screen.screen_width or 2 * margin >= screen.screen_height:
        raise SystemExit("--cursor-margin leaves no valid cursor area")

    min_x = margin
    max_x = screen.screen_width - margin - 1
    min_y = margin
    max_y = screen.screen_height - margin - 1
    rng = random.Random(f"{args.seed}:cursor")
    for _ in range(100):
        cursor = (rng.randint(min_x, max_x), rng.randint(min_y, max_y))
        if not point_in_box(cursor, box):
            return cursor

    candidates = [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)]
    outside = [
        candidate for candidate in candidates if not point_in_box(candidate, box)
    ]
    if not outside:
        raise SystemExit("box leaves no valid cursor start outside the box")
    return max(outside, key=lambda candidate: distance_to_box(candidate, box))


def validate_box(box: TargetBox, screen: ScreenConfig) -> None:
    if box.x1 > box.x2 or box.y1 > box.y2:
        raise SystemExit("--box must satisfy X1 <= X2 and Y1 <= Y2")
    if box.x1 < 0 or box.y1 < 0:
        raise SystemExit("--box must be inside the screen")
    if box.x2 >= screen.screen_width or box.y2 >= screen.screen_height:
        raise SystemExit("--box must be inside the screen")


def validate_cursor(cursor: tuple[int, int], screen: ScreenConfig) -> None:
    x, y = cursor
    if not (0 <= x < screen.screen_width and 0 <= y < screen.screen_height):
        raise SystemExit("--cursor must be inside the screen")


def initial_messages(
    state: SyntheticState, args: argparse.Namespace
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(state, args)}
    ]
    messages.append(
        user_message(state, args, prefix="Initial observation.", step_idx=0)
    )
    return messages


def write_prompt_snapshot(
    prompts_dir: Path,
    *,
    step_idx: int,
    messages: list[dict[str, Any]],
) -> None:
    (prompts_dir / f"step_{step_idx:03d}.txt").write_text(
        format_prompt(messages),
        encoding="utf-8",
    )


def print_prompt_for_turn(
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    step_idx: int,
) -> None:
    if args.quiet_prompt and not args.print_prompt:
        return
    print(f"\n=== Turn {step_idx} request ===")
    print(format_prompt(messages))
    print(format_tools(tools))


def apply_screenshot_history(
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.screenshot_history == "latest":
        return cast(
            list[dict[str, Any]],
            keep_latest_image_url(cast(Any, messages)),
        )
    return messages


def system_prompt(state: SyntheticState, args: argparse.Namespace) -> str:
    template = (
        read_text(args.system_prompt_file)
        if args.system_prompt_file
        else DEFAULT_SYSTEM_PROMPT
    )
    return render_text_template(template, state, step_idx=0, max_steps=args.max_steps)


def user_message(
    state: SyntheticState,
    args: argparse.Namespace,
    *,
    prefix: str,
    step_idx: int,
) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": turn_prompt(state, args, prefix=prefix, step_idx=step_idx),
            },
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
        ],
    }


def turn_prompt(
    state: SyntheticState,
    args: argparse.Namespace,
    *,
    prefix: str,
    step_idx: int,
) -> str:
    if args.turn_prompt_file:
        template = read_text(args.turn_prompt_file)
        return render_text_template(
            template,
            state,
            step_idx=step_idx,
            max_steps=args.max_steps,
        )
    return "\n".join(
        [
            f"Instruction: {args.instruction}",
            f"Screen resolution: {state.screen.screen_width}x{state.screen.screen_height}.",
            prefix,
        ]
    )


def render_text_template(
    template: str,
    state: SyntheticState,
    *,
    step_idx: int,
    max_steps: int,
) -> str:
    center_x, center_y = state.box_center
    values = {
        "color": state.color,
        "box_center_field": f"{state.color}_box_center_estimate",
        "width": state.screen.screen_width,
        "height": state.screen.screen_height,
        "step": step_idx,
        "max_steps": max_steps,
        "box_center_x": center_x,
        "box_center_y": center_y,
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def replace_latest_user_image(
    messages: list[dict[str, Any]], screenshot: bytes
) -> None:
    url = image_data_url(screenshot)
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in reversed(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                part["image_url"] = {"url": url}
                return
    raise RuntimeError("latest user message has no image_url part")


def render_observation(
    state: SyntheticState,
    args: argparse.Namespace,
    steps_dir: Path,
    step_idx: int,
) -> RenderedObservation:
    image = base_desktop_image(state, args)
    draw = ImageDraw.Draw(image)
    draw_target_box(draw, state.box, state.color, width=args.box_outline_width)
    draw_cursor(draw, *state.cursor)

    output = io.BytesIO()
    image.save(output, format="PNG")
    screenshot = output.getvalue()
    path = steps_dir / f"step_{step_idx:03d}.png"
    path.write_bytes(screenshot)
    return RenderedObservation(screenshot=screenshot, path=path)


def base_desktop_image(state: SyntheticState, args: argparse.Namespace) -> Image.Image:
    width = state.screen.screen_width
    height = state.screen.screen_height
    if args.background_image:
        image = Image.open(args.background_image.expanduser()).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height))
        return image

    image = Image.new("RGB", (width, height), (236, 238, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, 34], fill=(42, 45, 49))
    draw.rectangle([0, 34, 58, height], fill=(52, 55, 61))
    draw.rectangle(
        [92, 86, width - 92, height - 92],
        fill=(249, 249, 247),
        outline=(199, 204, 210),
        width=2,
    )
    draw.rectangle([92, 86, width - 92, 126], fill=(224, 228, 232))
    for x in range(120, min(width - 120, 520), 86):
        draw.rounded_rectangle(
            [x, 150, x + 52, 202],
            radius=8,
            fill=(219, 224, 229),
            outline=(191, 198, 205),
        )
    if args.grid:
        draw_grid(draw, width, height)
    return image


def draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    for x in range(0, width, 160):
        draw.line([(x, 0), (x, height)], fill=(216, 221, 226), width=1)
    for y in range(0, height, 120):
        draw.line([(0, y), (width, y)], fill=(216, 221, 226), width=1)


def draw_target_box(
    draw: ImageDraw.ImageDraw,
    box: TargetBox,
    color: str,
    *,
    width: int,
) -> None:
    draw.rectangle(
        [box.x1, box.y1, box.x2, box.y2], outline=BOX_COLORS[color], width=width
    )


def draw_cursor(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    points = [
        (x, y),
        (x, y + 35),
        (x + 9, y + 27),
        (x + 17, y + 45),
        (x + 28, y + 40),
        (x + 20, y + 24),
        (x + 34, y + 24),
    ]
    shadow = [(px + 3, py + 4) for px, py in points]
    draw.polygon(shadow, fill=(76, 76, 76))
    draw.polygon(points, fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line(points + [points[0]], fill=(0, 0, 0), width=2)


def apply_action(state: SyntheticState, action: ParsedComputerUseAction) -> StepOutcome:
    before = state.cursor
    distance_before = distance_to_box(before, state.box)
    stop_reason: str | None = None

    if action.terminate:
        inside = point_in_box(before, state.box)
        if action.terminate_status == "success" and inside:
            stop_reason = "target_box_success"
        elif action.terminate_status == "failure":
            stop_reason = "target_box_agent_failure"
        else:
            stop_reason = "target_box_bad_terminate"
        return StepOutcome(
            action=action,
            cursor_before=before,
            cursor_after=before,
            distance_before=distance_before,
            distance_after=distance_before,
            moved_closer=False,
            inside_after=inside,
            stop_reason=stop_reason,
        )

    if action.action != "mouse_move" or action.delta is None:
        return StepOutcome(
            action=action,
            cursor_before=before,
            cursor_after=before,
            distance_before=distance_before,
            distance_after=distance_before,
            moved_closer=False,
            inside_after=point_in_box(before, state.box),
            stop_reason="target_box_invalid_action",
        )

    dx, dy = action.delta
    after = clamp_cursor(
        (before[0] + dx, before[1] + dy),
        width=state.screen.screen_width,
        height=state.screen.screen_height,
    )
    distance_after = distance_to_box(after, state.box)
    return StepOutcome(
        action=action,
        cursor_before=before,
        cursor_after=after,
        distance_before=distance_before,
        distance_after=distance_after,
        moved_closer=distance_after < distance_before,
        inside_after=point_in_box(after, state.box),
        stop_reason=None,
    )


def clamp_cursor(
    cursor: tuple[int, int],
    *,
    width: int,
    height: int,
) -> tuple[int, int]:
    return (
        max(0, min(width - 1, int(cursor[0]))),
        max(0, min(height - 1, int(cursor[1]))),
    )


def parse_model_actions(
    message: Any,
) -> tuple[list[ParsedComputerUseAction], list[ToolCall]]:
    native_tool_calls = openai_tool_calls_to_verifiers(
        getattr(message, "tool_calls", None)
    )
    content = str(getattr(message, "content", "") or "")
    tool_calls = native_tool_calls or computer_use_tool_calls_from_text(content)
    return parse_computer_use_tool_calls(tool_calls), tool_calls


def openai_tool_calls_to_verifiers(raw_tool_calls: Any) -> list[ToolCall]:
    if not raw_tool_calls:
        return []
    tool_calls: list[ToolCall] = []
    for raw in raw_tool_calls:
        function = getattr(raw, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        tool_call_id = getattr(raw, "id", None)
        if isinstance(name, str) and isinstance(arguments, str):
            tool_calls.append(
                ToolCall(
                    id=str(tool_call_id or f"tool-call-{len(tool_calls)}"),
                    name=name,
                    arguments=arguments,
                )
            )
    return tool_calls


def assistant_history_message(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": str(getattr(message, "content", "") or ""),
    }
    raw_tool_calls = getattr(message, "tool_calls", None)
    if raw_tool_calls:
        tool_calls = []
        for raw in raw_tool_calls:
            function = getattr(raw, "function", None)
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
            if isinstance(name, str) and isinstance(arguments, str):
                tool_calls.append(
                    {
                        "type": "function",
                        "id": str(getattr(raw, "id", f"tool-call-{len(tool_calls)}")),
                        "function": {"name": name, "arguments": arguments},
                    }
                )
        if tool_calls:
            payload["tool_calls"] = tool_calls
    return payload


def openai_tools() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": tool.model_dump(exclude_none=True)}
        for tool in TARGET_BOX_COMPUTER_USE_TOOL_DEFS
    ]


def record_step(
    transcript_path: Path,
    *,
    step_idx: int,
    response_text: str,
    tool_calls: list[ToolCall],
    outcome: StepOutcome,
    screenshot: Path | None,
) -> None:
    append_jsonl(
        transcript_path,
        {
            "step": step_idx,
            "response": response_text,
            "tool_calls": serialize_tool_calls(tool_calls),
            "action": action_payload(outcome.action),
            "cursor_before": list(outcome.cursor_before),
            "cursor_after": list(outcome.cursor_after),
            "distance_before": round(outcome.distance_before, 3),
            "distance_after": round(outcome.distance_after, 3),
            "moved_closer": outcome.moved_closer,
            "inside_after": outcome.inside_after,
            "stop_reason": outcome.stop_reason,
            "screenshot": str(screenshot) if screenshot else None,
        },
    )


def record_parse_stop(
    transcript_path: Path,
    *,
    step_idx: int,
    response_text: str,
    tool_calls: list[ToolCall],
    stop_reason: str,
) -> None:
    append_jsonl(
        transcript_path,
        {
            "step": step_idx,
            "response": response_text,
            "tool_calls": serialize_tool_calls(tool_calls),
            "stop_reason": stop_reason,
        },
    )


def write_summary(output_dir: Path, state: SyntheticState, final_reason: str) -> None:
    summary = {
        "final_reason": final_reason,
        "final_cursor": list(state.cursor),
        "target_box": state.box.as_dict(),
        "target_box_center": list(state.box_center),
        "target_box_color": state.color,
        "target_box_inside": point_in_box(state.cursor, state.box),
        "target_box_distance": round(distance_to_box(state.cursor, state.box), 3),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def action_payload(action: ParsedComputerUseAction) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": action.action}
    if action.delta is not None:
        payload["delta"] = list(action.delta)
    if action.terminate_status is not None:
        payload["status"] = action.terminate_status
    return payload


def serialize_tool_calls(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [
        {"id": tool_call.id, "name": tool_call.name, "arguments": tool_call.arguments}
        for tool_call in tool_calls
    ]


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def image_data_url(screenshot: bytes) -> str:
    encoded = base64.b64encode(screenshot).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def read_text(path: Path) -> str:
    return path.expanduser().read_text(encoding="utf-8")


def print_run_header(
    state: SyntheticState,
    args: argparse.Namespace,
    output_dir: Path,
    initial_screenshot: Path,
) -> None:
    print("=== Synthetic target-box prompt lab ===")
    print(f"model: {args.model}")
    print(f"base_url: {args.base_url}")
    print(f"screen: {state.screen.screen_width}x{state.screen.screen_height}")
    print(f"box_color: {state.color}")
    print(f"target_box: {state.box.as_dict()}")
    print(f"target_box_center: {list(state.box_center)}")
    print(f"start_cursor: {list(state.cursor)}")
    print(f"max_steps: {args.max_steps}")
    print(f"screenshot_history: {args.screenshot_history}")
    print(f"initial_screenshot: {initial_screenshot}")
    print(f"output_dir: {output_dir}\n")


def print_step_summary(step_idx: int, outcome: StepOutcome, response_text: str) -> None:
    action = action_payload(outcome.action)
    print(f"step {step_idx}:")
    if response_text.strip():
        print(indent_block(response_text.strip(), prefix="  response: "))
    print(f"  action: {action}")
    print(f"  cursor: {list(outcome.cursor_before)} -> {list(outcome.cursor_after)}")
    print(
        "  distance: "
        f"{outcome.distance_before:.1f}px -> {outcome.distance_after:.1f}px "
        f"(closer={outcome.moved_closer})"
    )
    print(f"  inside_after: {outcome.inside_after}")
    if outcome.stop_reason:
        print(f"  stop_reason: {outcome.stop_reason}")


def indent_block(text: str, *, prefix: str) -> str:
    lines = text.splitlines() or [""]
    first, *rest = lines
    return "\n".join(
        [f"{prefix}{first}", *[f"{' ' * len(prefix)}{line}" for line in rest]]
    )


if __name__ == "__main__":
    raise SystemExit(main())
