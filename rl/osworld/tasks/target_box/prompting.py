from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any, Protocol, cast

from verifiers.v1 import (
    AssistantMessage,
    Messages,
    SystemMessage,
    ToolCall,
    UserMessage,
)

SYSTEM_PROMPT = """You control an Ubuntu desktop with exactly one computer_use tool call per step.

Only use action=mouse_move or action=terminate.
Use the newest screenshot as the current desktop state; older screenshots are only history.
The important visual clues are the cursor and the green box/square.
Before doing arithmetic, write four words-only visual lines.
These four visual lines must not contain digits, brackets, pixel values, or coordinate pairs.
inside: describe whether you think that the cursor is currently inside or outside of the green box/square.
cursor_visual: describe the cursor tip as a screen region using words only.
box_visual: describe where the green square is placed on the screen using words only.
move_visual: one sentence saying how the cursor should move visually to land in the box center, using words only.
After those visual lines, write three estimate lines:
cursor_position_estimate: [cx, cy]
green_box_center_estimate: [bx, by]
delta_estimate: [dx, dy]
Compute delta_estimate as green_box_center_estimate minus cursor_position_estimate.
Positive dx moves right, negative dx moves left; positive dy moves down, negative dy moves up.
If the visual move sentence disagrees with the delta signs, fix the estimates before calling the tool.
Aim mouse_move at the middle of the green box.
When the cursor is inside the green box, write delta_estimate: [0, 0] and call terminate with status=success.
"""


class ScreenConfig(Protocol):
    @property
    def screen_width(self) -> int: ...

    @property
    def screen_height(self) -> int: ...


TARGET_BOX_INSTRUCTION = (
    "Move the cursor into the green box. If the cursor is already inside the box, "
    "terminate with success. If the cursor is outside the box, move it towards "
    "the box center."
)


def target_box_initial_user_messages(
    obs: Mapping[str, Any],
    config: ScreenConfig,
) -> Messages:
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        *target_box_few_shot_messages(config),
        UserMessage(
            content=_target_box_user_content(
                obs,
                config,
                prefix="Initial observation.",
            )
        ),
    ]


def target_box_observation_messages(
    obs: Mapping[str, Any],
    config: ScreenConfig,
) -> Messages:
    return [
        UserMessage(
            content=_target_box_user_content(
                obs,
                config,
                prefix="Newest observation.",
            )
        )
    ]


def keep_latest_image_url[MessageT](messages: list[MessageT]) -> list[MessageT]:
    latest_image_index = None
    for index, message in enumerate(messages):
        if _message_has_image_url(message):
            latest_image_index = index

    if latest_image_index is None:
        return messages

    return [
        message if index == latest_image_index else _replace_image_urls(message)
        for index, message in enumerate(messages)
    ]


def _target_box_user_text(
    config: ScreenConfig,
    *,
    prefix: str,
) -> str:
    if prefix != "Initial observation.":
        return prefix

    return "\n".join(
        [
            f"Instruction: {TARGET_BOX_INSTRUCTION}",
            f"Screen resolution: {config.screen_width}x{config.screen_height}.",
            prefix,
        ]
    )


def _target_box_user_content(
    obs: Mapping[str, Any],
    config: ScreenConfig,
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": _target_box_user_text(config, prefix=prefix)},
        {"type": "image_url", "image_url": {"url": obs_image_url(obs)}},
    ]


def target_box_few_shot_messages(config: ScreenConfig) -> Messages:
    width = config.screen_width
    height = config.screen_height
    examples = [
        _target_box_move_example(
            width,
            height,
            cursor_fraction=(0.981, 0.854),
            box_center_fraction=(0.809, 0.744),
        ),
        _target_box_terminate_example(width, height),
    ]

    messages: Messages = []
    for index, (text, assistant_text, arguments) in enumerate(examples, start=1):
        messages.extend(
            [
                UserMessage(content=[{"type": "text", "text": text}]),
                AssistantMessage(
                    content=assistant_text,
                    tool_calls=[
                        ToolCall(
                            id=f"target-box-example-{index}",
                            name="computer_use",
                            arguments=json.dumps(arguments),
                        )
                    ],
                ),
            ]
        )
    return messages


def _target_box_move_example(
    width: int,
    height: int,
    *,
    cursor_fraction: tuple[float, float],
    box_center_fraction: tuple[float, float],
) -> tuple[str, str, dict[str, Any]]:
    cursor = _fractional_position(width, height, cursor_fraction)
    box_center = _fractional_position(width, height, box_center_fraction)
    delta = [box_center[0] - cursor[0], box_center[1] - cursor[1]]
    text = "\n".join(
        [
            "Formatting example observation.",
            f"Instruction: {TARGET_BOX_INSTRUCTION}",
            f"Screen resolution: {width}x{height}.",
            (
                "The screenshot shows the cursor tip near the bottom right corner "
                "and the green box also in the bottom right quadrant but closer "
                "to the screen center."
            ),
        ]
    )
    assistant_text = "\n".join(
        [
            "inside: cursor sits outside of the box",
            (
                "cursor_visual: the cursor tip is near the bottom right corner "
                "of the screen"
            ),
            (
                "box_visual: the green box is also in the bottom right quadrant "
                "of the screen but closer to the center"
            ),
            (
                "move_visual: move the cursor up and left toward the box center "
                "without overshooting"
            ),
            f"cursor_position_estimate: {cursor}",
            f"green_box_center_estimate: {box_center}",
            f"delta_estimate: {delta}",
        ]
    )
    return text, assistant_text, {"action": "mouse_move", "delta": delta}


def _target_box_terminate_example(
    width: int,
    height: int,
) -> tuple[str, str, dict[str, Any]]:
    cursor = _fractional_position(width, height, (0.5, 0.5))
    text = "\n".join(
        [
            "Formatting example observation.",
            f"Instruction: {TARGET_BOX_INSTRUCTION}",
            f"Screen resolution: {width}x{height}.",
            "The screenshot shows the cursor tip already inside the green box.",
        ]
    )
    assistant_text = "\n".join(
        [
            "inside: cursor sits inside the box",
            "cursor_visual: the cursor tip is inside the green box",
            "box_visual: the green box surrounds the cursor tip",
            (
                "move_visual: no movement is needed because the cursor is already "
                "in the box"
            ),
            f"cursor_position_estimate: {cursor}",
            f"green_box_center_estimate: {cursor}",
            "delta_estimate: [0, 0]",
        ]
    )
    return text, assistant_text, {"action": "terminate", "status": "success"}


def _fractional_position(
    width: int,
    height: int,
    fraction: tuple[float, float],
) -> list[int]:
    return [
        round(fraction[0] * (width - 1)),
        round(fraction[1] * (height - 1)),
    ]


def _message_has_image_url(message: Any) -> bool:
    content = _message_content(message)
    return isinstance(content, list) and any(
        _is_image_url_part(part) for part in content
    )


def _replace_image_urls(message: Any) -> Any:
    content = _message_content(message)
    if not isinstance(content, list):
        return message

    changed = False
    replacement_used = False
    updated_content: list[Any] = []
    for part in content:
        if _is_image_url_part(part):
            changed = True
            if not replacement_used:
                updated_content.append(
                    {"type": "text", "text": "Previous screenshot omitted."}
                )
                replacement_used = True
            continue
        updated_content.append(part)

    if not changed:
        return message
    return _copy_message_with_content(message, updated_content)


def _message_content(message: Any) -> Any:
    if isinstance(message, Mapping):
        return message.get("content")
    return getattr(message, "content", None)


def _is_image_url_part(part: Any) -> bool:
    data = _content_part_mapping(part)
    return data is not None and data.get("type") == "image_url"


def _content_part_mapping(part: Any) -> Mapping[str, Any] | None:
    if isinstance(part, Mapping):
        return part
    if hasattr(part, "model_dump"):
        return cast(Mapping[str, Any], part.model_dump(exclude_none=True))
    return None


def _copy_message_with_content(message: Any, content: list[Any]) -> Any:
    if isinstance(message, Mapping):
        updated = dict(message)
        updated["content"] = content
        return updated
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
        data["content"] = content
        return type(message)(**data)
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    return message


def obs_image_url(obs: Mapping[str, Any]) -> str:
    screenshot = obs.get("screenshot")
    if not screenshot:
        raise ValueError("OSWorld observation has no screenshot")
    encoded = base64.b64encode(cast(bytes, screenshot)).decode("ascii")
    return f"data:image/png;base64,{encoded}"
