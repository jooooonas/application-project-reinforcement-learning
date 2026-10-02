import asyncio
import io
from types import SimpleNamespace

import pytest
from PIL import Image

import rl.osworld.desktop.readiness as desktop_env_module
from rl.computer_use import ParsedComputerUseAction
from rl.osworld.tasks.target_box.actions import execute_target_box_action
from rl.osworld.tasks.target_box.geometry import (
    TargetBox,
    TargetBoxConfig,
    annotate_screenshot_bytes,
    point_in_box,
    sample_cursor_start,
    sample_target_box,
)
from rl.osworld.tasks.target_box.prompting import (
    TARGET_BOX_INSTRUCTION,
    keep_latest_image_url,
    target_box_initial_user_messages,
    target_box_observation_messages,
)


def _png_bytes(color=(255, 255, 255), size=(120, 80)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def _screen_config() -> SimpleNamespace:
    return SimpleNamespace(screen_width=120, screen_height=80)


def _content_parts(message):
    content = getattr(message, "content", None)
    return content if isinstance(content, list) else []


def _part_data(part):
    if isinstance(part, dict):
        return part
    if hasattr(part, "model_dump"):
        return part.model_dump(exclude_none=True)
    return {}


def _text_content(message) -> str:
    return "\n".join(
        data["text"]
        for data in (_part_data(part) for part in _content_parts(message))
        if data.get("type") == "text"
    )


def _image_url_count(messages) -> int:
    return sum(
        1
        for message in messages
        for part in _content_parts(message)
        if _part_data(part).get("type") == "image_url"
    )


def test_target_box_sampling_is_deterministic_and_in_bounds():
    config = TargetBoxConfig(box_width=20, box_height=10, margin=5, seed=7)
    box_a = sample_target_box(
        config,
        screen_width=120,
        screen_height=80,
        instance_key="task",
    )
    box_b = sample_target_box(
        config,
        screen_width=120,
        screen_height=80,
        instance_key="task",
    )

    assert box_a == box_b
    assert 5 <= box_a.x1 <= box_a.x2 < 115
    assert 5 <= box_a.y1 <= box_a.y2 < 75
    assert box_a.x2 - box_a.x1 + 1 == 20
    assert box_a.y2 - box_a.y1 + 1 == 10


def test_target_cursor_start_is_outside_box():
    config = TargetBoxConfig(box_width=20, box_height=10, margin=5, seed=7)
    box = TargetBox(10, 10, 29, 19)

    cursor = sample_cursor_start(
        config,
        box,
        screen_width=120,
        screen_height=80,
        instance_key="task",
    )

    assert not point_in_box(cursor, box)


def test_target_cursor_start_raises_when_no_outside_point_exists():
    config = TargetBoxConfig(
        box_width=120,
        box_height=80,
        margin=0,
        cursor_margin=0,
    )

    with pytest.raises(ValueError, match="no valid cursor start"):
        sample_cursor_start(
            config,
            TargetBox(0, 0, 119, 79),
            screen_width=120,
            screen_height=80,
            instance_key="task",
        )


def test_point_in_box_uses_inclusive_bounds():
    box = TargetBox(10, 20, 30, 40)

    assert point_in_box((10, 20), box)
    assert point_in_box((30, 40), box)
    assert not point_in_box((9, 20), box)
    assert not point_in_box((30, 41), box)


def test_annotation_keeps_size_and_draws_green_border():
    box = TargetBox(10, 10, 40, 30)
    annotated = annotate_screenshot_bytes(_png_bytes(), box)

    image = Image.open(io.BytesIO(annotated)).convert("RGB")
    assert image.size == (120, 80)
    assert image.getpixel((10, 10)) == (0, 255, 0)


class _MissingScreenshotEnv:
    def observe(self, *, request_timeout=None):
        return {}


def test_desktop_readiness_timeout_fails_without_screenshot(monkeypatch):
    monkeypatch.setattr(desktop_env_module, "_DESKTOP_READY_INITIAL_DELAY_S", 0.0)
    monkeypatch.setattr(desktop_env_module, "_DESKTOP_READY_TIMEOUT_S", 0.0)

    with pytest.raises(TimeoutError, match="without screenshot bytes"):
        asyncio.run(desktop_env_module.wait_for_desktop_ready(_MissingScreenshotEnv()))


def test_target_box_prompt_includes_estimation_instructions_and_examples():
    obs = {
        "screenshot": _png_bytes(),
        "target_box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
        "target_box_center": [20, 30],
        "cursor_position": [5, 10],
    }

    messages = target_box_initial_user_messages(obs, _screen_config())

    system_text = getattr(messages[0], "content", "")
    assert "Use the newest screenshot as the current desktop state" in system_text
    assert "cursor_visual" in system_text
    assert "box_visual" in system_text
    assert "move_visual" in system_text
    assert "must not contain digits, brackets" in system_text
    assert "using words only" in system_text
    assert "cursor_position_estimate" in system_text
    assert "green_box_center_estimate" in system_text
    assert "delta_estimate" in system_text
    assert "If the visual move sentence disagrees with the delta signs" in system_text
    assert "decision" not in system_text
    assert "Screen resolution" not in system_text

    text = _text_content(messages[-1])
    assert text == "\n".join(
        [
            f"Instruction: {TARGET_BOX_INSTRUCTION}",
            "Screen resolution: 120x80.",
            "Initial observation.",
        ]
    )

    move_example_response = getattr(messages[2], "content", "")
    assert "inside: cursor sits outside of the box" in move_example_response
    assert (
        "cursor_visual: the cursor tip is near the bottom right corner of the screen"
        in move_example_response
    )
    assert (
        "box_visual: the green box is also in the bottom right quadrant"
        in move_example_response
    )
    assert "move_visual: move the cursor up and left" in move_example_response
    assert "cursor_position_estimate: [117, 67]" in move_example_response
    assert "green_box_center_estimate: [96, 59]" in move_example_response
    assert "delta_estimate: [-21, -8]" in move_example_response

    move_example_tool_calls = getattr(messages[2], "tool_calls", [])
    assert len(move_example_tool_calls) == 1
    assert move_example_tool_calls[0].name == "computer_use"
    assert move_example_tool_calls[0].arguments == (
        '{"action": "mouse_move", "delta": [-21, -8]}'
    )

    terminate_example_response = getattr(messages[4], "content", "")
    assert "inside: cursor sits inside the box" in terminate_example_response
    assert (
        "cursor_visual: the cursor tip is inside the green box"
        in terminate_example_response
    )
    assert "delta_estimate: [0, 0]" in terminate_example_response

    terminate_example_tool_calls = getattr(messages[4], "tool_calls", [])
    assert len(terminate_example_tool_calls) == 1
    assert terminate_example_tool_calls[0].name == "computer_use"
    assert terminate_example_tool_calls[0].arguments == (
        '{"action": "terminate", "status": "success"}'
    )
    assert len(messages) == 6


def test_prompt_pruning_keeps_only_newest_screenshot():
    config = _screen_config()
    old_obs = {
        "screenshot": _png_bytes(),
        "target_box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
        "target_box_center": [20, 30],
        "cursor_position": [5, 10],
    }
    new_obs = {
        "screenshot": _png_bytes(color=(200, 200, 200)),
        "target_box": {"x1": 50, "y1": 25, "x2": 70, "y2": 45},
        "target_box_center": [60, 35],
        "cursor_position": [20, 30],
    }
    initial_messages = target_box_initial_user_messages(old_obs, config)
    old_observation_index = len(initial_messages) - 1
    messages = [
        *initial_messages,
        *target_box_observation_messages(new_obs, config),
    ]

    pruned = keep_latest_image_url(messages)

    assert _image_url_count(pruned) == 1
    assert "Previous screenshot omitted." in _text_content(
        pruned[old_observation_index]
    )
    assert _text_content(pruned[-1]) == "Newest observation."


class _FakeTargetEnv:
    def __init__(self, cursor_position=(0, 0), done=False):
        self._cursor_position = cursor_position
        self.done = done
        self.actions = []
        self.cursor_position_calls = 0

    def cursor_position(self):
        self.cursor_position_calls += 1
        return self._cursor_position

    def step(self, action, pause):
        self.actions.append((action, pause))
        return {"screenshot": _png_bytes()}, 0.0, self.done, {}


def test_target_terminate_uses_observed_cursor_without_requery():
    env = _FakeTargetEnv(cursor_position=(5, 5))
    action = ParsedComputerUseAction(
        action="terminate",
        terminate_status="success",
    )

    result = asyncio.run(
        execute_target_box_action(
            env,
            action,
            {"screenshot": _png_bytes(), "cursor_position": [15, 15]},
            target_box=TargetBox(10, 10, 20, 20),
        )
    )

    assert result.stop_reason == "target_box_success"
    assert result.reward == 1.0
    assert env.cursor_position_calls == 0


def test_target_terminate_success_outside_box_fails():
    env = _FakeTargetEnv(cursor_position=(15, 15))
    action = ParsedComputerUseAction(
        action="terminate",
        terminate_status="success",
    )

    result = asyncio.run(
        execute_target_box_action(
            env,
            action,
            {"screenshot": _png_bytes(), "cursor_position": [5, 5]},
            target_box=TargetBox(10, 10, 20, 20),
        )
    )

    assert result.stop_reason == "target_box_bad_terminate"
    assert result.reward == 0.0
    assert env.cursor_position_calls == 0


def test_target_terminate_missing_cursor_fails_without_requery():
    env = _FakeTargetEnv(cursor_position=(15, 15))
    action = ParsedComputerUseAction(
        action="terminate",
        terminate_status="success",
    )

    result = asyncio.run(
        execute_target_box_action(
            env,
            action,
            {"screenshot": _png_bytes()},
            target_box=TargetBox(10, 10, 20, 20),
        )
    )

    assert result.stop_reason == "target_box_missing_cursor"
    assert result.reward == 0.0
    assert result.info["cursor_position_missing"] is True
    assert env.cursor_position_calls == 0
    assert env.actions == []


def test_target_terminate_failure_does_not_need_cursor():
    env = _FakeTargetEnv(cursor_position=(15, 15))
    action = ParsedComputerUseAction(
        action="terminate",
        terminate_status="failure",
    )

    result = asyncio.run(
        execute_target_box_action(
            env,
            action,
            {"screenshot": _png_bytes()},
            target_box=TargetBox(10, 10, 20, 20),
        )
    )

    assert result.stop_reason == "target_box_agent_failure"
    assert result.reward == 0.0
    assert result.info["cursor_position_missing"] is True
    assert env.cursor_position_calls == 0
    assert env.actions == []


def test_target_move_inside_box_does_not_auto_complete():
    env = _FakeTargetEnv(cursor_position=(15, 15))
    action = ParsedComputerUseAction(
        action="mouse_move",
        pyautogui_code="pyautogui.moveRel(5, 5)",
        delta=(5, 5),
    )

    result = asyncio.run(
        execute_target_box_action(
            env,
            action,
            {"screenshot": _png_bytes(), "cursor_position": [10, 10]},
            target_box=TargetBox(10, 10, 20, 20),
        )
    )

    assert not result.rollout_terminated
    assert result.stop_reason is None
    assert result.info["target_box_inside"] is True
    assert result.info["observed_cursor_position"] == [10, 10]
    assert result.info["pre_action_cursor_position"] == [15, 15]
    assert result.info["pre_action_cursor_drift"] == [5, 5]
    assert result.info["anchored_mouse_move_destination"] == [15, 15]
    assert env.actions == [("pyautogui.moveTo(15, 15)", 1.0)]


def test_target_move_falls_back_to_relative_without_observed_cursor():
    env = _FakeTargetEnv(cursor_position=(15, 15))
    action = ParsedComputerUseAction(
        action="mouse_move",
        pyautogui_code="pyautogui.moveRel(5, 5)",
        delta=(5, 5),
    )

    result = asyncio.run(
        execute_target_box_action(
            env,
            action,
            {"screenshot": _png_bytes()},
            target_box=TargetBox(10, 10, 20, 20),
        )
    )

    assert not result.rollout_terminated
    assert result.info["target_box_inside"] is True
    assert "pre_action_cursor_position" not in result.info
    assert env.actions == [("pyautogui.moveRel(5, 5)", 1.0)]


def test_target_invalid_action_fails_without_execution():
    env = _FakeTargetEnv(cursor_position=(15, 15))
    action = ParsedComputerUseAction(
        action="left_click",
        pyautogui_code="pyautogui.click()",
    )

    result = asyncio.run(
        execute_target_box_action(
            env,
            action,
            {"screenshot": _png_bytes()},
            target_box=TargetBox(10, 10, 20, 20),
        )
    )

    assert result.stop_reason == "target_box_invalid_action"
    assert result.reward == 0.0
    assert env.actions == []
