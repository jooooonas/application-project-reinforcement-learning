import json

import pytest
from verifiers.v1 import ToolCall

from rl.computer_use import (
    TARGET_BOX_COMPUTER_USE_TOOL,
    ParsedComputerUseAction,
    computer_use_arguments_to_action,
    parse_computer_use_tool_calls,
)


def _action(arguments: dict) -> ParsedComputerUseAction:
    parsed = computer_use_arguments_to_action(arguments)
    assert parsed is not None
    return parsed


def _parse_tool_call(arguments: dict) -> list[ParsedComputerUseAction]:
    return parse_computer_use_tool_calls(
        [
            ToolCall(
                id="test-call",
                name="computer_use",
                arguments=json.dumps(arguments),
            )
        ]
    )


@pytest.mark.parametrize(
    ("arguments", "expected_code"),
    [
        ({"action": "mouse_move", "delta": [25, -10]}, "pyautogui.moveRel(25, -10)"),
        (
            {"action": "left_click_drag", "delta": [-120, 40]},
            "pyautogui.dragRel(-120, 40, duration=0.5, button='left')",
        ),
        ({"action": "mouse_move", "delta": [1.4, -1.6]}, "pyautogui.moveRel(1, -2)"),
    ],
)
def test_relative_mouse_actions_use_delta(arguments, expected_code):
    assert _action(arguments).pyautogui_code == expected_code


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "mouse_move"},
        {"action": "mouse_move", "delta": [10]},
        {"action": "mouse_move", "delta": ["bad", 1]},
        {"action": "mouse_move", "coordinate": [10, 20]},
        {"action": "left_click_drag", "coordinate": [10, 20]},
        {"action": "left_click", "coordinate": [10, 20]},
        {"action": "left_click", "delta": [10, 20]},
    ],
)
def test_mouse_coordinate_and_bad_delta_actions_are_rejected(arguments):
    assert _parse_tool_call(arguments) == []


@pytest.mark.parametrize(
    ("arguments", "expected_code"),
    [
        ({"action": "left_click"}, "pyautogui.click()"),
        ({"action": "right_click"}, "pyautogui.click(button='right')"),
        ({"action": "middle_click"}, "pyautogui.click(button='middle')"),
        ({"action": "double_click"}, "pyautogui.doubleClick()"),
        ({"action": "triple_click"}, "pyautogui.tripleClick()"),
    ],
)
def test_click_actions_use_current_cursor_position(arguments, expected_code):
    assert _action(arguments).pyautogui_code == expected_code


@pytest.mark.parametrize(
    ("arguments", "expected_code"),
    [
        ({"action": "type", "text": "hello"}, 'pyautogui.write("hello")'),
        ({"action": "key", "keys": ["ctrl", "c"]}, 'pyautogui.hotkey("ctrl", "c")'),
        ({"action": "scroll", "pixels": -5}, "pyautogui.scroll(-5)"),
        ({"action": "hscroll", "pixels": 3}, "pyautogui.hscroll(3)"),
        ({"action": "wait", "time": 0.25}, "time.sleep(0.250)"),
    ],
)
def test_non_mouse_actions_keep_existing_output(arguments, expected_code):
    assert _action(arguments).pyautogui_code == expected_code


def test_terminate_action_keeps_status():
    action = _action({"action": "terminate", "status": "success"})
    assert action.terminate
    assert action.terminate_status == "success"
    assert action.pyautogui_code is None


def test_mouse_move_keeps_parsed_delta_for_anchored_execution():
    action = _action({"action": "mouse_move", "delta": [7, -3]})

    assert action.pyautogui_code == "pyautogui.moveRel(7, -3)"
    assert action.delta == (7, -3)


def test_target_box_tool_exposes_only_supported_actions():
    assert TARGET_BOX_COMPUTER_USE_TOOL.name == "computer_use"
    assert TARGET_BOX_COMPUTER_USE_TOOL.parameters["properties"]["action"]["enum"] == [
        "mouse_move",
        "terminate",
    ]
