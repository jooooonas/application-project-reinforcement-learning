from __future__ import annotations

from verifiers.v1 import Tool

COMPUTER_USE_TOOL = Tool(
    name="computer_use",
    description=(
        "Control the Ubuntu desktop with exactly one action. Mouse movement "
        "uses relative deltas from the current cursor position."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "key",
                    "type",
                    "mouse_move",
                    "left_click",
                    "left_click_drag",
                    "right_click",
                    "middle_click",
                    "double_click",
                    "triple_click",
                    "scroll",
                    "hscroll",
                    "wait",
                    "terminate",
                ],
            },
            "delta": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Relative [dx, dy] for mouse_move or left_click_drag.",
            },
            "keys": {
                "description": "Key or key chord for action=key.",
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
            },
            "text": {"type": "string", "description": "Text for action=type."},
            "pixels": {
                "type": "number",
                "description": "Scroll amount for action=scroll or action=hscroll.",
            },
            "time": {"type": "number", "description": "Seconds for action=wait."},
            "status": {
                "type": "string",
                "enum": ["success", "failure"],
                "description": "Completion status for action=terminate.",
            },
        },
        "required": ["action"],
    },
)

COMPUTER_USE_TOOL_DEFS = [COMPUTER_USE_TOOL]

TARGET_BOX_COMPUTER_USE_TOOL = Tool(
    name="computer_use",
    description=(
        "Move the cursor by a relative delta, or terminate the target-box task."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["mouse_move", "terminate"],
            },
            "delta": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Relative [dx, dy] from the current cursor position.",
            },
            "status": {
                "type": "string",
                "enum": ["success", "failure"],
                "description": "Completion status for action=terminate.",
            },
        },
        "required": ["action"],
    },
)

TARGET_BOX_COMPUTER_USE_TOOL_DEFS = [TARGET_BOX_COMPUTER_USE_TOOL]
