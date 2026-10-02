from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from rl.computer_use import ParsedComputerUseAction

from .geometry import TargetBox, point_in_box
from .observations import (
    cursor_position_from_obs,
    target_box_info,
    with_target_box_context,
)


class TargetBoxDesktopEnv(Protocol):
    def cursor_position(self) -> tuple[int, int]: ...

    def step(
        self,
        action: str,
        pause: float,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]: ...


@dataclass(frozen=True)
class ActionExecution:
    obs: dict[str, Any]
    executed_action: str | None
    reward: float
    rollout_terminated: bool
    info: dict[str, Any]
    stop_reason: str | None


async def execute_target_box_action(
    env: TargetBoxDesktopEnv,
    action: ParsedComputerUseAction,
    obs: dict[str, Any],
    target_box: TargetBox,
) -> ActionExecution:
    if action.action not in {"mouse_move", "terminate"}:
        return ActionExecution(
            obs=obs,
            executed_action=None,
            reward=0.0,
            rollout_terminated=True,
            info={"invalid_action": action.action},
            stop_reason="target_box_invalid_action",
        )

    if action.terminate:
        cursor_position = cursor_position_from_obs(obs)
        if action.terminate_status == "failure":
            info = (
                target_box_info(cursor_position, target_box)
                if cursor_position is not None
                else {
                    "cursor_position_missing": True,
                    "target_box": target_box.as_dict(),
                }
            )
            return ActionExecution(
                obs=obs,
                executed_action=None,
                reward=0.0,
                rollout_terminated=True,
                info=info,
                stop_reason="target_box_agent_failure",
            )
        if cursor_position is None:
            return ActionExecution(
                obs=obs,
                executed_action=None,
                reward=0.0,
                rollout_terminated=True,
                info={
                    "cursor_position_missing": True,
                    "target_box": target_box.as_dict(),
                },
                stop_reason="target_box_missing_cursor",
            )
        inside = point_in_box(cursor_position, target_box)
        success = inside and action.terminate_status == "success"
        stop_reason = "target_box_success" if success else "target_box_bad_terminate"
        return ActionExecution(
            obs=obs,
            executed_action=None,
            reward=1.0 if success else 0.0,
            rollout_terminated=True,
            info=target_box_info(cursor_position, target_box),
            stop_reason=stop_reason,
        )

    if action.pyautogui_code is None:
        return ActionExecution(
            obs=obs,
            executed_action=None,
            reward=0.0,
            rollout_terminated=True,
            info={"invalid_action": action.action},
            stop_reason="target_box_invalid_action",
        )

    executed_action = action.pyautogui_code
    pre_action_info: dict[str, Any] = {}
    if action.action == "mouse_move" and action.delta is not None:
        observed_cursor = cursor_position_from_obs(obs)
        if observed_cursor is not None:
            pre_action_cursor = await asyncio.to_thread(env.cursor_position)
            dx, dy = action.delta
            destination = (observed_cursor[0] + dx, observed_cursor[1] + dy)
            executed_action = f"pyautogui.moveTo({destination[0]}, {destination[1]})"
            pre_action_info = {
                "observed_cursor_position": list(observed_cursor),
                "pre_action_cursor_position": list(pre_action_cursor),
                "pre_action_cursor_drift": [
                    pre_action_cursor[0] - observed_cursor[0],
                    pre_action_cursor[1] - observed_cursor[1],
                ],
                "anchored_mouse_move_destination": list(destination),
            }

    next_obs, _reward, environment_terminated, info = await asyncio.to_thread(
        env.step,
        executed_action,
        1.0,
    )
    cursor_position = await asyncio.to_thread(env.cursor_position)
    target_info = target_box_info(cursor_position, target_box)
    target_info.update(pre_action_info)
    target_info.update(info or {})
    next_obs = with_target_box_context(next_obs, target_box, cursor_position)
    return ActionExecution(
        obs=next_obs,
        executed_action=executed_action,
        reward=0.0,
        rollout_terminated=environment_terminated,
        info=target_info,
        stop_reason="env_done" if environment_terminated else None,
    )
