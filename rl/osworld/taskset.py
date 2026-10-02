from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

import verifiers.v1 as vf

from rl.osworld.task_loading import load_json, resolve_task_paths

__all__ = [
    "OSWorldState",
    "OSWorldTask",
    "OSWorldTaskData",
    "OSWorldTaskset",
    "OSWorldTasksetConfig",
]


class OSWorldTaskData(vf.TaskData):
    task_id: str
    instruction: str
    path: str


class OSWorldState(vf.State):
    success: bool = False
    outcome: str | None = None
    steps: int = 0


class _RewardTrace(Protocol):
    state: OSWorldState
    info: dict[str, Any]


class OSWorldTask(vf.Task[OSWorldTaskData, OSWorldState]):
    @vf.reward
    async def target_box(self, trace: _RewardTrace) -> float:
        state = trace.state
        result = trace.info.get("osworld_target_box")
        if not isinstance(result, Mapping):
            raise RuntimeError("OSWorld rollout did not publish a target-box result")
        if result.get("outcome") != state.outcome:
            raise RuntimeError("OSWorld state and trace info outcomes disagree")
        return float(state.success and state.outcome == "target_box_success")


class OSWorldTasksetConfig(vf.TasksetConfig):
    base_path: str
    max_tasks: int = 0
    shuffle_seed: int = -1


class OSWorldTaskset(vf.Taskset[OSWorldTask, OSWorldTasksetConfig]):
    def load(self) -> Iterable[OSWorldTask]:
        for idx, task_path in enumerate(resolve_task_paths(self.config)):
            task = load_json(task_path)
            task_id = task.get("id")
            instruction = task.get("instruction")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"OSWorld task has no string id: {task_path}")
            if not isinstance(instruction, str) or not instruction:
                raise ValueError(f"OSWorld task has no string instruction: {task_path}")
            yield OSWorldTask(
                OSWorldTaskData(
                    idx=idx,
                    name=task_id,
                    prompt=instruction,
                    task_id=task_id,
                    instruction=instruction,
                    path=str(task_path),
                ),
                self.config.task,
            )
