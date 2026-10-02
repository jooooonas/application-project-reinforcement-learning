from __future__ import annotations

import asyncio
import json
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import verifiers.v1 as vf
from openai import APIStatusError, AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionToolParam,
)
from pydantic import Field
from verifiers.v1.dialects.chat import message_to_wire

from rl.computer_use.actions import (
    ParsedComputerUseAction,
    computer_use_arguments_to_action,
)
from rl.computer_use.parsing import computer_use_tool_calls_from_text
from rl.computer_use.tools import TARGET_BOX_COMPUTER_USE_TOOL
from rl.osworld.config import OSWorldDesktopRuntimeConfig
from rl.osworld.desktop.factory import create_desktop_env_proxy
from rl.osworld.desktop.pool import CheckedOutDesktopSession, DesktopSessionPool
from rl.osworld.desktop.proxy import DesktopEnvProxy
from rl.osworld.desktop.readiness import wait_for_desktop_ready
from rl.osworld.task_loading import load_json
from rl.osworld.tasks.target_box.actions import execute_target_box_action
from rl.osworld.tasks.target_box.geometry import (
    TargetBox,
    TargetBoxConfig,
    annotate_observation,
    sample_cursor_start,
    sample_target_box,
    screenshot_size,
)
from rl.osworld.tasks.target_box.prompting import (
    keep_latest_image_url,
    target_box_initial_user_messages,
    target_box_observation_messages,
)
from rl.osworld.taskset import OSWorldState, OSWorldTaskData

__all__ = ["OSWorldHarness", "OSWorldHarnessConfig"]


class OSWorldHarnessConfig(vf.HarnessConfig):
    max_steps: int = Field(default=10, ge=1)
    desktop: OSWorldDesktopRuntimeConfig = Field(
        default_factory=OSWorldDesktopRuntimeConfig
    )
    target_box: TargetBoxConfig = Field(default_factory=TargetBoxConfig)


@dataclass(frozen=True)
class _FlatToolCall:
    id: str
    name: str
    arguments: str
    native: bool


def _openai_tool_schema() -> list[ChatCompletionToolParam]:
    tool = TARGET_BOX_COMPUTER_USE_TOOL
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
    ]


def _messages_to_openai(messages: vf.Messages) -> list[ChatCompletionMessageParam]:
    return [message_to_wire(message) for message in messages]


def _native_function_calls_to_flat(
    tool_calls: list[ChatCompletionMessageToolCallUnion],
) -> list[_FlatToolCall]:
    """Flatten valid function calls, skipping unsupported or malformed calls."""
    calls: list[_FlatToolCall] = []
    for tool_call in tool_calls:
        if tool_call.type != "function":
            continue
        function = tool_call.function
        calls.append(
            _FlatToolCall(
                id=tool_call.id,
                name=function.name,
                arguments=function.arguments,
                native=True,
            )
        )
    return calls


def _text_tool_calls_to_flat(content: str | None) -> list[_FlatToolCall]:
    if content is None:
        return []
    return [
        _FlatToolCall(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            native=False,
        )
        for call in computer_use_tool_calls_from_text(content)
    ]


def _parse_action(call: _FlatToolCall) -> ParsedComputerUseAction | None:
    if call.name != TARGET_BOX_COMPUTER_USE_TOOL.name:
        return None
    try:
        arguments = json.loads(call.arguments or "{}")
        if not isinstance(arguments, dict):
            return None
        return computer_use_arguments_to_action(arguments)
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
        return None


def _assistant_message(message: ChatCompletionMessage) -> vf.AssistantMessage:
    calls = _native_function_calls_to_flat(message.tool_calls or [])
    content = getattr(message, "content", None)
    payload = (
        message.model_dump(exclude_none=True)
        if hasattr(message, "model_dump")
        else vars(message)
    )
    reasoning_content = next(
        (
            value
            for field in ("reasoning", "reasoning_content")
            if isinstance(value := payload.get(field), str) and value
        ),
        None,
    )
    reasoning_details = payload.get("reasoning_details")
    provider_state = (
        [dict(item) for item in reasoning_details if isinstance(item, Mapping)]
        if isinstance(reasoning_details, list)
        else None
    )
    return vf.AssistantMessage(
        content=content if isinstance(content, str) else None,
        reasoning_content=reasoning_content,
        provider_state=provider_state or None,
        tool_calls=(
            [
                vf.ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                )
                for call in calls
            ]
            or None
        ),
    )


def _build_desktop_pool(
    config: OSWorldDesktopRuntimeConfig,
) -> DesktopSessionPool[DesktopEnvProxy]:
    pool_config = config.desktop_pool_config
    return DesktopSessionPool(
        config=pool_config,
        root_dir=pool_config.root_dir or config.output_dir / "pool",
        session_factory=lambda lease: create_desktop_env_proxy(config, lease),
    )


def _close_desktop_pool(pool: DesktopSessionPool[DesktopEnvProxy]) -> None:
    pool.close()


def _release_cancelled_checkout(
    checkout_task: asyncio.Task[CheckedOutDesktopSession[DesktopEnvProxy]],
) -> None:
    try:
        checked_out = checkout_task.result()
    except BaseException:
        return
    checked_out.release(
        failed=True,
        error="rollout cancelled during desktop checkout",
    )


def _openai_client(endpoint: str, secret: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=endpoint, api_key=secret)


async def _close_client(client: AsyncOpenAI) -> None:
    await client.close()


def _annotated_observation(
    obs: Mapping[str, Any],
    target_box: TargetBox,
    config: OSWorldDesktopRuntimeConfig,
) -> dict[str, Any]:
    actual_size = screenshot_size(obs)
    expected_size = (config.screen_width, config.screen_height)
    if actual_size != expected_size:
        raise ValueError(
            "OSWorld screenshot size "
            f"{actual_size[0]}x{actual_size[1]} does not match configured screen size "
            f"{expected_size[0]}x{expected_size[1]}"
        )
    return annotate_observation(obs, target_box)


class OSWorldHarness(vf.Harness[OSWorldHarnessConfig]):
    SUPPORTS_MESSAGE_PROMPT = True

    def __init__(self, config: OSWorldHarnessConfig) -> None:
        super().__init__(config)
        self._desktop_pool = _build_desktop_pool(config.desktop)
        self._late_checkout_tasks: set[
            asyncio.Task[CheckedOutDesktopSession[DesktopEnvProxy]]
        ] = set()
        self._pool_finalizer = weakref.finalize(
            self,
            _close_desktop_pool,
            self._desktop_pool,
        )
        # Env-server workers must publish pool status and begin prewarming before their first rollout.
        self._desktop_pool.start()

    def close(self) -> None:
        """Close the process-owned desktop pool exactly once."""
        self._pool_finalizer()

    def _release_cancelled_checkout(
        self,
        checkout_task: asyncio.Task[CheckedOutDesktopSession[DesktopEnvProxy]],
    ) -> None:
        self._late_checkout_tasks.discard(checkout_task)
        _release_cancelled_checkout(checkout_task)

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        del runtime, mcp_urls
        task = trace.task.data
        if not isinstance(task, OSWorldTaskData):
            raise TypeError("OSWorld harness requires OSWorldTaskData")
        if not isinstance(trace.state, OSWorldState):
            raise TypeError("OSWorld harness requires OSWorldState")

        checked_out: CheckedOutDesktopSession[DesktopEnvProxy] | None = None
        failed = True
        error: str | None = None
        try:
            checkout_task = asyncio.create_task(
                asyncio.to_thread(self._desktop_pool.checkout)
            )
            try:
                checked_out = await asyncio.shield(checkout_task)
            except asyncio.CancelledError:
                self._late_checkout_tasks.add(checkout_task)
                checkout_task.add_done_callback(self._release_cancelled_checkout)
                raise
            assert checked_out is not None
            env = checked_out.tracked_env()
            await self._run_target_box(ctx, trace, env, endpoint, secret)
            failed = False
            return vf.ProgramResult(0, "", "")
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            if checked_out is not None:
                await asyncio.to_thread(
                    checked_out.release,
                    failed=failed,
                    error=error,
                )

    async def _run_target_box(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        env: DesktopEnvProxy,
        endpoint: str,
        secret: str,
    ) -> None:
        task = trace.task.data
        if not isinstance(task, OSWorldTaskData):
            raise TypeError("OSWorld harness requires OSWorldTaskData")
        target_box, cursor_start = self._target_box(task)
        task_config = load_json(task.path)

        obs = await asyncio.to_thread(env.reset, task_config=task_config)
        obs = await wait_for_desktop_ready(env, initial_obs=obs)
        await asyncio.to_thread(env.move_cursor_to, *cursor_start)
        obs = await asyncio.to_thread(env.observe)
        obs["cursor_position"] = list(await asyncio.to_thread(env.cursor_position))
        prompt_obs = _annotated_observation(obs, target_box, self.config.desktop)
        messages = keep_latest_image_url(
            target_box_initial_user_messages(prompt_obs, self.config.desktop)
        )
        client = _openai_client(endpoint, secret)
        try:
            await self._run_model_loop(
                client,
                ctx,
                trace,
                env,
                target_box,
                obs,
                messages,
            )
        finally:
            await _close_client(client)

    async def _run_model_loop(
        self,
        client: AsyncOpenAI,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        env: DesktopEnvProxy,
        target_box: TargetBox,
        obs: dict[str, Any],
        messages: vf.Messages,
    ) -> None:
        for step_idx in range(1, self.config.max_steps + 1):
            try:
                completion = await client.chat.completions.create(
                    model=ctx.model,
                    messages=_messages_to_openai(messages),
                    tools=_openai_tool_schema(),
                )
            except APIStatusError:
                stop_condition = trace.stop_condition
                if stop_condition is None:
                    raise
                self._finish(
                    trace,
                    stop_condition,
                    step_idx - 1,
                    stop_condition=stop_condition,
                )
                return

            message = completion.choices[0].message
            messages.append(_assistant_message(message))
            native_calls = message.tool_calls or []
            if native_calls:
                calls = _native_function_calls_to_flat(native_calls)
                call_count = len(native_calls)
            else:
                calls = _text_tool_calls_to_flat(message.content)
                call_count = len(calls)

            if call_count == 0:
                self._finish(trace, "no_actions_parsed", step_idx)
                return
            if call_count != 1:
                self._finish(trace, "multiple_actions_parsed", step_idx)
                return
            if not calls:
                self._finish(trace, "no_actions_parsed", step_idx)
                return

            call = calls[0]
            action = _parse_action(call)
            if action is None:
                self._finish(trace, "no_actions_parsed", step_idx)
                return
            execution = await execute_target_box_action(
                env,
                action,
                obs,
                target_box,
            )
            obs = execution.obs
            if execution.rollout_terminated:
                reason = execution.stop_reason
                if reason is None:
                    raise RuntimeError(
                        "terminated target-box action has no stop reason"
                    )
                self._finish(
                    trace,
                    reason,
                    step_idx,
                    success=execution.reward == 1.0,
                    action_info=execution.info,
                )
                return

            if call.native:
                messages.append(
                    vf.ToolMessage(
                        tool_call_id=call.id,
                        name=call.name,
                        content="Action executed.",
                    )
                )
            prompt_obs = _annotated_observation(
                obs,
                target_box,
                self.config.desktop,
            )
            messages.extend(
                target_box_observation_messages(prompt_obs, self.config.desktop)
            )
            messages = keep_latest_image_url(messages)

        self._finish(
            trace,
            "max_steps",
            self.config.max_steps,
            stop_condition="max_turns",
        )

    def _target_box(
        self,
        task: OSWorldTaskData,
    ) -> tuple[TargetBox, tuple[int, int]]:
        instance_key = f"{task.task_id}:{task.path}"
        desktop = self.config.desktop
        box = sample_target_box(
            self.config.target_box,
            screen_width=desktop.screen_width,
            screen_height=desktop.screen_height,
            instance_key=instance_key,
        )
        cursor = sample_cursor_start(
            self.config.target_box,
            box,
            screen_width=desktop.screen_width,
            screen_height=desktop.screen_height,
            instance_key=instance_key,
        )
        return box, cursor

    @staticmethod
    def _finish(
        trace: vf.Trace,
        outcome: str,
        steps: int,
        *,
        success: bool = False,
        action_info: Mapping[str, Any] | None = None,
        stop_condition: str | None = None,
    ) -> None:
        state = trace.state
        if not isinstance(state, OSWorldState):
            raise TypeError("OSWorld harness requires OSWorldState")
        state.success = success
        state.outcome = outcome
        state.steps = steps
        result: dict[str, Any] = {
            "success": success,
            "outcome": outcome,
            "steps": steps,
        }
        if action_info is not None:
            result["action_info"] = dict(action_info)
        trace.info["osworld_target_box"] = result
        trace.stop(stop_condition or outcome)
