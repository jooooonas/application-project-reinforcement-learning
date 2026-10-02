from __future__ import annotations

import asyncio
import io
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import verifiers.v1 as vf
from openai import APIStatusError
from PIL import Image

import rl
import rl.osworld.harness as harness_module
from rl.osworld.config import OSWorldDesktopRuntimeConfig
from rl.osworld.harness import OSWorldHarness, OSWorldHarnessConfig
from rl.osworld.tasks.target_box.geometry import TargetBox, TargetBoxConfig
from rl.osworld.tasks.target_box.prompting import target_box_initial_user_messages
from rl.osworld.taskset import (
    OSWorldState,
    OSWorldTask,
    OSWorldTaskData,
    OSWorldTaskset,
    OSWorldTasksetConfig,
)


def _task_file(path: Path, task_id: str, instruction: str) -> Path:
    path.write_text(
        json.dumps({"id": task_id, "instruction": instruction}),
        encoding="utf-8",
    )
    return path


def _task_data(path: Path) -> OSWorldTaskData:
    return OSWorldTaskData(
        idx=0,
        name="task-1",
        prompt="Move the cursor.",
        task_id="task-1",
        instruction="Move the cursor.",
        path=str(path),
    )


def _trace(data: OSWorldTaskData) -> vf.Trace:
    return vf.Trace(
        task=vf.TraceTask(type="OSWorldTask", data=data),
        state=OSWorldState(),
    )


def _screenshot(width: int = 120, height: int = 80) -> bytes:
    image = Image.new("RGB", (width, height), "black")
    for x in range(width // 2, width):
        for y in range(height):
            image.putpixel((x, y), (255, 255, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _desktop_config(tmp_path: Path) -> OSWorldDesktopRuntimeConfig:
    return OSWorldDesktopRuntimeConfig(
        screen_width=120,
        screen_height=80,
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
        qcow_path=tmp_path / "Ubuntu.qcow2",
    )


def _harness_config(
    tmp_path: Path,
    *,
    max_steps: int = 3,
) -> OSWorldHarnessConfig:
    return OSWorldHarnessConfig(
        id="rl",
        max_steps=max_steps,
        desktop=_desktop_config(tmp_path),
        target_box=TargetBoxConfig(
            box_width=20,
            box_height=20,
            margin=5,
            cursor_margin=2,
            seed=7,
        ),
    )


def test_rl_plugin_loader_resolves_canonical_generic_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert rl.__all__ == ["OSWorldHarness", "OSWorldTaskset"]
    assert vf.taskset_config_type("rl") is OSWorldTasksetConfig
    assert vf.harness_config_type("rl") is OSWorldHarnessConfig
    assert vf.task_type("rl") is OSWorldTask
    task_path = _task_file(tmp_path / "task.json", "task-1", "Move.")
    taskset = vf.load_taskset(OSWorldTasksetConfig(id="rl", base_path=str(task_path)))
    pool = _FakePool()
    monkeypatch.setattr(
        harness_module,
        "_build_desktop_pool",
        lambda _config: pool,
    )
    harness = vf.load_harness(OSWorldHarnessConfig(id="rl"))
    assert isinstance(taskset, OSWorldTaskset)
    assert isinstance(harness, OSWorldHarness)
    assert pool.started == 1
    harness.close()
    assert pool.closed == 1


def test_taskset_loads_deterministically_and_serializes_typed_data(
    tmp_path: Path,
) -> None:
    _task_file(tmp_path / "b.json", "task-b", "Instruction B")
    _task_file(tmp_path / "a.json", "task-a", "Instruction A")
    config = OSWorldTasksetConfig(id="rl", base_path=str(tmp_path))

    tasks = OSWorldTaskset(config).select()

    assert [task.data.task_id for task in tasks] == ["task-a", "task-b"]
    assert tasks[0].data.idx == 0
    assert tasks[0].data.name == "task-a"
    assert tasks[0].data.prompt == "Instruction A"
    assert tasks[0].data.image is None
    assert tasks[0].data.model_dump(mode="json")["path"] == str(tmp_path / "a.json")


@pytest.mark.asyncio
async def test_task_reward_reads_typed_state_and_trace_info(tmp_path: Path) -> None:
    data = _task_data(_task_file(tmp_path / "task.json", "task-1", "Move."))
    task = OSWorldTask(data)
    trace = _trace(data)
    OSWorldHarness._finish(
        trace,
        "target_box_success",
        2,
        success=True,
    )

    await task.score(trace)

    assert trace.rewards == {"target_box": 1.0}


def test_multimodal_messages_and_tool_schema_use_openai_wire_format(
    tmp_path: Path,
) -> None:
    desktop = _desktop_config(tmp_path)
    obs = harness_module._annotated_observation(
        {"screenshot": _screenshot(), "cursor_position": [5, 5]},
        TargetBox(30, 20, 49, 39),
        desktop,
    )
    messages = target_box_initial_user_messages(obs, desktop)

    wire = harness_module._messages_to_openai(messages)
    image = wire[-1]["content"][-1]
    tool = harness_module._openai_tool_schema()[0]

    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "computer_use"
    assert tool["function"]["parameters"]["properties"]["action"]["enum"] == [
        "mouse_move",
        "terminate",
    ]

    sdk_message = _model_message("mouse_move", {"delta": [4, -2]})
    call = harness_module._native_function_calls_to_flat(sdk_message.tool_calls)[0]
    action = harness_module._parse_action(call)
    assert call.name == "computer_use"
    assert action is not None and action.delta == (4, -2)

    sdk_message.reasoning_content = "private reasoning"
    sdk_message.reasoning_details = [{"type": "reasoning", "data": "opaque"}]
    assistant = harness_module._assistant_message(sdk_message)
    assert assistant.reasoning_content == "private reasoning"
    assert assistant.provider_state == [{"type": "reasoning", "data": "opaque"}]
    assistant_wire = harness_module._messages_to_openai([assistant])[0]
    assert assistant_wire["reasoning_details"] == [
        {"type": "reasoning", "data": "opaque"}
    ]


class _FakeDesktop:
    def __init__(self) -> None:
        self.cursor = (0, 0)
        self.png = _screenshot()
        self.steps = 0

    def reset(self, *, task_config: dict[str, Any]) -> dict[str, Any]:
        assert task_config["id"] == "task-1"
        return {"screenshot": self.png}

    def observe(self, *, request_timeout: float | None = None) -> dict[str, Any]:
        del request_timeout
        return {"screenshot": self.png}

    def move_cursor_to(self, x: int, y: int) -> None:
        self.cursor = (x, y)

    def cursor_position(self) -> tuple[int, int]:
        return self.cursor

    def step(
        self,
        action: str,
        pause: float,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        del pause
        self.steps += 1
        match = re.fullmatch(r"pyautogui\.moveTo\((-?\d+), (-?\d+)\)", action)
        assert match is not None
        self.cursor = (int(match.group(1)), int(match.group(2)))
        return {"screenshot": self.png}, 0.0, False, {}


class _FakeCheckout:
    def __init__(self, env: _FakeDesktop) -> None:
        self.env = env
        self.releases: list[tuple[bool, str | None]] = []
        self.released = threading.Event()

    def tracked_env(self) -> _FakeDesktop:
        return self.env

    def release(self, *, failed: bool, error: str | None) -> None:
        self.releases.append((failed, error))
        self.released.set()


class _FakePool:
    def __init__(self) -> None:
        self.checkout_session = _FakeCheckout(_FakeDesktop())
        self.started = 0
        self.closed = 0
        self.checkouts = 0

    def start(self) -> None:
        self.started += 1

    def checkout(self) -> _FakeCheckout:
        self.checkouts += 1
        return self.checkout_session

    def close(self) -> None:
        self.closed += 1


class _DelayedCheckoutPool(_FakePool):
    def __init__(self) -> None:
        super().__init__()
        self.checkout_started = threading.Event()
        self.allow_checkout = threading.Event()

    def checkout(self) -> _FakeCheckout:
        self.checkout_started.set()
        if not self.allow_checkout.wait(timeout=5.0):
            raise TimeoutError("test checkout was not released")
        return super().checkout()


def _model_message(action: str, arguments: dict[str, Any]) -> Any:
    payload = {"action": action, **arguments}
    return SimpleNamespace(
        content=f"taking action {action}",
        tool_calls=[
            SimpleNamespace(
                id=f"call-{action}",
                type="function",
                function=SimpleNamespace(
                    name="computer_use",
                    arguments=json.dumps(payload),
                ),
            )
        ],
    )


class _FakeCompletions:
    def __init__(
        self, messages: list[Any] | None = None, error: Exception | None = None
    ):
        self.messages = list(messages or [])
        self.error = error
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        message = self.messages.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAI:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


def _ctx() -> Any:
    return SimpleNamespace(model="test-model", sampling=vf.Sampling())


@pytest.mark.asyncio
async def test_launch_uses_interception_and_releases_desktop_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_path = _task_file(tmp_path / "task.json", "task-1", "Move.")
    data = _task_data(task_path)
    trace = _trace(data)
    second_trace = _trace(data)
    pool = _FakePool()
    monkeypatch.setattr(
        harness_module,
        "_build_desktop_pool",
        lambda _config: pool,
    )
    harness = OSWorldHarness(_harness_config(tmp_path))
    target_box, cursor = harness._target_box(data)
    center = (
        (target_box.x1 + target_box.x2) // 2,
        (target_box.y1 + target_box.y2) // 2,
    )
    completions = _FakeCompletions(
        [
            _model_message(
                "mouse_move",
                {"delta": [center[0] - cursor[0], center[1] - cursor[1]]},
            ),
            _model_message("terminate", {"status": "success"}),
        ]
    )
    clients = [
        _FakeOpenAI(completions),
        _FakeOpenAI(
            _FakeCompletions(
                [
                    _model_message(
                        "mouse_move",
                        {
                            "delta": [
                                center[0] - cursor[0],
                                center[1] - cursor[1],
                            ]
                        },
                    ),
                    _model_message("terminate", {"status": "success"}),
                ]
            )
        ),
    ]
    client_args: list[tuple[str, str]] = []

    def client_factory(endpoint: str, secret: str) -> _FakeOpenAI:
        client_args.append((endpoint, secret))
        return clients[len(client_args) - 1]

    monkeypatch.setattr(harness_module, "_openai_client", client_factory)

    result = await harness.launch(
        _ctx(),
        trace,
        SimpleNamespace(),
        "http://interception.example/v1",
        "rollout-secret",
        {},
    )
    second_result = await harness.launch(
        _ctx(),
        second_trace,
        SimpleNamespace(),
        "http://interception.example/v1",
        "rollout-secret",
        {},
    )

    assert result == vf.ProgramResult(0, "", "")
    assert second_result == vf.ProgramResult(0, "", "")
    assert client_args == [
        ("http://interception.example/v1", "rollout-secret"),
        ("http://interception.example/v1", "rollout-secret"),
    ]
    assert all(request["model"] == "test-model" for request in completions.requests)
    assert trace.stop_condition == "target_box_success"
    assert second_trace.stop_condition == "target_box_success"
    assert trace.state.success is True
    assert trace.state.steps == 2
    assert pool.started == 1
    assert pool.checkouts == 2
    assert pool.checkout_session.releases == [(False, None), (False, None)]
    assert pool.closed == 0
    assert all(client.closed == 1 for client in clients)
    harness.close()
    harness.close()
    assert pool.closed == 1


@pytest.mark.asyncio
async def test_launch_releases_and_closes_desktop_on_model_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_path = _task_file(tmp_path / "task.json", "task-1", "Move.")
    trace = _trace(_task_data(task_path))
    pool = _FakePool()
    monkeypatch.setattr(
        harness_module,
        "_build_desktop_pool",
        lambda _config: pool,
    )
    harness = OSWorldHarness(_harness_config(tmp_path))
    client = _FakeOpenAI(_FakeCompletions(error=RuntimeError("model failed")))
    monkeypatch.setattr(harness_module, "_openai_client", lambda *_args: client)

    with pytest.raises(RuntimeError, match="model failed"):
        await harness.launch(
            _ctx(),
            trace,
            SimpleNamespace(),
            "http://interception.example/v1",
            "rollout-secret",
            {},
        )

    assert pool.checkout_session.releases[0][0] is True
    assert "model failed" in (pool.checkout_session.releases[0][1] or "")
    assert pool.closed == 0
    assert client.closed == 1
    assert trace.stop_condition is None
    harness.close()
    assert pool.closed == 1


@pytest.mark.asyncio
async def test_launch_releases_late_checkout_after_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_path = _task_file(tmp_path / "task.json", "task-1", "Move.")
    trace = _trace(_task_data(task_path))
    pool = _DelayedCheckoutPool()
    monkeypatch.setattr(
        harness_module,
        "_build_desktop_pool",
        lambda _config: pool,
    )
    harness = OSWorldHarness(_harness_config(tmp_path))
    launch_task = asyncio.create_task(
        harness.launch(
            _ctx(),
            trace,
            SimpleNamespace(),
            "http://interception.example/v1",
            "rollout-secret",
            {},
        )
    )

    assert await asyncio.to_thread(pool.checkout_started.wait, 1.0)
    launch_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await launch_task

    assert pool.checkout_session.releases == []
    pool.allow_checkout.set()
    assert await asyncio.to_thread(pool.checkout_session.released.wait, 1.0)
    assert pool.checkout_session.releases == [
        (True, "rollout cancelled during desktop checkout")
    ]
    assert harness._late_checkout_tasks == set()
    harness.close()


@pytest.mark.asyncio
async def test_framework_refusal_is_a_clean_zero_reward_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_path = _task_file(tmp_path / "task.json", "task-1", "Move.")
    data = _task_data(task_path)
    trace = _trace(data)
    trace.stop("max_input_tokens")
    response = httpx.Response(
        400,
        request=httpx.Request(
            "POST", "http://interception.example/v1/chat/completions"
        ),
    )
    refusal = APIStatusError("turn refused", response=response, body={})
    client = _FakeOpenAI(_FakeCompletions(error=refusal))
    pool = _FakePool()
    monkeypatch.setattr(
        harness_module,
        "_build_desktop_pool",
        lambda _config: pool,
    )
    monkeypatch.setattr(harness_module, "_openai_client", lambda *_args: client)
    harness = OSWorldHarness(_harness_config(tmp_path))

    result = await harness.launch(
        _ctx(),
        trace,
        SimpleNamespace(),
        "http://interception.example/v1",
        "rollout-secret",
        {},
    )
    await OSWorldTask(data).score(trace)

    assert result == vf.ProgramResult(0, "", "")
    assert trace.stop_condition == "max_input_tokens"
    assert trace.state.outcome == "max_input_tokens"
    assert trace.state.steps == 0
    assert trace.rewards == {"target_box": 0.0}
    assert pool.checkout_session.releases == [(False, None)]
    harness.close()


@pytest.mark.asyncio
async def test_max_steps_uses_canonical_truncation_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_path = _task_file(tmp_path / "task.json", "task-1", "Move.")
    data = _task_data(task_path)
    trace = _trace(data)
    pool = _FakePool()
    monkeypatch.setattr(
        harness_module,
        "_build_desktop_pool",
        lambda _config: pool,
    )
    harness = OSWorldHarness(_harness_config(tmp_path, max_steps=1))
    target_box, cursor = harness._target_box(data)
    center = (
        (target_box.x1 + target_box.x2) // 2,
        (target_box.y1 + target_box.y2) // 2,
    )
    client = _FakeOpenAI(
        _FakeCompletions(
            [
                _model_message(
                    "mouse_move",
                    {"delta": [center[0] - cursor[0], center[1] - cursor[1]]},
                )
            ]
        )
    )
    monkeypatch.setattr(harness_module, "_openai_client", lambda *_args: client)

    await harness.launch(
        _ctx(),
        trace,
        SimpleNamespace(),
        "http://interception.example/v1",
        "rollout-secret",
        {},
    )

    assert trace.state.outcome == "max_steps"
    assert trace.state.steps == 1
    assert trace.info["osworld_target_box"]["outcome"] == "max_steps"
    assert trace.stop_condition == "max_turns"
    assert trace.is_truncated is True
    harness.close()


@pytest.mark.asyncio
async def test_mixed_function_and_custom_calls_stop_without_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_path = _task_file(tmp_path / "task.json", "task-1", "Move.")
    data = _task_data(task_path)
    trace = _trace(data)
    pool = _FakePool()
    monkeypatch.setattr(
        harness_module,
        "_build_desktop_pool",
        lambda _config: pool,
    )
    harness = OSWorldHarness(_harness_config(tmp_path))
    message = _model_message("mouse_move", {"delta": [1, 1]})
    message.tool_calls.append(
        SimpleNamespace(
            id="call-custom",
            type="custom",
            custom=SimpleNamespace(name="custom", input="ignored"),
        )
    )
    client = _FakeOpenAI(_FakeCompletions([message]))
    monkeypatch.setattr(harness_module, "_openai_client", lambda *_args: client)

    await harness.launch(
        _ctx(),
        trace,
        SimpleNamespace(),
        "http://interception.example/v1",
        "rollout-secret",
        {},
    )

    assert len(harness_module._native_function_calls_to_flat(message.tool_calls)) == 1
    assert trace.state.outcome == "multiple_actions_parsed"
    assert pool.checkout_session.env.steps == 0
    harness.close()
