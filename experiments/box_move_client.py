from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from client_common import (
    API_KEY,
    BASE_URL,
    MODEL,
    format_prompt,
    format_tools,
    image_size,
    openai_messages,
    openai_tools,
    print_model_response,
)
from openai import APIConnectionError, BadRequestError, OpenAI
from verifiers.v1 import AssistantMessage, Messages, ToolCall

from rl.osworld.tasks.target_box.prompting import (
    target_box_initial_user_messages,
    target_box_observation_messages,
)


@dataclass(frozen=True)
class ScreenConfig:
    screen_width: int
    screen_height: int


@dataclass(frozen=True)
class EpisodeArtifacts:
    episode_dir: Path | None
    trajectory_steps: dict[int, dict[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an OSWorld step screenshot.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", MODEL))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", BASE_URL),
        help="OpenAI-compatible endpoint. Defaults to OPENAI_BASE_URL or localhost.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompt only.")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    messages = build_training_messages(image_path)
    prompt_payload = openai_messages(messages)
    tool_payload = openai_tools()

    print(format_prompt(prompt_payload))
    print(format_tools(tool_payload))

    if args.dry_run:
        return

    client = OpenAI(base_url=args.base_url, api_key=API_KEY)
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=prompt_payload,
            tools=tool_payload,
            temperature=0.2,
            max_tokens=512,
        )
    except APIConnectionError as exc:
        raise SystemExit(
            f"Could not connect to {args.base_url}. Start vLLM on this node, "
            "forward the port, or pass --base-url/OPENAI_BASE_URL."
        ) from exc
    except BadRequestError as exc:
        if "enable-auto-tool-choice" in str(exc):
            raise SystemExit(
                "vLLM rejected the request because native tools were sent, but "
                "the server was not started with auto tool choice enabled. "
                "Restart vLLM with --enable-auto-tool-choice and a compatible "
                "--tool-call-parser, or run this script against a server that "
                "already supports OpenAI tool calls."
            ) from exc
        raise

    print_model_response(resp)


def build_training_messages(image_path: Path) -> Messages:
    artifacts = load_episode_artifacts(image_path)
    step_idx = step_index(image_path)
    config = screen_config(image_path)

    messages = target_box_initial_user_messages(
        observation_for_step(image_path, 0),
        config,
    )
    if step_idx == 0:
        return messages

    for prior_step_idx in range(1, step_idx + 1):
        step = artifacts.trajectory_steps.get(prior_step_idx)
        if step is None:
            raise ValueError(
                "Cannot reconstruct the exact RL prompt: missing trajectory "
                f"record for step {prior_step_idx}."
            )
        if step.get("rollout_terminated"):
            raise ValueError(
                "No RL model request is sent after terminal "
                f"step {prior_step_idx}; {image_path} has no exact prompt."
            )

        messages.append(assistant_message_from_step(step))
        messages.extend(
            target_box_observation_messages(
                observation_for_step(image_path, prior_step_idx),
                config,
            )
        )

    return messages


def load_episode_artifacts(image_path: Path) -> EpisodeArtifacts:
    episode_dir = (
        image_path.parent.parent if image_path.parent.name == "steps" else None
    )
    return EpisodeArtifacts(
        episode_dir=episode_dir,
        trajectory_steps=read_trajectory_steps(episode_dir),
    )


def observation_for_step(
    target_image_path: Path,
    step_idx: int,
) -> dict[str, Any]:
    image_path = step_image_path(target_image_path, step_idx)
    return {"screenshot": image_path.read_bytes()}


def assistant_message_from_step(step: Mapping[str, Any]) -> AssistantMessage:
    raw_tool_calls = step.get("tool_calls", [])
    if not isinstance(raw_tool_calls, list):
        raw_tool_calls = []

    tool_calls = [
        ToolCall(
            id=str(tool_call["id"]),
            name=str(tool_call["name"]),
            arguments=str(tool_call["arguments"]),
        )
        for tool_call in raw_tool_calls
        if isinstance(tool_call, Mapping)
        and all(key in tool_call for key in ("id", "name", "arguments"))
    ]
    response = step.get("response", "")
    return AssistantMessage(
        content=response if isinstance(response, str) else "",
        tool_calls=tool_calls or None,
    )


def screen_config(image_path: Path) -> ScreenConfig:
    width, height = image_size(image_path)
    return ScreenConfig(screen_width=width, screen_height=height)


def step_image_path(target_image_path: Path, step_idx: int) -> Path:
    if target_image_path.parent.name != "steps":
        return target_image_path
    return target_image_path.parent / f"step_{step_idx:03d}.png"


def step_index(image_path: Path) -> int:
    match = re.fullmatch(r"step_(\d+)\.png", image_path.name)
    if match is None:
        raise ValueError("Expected screenshot filename like step_000.png")
    return int(match.group(1))


def read_trajectory_steps(episode_dir: Path | None) -> dict[int, dict[str, Any]]:
    if episode_dir is None:
        return {}
    path = episode_dir / "traj.jsonl"
    if not path.exists():
        return {}

    steps: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step_num = record.get("step_num")
        if isinstance(step_num, int):
            steps[step_num] = record
    return steps


if __name__ == "__main__":
    main()
