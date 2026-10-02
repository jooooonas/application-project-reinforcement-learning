from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .specs import EnvServerSpec


def write_env_server_config(
    spec: EnvServerSpec,
    args: Any,
    metadata: Mapping[str, Any],
) -> None:
    harness = deepcopy(metadata["harness"])
    if spec.pool_status_dir:
        desktop = harness.setdefault("desktop", {})
        pool_config = desktop.setdefault("desktop_pool_config", {})
        pool_config["status_dir"] = spec.pool_status_dir
    taskset = {
        "id": args.env_id,
        "base_path": str(args.task_base_path),
        "max_tasks": args.max_tasks,
        "shuffle_seed": args.shuffle_seed,
    }
    harness["id"] = args.env_id
    payload = {
        "output_dir": str(args.run_root / "server_output"),
        "log": {"level": "INFO"},
        "env": {
            "name": spec.name,
            "address": spec.bind_address,
            "taskset": taskset,
            "harness": harness,
            "pool": {"type": "static", "num_workers": spec.num_workers},
            "timeout": {"rollout": args.rollout_timeout},
            "retries": {"rollout": {"max_retries": args.env_max_retries}},
            "max_turns": args.max_steps,
        },
    }
    Path(spec.config_path).write_text(to_toml(payload), encoding="utf-8")


def to_toml(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    scalar_items = {
        key: value for key, value in payload.items() if not isinstance(value, Mapping)
    }
    for key, value in scalar_items.items():
        lines.append(f"{key} = {toml_literal(value)}")
    for section, value in payload.items():
        if not isinstance(value, Mapping):
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, item in value.items():
            lines.append(f"{key} = {toml_literal(item)}")
    return "\n".join(lines).strip() + "\n"


def toml_literal(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Mapping):
        items = ", ".join(
            f"{key} = {toml_literal(item)}" for key, item in value.items()
        )
        return f"{{ {items} }}"
    if isinstance(value, list):
        return "[" + ", ".join(toml_literal(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")
