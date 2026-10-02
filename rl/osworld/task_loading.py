from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol


class TaskPathConfig(Protocol):
    base_path: str
    max_tasks: int
    shuffle_seed: int


def resolve_task_paths(config: TaskPathConfig) -> list[Path]:
    if config.max_tasks < 0:
        raise ValueError("max_tasks must be non-negative")

    task_paths = list(iter_task_paths(config.base_path))
    if config.shuffle_seed >= 0:
        random.Random(config.shuffle_seed).shuffle(task_paths)
    if config.max_tasks:
        task_paths = task_paths[: config.max_tasks]
    return task_paths


def iter_task_paths(base_path: str) -> Iterable[Path]:
    path = Path(base_path)
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"OSWorld task path does not exist: {path}")

    task_paths = sorted(
        candidate for candidate in path.rglob("*.json") if candidate.is_file()
    )
    if not task_paths:
        raise FileNotFoundError(f"No OSWorld task JSON files found under: {path}")
    yield from task_paths


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"OSWorld JSON file must contain an object: {path}")
    return payload
