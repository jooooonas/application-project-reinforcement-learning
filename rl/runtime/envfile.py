from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

from dotenv import dotenv_values

from rl.runtime.paths import repo_root, require_absolute_path


def load_runtime_env_file(
    env: MutableMapping[str, str] = os.environ,
    *,
    path: str | Path | None = None,
) -> Path | None:
    """Load runtime defaults without replacing explicit environment values."""
    resolved = _runtime_env_file_path(env=env, path=path)
    if resolved is None or not resolved.is_file():
        return None

    values = _runtime_env_values(resolved)
    for key, value in values.items():
        env.setdefault(key, value)
    return resolved


def _runtime_env_file_path(
    env: MutableMapping[str, str] = os.environ,
    path: str | Path | None = None,
) -> Path | None:
    if path is not None:
        return require_absolute_path(path, name="runtime env file path")
    explicit = env.get("RL_RUNTIME_ENV_FILE")
    if explicit is not None:
        if not explicit:
            return None
        return require_absolute_path(explicit, name="RL_RUNTIME_ENV_FILE")
    return repo_root() / ".env"


def _runtime_env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    parsed = dotenv_values(path)
    for key, value in parsed.items():
        if value is None:
            raise ValueError(f"{path}: expected KEY=value for {key!r}")
        values[key] = value
    return values
