from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prime_rl_root() -> Path:
    return repo_root() / "deps" / "prime-rl"


def require_absolute_path(
    value: str | Path,
    *,
    name: str = "path",
) -> Path:
    raw_value = str(value)
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(
            f"{name} must be an absolute path; got {raw_value!r}. "
            "Do not use '~', relative paths, or unexpanded shell variables."
        )
    return path


def scratch_root(env: Mapping[str, str] = os.environ) -> Path:
    scratch = env.get("SCRATCH")
    if scratch:
        return require_absolute_path(scratch, name="SCRATCH")
    return repo_root() / ".scratch"


def scratch_subdir(*parts: str, env: Mapping[str, str] = os.environ) -> Path:
    return scratch_root(env).joinpath(*parts)


def osworld_root() -> Path:
    """Return the pinned OSWorld submodule checkout."""
    return repo_root() / "deps" / "OSWorldRL"


def osworld_task_base_path() -> Path:
    return (
        osworld_root() / "evaluation_examples" / "examples" / "target_box_empty_desktop"
    )


def osworld_apptainer_image() -> Path:
    return repo_root() / "apptainer" / "images" / "osworld.sif"


def osworld_qcow_path(env: Mapping[str, str] = os.environ) -> Path:
    value = env.get("OSWORLD_QCOW_PATH")
    if value:
        return require_absolute_path(value, name="OSWORLD_QCOW_PATH")
    return repo_root().parent / "osworld_deployment" / "Ubuntu.qcow2"


def osworld_asset_cache_dir(env: Mapping[str, str] = os.environ) -> Path:
    return scratch_subdir("osworld_asset_cache", env=env)


def slurm_run_id(env: Mapping[str, str] = os.environ) -> str:
    return env.get("SLURM_JOB_ID") or "manual"
