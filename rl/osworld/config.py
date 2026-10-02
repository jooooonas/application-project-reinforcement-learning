from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from pydantic import Field

from rl.runtime.paths import (
    osworld_apptainer_image,
    osworld_asset_cache_dir,
    osworld_qcow_path,
    require_absolute_path,
    scratch_subdir,
    slurm_run_id,
)

from .desktop.pool import DesktopPoolConfig


@dataclass(frozen=True)
class OSWorldDesktopRuntimeConfig:
    screen_width: int = 1920
    screen_height: int = 1080
    # This is a quick workaround because verifiers doesnt ignore fields with init=False.
    apptainer_image: Annotated[Path, Field(exclude=True)] = field(
        default_factory=osworld_apptainer_image,
        init=False,
    )
    apptainer_ready_timeout: int = 1800
    screenshot_timeout: float = 60.0
    apptainer_cpu_cores: int = 4
    apptainer_ram_size: str = "8G"
    apptainer_qemu_snapshot_name: str = "osworld_ready"
    apptainer_qemu_snapshot_timeout: float = 600.0
    qcow_path: Path = field(default_factory=osworld_qcow_path)
    cache_dir: Path = field(default_factory=osworld_asset_cache_dir)
    output_dir: Path = field(
        default_factory=lambda: scratch_subdir("osworld_rl", slurm_run_id())
    )
    desktop_pool_config: DesktopPoolConfig = field(default_factory=DesktopPoolConfig)

    def __post_init__(self) -> None:
        if isinstance(self.desktop_pool_config, Mapping):
            object.__setattr__(
                self,
                "desktop_pool_config",
                DesktopPoolConfig(**self.desktop_pool_config),
            )
        for field_name in (
            "apptainer_image",
            "qcow_path",
            "cache_dir",
            "output_dir",
        ):
            object.__setattr__(
                self,
                field_name,
                require_absolute_path(getattr(self, field_name), name=field_name),
            )
        if self.apptainer_qemu_snapshot_timeout <= 0:
            raise ValueError("apptainer_qemu_snapshot_timeout must be positive")
