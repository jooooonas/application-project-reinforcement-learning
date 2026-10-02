from __future__ import annotations

import os
from typing import TYPE_CHECKING

from rl.osworld.desktop.deployment import build_process_env
from rl.osworld.desktop.proxy import DesktopEnvProxy
from rl.runtime.paths import osworld_root
from rl.runtime.ports import PortLease

if TYPE_CHECKING:
    from rl.osworld.config import OSWorldDesktopRuntimeConfig


def create_desktop_env_proxy(
    config: OSWorldDesktopRuntimeConfig,
    lease: PortLease,
) -> DesktopEnvProxy:
    root = osworld_root()
    process_env = build_process_env(
        base_env=os.environ,
        osworld_root=root,
        apptainer_image=config.apptainer_image,
        apptainer_ready_timeout=config.apptainer_ready_timeout,
        screenshot_timeout=config.screenshot_timeout,
        apptainer_cpu_cores=config.apptainer_cpu_cores,
        apptainer_ram_size=config.apptainer_ram_size,
        apptainer_qemu_snapshot_name=config.apptainer_qemu_snapshot_name,
        apptainer_qemu_snapshot_timeout=config.apptainer_qemu_snapshot_timeout,
        lease=lease,
    )
    osworld_python = root / ".venv" / "bin" / "python"
    return DesktopEnvProxy(
        python=str(osworld_python),
        process_env=process_env,
        path_to_vm=str(config.qcow_path),
        screen_size=(config.screen_width, config.screen_height),
        cache_dir=str(config.cache_dir),
        init_timeout_s=config.desktop_pool_config.startup_timeout_s,
    )


_create_desktop_env_proxy = create_desktop_env_proxy
