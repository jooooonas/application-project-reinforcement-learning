from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rl.runtime.ports import PortLease

from .qemu_snapshot import configure_qemu_snapshot_env


def build_process_env(
    *,
    base_env: Mapping[str, str],
    osworld_root: Path,
    apptainer_image: Path,
    apptainer_ready_timeout: int,
    screenshot_timeout: float,
    apptainer_cpu_cores: int,
    apptainer_ram_size: str,
    lease: PortLease,
    apptainer_qemu_snapshot_name: str = "osworld_ready",
    apptainer_qemu_snapshot_timeout: float = 600.0,
) -> dict[str, str]:
    process_env = dict(base_env)
    logdir = lease.logdir if lease.logdir is not None else lease.workdir
    vm_logdir = logdir / "vm"
    lease.workdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    vm_logdir.mkdir(parents=True, exist_ok=True)
    persistent_logdir = logdir.resolve()
    persistent_vm_logdir = persistent_logdir / "vm"
    persistent_vm_logdir.mkdir(parents=True, exist_ok=True)
    process_env.setdefault(
        "PROXY_CONFIG_FILE",
        str(osworld_root / "evaluation_examples/settings/proxy/dataimpulse.json"),
    )
    _prepend_apptainer_bind_paths(process_env, [persistent_logdir])
    process_env["OSWORLD_APPTAINER_IMAGE"] = str(apptainer_image)
    process_env["OSWORLD_APPTAINER_READY_TIMEOUT"] = str(apptainer_ready_timeout)
    process_env["OSWORLD_SCREENSHOT_TIMEOUT"] = str(screenshot_timeout)
    process_env["OSWORLD_APPTAINER_LOG"] = str(persistent_logdir / "apptainer.log")
    process_env["OSWORLD_APPTAINER_PIDFILE"] = str(lease.workdir / "apptainer.pid.json")
    process_env["OSWORLD_APPTAINER_SERVER_PORT"] = str(lease.ports.server)
    process_env["OSWORLD_APPTAINER_CHROMIUM_PORT"] = str(lease.ports.chromium)
    process_env["OSWORLD_APPTAINER_VNC_PORT"] = str(lease.ports.vnc)
    process_env["OSWORLD_APPTAINER_VLC_PORT"] = str(lease.ports.vlc)
    process_env["APPTAINERENV_QEMU_VNC_PORT"] = str(lease.ports.qemu_vnc)
    process_env["OSWORLD_APPTAINER_CPU_CORES"] = str(apptainer_cpu_cores)
    process_env["OSWORLD_APPTAINER_RAM_SIZE"] = str(apptainer_ram_size)
    process_env["APPTAINERENV_OSWORLD_WORKDIR"] = str(lease.workdir / "vm")
    process_env["APPTAINERENV_OSWORLD_LOG_DIR"] = str(persistent_vm_logdir)
    process_env["APPTAINERENV_READY_TIMEOUT"] = str(apptainer_ready_timeout)
    configure_qemu_snapshot_env(
        process_env,
        snapshot_name=apptainer_qemu_snapshot_name,
        timeout_s=apptainer_qemu_snapshot_timeout,
    )
    existing_pythonpath = process_env.get("PYTHONPATH")
    process_env["PYTHONPATH"] = (
        f"{osworld_root}:{existing_pythonpath}"
        if existing_pythonpath
        else str(osworld_root)
    )

    return process_env


def _prepend_apptainer_bind_paths(
    process_env: dict[str, str],
    paths: list[Path],
) -> None:
    binds = [f"{path.resolve()}:{path}" for path in paths]
    existing = process_env.get("APPTAINER_BINDPATH")
    if existing:
        binds.append(existing)
    process_env["APPTAINER_BINDPATH"] = ",".join(binds)
