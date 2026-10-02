from __future__ import annotations


def configure_qemu_snapshot_env(
    process_env: dict[str, str],
    *,
    snapshot_name: str = "osworld_ready",
    timeout_s: float = 600.0,
) -> None:
    """Populate host and Apptainer env vars for OSWorld QEMU ready snapshots."""
    timeout_text = _format_seconds(timeout_s)
    process_env["OSWORLD_QEMU_SNAPSHOT_NAME"] = snapshot_name
    process_env["OSWORLD_QEMU_MONITOR_TIMEOUT"] = timeout_text
    process_env["OSWORLD_QEMU_SNAPSHOT_READY_TIMEOUT"] = timeout_text
    process_env["APPTAINERENV_OSWORLD_QEMU_READY_SNAPSHOT"] = "1"
    process_env["APPTAINERENV_OSWORLD_QEMU_SNAPSHOT_NAME"] = snapshot_name
    process_env["APPTAINERENV_OSWORLD_QEMU_SNAPSHOT_TIMEOUT"] = timeout_text


def _format_seconds(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(float(value))
