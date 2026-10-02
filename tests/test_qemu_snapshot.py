from __future__ import annotations

from rl.osworld.desktop.qemu_snapshot import configure_qemu_snapshot_env


def test_configure_qemu_snapshot_env_sets_host_and_container_values():
    env: dict[str, str] = {}

    configure_qemu_snapshot_env(
        env,
        snapshot_name="ready",
        timeout_s=45.0,
    )

    assert env["OSWORLD_QEMU_SNAPSHOT_NAME"] == "ready"
    assert env["OSWORLD_QEMU_MONITOR_TIMEOUT"] == "45"
    assert env["APPTAINERENV_OSWORLD_QEMU_READY_SNAPSHOT"] == "1"
    assert env["APPTAINERENV_OSWORLD_QEMU_SNAPSHOT_TIMEOUT"] == "45"


def test_configure_qemu_snapshot_env_uses_default_ready_snapshot():
    env: dict[str, str] = {}

    configure_qemu_snapshot_env(env)

    assert env["OSWORLD_QEMU_SNAPSHOT_NAME"] == "osworld_ready"
    assert env["APPTAINERENV_OSWORLD_QEMU_READY_SNAPSHOT"] == "1"
