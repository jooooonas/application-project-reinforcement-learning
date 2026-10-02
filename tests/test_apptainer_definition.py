from __future__ import annotations

from pathlib import Path


def test_snapshot_markers_stay_next_to_qemu_monitor_sock():
    definition = Path("apptainer/osworld.def").read_text(encoding="utf-8")

    assert 'MONITOR="${WORKDIR}/qemu-monitor.sock"' in definition
    assert 'SNAPSHOT_OK="${WORKDIR}/ready-snapshot.ok"' in definition
    assert 'SNAPSHOT_FAILED="${WORKDIR}/ready-snapshot.failed"' in definition
    assert 'SNAPSHOT_LOG="${LOGDIR}/ready-snapshot.log"' in definition
    assert 'SNAPSHOT_OK="${LOGDIR}/ready-snapshot.ok"' not in definition
    assert 'SNAPSHOT_FAILED="${LOGDIR}/ready-snapshot.failed"' not in definition
