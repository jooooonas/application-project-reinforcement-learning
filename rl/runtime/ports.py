import fcntl
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO


@dataclass(frozen=True)
class WorkerPorts:
    server: int
    chromium: int
    vnc: int
    vlc: int
    qemu_vnc: int


@dataclass
class PortLease:
    ports: WorkerPorts
    slot: int
    workdir: Path
    _lock_file: TextIO
    logdir: Path | None = None
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        if fcntl is not None:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()
        self._released = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def ports_for_worker(base: int, worker_id: int, stride: int = 10) -> WorkerPorts:
    start = base + worker_id * stride
    ports = WorkerPorts(
        server=start,
        chromium=start + 1,
        vnc=start + 2,
        vlc=start + 3,
        qemu_vnc=start + 4,
    )
    for port in (ports.server, ports.chromium, ports.vnc, ports.vlc, ports.qemu_vnc):
        if port > 65535:
            raise ValueError(f"Worker port {port} exceeds TCP port range")
    return ports


def assert_ports_available(ports: WorkerPorts) -> None:
    for port in (ports.server, ports.chromium, ports.vnc, ports.vlc, ports.qemu_vnc):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is already in use") from exc


def get_port_base(slurm_job_id: str | None) -> int:
    if slurm_job_id and slurm_job_id.isdigit():
        return 20000 + int(slurm_job_id) % 10000
    return 20000


def allocate_worker_ports(
    *,
    lock_dir: str | Path,
    work_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    stride: int = 10,
    max_slots: int = 512,
) -> PortLease:
    """Reserve a per-rollout OSWorld port block with an advisory file lock."""

    base = get_port_base(os.environ.get("SLURM_JOB_ID"))
    root = Path(lock_dir)
    work_root = Path(work_dir) if work_dir is not None else root
    log_root = Path(log_dir) if log_dir is not None else None
    root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    if log_root is not None:
        log_root.mkdir(parents=True, exist_ok=True)
    for slot in range(max_slots):
        ports = ports_for_worker(base, slot, stride=stride)
        lock_path = root / f"ports_{slot:04d}.lock"
        lock_file = lock_path.open("a+")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            continue
        try:
            assert_ports_available(ports)
        except Exception:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            continue
        lease_name = f"w{os.getpid()}_{slot:x}"
        workdir = work_root / lease_name
        logdir = log_root / lease_name if log_root is not None else None
        workdir.mkdir(parents=True, exist_ok=True)
        if logdir is not None:
            logdir.mkdir(parents=True, exist_ok=True)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} ports={ports}\n")
        lock_file.flush()
        return PortLease(
            ports=ports,
            slot=slot,
            workdir=workdir,
            _lock_file=lock_file,
            logdir=logdir,
        )
    raise RuntimeError(
        f"No available OSWorld port blocks under {root} from base {base}"
    )
