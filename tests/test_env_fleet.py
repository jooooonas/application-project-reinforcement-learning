from __future__ import annotations

import json
import os
import sys
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl.runtime.envfile import load_runtime_env_file
from rl.runtime.fleet import (
    FleetRunLayout,
    SlurmJob,
    confirm_cancel,
    default_public_host,
    make_server_specs,
    parse_squeue,
    read_registry,
    select_cancel_job,
    upsert_registry,
)
from rl.runtime.fleet.config_rendering import toml_literal, write_env_server_config
from rl.runtime.fleet.readiness import readiness_summary
from rl.runtime.paths import (
    osworld_apptainer_image,
    osworld_asset_cache_dir,
    osworld_qcow_path,
    osworld_root,
    prime_rl_root,
    repo_root,
    scratch_root,
    scratch_subdir,
)
from scripts.osworld_fleet import (
    build_prefetch_command,
    build_sbatch_command,
    default_asset_cache_dir,
    default_task_base_path,
    format_status_report,
    format_submit_report,
    parse_sbatch_job_id,
    prepare_slurm_log_directories,
    read_registry_for_status,
    read_registry_optional,
    registry_path_for_args,
)
from scripts.osworld_fleet import (
    parse_args as parse_osworld_fleet_args,
)
from scripts.prepare_env_fleet import (
    expected_ready_sessions,
    gateway_config,
    harness_config,
    registry_metadata,
)
from scripts.prepare_env_fleet import (
    parse_args as parse_prepare_env_fleet_args,
)
from scripts.prime_rl import (
    absolutize_config_args,
    prepend_path,
    prime_rl_setup_error,
    subprocess_env,
    with_cluster_profile_defaults,
    with_forced_option,
    with_runtime_env_defaults,
)
from scripts.render_prime_rl_fleet_config import (
    configure_external_fleet,
    external_env_config,
)
from scripts.supervise_osworld_env_fleet import (
    PoolHealth,
    ReplicaRuntime,
    SupervisorPolicy,
    gateway_script_path,
    owned_process_group_ids,
    read_pool_health,
    restart_reason,
    safe_process_group_ids,
)


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", "")


def test_fleet_run_layout_uses_readable_default_paths(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="12345",
        run_base=tmp_path / "osworld_rl",
    )

    assert layout.run_root == tmp_path / "osworld_rl" / "12345" / "env_fleet"
    assert layout.registry_path == layout.run_root / "env_registry.json"
    assert layout.pool_root == tmp_path / "osworld_rl" / "12345" / "pool"
    assert layout.pool_status_dir == layout.pool_root / "status"
    assert layout.logs_dir == layout.run_root / "logs"
    assert layout.configs_dir == layout.run_root / "configs"
    assert layout.prime_rl_config_path == (
        tmp_path / "osworld_rl" / "12345" / "prime_rl_fleet.toml"
    )
    assert layout.prime_rl_output_dir == tmp_path / "osworld_rl" / "12345" / "prime_rl"


def test_fleet_run_layout_rejects_relative_constructor_paths(tmp_path):
    with pytest.raises(ValueError, match="run_base must be an absolute path"):
        FleetRunLayout.for_run(run_id="run-a", run_base="relative/base")

    with pytest.raises(ValueError, match="registry_path must be an absolute path"):
        FleetRunLayout.for_run(
            run_id="run-a",
            run_base=tmp_path / "base",
            registry_path="relative/registry.json",
        )


def test_fleet_run_layout_honors_environment_overrides(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_FLEET_RUN_ID": "run-a",
        "OSWORLD_RUN_BASE": str(tmp_path / "base"),
        "OSWORLD_FLEET_RUN_ROOT": str(tmp_path / "custom-env-fleet"),
        "OSWORLD_DESKTOP_POOL_ROOT": str(tmp_path / "custom-desktop-pool"),
        "OSWORLD_ENV_FLEET_REGISTRY": str(tmp_path / "registry.json"),
        "OSWORLD_PRIME_RL_CONFIG_PATH": str(tmp_path / "prime_rl.toml"),
        "OSWORLD_PRIME_RL_OUTPUT_DIR": str(tmp_path / "prime_rl"),
    }

    layout = FleetRunLayout.from_env(env)

    assert layout.run_id == "run-a"
    assert layout.run_base == tmp_path / "base"
    assert layout.run_root == tmp_path / "custom-env-fleet"
    assert layout.pool_root == tmp_path / "custom-desktop-pool"
    assert layout.pool_status_dir == tmp_path / "custom-desktop-pool" / "status"
    assert layout.registry_path == tmp_path / "registry.json"
    assert layout.prime_rl_config_path == tmp_path / "prime_rl.toml"
    assert layout.prime_rl_output_dir == tmp_path / "prime_rl"


def test_fleet_run_layout_defaults_to_short_scratch_run_base(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_FLEET_RUN_ID": "run-a",
    }

    layout = FleetRunLayout.from_env(env)

    assert layout.run_base == tmp_path / "scratch"
    assert layout.pool_root == tmp_path / "scratch" / "run-a" / "pool"


def test_fleet_run_layout_keeps_explicit_paths_literal(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_FLEET_RUN_ID": "13969570",
        "OSWORLD_RUN_BASE": str(tmp_path / "shared" / "osworld_rl"),
        "OSWORLD_FLEET_RUN_ROOT": str(tmp_path / "custom" / "env_fleet"),
        "OSWORLD_ENV_FLEET_REGISTRY": str(tmp_path / "custom" / "registry.json"),
        "OSWORLD_DESKTOP_POOL_ROOT": str(tmp_path / "custom" / "desktop_pool"),
        "OSWORLD_DESKTOP_POOL_STATUS_DIR": str(tmp_path / "custom" / "status"),
        "OSWORLD_PRIME_RL_CONFIG_PATH": str(tmp_path / "custom" / "prime_rl.toml"),
        "OSWORLD_PRIME_RL_OUTPUT_DIR": str(tmp_path / "custom" / "prime_rl_output"),
    }

    layout = FleetRunLayout.from_env(env)

    assert layout.run_base == tmp_path / "shared" / "osworld_rl"
    assert layout.run_root == tmp_path / "custom" / "env_fleet"
    assert layout.registry_path == tmp_path / "custom" / "registry.json"
    assert layout.pool_root == tmp_path / "custom" / "desktop_pool"
    assert layout.pool_status_dir == tmp_path / "custom" / "status"
    assert layout.prime_rl_config_path == tmp_path / "custom" / "prime_rl.toml"
    assert layout.prime_rl_output_dir == tmp_path / "custom" / "prime_rl_output"


def test_fleet_run_layout_defaults_without_scratch_or_project(monkeypatch):
    monkeypatch.delenv("SCRATCH", raising=False)

    layout = FleetRunLayout.from_env({}, run_id="manual")

    assert layout.run_base == repo_root() / ".scratch"
    assert layout.registry_path == (
        repo_root() / ".scratch" / "manual" / "env_fleet" / "env_registry.json"
    )


def test_make_server_specs_assigns_ports_and_replicas(tmp_path):
    specs = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=2,
        servers_per_node=2,
        workers_per_server=3,
        replica_count=8,
        replica_offset=4,
        name_prefix="osworld",
        config_dir=tmp_path / "configs",
        log_dir=tmp_path / "logs",
        pool_status_root=tmp_path / "status",
    )

    assert [spec.name for spec in specs] == ["osworld-0004", "osworld-0005"]
    assert [spec.port for spec in specs] == [5204, 5205]
    assert [spec.replica_index for spec in specs] == [4, 5]
    assert [spec.replica_count for spec in specs] == [8, 8]
    assert specs[0].public_address == "tcp://node001:5204"
    assert specs[0].num_workers == 3
    assert specs[0].pool_status_dir == str(tmp_path / "status" / "osworld-0004")


def test_default_public_host_prefers_explicit_env():
    host = default_public_host(
        {"OSWORLD_FLEET_HOST": "custom-host"},
        run_command=lambda _command: "NodeAddr=slurm-host",
    )

    assert host == "custom-host"


def test_default_public_host_prefers_slurm_node_addr():
    commands: list[list[str]] = []

    def run_command(command: list[str]) -> str:
        commands.append(command)
        return "NodeName=jwb0127 Arch=x86_64 NodeAddr=jwb0127i NodeHostName=jwb0127"

    host = default_public_host(
        {"SLURMD_NODENAME": "jwb0127"},
        run_command=run_command,
        fqdn_func=lambda: "jwb0127.juwels",
        hostname_func=lambda: "jwb0127",
    )

    assert host == "jwb0127i"
    assert commands == [["scontrol", "show", "node", "jwb0127"]]


def test_default_public_host_falls_back_to_fqdn():
    host = default_public_host(
        {},
        run_command=lambda _command: "",
        fqdn_func=lambda: "jwb0127.juwels",
        hostname_func=lambda: "jwb0127",
    )

    assert host == "jwb0127.juwels"


def test_registry_upsert_merges_by_server_name(tmp_path):
    registry_path = tmp_path / "registry.json"
    first = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=1,
        replica_count=2,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=tmp_path,
        log_dir=tmp_path,
    )
    second = make_server_specs(
        host="node002",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=1,
        servers_per_node=1,
        workers_per_server=1,
        replica_count=2,
        replica_offset=1,
        name_prefix="osworld",
        config_dir=tmp_path,
        log_dir=tmp_path,
    )

    upsert_registry(
        path=registry_path,
        run_id="run",
        metadata={"env_id": "rl"},
        servers=first,
    )
    upsert_registry(
        path=registry_path,
        run_id="run",
        metadata={"task_base_path": "/tasks"},
        servers=second,
    )

    registry = read_registry(registry_path)
    assert [server.name for server in registry.servers] == [
        "osworld-0000",
        "osworld-0001",
    ]
    assert registry.metadata["env_id"] == "rl"
    assert registry.metadata["task_base_path"] == "/tasks"
    json.dumps(registry.as_dict())


def test_render_prime_rl_fleet_config_uses_v1_schema(tmp_path):
    metadata = {
        "env_id": "rl",
        "env_name_prefix": "osworld",
        "task_base_path": "/tasks",
        "max_tasks": 4,
        "shuffle_seed": 7,
        "gateway": {"public_address": "tcp://node001:5200"},
        "harness": {
            "max_steps": 4,
            "desktop": {
                "output_dir": str(tmp_path / "worker-output"),
                "cache_dir": "/scratch/user/cache",
                "desktop_pool_config": {
                    "min_ready_sessions": 1,
                    "max_sessions": 3,
                },
            },
        },
    }
    config = {
        "output_dir": "/old",
        "orchestrator": {
            "max_inflight_rollouts": 1,
            "train": {"env": [{"name": "old"}]},
        },
        "inference": {"gpu_memory_utilization": 0.85},
    }

    configure_external_fleet(
        config,
        metadata=metadata,
        output_dir=tmp_path / "trainer-output",
        max_inflight_rollouts=2,
        rollout_timeout=3600,
        max_retries=1,
    )

    env = config["orchestrator"]["train"]["env"][0]
    assert config["output_dir"] == str(tmp_path / "trainer-output")
    assert config["orchestrator"]["max_inflight_rollouts"] == 2
    assert env["address"] == "tcp://node001:5200"
    assert env["taskset"] == {
        "id": "rl",
        "base_path": "/tasks",
        "max_tasks": 4,
        "shuffle_seed": 7,
    }
    assert env["harness"]["id"] == "rl"
    assert env["harness"]["desktop"]["desktop_pool_config"] == {
        "min_ready_sessions": 0,
        "max_sessions": 3,
    }
    assert env["timeout"] == {"rollout": 3600}
    assert env["retries"] == {"rollout": {"max_retries": 1}}
    assert env["max_turns"] == 4
    assert config["inference"] == {"gpu_memory_utilization": 0.85}


def test_render_prime_rl_fleet_config_requires_gateway_address(tmp_path):
    metadata = {
        "env_id": "rl",
        "env_name_prefix": "osworld",
        "task_base_path": "/tasks",
        "max_tasks": 4,
        "shuffle_seed": 7,
        "harness": {"max_steps": 4},
    }

    with pytest.raises(ValueError, match="gateway.public_address"):
        external_env_config(
            metadata,
            rollout_timeout=3600,
            max_retries=1,
        )


def test_render_prime_rl_fleet_config_uses_gateway_address(tmp_path):
    metadata = {
        "env_id": "rl",
        "env_name_prefix": "osworld",
        "task_base_path": "/tasks",
        "max_tasks": 4,
        "shuffle_seed": 7,
        "harness": {"max_steps": 4},
        "gateway": {
            "bind_address": "tcp://0.0.0.0:5202",
            "public_address": "tcp://node001:5202",
            "backend_addresses": ["tcp://node001:5200", "tcp://node001:5201"],
        },
    }

    rendered = external_env_config(
        metadata,
        rollout_timeout=3600,
        max_retries=1,
    )

    assert rendered["address"] == "tcp://node001:5202"
    assert "tcp://node001:5200" not in json.dumps(rendered)


def test_prepare_env_fleet_harness_config_includes_desktop_pool(tmp_path):
    args = SimpleNamespace(
        max_steps=4,
        screen_width=1920,
        screen_height=1080,
        screenshot_timeout=60.0,
        artifact_output_dir=tmp_path / "custom-artifacts",
        cache_dir=tmp_path / "cache",
        run_root=tmp_path / "run",
        qcow_path=tmp_path / "Ubuntu.qcow2",
        desktop_pool_min_ready_sessions=1,
        desktop_pool_max_sessions=1,
        desktop_pool_max_rollouts_per_session=1,
        desktop_pool_checkout_timeout=900.0,
        desktop_pool_lease_timeout=300.0,
        desktop_pool_startup_timeout=840.0,
        desktop_pool_startup_retry_backoff=30.0,
        desktop_pool_startup_retry_backoff_max=300.0,
        desktop_pool_status_heartbeat_interval=10.0,
        desktop_pool_root=tmp_path / "run" / "desktop_pool",
        desktop_pool_runtime_dir=tmp_path / "runtime" / "pool",
        desktop_pool_log_runtime_dir=tmp_path / "log",
        workers_per_server=2,
    )

    config = harness_config(args)

    assert config["max_steps"] == 4
    desktop = config["desktop"]
    assert "osworld_root" not in desktop
    assert desktop["output_dir"] == str(tmp_path / "custom-artifacts")
    assert desktop["desktop_pool_config"] == {
        "min_ready_sessions": 1,
        "max_sessions": 1,
        "max_rollouts_per_session": 1,
        "checkout_timeout_s": 900.0,
        "lease_timeout_s": 300.0,
        "startup_timeout_s": 840.0,
        "startup_retry_backoff_s": 30.0,
        "startup_retry_backoff_max_s": 300.0,
        "status_heartbeat_interval_s": 10.0,
        "root_dir": str(tmp_path / "run" / "desktop_pool"),
        "runtime_dir": str(tmp_path / "runtime" / "pool"),
        "log_runtime_dir": str(tmp_path / "log"),
    }
    assert expected_ready_sessions(args, replica_count=3) == 6


def test_prepare_env_fleet_uses_runtime_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["prepare_env_fleet.py"])
    monkeypatch.setenv("PROJECT", str(tmp_path / "project"))
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setenv("OSWORLD_FLEET_RUN_ID", "run-a")
    monkeypatch.setenv(
        "OSWORLD_DESKTOP_POOL_RUNTIME_DIR",
        str(tmp_path / "runtime" / "pool"),
    )
    monkeypatch.setenv(
        "OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR",
        str(tmp_path / "log"),
    )
    monkeypatch.delenv("OSWORLD_QCOW_PATH", raising=False)

    args = parse_prepare_env_fleet_args()

    assert args.qcow_path == repo_root().parent / "osworld_deployment" / "Ubuntu.qcow2"
    assert args.desktop_pool_runtime_dir == tmp_path / "runtime" / "pool"
    assert args.desktop_pool_startup_timeout == 840.0
    assert args.desktop_pool_log_runtime_dir == tmp_path / "log"


def test_prepare_env_fleet_registry_metadata_includes_layout(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="run",
        run_base=tmp_path / "osworld_rl",
    )
    args = SimpleNamespace(
        run_id="run",
        env_id="rl",
        env_name_prefix="osworld",
        task_base_path=tmp_path / "tasks",
        max_tasks=5,
        shuffle_seed=11,
        max_steps=4,
        screen_width=1920,
        screen_height=1080,
        screenshot_timeout=60.0,
        artifact_output_dir=None,
        cache_dir=tmp_path / "cache",
        run_root=layout.run_root,
        qcow_path=tmp_path / "Ubuntu.qcow2",
        desktop_pool_min_ready_sessions=2,
        desktop_pool_max_sessions=2,
        desktop_pool_max_rollouts_per_session=1,
        desktop_pool_checkout_timeout=900.0,
        desktop_pool_lease_timeout=300.0,
        desktop_pool_startup_timeout=840.0,
        desktop_pool_startup_retry_backoff=30.0,
        desktop_pool_startup_retry_backoff_max=300.0,
        desktop_pool_status_heartbeat_interval=10.0,
        desktop_pool_root=layout.pool_root,
        desktop_pool_runtime_dir=tmp_path / "runtime" / "pool",
        desktop_pool_log_runtime_dir=tmp_path / "log",
        workers_per_server=3,
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        servers_per_node=2,
        replica_hosts="node001,node002",
        gateway_host=None,
        gateway_bind_host=None,
        gateway_port=0,
    )

    metadata = registry_metadata(args, layout, replica_count=4)

    assert metadata["layout"] == layout.as_metadata()
    assert metadata["env_name_prefix"] == "osworld"
    assert "pool_root" not in metadata
    assert "pool_status_dir" not in metadata
    assert metadata["max_tasks"] == 5
    desktop = metadata["harness"]["desktop"]
    assert desktop["output_dir"] == str(layout.run_root / "artifacts")
    assert desktop["desktop_pool_config"]["runtime_dir"] == str(
        tmp_path / "runtime" / "pool"
    )
    assert desktop["desktop_pool_config"]["log_runtime_dir"] == str(tmp_path / "log")
    assert metadata["gateway"] == {
        "bind_address": "tcp://0.0.0.0:5204",
        "public_address": "tcp://node001:5204",
        "backend_addresses": [
            "tcp://node001:5200",
            "tcp://node001:5201",
            "tcp://node002:5202",
            "tcp://node002:5203",
        ],
    }
    assert metadata["expected_env_servers"] == 4
    assert metadata["expected_env_workers"] == 12
    assert "expected_server_count" not in metadata
    assert "expected_worker_count" not in metadata
    assert metadata["expected_ready_sessions"] == 24


def test_gateway_config_uses_explicit_gateway_port():
    args = SimpleNamespace(
        gateway_port=18084,
        base_port=5200,
        replica_hosts="node001",
        gateway_host=None,
        host="node001",
        gateway_bind_host=None,
        bind_host="0.0.0.0",
        servers_per_node=4,
    )

    config = gateway_config(args, replica_count=4)

    assert config == {
        "bind_address": "tcp://0.0.0.0:18084",
        "public_address": "tcp://node001:18084",
        "backend_addresses": [
            "tcp://node001:5200",
            "tcp://node001:5201",
            "tcp://node001:5202",
            "tcp://node001:5203",
        ],
    }


def test_prepare_env_fleet_writes_taskset_without_partition_keys(tmp_path):
    server = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=2,
        replica_count=4,
        replica_offset=2,
        name_prefix="osworld",
        config_dir=tmp_path,
        log_dir=tmp_path,
        pool_status_root=tmp_path / "status",
    )[0]
    args = SimpleNamespace(
        task_base_path=tmp_path / "tasks",
        max_tasks=5,
        shuffle_seed=11,
        max_steps=4,
        run_root=tmp_path / "run",
        env_id="rl",
        rollout_timeout=3600.0,
        env_max_retries=2,
    )

    write_env_server_config(
        server,
        args,
        {
            "harness": {
                "max_steps": 4,
                "desktop": {
                    "desktop_pool_config": {
                        "runtime_dir": "/tmp/osworld-runtime",
                    },
                },
            }
        },
    )

    with Path(server.config_path).open("rb") as file:
        config = tomllib.load(file)
    env = config["env"]
    assert env["taskset"] == {
        "id": "rl",
        "base_path": str(tmp_path / "tasks"),
        "max_tasks": 5,
        "shuffle_seed": 11,
    }
    assert env["pool"] == {"type": "static", "num_workers": 2}
    assert env["timeout"] == {"rollout": 3600.0}
    assert env["retries"] == {"rollout": {"max_retries": 2}}
    assert env["max_turns"] == 4
    pool = env["harness"]["desktop"]["desktop_pool_config"]
    assert pool["status_dir"] == server.pool_status_dir
    assert pool["runtime_dir"] == "/tmp/osworld-runtime"


def test_wait_env_fleet_ready_aggregates_registry_and_pool_status(tmp_path):
    registry_path = tmp_path / "registry.json"
    server = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=2,
        replica_count=1,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=tmp_path,
        log_dir=tmp_path,
        pool_status_root=tmp_path / "desktop_pool" / "status",
    )[0]
    upsert_registry(
        path=registry_path,
        run_id="run",
        metadata={
            "expected_env_servers": 1,
            "expected_ready_sessions": 2,
        },
        servers=[server],
    )
    status_dir = tmp_path / "desktop_pool" / "status"
    server_status_dir = Path(server.pool_status_dir)
    server_status_dir.mkdir(parents=True)
    (server_status_dir / "worker-a.json").write_text(
        json.dumps({"closed": False, "ready": 2, "starting": 0}),
        encoding="utf-8",
    )

    summary = readiness_summary(
        SimpleNamespace(
            registry=registry_path,
            status_dir=status_dir,
            pool_status_dir=None,
            run_root=registry_path.parent,
            min_ready_sessions=-1,
            expected_servers=0,
        )
    )

    assert summary["registry_ready"] is True
    assert summary["min_ready"] == 2
    assert summary["ready"] == 2
    assert summary["server_summaries"][0]["ready"] == 2


def test_wait_env_fleet_ready_derives_status_dir_from_registry_layout(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="run",
        run_base=tmp_path / "osworld_rl",
    )
    server = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=1,
        replica_count=1,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=layout.configs_dir,
        log_dir=layout.logs_dir,
        pool_status_root=layout.pool_status_dir,
    )[0]
    upsert_registry(
        path=layout.registry_path,
        run_id="run",
        metadata={
            "layout": layout.as_metadata(),
            "expected_env_servers": 1,
            "expected_ready_sessions": 1,
        },
        servers=[server],
    )
    server_status_dir = Path(server.pool_status_dir)
    server_status_dir.mkdir(parents=True)
    (server_status_dir / "worker-a.json").write_text(
        json.dumps(
            {
                "closed": False,
                "ready": 1,
                "total_failed": 3,
                "retry_scheduled": True,
                "consecutive_start_failures": 2,
                "startup_cooldown_remaining_s": 12.5,
            }
        ),
        encoding="utf-8",
    )

    summary = readiness_summary(
        SimpleNamespace(
            registry=layout.registry_path,
            status_dir=None,
            pool_status_dir=None,
            run_root=layout.run_root,
            min_ready_sessions=-1,
            expected_servers=0,
        )
    )

    assert summary["registry_ready"] is True
    assert summary["status_dir"] == str(layout.pool_status_dir)
    assert summary["ready"] == 1
    assert summary["total_failed"] == 3
    assert summary["retry_scheduled_workers"] == 1
    assert summary["cooling_down_workers"] == 1
    assert summary["consecutive_start_failures"] == 2
    assert summary["startup_cooldown_remaining_s"] == 12.5
    assert summary["server_summaries"][0]["name"] == "osworld-0000"
    assert summary["server_summaries"][0]["total_failed"] == 3


def test_readiness_summary_ignores_stale_status_files(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="run",
        run_base=tmp_path / "osworld_rl",
    )
    server = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=1,
        replica_count=1,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=layout.configs_dir,
        log_dir=layout.logs_dir,
        pool_status_root=layout.pool_status_dir,
    )[0]
    upsert_registry(
        path=layout.registry_path,
        run_id="run",
        metadata={
            "layout": layout.as_metadata(),
            "expected_env_servers": 1,
            "expected_ready_sessions": 1,
        },
        servers=[server],
    )
    server_status_dir = Path(server.pool_status_dir)
    server_status_dir.mkdir(parents=True)
    now = time.time()
    (server_status_dir / "worker-stale.json").write_text(
        json.dumps({"closed": False, "updated_at": now - 1000.0, "ready": 5}),
        encoding="utf-8",
    )
    (server_status_dir / "worker-fresh.json").write_text(
        json.dumps({"closed": False, "updated_at": now, "ready": 1}),
        encoding="utf-8",
    )

    summary = readiness_summary(
        SimpleNamespace(
            registry=layout.registry_path,
            status_dir=None,
            pool_status_dir=None,
            run_root=layout.run_root,
            min_ready_sessions=-1,
            expected_servers=0,
            status_stale_after_s=120.0,
        )
    )

    assert summary["ready"] == 1
    assert summary["active_status_files"] == 1
    assert summary["stale_status_files"] == 1
    assert summary["server_summaries"][0]["stale_status_files"] == 1


def test_supervisor_fresh_starting_capacity_is_recovering_after_grace(tmp_path):
    process = SimpleNamespace(poll=lambda: None, pid=123)
    replica = ReplicaRuntime(
        name="osworld-0000",
        config_path=tmp_path / "config.toml",
        log_path=tmp_path / "server.log",
        status_dir=tmp_path / "status",
        command=[],
        process=process,
        started_at=0.0,
        unhealthy_since=0.0,
    )
    policy = SupervisorPolicy(
        poll_s=5.0,
        startup_grace_s=10.0,
        replica_unhealthy_s=10.0,
        failure_window_s=300.0,
        max_failures_per_window=0,
        restart_backoff_s=10.0,
        fleet_unhealthy_s=300.0,
        max_fleet_restarts=3,
        terminate_timeout_s=30.0,
        status_stale_after_s=120.0,
    )
    health = PoolHealth(
        status_files=1,
        active_status_files=1,
        stale_status_files=0,
        ready=0,
        starting=4,
        fresh_starting=4,
        stale_starting=0,
        oldest_starting_age_s=30.0,
        leased=0,
        total_failed=0,
        last_errors=[],
    )

    assert restart_reason(replica, health, now=30.0, policy=policy) is None


def test_supervisor_stale_starting_capacity_is_unhealthy_after_grace(tmp_path):
    process = SimpleNamespace(poll=lambda: None, pid=123)
    replica = ReplicaRuntime(
        name="osworld-0000",
        config_path=tmp_path / "config.toml",
        log_path=tmp_path / "server.log",
        status_dir=tmp_path / "status",
        command=[],
        process=process,
        started_at=0.0,
        unhealthy_since=0.0,
    )
    policy = SupervisorPolicy(
        poll_s=5.0,
        startup_grace_s=10.0,
        replica_unhealthy_s=10.0,
        failure_window_s=300.0,
        max_failures_per_window=0,
        restart_backoff_s=10.0,
        fleet_unhealthy_s=300.0,
        max_fleet_restarts=3,
        terminate_timeout_s=30.0,
        status_stale_after_s=120.0,
    )
    health = PoolHealth(
        status_files=1,
        active_status_files=1,
        stale_status_files=0,
        ready=0,
        starting=4,
        fresh_starting=0,
        stale_starting=4,
        oldest_starting_age_s=901.0,
        leased=0,
        total_failed=0,
        last_errors=[],
    )

    assert (
        restart_reason(replica, health, now=30.0, policy=policy)
        == "4 desktop sessions stuck starting for up to 901.0s"
    )


def test_supervisor_mixed_starting_capacity_is_unhealthy_after_grace(tmp_path):
    process = SimpleNamespace(poll=lambda: None, pid=123)
    replica = ReplicaRuntime(
        name="osworld-0000",
        config_path=tmp_path / "config.toml",
        log_path=tmp_path / "server.log",
        status_dir=tmp_path / "status",
        command=[],
        process=process,
        started_at=0.0,
        unhealthy_since=0.0,
    )
    policy = SupervisorPolicy(
        poll_s=5.0,
        startup_grace_s=10.0,
        replica_unhealthy_s=10.0,
        failure_window_s=300.0,
        max_failures_per_window=0,
        restart_backoff_s=10.0,
        fleet_unhealthy_s=300.0,
        max_fleet_restarts=3,
        terminate_timeout_s=30.0,
        status_stale_after_s=120.0,
    )
    health = PoolHealth(
        status_files=1,
        active_status_files=1,
        stale_status_files=0,
        ready=0,
        starting=4,
        fresh_starting=2,
        stale_starting=2,
        oldest_starting_age_s=901.0,
        leased=0,
        total_failed=0,
        last_errors=[],
    )

    assert (
        restart_reason(replica, health, now=30.0, policy=policy)
        == "2 desktop sessions stuck starting for up to 901.0s"
    )


def test_read_pool_health_splits_fresh_and_stale_starting_sessions(tmp_path):
    now = time.time()
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    (status_dir / "worker.json").write_text(
        json.dumps(
            {
                "updated_at": now,
                "closed": False,
                "ready": 0,
                "starting": 2,
                "leased": 0,
                "total_failed": 0,
                "startup_timeout_s": 100.0,
                "starting_sessions": [
                    {"session_id": "fresh", "created_at": now - 10.0},
                    {"session_id": "stale", "created_at": now - 101.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    health = read_pool_health(status_dir, status_stale_after_s=120.0)

    assert health.starting == 2
    assert health.fresh_starting == 1
    assert health.stale_starting == 1
    assert health.oldest_starting_age_s is not None
    assert health.oldest_starting_age_s >= 100.0


def test_owned_process_group_ids_reads_sessions_and_starting_pidfiles(tmp_path):
    status_dir = tmp_path / "status"
    workdir = tmp_path / "runtime" / "w0"
    status_dir.mkdir()
    workdir.mkdir(parents=True)
    pidfile = workdir / "apptainer.pid.json"
    pidfile.write_text(json.dumps({"pid": 123, "pgid": 456}), encoding="utf-8")
    (status_dir / "worker.json").write_text(
        json.dumps(
            {
                "updated_at": time.time(),
                "closed": False,
                "sessions": [{"health": {"vm_pgid": 111}}],
                "starting_sessions": [{"apptainer_pidfile": str(pidfile)}],
            }
        ),
        encoding="utf-8",
    )

    assert owned_process_group_ids(status_dir) == (111, 456)


def test_safe_process_group_ids_skip_supervisor_group():
    assert safe_process_group_ids((os.getpgrp(), 123456789, 123456789)) == (123456789,)


def test_osworld_fleet_submit_command_uses_slurm_options_and_exports(tmp_path):
    args = SimpleNamespace(
        script=Path("sbatch/run_osworld_env_fleet.sbatch"),
        account="research",
        partition="booster",
        time="00:30:00",
        nodes=2,
        cpus_per_task=16,
        mem=None,
        job_name="osworld_env_fleet",
        slurm_output=tmp_path / "logs" / "slurm-%x.%j.out",
        slurm_error=tmp_path / "logs" / "slurm-%x.%j.out",
        run_id="fleet-a",
        run_base=tmp_path / "runs",
        task_base_path=tmp_path / "tasks",
        asset_cache_dir=tmp_path / "asset-cache",
        servers_per_node=2,
        workers_per_server=4,
        base_port=5300,
        max_tasks=7,
        artifact_output_dir=tmp_path / "artifacts",
        desktop_pool_min_ready_sessions=2,
        desktop_pool_max_sessions=3,
        desktop_pool_max_rollouts_per_session=25,
        desktop_pool_checkout_timeout=120.0,
        desktop_pool_lease_timeout=300.0,
        desktop_pool_startup_timeout=840.0,
        desktop_pool_startup_retry_backoff=5.0,
        desktop_pool_startup_retry_backoff_max=60.0,
        desktop_pool_status_heartbeat_interval=7.0,
        desktop_pool_root=tmp_path / "desktop-pool",
        desktop_pool_runtime_dir=tmp_path / "desktop-runtime",
        desktop_pool_log_runtime_dir=tmp_path / "desktop-log",
        rollout_timeout=900.0,
        env_max_retries=2,
        replica_unhealthy_s=120.0,
        fleet_unhealthy_s=300.0,
        max_fleet_restarts=3,
        gateway_request_timeout_s=900.0,
        status_stale_after_s=120.0,
    )

    command = build_sbatch_command(args)

    assert command[:2] == ["sbatch", "--parsable"]
    assert "--partition" in command
    assert "booster" in command
    assert "--nodes" in command
    assert "2" in command
    assert command[command.index("--output") + 1] == str(
        tmp_path / "logs" / "slurm-%x.%j.out"
    )
    assert command[command.index("--error") + 1] == str(
        tmp_path / "logs" / "slurm-%x.%j.out"
    )
    assert any("OSWORLD_FLEET_RUN_ID=fleet-a" in item for item in command)
    assert not any("OSWORLD_TASK_BASE_PATH=" in item for item in command)
    assert not any(str(tmp_path / "asset-cache") in item for item in command)
    assert any("OSWORLD_ENV_WORKERS_PER_SERVER=4" in item for item in command)
    assert any("OSWORLD_MAX_TASKS=7" in item for item in command)
    assert any(
        f"OSWORLD_ARTIFACT_DIR={tmp_path / 'artifacts'}" in item for item in command
    )
    assert any("OSWORLD_DESKTOP_POOL_MIN_READY_SESSIONS=2" in item for item in command)
    assert any("OSWORLD_DESKTOP_POOL_MAX_SESSIONS=3" in item for item in command)
    assert any(
        "OSWORLD_DESKTOP_POOL_MAX_ROLLOUTS_PER_SESSION=25" in item for item in command
    )
    assert any(
        "OSWORLD_DESKTOP_POOL_CHECKOUT_TIMEOUT=120.0" in item for item in command
    )
    assert any("OSWORLD_DESKTOP_POOL_LEASE_TIMEOUT=300.0" in item for item in command)
    assert any("OSWORLD_DESKTOP_POOL_STARTUP_TIMEOUT=840.0" in item for item in command)
    assert any(
        "OSWORLD_DESKTOP_POOL_STARTUP_RETRY_BACKOFF=5.0" in item for item in command
    )
    assert any(
        "OSWORLD_DESKTOP_POOL_STARTUP_RETRY_BACKOFF_MAX=60.0" in item
        for item in command
    )
    assert any(
        "OSWORLD_DESKTOP_POOL_STATUS_HEARTBEAT_INTERVAL=7.0" in item for item in command
    )
    assert any(
        f"OSWORLD_DESKTOP_POOL_ROOT={tmp_path / 'desktop-pool'}" in item
        for item in command
    )
    assert any(
        f"OSWORLD_DESKTOP_POOL_RUNTIME_DIR={tmp_path / 'desktop-runtime'}" in item
        for item in command
    )
    assert any(
        f"OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR={tmp_path / 'desktop-log'}" in item
        for item in command
    )
    assert any("OSWORLD_ROLLOUT_TIMEOUT=900.0" in item for item in command)
    assert any("OSWORLD_ENV_MAX_RETRIES=2" in item for item in command)
    assert any("OSWORLD_SUPERVISOR_MAX_FLEET_RESTARTS=3" in item for item in command)
    assert any("OSWORLD_GATEWAY_REQUEST_TIMEOUT_S=900.0" in item for item in command)
    assert any("OSWORLD_STATUS_STALE_AFTER_S=120.0" in item for item in command)
    assert parse_sbatch_job_id("12345;juwels\n") == "12345"


def test_osworld_fleet_prefetch_command_uses_asset_cache(tmp_path):
    args = SimpleNamespace(
        task_base_path=tmp_path / "tasks",
        asset_cache_dir=tmp_path / "asset-cache",
        asset_source_root=None,
    )

    command = build_prefetch_command(args)

    assert command[:4] == ["uv", "run", "--no-sync", "python"]
    assert command[4].endswith("prefetch_osworld_assets.py")
    assert command[-4:] == [
        "--tasks",
        str(tmp_path / "tasks"),
        "--cache-dir",
        str(tmp_path / "asset-cache"),
    ]


def test_osworld_fleet_default_task_base_uses_pinned_osworld():
    assert default_task_base_path() == (
        repo_root()
        / "deps"
        / "OSWorldRL"
        / "evaluation_examples"
        / "examples"
        / "target_box_empty_desktop"
    )


def test_osworld_fleet_parse_args_loads_runtime_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"export SCRATCH={tmp_path / 'scratch'}",
                "export OSWORLD_DESKTOP_POOL_RUNTIME_DIR=/tmp/osworld-runtime",
                "export OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR=/tmp/osworld-log",
                "export OSWORLD_FLEET_SLURM_PARTITION=cpu-partition",
                "export OSWORLD_FLEET_SLURM_TIME=08:00:00",
                "export OSWORLD_FLEET_SLURM_NODES=2",
                "export OSWORLD_FLEET_SLURM_CPUS_PER_TASK=24",
                "export OSWORLD_FLEET_SLURM_MEM_PER_NODE=96G",
                "export OSWORLD_ENV_SERVERS_PER_NODE=3",
                "export OSWORLD_ENV_WORKERS_PER_SERVER=2",
                "export OSWORLD_DESKTOP_POOL_MAX_SESSIONS=5",
                "export OSWORLD_DESKTOP_POOL_MAX_ROLLOUTS_PER_SESSION=6",
                "export OSWORLD_DESKTOP_POOL_CHECKOUT_TIMEOUT=700",
                "export SBATCH_ACCOUNT=fallback-project",
                "export SBATCH_PARTITION=fallback-partition",
                "export SBATCH_TIMELIMIT=01:00:00",
                "export SBATCH_NODES=3",
                "export SBATCH_CPUS_PER_TASK=12",
                "export SBATCH_MEM_PER_NODE=48G",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", str(env_file))
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.delenv("SBATCH_OUTPUT", raising=False)
    monkeypatch.delenv("SBATCH_ERROR", raising=False)

    args = parse_osworld_fleet_args(
        ["submit", "--dry-run", "--run-base", str(tmp_path / "custom-runs")]
    )

    assert args.run_base == tmp_path / "custom-runs"
    assert args.asset_cache_dir == tmp_path / "scratch" / "osworld_asset_cache"
    assert args.task_base_path == (
        repo_root()
        / "deps"
        / "OSWorldRL"
        / "evaluation_examples"
        / "examples"
        / "target_box_empty_desktop"
    )
    assert args.desktop_pool_runtime_dir == Path("/tmp/osworld-runtime")
    assert args.desktop_pool_log_runtime_dir == Path("/tmp/osworld-log")
    assert args.account == "fallback-project"
    assert args.partition == "cpu-partition"
    assert args.time == "08:00:00"
    assert args.nodes == 2
    assert args.cpus_per_task == 24
    assert args.mem == "96G"
    assert args.slurm_output == (tmp_path / "custom-runs" / "logs" / "slurm-%x.%j.out")
    assert args.slurm_error == args.slurm_output
    assert args.servers_per_node == 3
    assert args.workers_per_server == 2
    assert args.desktop_pool_max_sessions == 5
    assert args.desktop_pool_max_rollouts_per_session == 6
    assert args.desktop_pool_checkout_timeout == 700.0


def test_osworld_fleet_explicit_environment_wins_over_runtime_env_file(
    monkeypatch,
    tmp_path,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OSWORLD_DESKTOP_POOL_MAX_ROLLOUTS_PER_SESSION=4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", str(env_file))
    monkeypatch.setenv("OSWORLD_DESKTOP_POOL_MAX_ROLLOUTS_PER_SESSION", "32")
    monkeypatch.setenv("SBATCH_OUTPUT", str(tmp_path / "stdout" / "job-%j.log"))
    monkeypatch.setenv("SBATCH_ERROR", str(tmp_path / "stderr" / "job-%j.log"))

    args = parse_osworld_fleet_args(["submit", "--dry-run"])

    assert args.desktop_pool_max_rollouts_per_session == 32
    assert args.slurm_output == tmp_path / "stdout" / "job-%j.log"
    assert args.slurm_error == tmp_path / "stderr" / "job-%j.log"


def test_prepare_slurm_log_directories_creates_distinct_parents(tmp_path):
    stdout = tmp_path / "stdout" / "job-%j.log"
    stderr = tmp_path / "stderr" / "job-%j.log"

    prepare_slurm_log_directories(stdout, stderr)

    assert stdout.parent.is_dir()
    assert stderr.parent.is_dir()


def test_prepare_env_fleet_explicit_environment_wins_over_runtime_env_file(
    monkeypatch,
    tmp_path,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OSWORLD_DESKTOP_POOL_MAX_ROLLOUTS_PER_SESSION=4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", str(env_file))
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("OSWORLD_DESKTOP_POOL_MAX_ROLLOUTS_PER_SESSION", "32")
    monkeypatch.setattr(sys, "argv", ["prepare_env_fleet.py"])

    args = parse_prepare_env_fleet_args()

    assert args.desktop_pool_max_rollouts_per_session == 32


def test_osworld_fleet_default_asset_cache_uses_scratch(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
    }

    assert default_asset_cache_dir(env) == tmp_path / "scratch" / "osworld_asset_cache"


def test_osworld_fleet_submit_report_prints_next_steps(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="fleet-a",
        run_base=tmp_path / "runs",
    )

    with pytest.raises(SystemExit):
        parse_osworld_fleet_args(["render"])

    report = format_submit_report("12345", layout, SimpleNamespace())

    assert "Render config and launch PrimeRL trainer:" in report
    assert "scripts/render_prime_rl_fleet_config.py" in report
    assert "prime_rl_fleet.toml" in report
    assert "Readiness:" in report
    assert "uv run --no-sync python scripts/osworld_fleet.py status" in report
    assert "scripts/wait_env_fleet_ready.py" in report
    assert "uv run --no-sync python scripts/prime_rl.py" in report
    assert "scripts/prime_rl.py" in report
    assert f"{layout.prime_rl_config_path} --clean-output-dir" in report
    assert ".venv/bin/rl" not in report
    assert "rl/verifiers/prime-rl" not in report
    assert "Cancel:" in report


def test_prime_rl_forces_root_submodule_project_dir():
    args = with_forced_option(
        [
            "@",
            "config.toml",
            "--slurm.project-dir",
            "/old/deps/prime-rl",
            "--dry-run",
        ],
        "--slurm.project-dir",
        "/repo/deps/prime-rl",
    )

    assert args == [
        "@",
        "config.toml",
        "--dry-run",
        "--slurm.project-dir",
        "/repo/deps/prime-rl",
    ]
    assert prepend_path(Path("/repo"), None) == "/repo"
    assert prepend_path(Path("/repo"), "/existing") == "/repo:/existing"


def test_prime_rl_prepares_pythonpath():
    env = subprocess_env(
        Path("/repo"),
        {
            "PYTHONPATH": "/existing",
            "UNRELATED": "value",
        },
    )

    assert env["PYTHONPATH"] == "/repo:/existing"
    assert env["UNRELATED"] == "value"


def test_prime_rl_ignores_generic_sbatch_defaults():
    env = subprocess_env(
        Path("/repo"),
        {
            "SBATCH_ACCOUNT": "project",
            "SBATCH_PARTITION": "booster",
            "SBATCH_TIMELIMIT": "24:00:00",
            "SBATCH_CPUS_PER_TASK": "16",
            "SBATCH_MEM_PER_NODE": "64G",
            "SBATCH_EXCLUSIVE": "1",
        },
    )

    assert not any(name.startswith("SBATCH_") for name in env)


def test_prime_rl_ignores_generic_sbatch_cluster_profile_defaults():
    args = with_cluster_profile_defaults(
        ["@", "config.toml"],
        {
            "SBATCH_ACCOUNT": "project",
            "SBATCH_PARTITION": "booster",
            "SBATCH_TIMELIMIT": "24:00:00",
        },
    )

    assert args == ["@", "config.toml"]


def test_prime_rl_reports_submodule_and_environment_setup_separately(tmp_path):
    prime_rl_dir = tmp_path / "deps" / "prime-rl"
    rl_bin = prime_rl_dir / ".venv" / "bin" / "rl"

    assert "submodule update --init --recursive" in (
        prime_rl_setup_error(prime_rl_dir, rl_bin) or ""
    )
    prime_rl_dir.mkdir(parents=True)
    (prime_rl_dir / "pyproject.toml").write_text("[project]\nname='prime-rl'\n")
    assert "uv sync --project deps/prime-rl" in (
        prime_rl_setup_error(prime_rl_dir, rl_bin) or ""
    )
    rl_bin.parent.mkdir(parents=True)
    rl_bin.write_text("#!/bin/sh\n")
    assert prime_rl_setup_error(prime_rl_dir, rl_bin) is None


def test_prime_rl_absolutizes_config_args_from_submit_cwd(tmp_path):
    args = absolutize_config_args(
        ["@", "configs/rl.toml", "@other.toml", "--dry-run"],
        tmp_path,
    )

    assert args == [
        "@",
        str(tmp_path / "configs" / "rl.toml"),
        f"@{tmp_path / 'other.toml'}",
        "--dry-run",
    ]


def test_prime_rl_cluster_profile_defaults_become_cli_overrides():
    args = with_cluster_profile_defaults(
        ["@", "config.toml", "--slurm.time", "explicit"],
        {
            "PRIME_RL_SLURM_ACCOUNT": "project-123",
            "PRIME_RL_SLURM_PARTITION": "booster",
            "SBATCH_ACCOUNT": "fallback-project",
            "SBATCH_PARTITION": "fallback-partition",
            "SBATCH_TIMELIMIT": "12:00:00",
            "PRIME_RL_NUM_TRAIN_NODES": "2",
            "PRIME_RL_NUM_INFER_NODES": "3",
            "PRIME_RL_GPUS_PER_NODE": "4",
        },
    )

    assert args == [
        "@",
        "config.toml",
        "--slurm.time",
        "explicit",
        "--slurm.account",
        "project-123",
        "--slurm.partition",
        "booster",
        "--deployment.num-train-nodes",
        "2",
        "--deployment.num-infer-nodes",
        "3",
        "--deployment.gpus-per-node",
        "4",
    ]


def test_prime_rl_explicit_slurm_profile_wins_over_environment():
    args = with_cluster_profile_defaults(
        [
            "@",
            "config.toml",
            "--slurm.account=explicit-account",
            "--slurm.partition",
            "explicit-partition",
        ],
        {
            "PRIME_RL_SLURM_ACCOUNT": "prime-account",
            "PRIME_RL_SLURM_PARTITION": "prime-partition",
            "SBATCH_ACCOUNT": "environment-account",
            "SBATCH_PARTITION": "environment-partition",
        },
    )

    assert args == [
        "@",
        "config.toml",
        "--slurm.account=explicit-account",
        "--slurm.partition",
        "explicit-partition",
    ]


def test_prime_rl_runtime_env_defaults_become_cli_overrides():
    args = with_runtime_env_defaults(
        ["@", "config.toml", "--dry-run"],
        {"OSWORLD_PRIME_RL_OUTPUT_DIR": "/scratch/run/prime_rl"},
    )

    assert args == [
        "@",
        "config.toml",
        "--dry-run",
        "--output-dir",
        "/scratch/run/prime_rl",
    ]


def test_prime_rl_explicit_output_dir_wins_over_runtime_env_default():
    args = with_runtime_env_defaults(
        ["@", "config.toml", "--output-dir", "/explicit/out"],
        {"OSWORLD_PRIME_RL_OUTPUT_DIR": "/scratch/run/prime_rl"},
    )

    assert args == ["@", "config.toml", "--output-dir", "/explicit/out"]


def test_osworld_fleet_registry_helpers_keep_missing_distinct_from_corrupt(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="fleet-a",
        run_base=tmp_path / "runs",
    )
    args = SimpleNamespace(
        registry=None,
        run_id=layout.run_id,
        run_base=layout.run_base,
    )

    assert registry_path_for_args(args) == layout.registry_path
    assert (
        registry_path_for_args(
            SimpleNamespace(
                registry=str(tmp_path / "custom-registry.json"),
                run_id=layout.run_id,
                run_base=layout.run_base,
            )
        )
        == tmp_path / "custom-registry.json"
    )
    assert read_registry_optional(layout.registry_path) is None

    registry, error = read_registry_for_status(layout.registry_path)
    assert registry is None
    assert error == f"missing: {layout.registry_path}"

    layout.registry_path.parent.mkdir(parents=True)
    layout.registry_path.write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_registry_optional(layout.registry_path)

    registry, error = read_registry_for_status(layout.registry_path)
    assert registry is None
    assert error is not None
    assert str(layout.registry_path) in error


def test_registry_based_prime_rl_configuration():
    config = Path("configs/prime_rl/multi_node.toml").read_text(encoding="utf-8")
    single_node_config = Path("configs/prime_rl/single_node.toml").read_text(
        encoding="utf-8"
    )

    assert 'taskset = { id = "rl"' in config
    assert "timeout = { rollout = 1000 }" in config
    assert "desktop_pool_config = { min_ready_sessions = 0 }" in config
    assert "project_dir" not in config
    assert "image_cache_max = 5" in config
    assert "template_path" not in config
    assert (
        'template_path = "sbatch/prime_rl_single_node.sbatch.j2"' in single_node_config
    )
    assert "partition =" not in config
    assert "pre_run_command" not in config
    assert "\nmax_inflight_rollouts =" in config

    assert not Path("sbatch/single_node_rl_osworld.sbatch.j2").exists()
    assert not Path("sbatch/multi_node_rl_osworld.sbatch.j2").exists()
    assert Path("sbatch/prime_rl_single_node.sbatch.j2").is_file()
    assert Path(
        "deps/prime-rl/src/prime_rl/templates/single_node_rl.sbatch.j2"
    ).is_file()
    assert Path(
        "deps/prime-rl/src/prime_rl/templates/multi_node_rl.sbatch.j2"
    ).is_file()


def test_supervisor_gateway_script_path_is_repo_local():
    assert (
        gateway_script_path() == (Path("scripts") / "zmq_rollout_gateway.py").resolve()
    )


def test_osworld_fleet_sbatch_lets_python_choose_public_host():
    script = Path("sbatch/run_osworld_env_fleet.sbatch").read_text(encoding="utf-8")

    assert 'HOST="$(hostname -f 2>/dev/null || hostname)"' not in script
    assert '--host "$HOST"' not in script
    assert "scripts/prepare_env_fleet.py \\" in script
    assert '--replica-count "$TOTAL_REPLICAS" \\' in script
    assert '--replica-offset "$REPLICA_OFFSET"' in script
    assert "scripts/supervise_osworld_env_fleet.py" in script
    assert "--start-gateway" in script
    assert "load_runtime_env_file" not in script
    assert "RL_RUNTIME_ENV_FILE" not in script
    assert "export RL_CLUSTER_WORKLOAD=osworld" in script
    assert 'source "$CLUSTER_SETUP_SCRIPT"' in script
    assert 'export OSWORLD_ENV_MAX_RETRIES="${OSWORLD_ENV_MAX_RETRIES:-2}"' in script
    assert "OSWORLD_SUPERVISOR_MAX_FLEET_RESTARTS" in script
    assert "OSWORLD_STATUS_STALE_AFTER_S" in script
    assert "OSWORLD_DESKTOP_POOL_STARTUP_TIMEOUT" in script
    assert "OSWORLD_DESKTOP_POOL_STATUS_HEARTBEAT_INTERVAL" in script
    assert 'srun "${SRUN_SCOPE[@]}" \\' in script
    assert '--nodes="$OSWORLD_FLEET_ALLOCATED_NODES" \\' in script
    assert '--ntasks="$OSWORLD_FLEET_ALLOCATED_NODES" \\' in script
    assert 'SRUN_SCOPE+=("--het-group=$OSWORLD_SLURM_HET_GROUP")' in script
    assert "--cpu-bind" not in script
    assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"' in script
    assert 'export SCRATCH="${SCRATCH:-$ROOT_DIR/.scratch}"' in script
    assert 'export OSWORLD_RUN_BASE="${OSWORLD_RUN_BASE:-$SCRATCH}"' in script
    expected_pool_root_export = (
        'export OSWORLD_DESKTOP_POOL_ROOT="${OSWORLD_DESKTOP_POOL_ROOT:-'
        '$OSWORLD_RUN_BASE/$OSWORLD_FLEET_RUN_ID/pool}"'
    )
    assert expected_pool_root_export in script
    assert (
        'OSWORLD_DESKTOP_POOL_TMP_ROOT="${TMPDIR:?TMPDIR must be set by Slurm}/osworld-${SLURM_JOB_ID:-${USER:-user}-${OSWORLD_FLEET_RUN_ID:-manual}}"'
        in script
    )
    assert (
        'export OSWORLD_DESKTOP_POOL_RUNTIME_DIR="${OSWORLD_DESKTOP_POOL_RUNTIME_DIR:-$OSWORLD_DESKTOP_POOL_TMP_ROOT/runtime}"'
        in script
    )
    assert (
        'export OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR="${OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR:-$OSWORLD_DESKTOP_POOL_TMP_ROOT/log}"'
        in script
    )
    assert "${TMPDIR:?TMPDIR must be set by Slurm}/runtime" not in script
    assert "${TMPDIR:?TMPDIR must be set by Slurm}/log" not in script
    assert "ln -sfnT" not in script
    assert "desktop_runtime/node_" not in script
    assert script.index('mkdir -p "$OSWORLD_DESKTOP_POOL_RUNTIME_DIR"') < script.index(
        '"$PYTHON" scripts/prepare_env_fleet.py'
    )
    assert 'export OSWORLD_MAX_TASKS="${OSWORLD_MAX_TASKS:-0}"' in script
    assert 'PRIME_RL_DIR="${PRIME_RL_DIR:-$ROOT_DIR/deps/prime-rl}"' in script
    assert "personalize_project_scratch_env" not in script


def test_sbatch_cluster_workload_hints_are_set_before_cluster_setup():
    osworld_scripts = [Path("sbatch/run_osworld_env_fleet.sbatch")]
    for script_path in osworld_scripts:
        script = script_path.read_text(encoding="utf-8")
        assert script.index("export RL_CLUSTER_WORKLOAD=osworld") < script.index(
            'source "$CLUSTER_SETUP_SCRIPT"'
        )


def test_sbatch_scripts_do_not_load_runtime_env_file():
    script_paths = [Path("sbatch/run_osworld_env_fleet.sbatch")]

    for script_path in script_paths:
        script = script_path.read_text(encoding="utf-8")
        assert "load_runtime_env_file" not in script
        assert "RL_RUNTIME_ENV_FILE" not in script
        assert 'source "$env_file"' not in script


def test_juwels_cluster_profile_restores_workload_specific_modules():
    profile = Path("sbatch/clusters/juwels.sh").read_text(encoding="utf-8")

    assert 'case "${RL_CLUSTER_WORKLOAD:-}" in' in profile
    assert "prime_rl)" in profile
    assert "module load Stages/2025 CUDA/12 GCC/13.3.0" in profile
    assert "osworld)" in profile
    assert "module load Stages/2026" in profile
    assert 'if [[ -n "${JUTIL_PROJECT:-}" ]]; then' in profile
    assert 'jutil env activate -p "$JUTIL_PROJECT"' in profile
    assert "JUTIL_PROJECT is unset; skipping jutil env activate" in profile
    assert "module list" in profile


def test_haicore_cluster_profile_uses_slurm_interact_port():
    profile = Path("sbatch/clusters/haicore.sh").read_text(encoding="utf-8")

    assert "source /etc/profile.d/slurm.sh" in profile
    assert (
        'export OSWORLD_GATEWAY_PORT="${SLURM_INTERACT_PORT:'
        '?SLURM_INTERACT_PORT was not assigned}"' in profile
    )


def test_tooling_excludes_dependency_submodules():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'exclude = ["deps/**"]' in pyproject
    assert 'extend-exclude = ["deps"]' in pyproject
    assert '"deps/"' in pyproject
    assert 'testpaths = ["tests"]' in pyproject


def test_osworld_fleet_status_format_and_cancel_guards(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="12345",
        run_base=tmp_path / "osworld_rl",
    )
    server = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=2,
        replica_count=1,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=layout.configs_dir,
        log_dir=layout.logs_dir,
    )[0]
    registry = upsert_registry(
        path=layout.registry_path,
        run_id="12345",
        metadata={"layout": layout.as_metadata(), "expected_env_servers": 1},
        servers=[server],
    )
    summary = {
        "status_dir": str(layout.pool_status_dir),
        "registry_ready": True,
        "registered_servers": 1,
        "expected_servers": 1,
        "ready": 2,
        "min_ready": 2,
        "starting": 0,
        "leased": 0,
        "stale_status_files": 0,
        "total_failed": 1,
        "retry_scheduled_workers": 1,
        "cooling_down_workers": 1,
        "consecutive_start_failures": 2,
        "startup_cooldown_remaining_s": 12.5,
        "last_errors": [],
        "unhealthy_servers": 0,
        "server_summaries": [
            {
                "name": "osworld-0000",
                "ready": 2,
                "starting": 0,
                "leased": 0,
                "stale_status_files": 0,
                "total_failed": 1,
            }
        ],
    }
    job = SlurmJob(
        job_id="12345",
        user="user-a",
        name="osworld_env_fleet",
        state="RUNNING",
        reason="None",
        elapsed="1:00",
        nodes="1",
        cpus="16",
    )

    report = format_status_report(layout, registry, None, summary, [job])

    assert "Slurm: 12345 RUNNING" in report
    assert "ready=2/2" in report
    assert "ready=2 starting=0 leased=0 stale_status=0 failed=1" in report
    assert "Startup retry: scheduled_workers=1" in report
    assert "cooldown_remaining_s=12.5" in report
    assert "tcp://node001:5200" in report
    assert parse_squeue("12345|user-a|osworld_env_fleet|RUNNING|None|1:00|1|16|\n") == [
        job
    ]
    assert select_cancel_job([job], user="user-a", job_name="osworld_env_fleet") == job
    assert confirm_cancel(job, yes=False, input_fn=lambda _prompt: "n") is False
    assert confirm_cancel(job, yes=True) is True


def test_toml_literal_renders_nested_inline_tables():
    rendered = toml_literal({"config": {"taskset": {"base_path": "/tasks"}}})

    assert rendered == '{ config = { taskset = { base_path = "/tasks" } } }'


def test_path_helpers_default_to_repo_scratch_and_pinned_osworld(monkeypatch):
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.setenv("OSWORLD_ROOT", "/ignored/OSWorldRL")
    monkeypatch.delenv("OSWORLD_QCOW_PATH", raising=False)
    assert scratch_root({}) == repo_root() / ".scratch"
    assert (
        scratch_subdir("osworld_rl", env={}) == repo_root() / ".scratch" / "osworld_rl"
    )
    assert osworld_root() == repo_root() / "deps" / "OSWorldRL"
    assert prime_rl_root() == repo_root() / "deps" / "prime-rl"
    assert (
        osworld_apptainer_image()
        == repo_root() / "apptainer" / "images" / "osworld.sif"
    )
    assert osworld_qcow_path({}) == (
        repo_root().parent / "osworld_deployment" / "Ubuntu.qcow2"
    )
    assert osworld_asset_cache_dir({}) == (
        repo_root() / ".scratch" / "osworld_asset_cache"
    )


def test_path_helpers_honor_runtime_overrides(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_QCOW_PATH": str(tmp_path / "images" / "custom.qcow2"),
    }

    assert scratch_root(env) == tmp_path / "scratch"
    assert scratch_subdir("osworld_rl", env=env) == tmp_path / "scratch" / "osworld_rl"
    assert osworld_root() == repo_root() / "deps" / "OSWorldRL"
    assert (
        osworld_apptainer_image()
        == repo_root() / "apptainer" / "images" / "osworld.sif"
    )
    assert osworld_qcow_path(env) == tmp_path / "images" / "custom.qcow2"
    assert osworld_asset_cache_dir(env) == tmp_path / "scratch" / "osworld_asset_cache"


@pytest.mark.parametrize(
    ("name", "helper"),
    [
        ("SCRATCH", scratch_root),
        ("OSWORLD_QCOW_PATH", osworld_qcow_path),
    ],
)
def test_path_helpers_reject_non_absolute_runtime_overrides(name, helper):
    with pytest.raises(ValueError, match=f"{name} must be an absolute path"):
        helper({name: "~/not-expanded"})

    with pytest.raises(ValueError, match=f"{name} must be an absolute path"):
        helper({name: "relative/path"})

    with pytest.raises(ValueError, match=f"{name} must be an absolute path"):
        helper({name: f"${name}/path"})

    with pytest.raises(ValueError, match=f"{name} must be an absolute path"):
        helper({name: f"${{{name}}}/path"})


def test_fleet_run_layout_rejects_non_absolute_environment_paths(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_FLEET_RUN_ID": "run-a",
        "OSWORLD_RUN_BASE": "relative/run-base",
    }

    with pytest.raises(ValueError, match="OSWORLD_RUN_BASE must be an absolute path"):
        FleetRunLayout.from_env(env)

    env["OSWORLD_RUN_BASE"] = str(tmp_path / "run-base")
    env["OSWORLD_ENV_FLEET_REGISTRY"] = "${SCRATCH}/registry.json"

    with pytest.raises(
        ValueError,
        match="OSWORLD_ENV_FLEET_REGISTRY must be an absolute path",
    ):
        FleetRunLayout.from_env(env)


def test_runtime_env_file_rejects_non_absolute_path_overrides():
    with pytest.raises(ValueError, match="runtime env file path must be an absolute"):
        load_runtime_env_file(path="runtime.env")

    with pytest.raises(ValueError, match="RL_RUNTIME_ENV_FILE must be an absolute"):
        load_runtime_env_file(env={"RL_RUNTIME_ENV_FILE": "${SCRATCH}/runtime.env"})


def test_runtime_env_file_loads_shell_style_assignments(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# local runtime settings",
                "PROJECT_ROOT=/tmp/project",
                "export SCRATCH=/tmp/scratch",
                'HF_HOME="${SCRATCH}/huggingface"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    env: dict[str, str] = {}

    loaded = load_runtime_env_file(env=env, path=env_file)

    assert loaded == env_file
    assert env == {
        "PROJECT_ROOT": "/tmp/project",
        "SCRATCH": "/tmp/scratch",
        "HF_HOME": "/tmp/scratch/huggingface",
    }


def test_runtime_env_file_preserves_existing_environment_by_default(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SETTING=from-file\n", encoding="utf-8")
    env = {"SETTING": "from-environment"}

    assert load_runtime_env_file(env=env, path=env_file) == env_file
    assert env["SETTING"] == "from-environment"


def test_runtime_env_file_uses_python_dotenv_interpolation(monkeypatch, tmp_path):
    monkeypatch.delenv("SCRATCH", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('HF_HOME="${SCRATCH}/huggingface"\n', encoding="utf-8")
    env: dict[str, str] = {}

    assert load_runtime_env_file(env=env, path=env_file) == env_file
    assert env == {"HF_HOME": "/huggingface"}


def test_runtime_env_file_can_be_disabled():
    env = {"RL_RUNTIME_ENV_FILE": ""}

    assert load_runtime_env_file(env=env) is None


def test_runtime_env_file_rejects_invalid_lines(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("not-an-assignment\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected KEY=value"):
        load_runtime_env_file(path=env_file)
