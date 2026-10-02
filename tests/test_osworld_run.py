from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.osworld_run as osworld_run
from rl.runtime.fleet import SlurmJob
from rl.runtime.paths import repo_root
from scripts.osworld_fleet import run_in_allocation as run_fleet_in_allocation
from scripts.osworld_run import (
    PrimeResources,
    build_sbatch_command,
    cancel,
    load_prime_resources,
    parse_sbatch_job_id,
)
from scripts.prime_rl import allocation_runtime_env


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", "")


def prime_resource_args(**overrides):
    values = {
        "base_config": Path("configs/prime_rl/multi_node.toml"),
        "num_train_nodes": None,
        "num_infer_nodes": None,
        "gpus_per_node": None,
        "prime_partition": None,
        "account": None,
        "time": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def submit_args(tmp_path: Path, **overrides):
    values = {
        "script": Path("sbatch/run_osworld.sbatch"),
        "base_config": Path("configs/prime_rl/multi_node.toml"),
        "run_id": "run-a",
        "run_base": tmp_path / "runs",
        "job_name": "osworld_run",
        "fleet_partition": "standard",
        "fleet_nodes": 1,
        "fleet_cpus_per_task": 55,
        "fleet_mem": "512G",
        "prime_cpus_per_task": 32,
        "prime_mem": "256G",
        "inflight_per_worker": 2,
        "ready_timeout_s": 1800.0,
        "clean_output_dir": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_prime_resources_are_derived_from_the_prime_config():
    resources = load_prime_resources(prime_resource_args(), {})

    assert resources.num_train_nodes == 1
    assert resources.num_infer_nodes == 1
    assert resources.num_infer_replicas == 1
    assert resources.gpus_per_node == 2
    assert resources.total_nodes == 2
    assert resources.partition is None
    assert resources.time == "24:00:00"


def test_prime_resource_cli_overrides_match_the_generated_config():
    resources = load_prime_resources(
        prime_resource_args(
            num_train_nodes=2,
            num_infer_nodes=3,
            gpus_per_node=4,
            prime_partition="gpu",
            account="project",
            time="08:00:00",
        ),
        {},
    )

    assert resources.total_nodes == 5
    assert resources.gpus_per_node == 4
    assert resources.partition == "gpu"
    assert resources.account == "project"
    assert resources.time == "08:00:00"


def test_prime_resource_request_does_not_depend_on_a_custom_slurm_template(tmp_path):
    config = tmp_path / "prime.toml"
    config.write_text(
        "\n".join(
            [
                "[deployment]",
                'type = "multi_node"',
                "num_train_nodes = 1",
                "num_infer_nodes = 1",
                "gpus_per_node = 2",
                "[slurm]",
                'time = "12:00:00"',
            ]
        ),
        encoding="utf-8",
    )

    resources = load_prime_resources(prime_resource_args(base_config=config), {})

    assert resources.total_nodes == 2
    assert resources.time == "12:00:00"


def test_combined_sbatch_command_requests_two_heterogeneous_components(tmp_path):
    args = submit_args(tmp_path)
    resources = PrimeResources(
        num_train_nodes=1,
        num_infer_nodes=1,
        num_infer_replicas=1,
        gpus_per_node=2,
        partition="standard",
        account="project",
        time="24:00:00",
    )

    command = build_sbatch_command(args, resources)

    separator = command.index(":")
    fleet_options = command[:separator]
    prime_options = command[separator + 1 : -1]
    assert command[:2] == ["sbatch", "--parsable"]
    assert fleet_options[fleet_options.index("--nodes") + 1] == "1"
    assert fleet_options[fleet_options.index("--cpus-per-task") + 1] == "55"
    assert fleet_options[fleet_options.index("--mem") + 1] == "512G"
    assert prime_options[prime_options.index("--nodes") + 1] == "2"
    assert prime_options[prime_options.index("--gpus-per-node") + 1] == "2"
    assert prime_options[prime_options.index("--cpus-per-task") + 1] == "32"
    exports = next(item for item in fleet_options if item.startswith("--export="))
    assert "OSWORLD_RUN_ID=run-a" in exports
    assert "OSWORLD_RUN_PRIME_NODES=2" in exports
    assert "PRIME_RL_NUM_TRAIN_NODES=1" in exports
    assert "PRIME_RL_NUM_INFER_NODES=1" in exports
    assert "OSWORLD_PRIME_INFLIGHT_PER_WORKER=2" in exports
    assert command[-1] == str(repo_root() / "sbatch" / "run_osworld.sbatch")


def test_combined_job_id_parser_normalizes_component_and_cluster_suffixes():
    assert parse_sbatch_job_id("12345+0;cluster\n") == "12345"


def test_cancel_accepts_all_components_of_one_owned_heterogeneous_job(
    monkeypatch,
    tmp_path,
):
    jobs = [
        SlurmJob("12345+0", "alice", "osworld_run", "RUNNING", "None"),
        SlurmJob("12345+1", "alice", "osworld_run", "RUNNING", "None"),
    ]
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setattr(osworld_run, "query_squeue", lambda **_kwargs: jobs)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = cancel(
        SimpleNamespace(
            registry=tmp_path / "missing.json",
            run_id="12345",
            run_base=tmp_path,
            job_id=None,
            job_name="osworld_run",
            yes=True,
        )
    )

    assert result == 0
    assert commands == [["scancel", "12345"]]


def test_prime_runtime_selects_only_its_heterogeneous_component():
    env = allocation_runtime_env(
        {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NODELIST": "fleet-node",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_JOB_NODELIST_HET_GROUP_1": "gpu[01-02]",
            "SLURM_JOB_NUM_NODES_HET_GROUP_1": "2",
        },
        het_group=1,
        expected_nodes=2,
    )

    assert env["PRIME_RL_SLURM_HET_GROUP"] == "1"
    assert env["PRIME_RL_SLURM_NODELIST"] == "gpu[01-02]"
    assert env["PRIME_RL_SLURM_NUM_NODES"] == "2"
    assert env["SLURM_JOB_NODELIST"] == "gpu[01-02]"
    assert env["SLURM_JOB_NUM_NODES"] == "2"


def test_prime_runtime_rejects_a_mismatched_component_size():
    with pytest.raises(ValueError, match="has 1 nodes; expected 2"):
        allocation_runtime_env(
            {
                "SLURM_JOB_NODELIST_HET_GROUP_1": "gpu01",
                "SLURM_JOB_NUM_NODES_HET_GROUP_1": "1",
            },
            het_group=1,
            expected_nodes=2,
        )


def test_fleet_run_mode_executes_the_existing_payload_in_the_allocation(
    monkeypatch,
    tmp_path,
):
    script = tmp_path / "fleet.sbatch"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_fleet_in_allocation(
        SimpleNamespace(script=script, het_group=0),
    )

    assert result == 0
    assert captured["command"] == ["bash", str(script)]
    assert captured["kwargs"]["env"]["OSWORLD_SLURM_HET_GROUP"] == "0"
    assert captured["kwargs"]["env"]["ROOT_DIR"] == str(repo_root())


def test_component_srun_calls_are_scoped_when_used_by_the_umbrella():
    fleet_script = Path("sbatch/run_osworld_env_fleet.sbatch").read_text(
        encoding="utf-8"
    )
    prime_template = Path("sbatch/prime_rl_multi_node/rl.sbatch.j2").read_text(
        encoding="utf-8"
    )

    fleet_srun_lines = [
        line.strip()
        for line in fleet_script.splitlines()
        if line.lstrip().startswith("srun ")
    ]
    prime_srun_lines = [
        line.strip()
        for line in prime_template.splitlines()
        if line.lstrip().startswith("srun ")
    ]
    assert fleet_srun_lines
    assert len(prime_srun_lines) == 3
    assert all('srun "${SRUN_SCOPE[@]}"' in line for line in fleet_srun_lines)
    assert all('srun "${SRUN_SCOPE[@]}"' in line for line in prime_srun_lines)
    for option in (
        '"--ntasks=$PRIME_RL_SLURM_NUM_NODES"',
        '"--ntasks-per-node=1"',
    ):
        assert prime_template.count(option) == 1
