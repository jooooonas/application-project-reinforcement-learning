from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from prime_rl.configs.env_server import EnvServerConfig
from prime_rl.configs.rl import RLConfig
from verifiers.v1.serve.pool import env_config_data

from rl.runtime.fleet import make_server_specs
from rl.runtime.fleet.config_rendering import write_env_server_config
from scripts.render_prime_rl_fleet_config import configure_external_fleet, load_config


def _metadata(tmp_path: Path) -> dict[str, object]:
    return {
        "env_id": "rl",
        "env_name_prefix": "osworld-target-box",
        "task_base_path": str(tmp_path / "tasks"),
        "max_tasks": 4,
        "shuffle_seed": 7,
        "gateway": {"public_address": "tcp://node001:5300"},
        "harness": {
            "max_steps": 4,
            "desktop": {
                "qcow_path": str(tmp_path / "Ubuntu.qcow2"),
                "cache_dir": str(tmp_path / "cache"),
                "output_dir": str(tmp_path / "output"),
                "desktop_pool_config": {
                    "min_ready_sessions": 1,
                    "max_sessions": 2,
                    "root_dir": str(tmp_path / "pool"),
                },
            },
        },
    }


@pytest.mark.parametrize(
    ("path", "template_path"),
    [
        (
            Path("configs/prime_rl/single_node.toml"),
            Path("sbatch/prime_rl_single_node.sbatch.j2"),
        ),
        (
            Path("configs/prime_rl/multi_node.toml"),
            Path("sbatch/prime_rl_multi_node/rl.sbatch.j2"),
        ),
    ],
)
def test_base_prime_rl_configs_validate(path: Path, template_path: Path | None) -> None:
    config = RLConfig.model_validate(load_config(path))

    assert config.slurm is not None
    assert config.slurm.template_path == template_path
    assert "partition" not in config.slurm.model_fields_set
    assert config.slurm.pre_run_command is None
    assert config.inference is not None
    assert config.inference.server.advertise is (
        config.deployment.type == "single_node"
    )
    assert config.trainer.model.name == "Qwen/Qwen3.5-9B"
    assert config.trainer.model.impl == "custom"
    assert config.orchestrator.renderer.name == "qwen3.5"


def test_generated_trainer_config_validates(tmp_path: Path) -> None:
    config = load_config(Path("configs/prime_rl/single_node.toml"))
    configure_external_fleet(
        config,
        metadata=_metadata(tmp_path),
        output_dir=tmp_path / "trainer",
        max_inflight_rollouts=8,
        rollout_timeout=900,
        max_retries=2,
    )

    parsed = RLConfig.model_validate(config)

    env = parsed.orchestrator.train.env[0]
    assert parsed.orchestrator.max_inflight_rollouts == 16
    assert env.address == "tcp://node001:5300"
    assert env.taskset.id == "rl"
    assert env.harness.id == "rl"
    assert env.timeout.rollout == 900
    assert env.retries.rollout.max_retries == 2
    assert env.max_turns == 4
    assert parsed.orchestrator.renderer.image_cache_max == 5
    assert parsed.slurm is not None
    assert parsed.slurm.template_path == (
        Path("sbatch/prime_rl_single_node.sbatch.j2").resolve()
    )


def test_single_node_slurm_template_only_overrides_shared_resources() -> None:
    custom = Path("sbatch/prime_rl_single_node.sbatch.j2").read_text(encoding="utf-8")
    upstream = Path(
        "deps/prime-rl/src/prime_rl/templates/single_node_rl.sbatch.j2"
    ).read_text(encoding="utf-8")
    expected = upstream.replace(
        "#SBATCH --gres=gpu:{{ gpus_per_node }}\n#SBATCH --exclusive\n",
        "#SBATCH --gres=gpu:{{ gpus_per_node }}\n"
        "#SBATCH --cpus-per-task=16\n"
        "#SBATCH --mem=128G\n",
    )

    assert custom == expected


def test_multi_node_slurm_template_only_has_documented_local_overrides() -> None:
    custom_dir = Path("sbatch/prime_rl_multi_node")
    upstream_dir = Path("deps/prime-rl/src/prime_rl/templates")
    custom = (custom_dir / "rl.sbatch.j2").read_text(encoding="utf-8")
    upstream = (upstream_dir / "multi_node_rl.sbatch.j2").read_text(encoding="utf-8")
    expected = upstream.replace(
        "#SBATCH --gres=gpu:{{ gpus_per_node }}\n",
        "#SBATCH --gres=gpu:{{ gpus_per_node }}\n"
        "#SBATCH --cpus-per-task=32\n"
        "#SBATCH --mem=256G\n",
        1,
    )
    expected = expected.replace("#SBATCH --exclusive\n", "", 1)
    expected = expected.replace(
        "set -e\n",
        "set -e\n\n"
        "# The combined OSWorld launcher runs this payload inside a heterogeneous Slurm\n"
        "# allocation. Scope every job step to the PrimeRL component when requested;\n"
        "# standalone jobs leave this array empty and retain the usual srun behavior.\n"
        "SRUN_SCOPE=()\n"
        'if [[ -n "${PRIME_RL_SLURM_HET_GROUP:-}" ]]; then\n'
        "    SRUN_SCOPE+=(\n"
        '        "--het-group=$PRIME_RL_SLURM_HET_GROUP"\n'
        '        "--ntasks=$PRIME_RL_SLURM_NUM_NODES"\n'
        '        "--ntasks-per-node=1"\n'
        "    )\n"
        "fi\n",
        1,
    )
    for command in (
        "srun --ntasks-per-node=1",
        "srun bash -s",
        "srun --kill-on-bad-exit=1",
    ):
        expected = expected.replace(
            command,
            command.replace("srun", 'srun "${SRUN_SCOPE[@]}"', 1),
            1,
        )

    assert custom == expected

    for partial in (
        "_launch_rank.sh.j2",
        "_launch_router.sh.j2",
        "_mooncake_store.sh.j2",
        "llmd/endpoints.yaml.j2",
        "llmd/envoy.yaml.j2",
        "llmd/epp_estimate.yaml.j2",
        "llmd/epp_pd.yaml.j2",
    ):
        assert (custom_dir / partial).read_text(encoding="utf-8") == (
            upstream_dir / partial
        ).read_text(encoding="utf-8")


def test_generated_env_server_config_validates(tmp_path: Path) -> None:
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
        pool_status_root=tmp_path / "status",
    )[0]
    args = SimpleNamespace(
        env_id="rl",
        task_base_path=tmp_path / "tasks",
        max_tasks=4,
        shuffle_seed=7,
        max_steps=4,
        rollout_timeout=900,
        env_max_retries=2,
        run_root=tmp_path / "run",
    )
    write_env_server_config(server, args, _metadata(tmp_path))

    with Path(server.config_path).open("rb") as file:
        parsed = EnvServerConfig.model_validate(tomllib.load(file))

    assert parsed.env.address == "tcp://0.0.0.0:5200"
    assert parsed.env.taskset.id == "rl"
    assert parsed.env.harness.id == "rl"
    assert parsed.env.pool.type == "static"
    assert parsed.env.pool.num_workers == 2
    assert parsed.env.max_turns == 4

    worker_config = env_config_data(parsed.env)
    assert "apptainer_image" not in worker_config["harness"]["desktop"]
    assert vf.EnvConfig.model_validate(worker_config).harness.id == "rl"
