from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from rl.runtime.paths import require_absolute_path, scratch_root, slurm_run_id

# Keep the shared desktop-pool path component short as a fallback; QEMU AF_UNIX
# socket paths should use DesktopPoolConfig.runtime_dir when available.
DEFAULT_DESKTOP_POOL_DIR = "pool"


@dataclass(frozen=True)
class FleetRunLayout:
    run_id: str
    run_base: Path
    run_root: Path
    registry_path: Path
    pool_root: Path
    pool_status_dir: Path
    logs_dir: Path
    configs_dir: Path
    prime_rl_config_path: Path
    prime_rl_output_dir: Path

    @classmethod
    def for_run(
        cls,
        *,
        run_id: str,
        run_base: str | Path,
        run_root: str | Path | None = None,
        registry_path: str | Path | None = None,
        pool_root: str | Path | None = None,
        pool_status_dir: str | Path | None = None,
        logs_dir: str | Path | None = None,
        configs_dir: str | Path | None = None,
        prime_rl_config_path: str | Path | None = None,
        prime_rl_output_dir: str | Path | None = None,
    ) -> Self:
        resolved_run_id = str(run_id)
        resolved_run_base = require_absolute_path(run_base, name="run_base")
        resolved_run_root = Path(
            run_root or resolved_run_base / resolved_run_id / "env_fleet"
        )
        resolved_pool_root = Path(
            pool_root or resolved_run_base / resolved_run_id / DEFAULT_DESKTOP_POOL_DIR
        )
        run_dir = resolved_run_base / resolved_run_id
        return cls(
            run_id=resolved_run_id,
            run_base=resolved_run_base,
            run_root=require_absolute_path(resolved_run_root, name="run_root"),
            registry_path=require_absolute_path(
                registry_path or resolved_run_root / "env_registry.json",
                name="registry_path",
            ),
            pool_root=require_absolute_path(resolved_pool_root, name="pool_root"),
            pool_status_dir=require_absolute_path(
                pool_status_dir or resolved_pool_root / "status",
                name="pool_status_dir",
            ),
            logs_dir=require_absolute_path(
                logs_dir or resolved_run_root / "logs",
                name="logs_dir",
            ),
            configs_dir=require_absolute_path(
                configs_dir or resolved_run_root / "configs",
                name="configs_dir",
            ),
            prime_rl_config_path=require_absolute_path(
                prime_rl_config_path or run_dir / "prime_rl_fleet.toml",
                name="prime_rl_config_path",
            ),
            prime_rl_output_dir=require_absolute_path(
                prime_rl_output_dir or run_dir / "prime_rl",
                name="prime_rl_output_dir",
            ),
        )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] = os.environ,
        *,
        run_id: str | None = None,
        run_base: str | Path | None = None,
    ) -> Self:
        resolved_run_id = run_id or env.get("OSWORLD_FLEET_RUN_ID") or slurm_run_id(env)
        if run_base is not None:
            resolved_run_base = require_absolute_path(run_base, name="run_base")
        elif env.get("OSWORLD_RUN_BASE"):
            resolved_run_base = require_absolute_path(
                env["OSWORLD_RUN_BASE"],
                name="OSWORLD_RUN_BASE",
            )
        else:
            resolved_run_base = scratch_root(env)
        return cls.for_run(
            run_id=resolved_run_id,
            run_base=resolved_run_base,
            run_root=_env_scratch_path(env, "OSWORLD_FLEET_RUN_ROOT"),
            registry_path=_env_scratch_path(
                env,
                "OSWORLD_ENV_FLEET_REGISTRY",
            ),
            pool_root=_env_scratch_path(env, "OSWORLD_DESKTOP_POOL_ROOT"),
            pool_status_dir=_env_scratch_path(env, "OSWORLD_DESKTOP_POOL_STATUS_DIR"),
            logs_dir=_env_scratch_path(env, "OSWORLD_FLEET_LOGS_DIR"),
            configs_dir=_env_scratch_path(env, "OSWORLD_FLEET_CONFIGS_DIR"),
            prime_rl_config_path=_env_scratch_path(
                env,
                "OSWORLD_PRIME_RL_CONFIG_PATH",
            ),
            prime_rl_output_dir=_env_scratch_path(
                env,
                "OSWORLD_PRIME_RL_OUTPUT_DIR",
            ),
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        fallback: Self | None = None,
    ) -> Self | None:
        layout = metadata.get("layout")
        if not isinstance(layout, Mapping):
            return fallback

        def _value(name: str, default: object | None = None) -> object | None:
            return layout.get(name, default)

        run_id = _value("run_id", fallback.run_id if fallback else None)
        run_base = _value("run_base", fallback.run_base if fallback else None)
        if run_id is None or run_base is None:
            return fallback
        return cls.for_run(
            run_id=str(run_id),
            run_base=Path(str(run_base)),
            run_root=_path_value(_value("run_root")),
            registry_path=_path_value(_value("registry_path")),
            pool_root=_path_value(_value("pool_root")),
            pool_status_dir=_path_value(_value("pool_status_dir")),
            logs_dir=_path_value(_value("logs_dir")),
            configs_dir=_path_value(_value("configs_dir")),
            prime_rl_config_path=_path_value(_value("prime_rl_config_path")),
            prime_rl_output_dir=_path_value(_value("prime_rl_output_dir")),
        )

    def as_metadata(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "run_base": str(self.run_base),
            "run_root": str(self.run_root),
            "registry_path": str(self.registry_path),
            "pool_root": str(self.pool_root),
            "pool_status_dir": str(self.pool_status_dir),
            "logs_dir": str(self.logs_dir),
            "configs_dir": str(self.configs_dir),
            "prime_rl_config_path": str(self.prime_rl_config_path),
            "prime_rl_output_dir": str(self.prime_rl_output_dir),
        }

    def node_configs_dir(self, node_rank: int) -> Path:
        return self.configs_dir / f"node_{node_rank:04d}"

    def node_logs_dir(self, node_rank: int) -> Path:
        return self.logs_dir / f"node_{node_rank:04d}"


def _path_value(value: object | None) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _env_scratch_path(env: Mapping[str, str], *names: str) -> Path | None:
    for name in names:
        value = env.get(name)
        if value:
            return require_absolute_path(value, name=name)
    return None
