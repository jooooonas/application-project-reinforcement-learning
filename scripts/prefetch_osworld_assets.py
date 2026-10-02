#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from rl.runtime.envfile import load_runtime_env_file
from rl.runtime.paths import (
    osworld_asset_cache_dir,
    osworld_task_base_path,
)

DEFAULT_REPO_PREFIX = (
    "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/"
)


def main() -> int:
    args = parse_args()
    task_paths = list(iter_task_paths(args.tasks))
    if args.max_tasks is not None:
        task_paths = task_paths[: args.max_tasks]

    url_targets: dict[str, set[Path]] = {}
    for task_path in task_paths:
        task = load_json(task_path)
        task_id = str(task.get("id") or task_path.stem)
        task_cache = args.cache_dir / task_id
        for url, target_name in collect_osworld_cache_targets(task):
            url_targets.setdefault(url, set()).add(task_cache / target_name)

    print(
        f"Found {len(url_targets)} unique asset URL(s) "
        f"for {len(task_paths)} task JSON(s)."
    )
    if args.dry_run:
        for url, targets in sorted(url_targets.items()):
            print(url)
            for target in sorted(targets):
                print(f"  -> {target}")
        return 0

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root.resolve() if args.source_root else None
    copied = 0
    skipped = 0

    for url, targets in sorted(url_targets.items()):
        target_list = sorted(targets)
        missing = [target for target in target_list if not target.exists()]
        if not missing:
            skipped += len(target_list)
            continue

        existing_source = next(
            (target for target in target_list if target.exists()),
            None,
        )
        if existing_source is None:
            first_target = missing.pop(0)
            first_target.parent.mkdir(parents=True, exist_ok=True)
            local_source = (
                local_source_for_url(
                    url,
                    source_root=source_root,
                    repo_prefix=args.repo_prefix,
                )
                if source_root is not None
                else None
            )
            if local_source is not None:
                shutil.copy2(local_source, first_target)
                print(f"Copied {local_source} -> {first_target}")
            else:
                download_url(url, first_target)
                print(f"Downloaded {url} -> {first_target}")
            existing_source = first_target
            copied += 1

        for target in missing:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing_source, target)
            print(f"Copied {existing_source} -> {target}")
            copied += 1

    print(f"Done. Created {copied} file(s); skipped {skipped} existing file(s).")
    return 0


def parse_args() -> argparse.Namespace:
    load_runtime_env_file()
    env = os.environ
    parser = argparse.ArgumentParser(
        description=(
            "Populate an OSWorld DesktopEnv cache from task JSONs so compute "
            "nodes do not need to download Hugging Face task assets."
        )
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=default_task_base_path(),
        help="Task JSON file or directory containing OSWorld task JSONs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=osworld_asset_cache_dir(env=env),
        help="Destination cache_dir passed to OSWorld DesktopEnv.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Optional local mirror root containing paths such as "
            "gimp/<task-id>/computer.png. If omitted, assets are downloaded."
        ),
    )
    parser.add_argument(
        "--repo-prefix",
        default=DEFAULT_REPO_PREFIX,
        help="URL prefix to strip when resolving --source-root paths.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Only process the first N sorted task JSONs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print required cache targets without copying or downloading.",
    )
    return parser.parse_args()


def default_task_base_path() -> Path:
    return osworld_task_base_path()


def iter_task_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"Task path does not exist: {path}")
    yield from sorted(
        candidate for candidate in path.rglob("*.json") if candidate.is_file()
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def collect_osworld_cache_targets(task: dict[str, Any]) -> Iterable[tuple[str, Path]]:
    for step in task.get("config", []):
        if not isinstance(step, dict) or step.get("type") != "download":
            continue
        params = step.get("parameters") or {}
        for file_spec in params.get("files", []):
            if not isinstance(file_spec, dict):
                continue
            url = file_spec.get("url")
            vm_path = file_spec.get("path")
            if isinstance(url, str) and isinstance(vm_path, str):
                yield url, Path(setup_cache_filename(url, vm_path))

    yield from collect_cloud_file_targets(task.get("evaluator", {}))


def setup_cache_filename(url: str, vm_path: str) -> str:
    return f"{uuid.uuid5(uuid.NAMESPACE_URL, url)}_{Path(vm_path).name}"


def collect_cloud_file_targets(obj: Any) -> Iterable[tuple[str, Path]]:
    if isinstance(obj, dict):
        if obj.get("type") == "cloud_file" and "path" in obj and "dest" in obj:
            paths = as_list(obj["path"])
            dests = as_list(obj["dest"])
            if len(paths) != len(dests):
                raise ValueError(f"cloud_file path/dest length mismatch: {obj}")
            for url, dest in zip(paths, dests, strict=True):
                if isinstance(url, str) and isinstance(dest, str):
                    yield url, Path(dest)
        for value in obj.values():
            yield from collect_cloud_file_targets(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from collect_cloud_file_targets(item)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def local_source_for_url(
    url: str,
    *,
    source_root: Path,
    repo_prefix: str,
) -> Path | None:
    stripped = strip_query(url)
    if not stripped.startswith(repo_prefix):
        return None
    relative = unquote(stripped[len(repo_prefix) :])
    candidate = (source_root / relative).resolve()
    try:
        candidate.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            f"Resolved source path escapes source root: {candidate}"
        ) from exc
    if not candidate.exists():
        raise FileNotFoundError(f"Local source for {url} was not found at {candidate}")
    return candidate


def strip_query(url: str) -> str:
    parts = urlsplit(url)
    return parts._replace(query="", fragment="").geturl()


def download_url(url: str, target: Path) -> None:
    request = Request(
        url,
        headers={"User-Agent": "reinforcement-learning-osworld-prefetch"},
    )
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as output:
            with urlopen(request, timeout=300) as response:
                shutil.copyfileobj(response, output)
        Path(tmp_name).replace(target)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
