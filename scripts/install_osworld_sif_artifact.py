#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

IMAGE_NAME = "osworld.sif"


@dataclass(frozen=True)
class InstalledArtifact:
    image_path: Path


def default_image_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "apptainer" / "images"


def install_artifact(
    artifact_zip: Path,
    *,
    image_dir: Path | None = None,
    inspect_runner: Callable[[Path], None] | None = None,
) -> InstalledArtifact:
    target_dir = image_dir or default_image_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    inspect = inspect_runner or inspect_image

    with tempfile.TemporaryDirectory(prefix=".osworld-sif-", dir=target_dir) as tmp:
        tmp_dir = Path(tmp)
        tmp_image = tmp_dir / IMAGE_NAME

        with zipfile.ZipFile(artifact_zip) as archive:
            extract_unique_member(archive, IMAGE_NAME, tmp_image)

        inspect(tmp_image)

        target_image = target_dir / IMAGE_NAME
        tmp_image.replace(target_image)

    return InstalledArtifact(
        image_path=target_dir / IMAGE_NAME,
    )


def extract_unique_member(
    archive: zipfile.ZipFile,
    member_name: str,
    destination: Path,
) -> None:
    matches = [
        info
        for info in archive.infolist()
        if not info.is_dir() and Path(info.filename).name == member_name
    ]
    if not matches:
        raise FileNotFoundError(f"{member_name} was not found in {archive.filename}")
    if len(matches) > 1:
        names = ", ".join(info.filename for info in matches)
        raise ValueError(
            f"{archive.filename} contains multiple {member_name} files: {names}"
        )

    with archive.open(matches[0]) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def inspect_image(path: Path) -> None:
    subprocess.run(["apptainer", "inspect", str(path)], check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install a downloaded osworld-sif GitHub Actions artifact."
    )
    parser.add_argument("artifact_zip", type=Path)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir(),
        help="Destination directory for osworld.sif.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    installed = install_artifact(args.artifact_zip, image_dir=args.image_dir)
    print(f"installed image: {installed.image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
