from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.install_osworld_sif_artifact import install_artifact


def test_install_osworld_sif_artifact_extracts_and_verifies(tmp_path):
    image_bytes = b"fake sif bytes"
    artifact = tmp_path / "osworld-sif.zip"
    inspected: list[tuple[str, bytes]] = []

    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("osworld.sif", image_bytes)

    def inspect(path: Path) -> None:
        inspected.append((path.name, path.read_bytes()))

    installed = install_artifact(
        artifact,
        image_dir=tmp_path / "images",
        inspect_runner=inspect,
    )

    assert installed.image_path.read_bytes() == image_bytes
    assert inspected == [("osworld.sif", image_bytes)]


def test_install_osworld_sif_artifact_requires_image(tmp_path):
    artifact = tmp_path / "osworld-sif.zip"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("readme.txt", "not the artifact")

    with pytest.raises(FileNotFoundError, match="osworld.sif"):
        install_artifact(
            artifact,
            image_dir=tmp_path / "images",
            inspect_runner=lambda _path: None,
        )
