#!/usr/bin/env python3
"""Download and validate the DOTA v1.5 dataset from OpenDataLab."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DATASET_REPOSITORY = "OpenDataLab/DOTA_V1_dot_5"
_NESTED_LABEL_DIRECTORIES = {
    "train": "DOTA-v1.5_train",
    "val": "DOTA-v1.5_val",
}


@dataclass(frozen=True)
class DatasetStatus:
    """The raw DOTA files needed to prepare training data."""

    ready: bool
    trainimages: int
    trainlabels: int
    valimages: int
    errors: tuple[str, ...]


def _count_files(directory: Path, suffix: str | None = None) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        path.is_file() and (suffix is None or path.suffix == suffix)
        for path in directory.iterdir()
    )


def verify_dataset(dataset_dir: Path | str = "dataset/DOTA") -> DatasetStatus:
    """Report whether DOTA has the raw files required for preparation.

    Official DOTA v1.5 validation labels are optional, so readiness depends only
    on train images, train labels, and validation images.
    """

    dataset_path = Path(dataset_dir)
    trainimages = _count_files(dataset_path / "train" / "images")
    trainlabels = _count_files(dataset_path / "train" / "labelTxt", ".txt")
    valimages = _count_files(dataset_path / "val" / "images")
    errors: list[str] = []
    if trainimages == 0:
        errors.append("no training images found in train/images")
    if trainlabels == 0:
        errors.append("no training labels found in train/labelTxt")
    if valimages == 0:
        errors.append("no validation images found in val/images")
    return DatasetStatus(
        ready=trainimages > 0 and trainlabels > 0 and valimages > 0,
        trainimages=trainimages,
        trainlabels=trainlabels,
        valimages=valimages,
        errors=tuple(errors),
    )


def normalize_download_layout(dataset_dir: Path | str = "dataset/DOTA") -> DatasetStatus:
    """Flatten recognized OpenDataLab label directories without data loss."""

    dataset_path = Path(dataset_dir)
    for split, nested_name in _NESTED_LABEL_DIRECTORIES.items():
        label_directory = dataset_path / split / "labelTxt"
        nested_directory = label_directory / nested_name
        if not nested_directory.is_dir():
            continue

        label_files = [
            source
            for source in nested_directory.iterdir()
            if source.is_file() and source.suffix == ".txt"
        ]
        for source in label_files:
            destination = label_directory / source.name
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise FileExistsError(
                    f"refusing to overwrite distinct label: {destination}"
                )

        for source in label_files:
            destination = label_directory / source.name
            if destination.exists():
                source.unlink()
            else:
                source.rename(destination)
        if not any(nested_directory.iterdir()):
            nested_directory.rmdir()

    (dataset_path / "val" / "labelTxt").mkdir(parents=True, exist_ok=True)
    return verify_dataset(dataset_path)


def _resolve_openxlab_download() -> Callable[..., object]:
    try:
        from openxlab.dataset import download
    except ImportError as error:
        raise RuntimeError(
            "OpenDataLab SDK is unavailable; install the cloud extra before downloading."
        ) from error
    return download


def download_dota_sdk(
    target_dir: Path | str = "dataset/DOTA", download_fn: Callable[..., object] | None = None
) -> bool:
    """Download DOTA v1.5 through the SDK, returning whether the request succeeded."""

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    download = _resolve_openxlab_download() if download_fn is None else download_fn
    try:
        download(
            dataset_repo=DATASET_REPOSITORY,
            source_path="",
            target_path=str(target_path),
        )
    except Exception as error:
        print(f"DOTA download failed: {error}")
        return False
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "dataset/DOTA"
    if not download_dota_sdk(target):
        sys.exit(1)

    status = normalize_download_layout(target)
    if status.ready:
        print("Dataset ready for splitting.")
        sys.exit(0)

    print("Dataset structure incomplete:")
    for error in status.errors:
        print(f"- {error}")
    sys.exit(1)
