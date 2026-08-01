#!/usr/bin/env python3
"""Validate and atomically prepare DOTA image patches for training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Callable

from msdyolo.data.dota import DotaObject, parse_dota_label
from msdyolo.data.scripts.download_dota import normalize_download_layout
from msdyolo.data.scripts.split_dota import split_dataset, validate_split_arguments


STATE_FILE = ".msdyolo-split.json"
FORMAT_VERSION = "dota-pixel-v1"


@dataclass(frozen=True)
class SplitStatus:
    """Validation and reuse status for a prepared DOTA split tree."""

    ready: bool
    trainimages: int
    trainlabels: int
    valimages: int
    errors: tuple[str, ...]
    reused: bool = False


def _files(directory: Path, suffix: str | None = None) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and (suffix is None or path.suffix == suffix)
    )


def source_snapshot(dataset_dir: Path) -> dict[str, object]:
    """Return a deterministic size-and-mtime snapshot of raw DOTA inputs."""

    dataset_dir = Path(dataset_dir)
    groups = (
        ("train_images", dataset_dir / "train" / "images", None),
        ("train_labels", dataset_dir / "train" / "labelTxt", ".txt"),
        ("val_images", dataset_dir / "val" / "images", None),
        ("val_labels", dataset_dir / "val" / "labelTxt", ".txt"),
    )
    grouped_files = {name: _files(directory, suffix) for name, directory, suffix in groups}
    source_files = sorted(
        (path for paths in grouped_files.values() for path in paths),
        key=lambda path: path.relative_to(dataset_dir).as_posix(),
    )
    records = []
    for path in source_files:
        stat = path.stat()
        records.append(
            {
                "path": path.relative_to(dataset_dir).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "counts": {
            "train_images": len(grouped_files["train_images"]),
            "train_labels": len(grouped_files["train_labels"]),
            "val_images": len(grouped_files["val_images"]),
        },
        "files": records,
    }


def _object_coordinate_error(obj: DotaObject, path: Path, subsize: int) -> str | None:
    if any(coordinate < 0.0 or coordinate > subsize for coordinate in obj.coordinates):
        return f"{path}: coordinates must be within [0, {subsize}]"
    return None


def validate_split_tree(split_dir: Path, subsize: int) -> SplitStatus:
    """Validate all labels and required directories in a prepared split tree."""

    split_dir = Path(split_dir)
    train_images = _files(split_dir / "train" / "images")
    train_labels = _files(split_dir / "train" / "labelTxt", ".txt")
    val_images_directory = split_dir / "val" / "images"
    val_images = _files(val_images_directory)
    errors: list[str] = []
    if not train_images:
        errors.append(f"no split training images found in {split_dir / 'train' / 'images'}")
    if not train_labels:
        errors.append(f"no split training labels found in {split_dir / 'train' / 'labelTxt'}")
    if not val_images_directory.is_dir():
        errors.append(f"missing split validation image directory: {val_images_directory}")

    train_objects: list[DotaObject] = []
    for label in train_labels:
        try:
            objects = parse_dota_label(label)
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(str(error))
            continue
        train_objects.extend(objects)
        errors.extend(
            error
            for obj in objects
            if (error := _object_coordinate_error(obj, label, subsize)) is not None
        )

    for label in _files(split_dir / "val" / "labelTxt", ".txt"):
        try:
            objects = parse_dota_label(label)
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(str(error))
            continue
        errors.extend(
            error
            for obj in objects
            if (error := _object_coordinate_error(obj, label, subsize)) is not None
        )

    if not train_objects:
        errors.append("split training labels contain no objects")
    elif max(coordinate for obj in train_objects for coordinate in obj.coordinates) <= 1.0:
        errors.append("split training labels do not contain pixel coordinates greater than 1")

    return SplitStatus(
        ready=not errors,
        trainimages=len(train_images),
        trainlabels=len(train_labels),
        valimages=len(val_images),
        errors=tuple(errors),
    )


def _state(dataset_dir: Path, subsize: int, gap: int) -> dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "source": source_snapshot(dataset_dir),
        "subsize": subsize,
        "gap": gap,
    }


def _read_state(marker: Path) -> dict[str, object] | None:
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_state(marker: Path, state: dict[str, object]) -> None:
    marker.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_source_labels(dataset_dir: Path) -> None:
    errors = []
    for label in _files(dataset_dir / "train" / "labelTxt", ".txt"):
        try:
            parse_dota_label(label)
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(str(error))
    if errors:
        raise ValueError("invalid raw training labels:\n" + "\n".join(errors))


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def prepare_dataset(
    dataset_dir,
    split_dir,
    subsize,
    gap,
    num_process,
    force=False,
    splitter: Callable[..., object] = split_dataset,
) -> SplitStatus:
    """Normalize raw layout, then safely reuse or rebuild a validated split."""

    dataset_dir = Path(dataset_dir)
    split_dir = Path(split_dir)
    backup = split_dir.with_name(f".{split_dir.name}-backup")
    if _exists(backup):
        raise RuntimeError(
            f"stale backup exists at {backup}; recover it manually before retrying "
            f"(compare {split_dir} and {backup}, then keep the authoritative tree)"
        )

    validate_split_arguments(subsize, gap, num_process)
    dataset_status = normalize_download_layout(dataset_dir)
    if not dataset_status.ready:
        raise ValueError("invalid raw DOTA dataset: " + "; ".join(dataset_status.errors))
    _validate_source_labels(dataset_dir)
    expected_state = _state(dataset_dir, subsize, gap)

    marker = split_dir / STATE_FILE
    if not force and _read_state(marker) == expected_state:
        existing_status = validate_split_tree(split_dir, subsize)
        if existing_status.ready:
            return replace(existing_status, reused=True)

    split_dir.parent.mkdir(parents=True, exist_ok=True)
    candidate = Path(
        tempfile.mkdtemp(prefix=".split-candidate-", dir=split_dir.parent)
    )
    installed = False
    try:
        splitter(
            dataset_dir / "train" / "images",
            dataset_dir / "train" / "labelTxt",
            candidate / "train",
            subsize,
            gap,
            num_process,
        )
        splitter(
            dataset_dir / "val" / "images",
            dataset_dir / "val" / "labelTxt",
            candidate / "val",
            subsize,
            gap,
            num_process,
        )
        candidate_status = validate_split_tree(candidate, subsize)
        if not candidate_status.ready:
            raise ValueError("invalid prepared DOTA split: " + "; ".join(candidate_status.errors))
        _write_state(candidate / STATE_FILE, expected_state)

        if _exists(split_dir):
            split_dir.rename(backup)
        try:
            candidate.rename(split_dir)
            installed = True
        except Exception:
            if _exists(backup):
                backup.rename(split_dir)
            raise
        else:
            if _exists(backup):
                shutil.rmtree(backup)
        return replace(candidate_status, reused=False)
    finally:
        if not installed and candidate.exists():
            shutil.rmtree(candidate)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and atomically prepare DOTA training patches"
    )
    parser.add_argument("--dataset", default="dataset/DOTA", help="Raw DOTA directory")
    parser.add_argument(
        "--output", default="dataset/DOTA/split", help="Prepared split directory"
    )
    parser.add_argument("--subsize", type=int, default=1024, help="Patch size")
    parser.add_argument("--gap", type=int, default=200, help="Patch overlap")
    parser.add_argument("--num-process", type=int, default=8, help="Worker processes")
    parser.add_argument(
        "--force-resplit", action="store_true", help="Safely rebuild even if state matches"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        status = prepare_dataset(
            args.dataset,
            args.output,
            args.subsize,
            args.gap,
            args.num_process,
            force=args.force_resplit,
        )
        counts = source_snapshot(Path(args.dataset))["counts"]
    except Exception as error:
        print(f"DOTA preparation failed: {error}", file=sys.stderr)
        return 1

    action = "reused" if status.reused else "rebuilt"
    print(
        f"DOTA split {action}: "
        f"source train images={counts['train_images']}, "
        f"train labels={counts['train_labels']}, val images={counts['val_images']}; "
        f"split train images={status.trainimages}, "
        f"train labels={status.trainlabels}, val images={status.valimages}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
