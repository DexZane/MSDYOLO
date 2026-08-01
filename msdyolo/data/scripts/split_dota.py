#!/usr/bin/env python3
"""Split DOTA images into overlapping, pixel-labelled patches."""

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from msdyolo.data.dota import DotaObject, clip_object_to_patch, format_dota_object, parse_dota_label


def is_object_in_patch(obj: DotaObject, xstart: int, ystart: int, subsize: int) -> bool:
    """Return whether an object's bounding-box center belongs to a patch."""
    xend = xstart + subsize
    yend = ystart + subsize
    xs = obj.coordinates[::2]
    ys = obj.coordinates[1::2]
    centerx = (min(xs) + max(xs)) / 2
    centery = (min(ys) + max(ys)) / 2
    return xstart <= centerx < xend and ystart <= centery < yend


def has_distinct_points(obj: DotaObject) -> bool:
    """Return whether the polygon has more than one unique vertex."""
    return len(set(zip(obj.coordinates[::2], obj.coordinates[1::2]))) > 1


def split_single_image(args):
    """Split one image and return the number of generated patches."""
    image_path, label_path, output_dir, subsize, gap = args
    image_name = Path(image_path).stem
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Cannot read {image_path}")
        return 0

    height, width = image.shape[:2]
    objects = []
    if label_path and Path(label_path).exists():
        objects = parse_dota_label(Path(label_path))

    image_output = Path(output_dir) / "images"
    label_output = Path(output_dir) / "labelTxt"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    patch_count = 0
    for ystart in range(0, height, subsize - gap):
        for xstart in range(0, width, subsize - gap):
            xend = min(xstart + subsize, width)
            yend = min(ystart + subsize, height)
            if xend - xstart < subsize // 2 or yend - ystart < subsize // 2:
                continue

            patch = image[ystart:yend, xstart:xend]
            if patch.shape[0] < subsize or patch.shape[1] < subsize:
                padded = np.zeros((subsize, subsize, 3), dtype=np.uint8)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded

            patch_name = f"{image_name}_{xstart}_{ystart}"
            cv2.imwrite(str(image_output / f"{patch_name}.png"), patch)

            patch_objects = []
            for obj in objects:
                if is_object_in_patch(obj, xstart, ystart, subsize):
                    clipped = clip_object_to_patch(obj, xstart, ystart, subsize)
                    if has_distinct_points(clipped):
                        patch_objects.append(clipped)
            if patch_objects:
                output_label = label_output / f"{patch_name}.txt"
                output_label.write_text(
                    "\n".join(format_dota_object(obj) for obj in patch_objects) + "\n",
                    encoding="utf-8",
                )

            patch_count += 1
    return patch_count


def validate_split_arguments(subsize: int, gap: int, num_process: int) -> None:
    """Validate split geometry and worker-count arguments."""
    if subsize <= 0:
        raise ValueError("subsize must be positive")
    if subsize > 1024:
        raise ValueError("subsize must not exceed 1024 for pixel labels")
    if gap < 0 or gap >= subsize:
        raise ValueError("gap must satisfy 0 <= gap < subsize")
    if num_process <= 0:
        raise ValueError("num_process must be positive")


def split_dataset(image_dir, label_dir, output_dir, subsize=1024, gap=200, num_process=8):
    """Split all PNG and JPEG images in a DOTA dataset directory."""
    validate_split_arguments(subsize, gap, num_process)
    image_dir = Path(image_dir)
    label_dir = Path(label_dir) if label_dir else None
    image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images")
    print(f"Subsize: {subsize}, Gap: {gap}, Processes: {num_process}")

    args_list = []
    for image_path in image_files:
        label_path = label_dir / f"{image_path.stem}.txt" if label_dir else None
        args_list.append((image_path, label_path, output_dir, subsize, gap))
    with Pool(num_process) as pool:
        results = pool.map(split_single_image, args_list)

    total_patches = sum(results)
    print(f"Split complete: {total_patches} patches generated")
    return total_patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split DOTA images into patches")
    parser.add_argument("--imageset", required=True, help="Path to images directory")
    parser.add_argument("--labelset", default=None, help="Path to labels directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--subsize", type=int, default=1024, help="Patch size")
    parser.add_argument("--gap", type=int, default=200, help="Overlap between patches")
    parser.add_argument("--num_process", type=int, default=8, help="Number of processes")
    args = parser.parse_args()
    split_dataset(
        image_dir=args.imageset,
        label_dir=args.labelset,
        output_dir=args.output,
        subsize=args.subsize,
        gap=args.gap,
        num_process=args.num_process,
    )
