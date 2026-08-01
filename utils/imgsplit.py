#!/usr/bin/env python3
"""
DOTA Image Splitting Tool
Splits large DOTA images (e.g., 4000x4000) into smaller patches with overlap
"""

import os
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import argparse


def parse_dota_label(label_file):
    """Parse DOTA format label file"""
    objects = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:  # Skip first 2 lines (metadata)
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            obj = {
                'x1': float(parts[0]),
                'y1': float(parts[1]),
                'x2': float(parts[2]),
                'y2': float(parts[3]),
                'x3': float(parts[4]),
                'y3': float(parts[5]),
                'x4': float(parts[6]),
                'y4': float(parts[7]),
                'class': parts[8],
                'difficult': int(parts[9]) if len(parts) > 9 else 0
            }
            objects.append(obj)
    return objects


def is_object_in_patch(obj, x_start, y_start, patch_size):
    """Check if object is within patch boundaries"""
    x_end = x_start + patch_size
    y_end = y_start + patch_size

    # Get object bounding box
    xs = [obj['x1'], obj['x2'], obj['x3'], obj['x4']]
    ys = [obj['y1'], obj['y2'], obj['y3'], obj['y4']]

    obj_xmin, obj_xmax = min(xs), max(xs)
    obj_ymin, obj_ymax = min(ys), max(ys)

    # Check if object center is in patch
    center_x = (obj_xmin + obj_xmax) / 2
    center_y = (obj_ymin + obj_ymax) / 2

    return (x_start <= center_x < x_end and y_start <= center_y < y_end)


def transform_object_coords(obj, x_start, y_start):
    """Transform object coordinates to patch coordinate system"""
    return {
        'x1': obj['x1'] - x_start,
        'y1': obj['y1'] - y_start,
        'x2': obj['x2'] - x_start,
        'y2': obj['y2'] - y_start,
        'x3': obj['x3'] - x_start,
        'y3': obj['y3'] - y_start,
        'x4': obj['x4'] - x_start,
        'y4': obj['y4'] - y_start,
        'class': obj['class'],
        'difficult': obj['difficult']
    }


def split_single_image(args):
    """Split a single image into patches"""
    image_path, label_path, output_dir, subsize, gap = args

    image_name = Path(image_path).stem
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Warning: Cannot read {image_path}")
        return 0

    height, width = img.shape[:2]

    # Parse labels if exist
    objects = []
    if label_path and Path(label_path).exists():
        objects = parse_dota_label(label_path)

    # Create output directories
    img_output = Path(output_dir) / 'images'
    label_output = Path(output_dir) / 'labels'
    img_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    patch_count = 0

    # Sliding window
    for y in range(0, height, subsize - gap):
        for x in range(0, width, subsize - gap):
            # Adjust patch size at boundaries
            x_end = min(x + subsize, width)
            y_end = min(y + subsize, height)

            # Skip if patch too small
            if (x_end - x) < subsize // 2 or (y_end - y) < subsize // 2:
                continue

            # Extract patch
            patch = img[y:y_end, x:x_end]

            # Pad if necessary
            if patch.shape[0] < subsize or patch.shape[1] < subsize:
                patch_padded = np.zeros((subsize, subsize, 3), dtype=np.uint8)
                patch_padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = patch_padded

            # Save patch
            patch_name = f"{image_name}_{x}_{y}"
            patch_path = img_output / f"{patch_name}.png"
            cv2.imwrite(str(patch_path), patch)

            # Find objects in this patch
            patch_objects = []
            for obj in objects:
                if is_object_in_patch(obj, x, y, subsize):
                    transformed = transform_object_coords(obj, x, y)
                    patch_objects.append(transformed)

            # Save labels
            if patch_objects:
                label_path_out = label_output / f"{patch_name}.txt"
                with open(label_path_out, 'w') as f:
                    f.write("imagesource:DOTA-v1.5\n")
                    f.write(f"gsd:null\n")
                    for obj in patch_objects:
                        f.write(f"{obj['x1']:.1f} {obj['y1']:.1f} "
                               f"{obj['x2']:.1f} {obj['y2']:.1f} "
                               f"{obj['x3']:.1f} {obj['y3']:.1f} "
                               f"{obj['x4']:.1f} {obj['y4']:.1f} "
                               f"{obj['class']} {obj['difficult']}\n")

            patch_count += 1

    return patch_count


def split_dataset(image_dir, label_dir, output_dir, subsize=1024, gap=200, num_process=8):
    """Split entire dataset"""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir) if label_dir else None

    # Collect image files
    image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))

    print(f"Found {len(image_files)} images")
    print(f"Subsize: {subsize}, Gap: {gap}, Processes: {num_process}")

    # Prepare arguments
    args_list = []
    for img_path in image_files:
        label_path = None
        if label_dir:
            label_path = label_dir / (img_path.stem + '.txt')
        args_list.append((img_path, label_path, output_dir, subsize, gap))

    # Process with multiprocessing
    with Pool(num_process) as pool:
        results = pool.map(split_single_image, args_list)

    total_patches = sum(results)
    print(f"Split complete: {total_patches} patches generated")

    return total_patches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split DOTA images into patches')
    parser.add_argument('--imageset', required=True, help='Path to images directory')
    parser.add_argument('--labelset', default=None, help='Path to labels directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--subsize', type=int, default=1024, help='Patch size')
    parser.add_argument('--gap', type=int, default=200, help='Overlap between patches')
    parser.add_argument('--num_process', type=int, default=8, help='Number of processes')

    args = parser.parse_args()

    split_dataset(
        image_dir=args.imageset,
        label_dir=args.labelset,
        output_dir=args.output,
        subsize=args.subsize,
        gap=args.gap,
        num_process=args.num_process
    )
