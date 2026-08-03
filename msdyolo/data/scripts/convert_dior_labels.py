#!/usr/bin/env python3
"""Convert DIOR-R labels from YOLOv11 OBB format to YOLOv5-OBB format.

YOLOv11 OBB format (DIOR-R): class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 columns)
YOLOv5-OBB format (MSDYOLO): x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty (10 columns)
"""

from pathlib import Path
import sys

# DIOR class names (in order of class_id 0-19)
# Match DIOR-R dataset exactly (no hyphens)
CLASS_NAMES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt',
    'bridge', 'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
    'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]


def convert_label_file(input_path: Path, output_path: Path):
    """Convert one label file from YOLOv11 to YOLOv5-OBB format."""
    with open(input_path, 'r') as f:
        lines = f.readlines()

    converted = []
    skipped = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 9:
            print(f"Warning: {input_path.name} has {len(parts)} columns, expected 9")
            continue

        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:9]]

        # Clip coordinates to [0, 1] range to fix boundary issues
        coords = [max(0.0, min(1.0, x)) for x in coords]

        # Check if any coordinate was clipped
        original_coords = [float(x) for x in parts[1:9]]
        if coords != original_coords:
            skipped += 1

        class_name = CLASS_NAMES[class_id]
        difficulty = '0'  # DIOR-R doesn't have difficulty, default to 0

        # YOLOv5-OBB format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
        coords_str = [f"{x:.6f}" for x in coords]
        converted_line = ' '.join(coords_str + [class_name, difficulty])
        converted.append(converted_line)

    if skipped > 0:
        print(f"  {input_path.name}: clipped {skipped} objects with out-of-bound coordinates")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(converted) + '\n')


def convert_dataset(dataset_root: Path):
    """Convert all label files in train/val/test splits."""
    for split in ['train', 'val', 'test']:
        labels_dir = dataset_root / split / 'labels'
        if not labels_dir.exists():
            print(f"Skipping {split}: {labels_dir} not found")
            continue

        label_files = list(labels_dir.glob('*.txt'))
        print(f"Converting {len(label_files)} {split} labels...")

        for label_file in label_files:
            convert_label_file(label_file, label_file)

        print(f"  {split}: Done")


if __name__ == '__main__':
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('dataset/DIOR')

    if not dataset_root.exists():
        print(f"Error: Dataset root {dataset_root} not found")
        sys.exit(1)

    print(f"Converting DIOR-R labels in {dataset_root}...")
    convert_dataset(dataset_root)
    print("Conversion complete!")
