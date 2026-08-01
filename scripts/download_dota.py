#!/usr/bin/env python3
"""
DOTA v1.5 Dataset Downloader using OpenDataLab SDK
Automatically downloads and organizes DOTA dataset structure
"""

import os
import sys
from pathlib import Path

def download_dota_sdk(target_dir='dataset/DOTA'):
    """Download DOTA v1.5 using OpenDataLab SDK"""
    try:
        from openxlab.dataset import download
        print("✓ OpenDataLab SDK imported")
    except ImportError:
        print("✗ OpenDataLab SDK not found, installing...")
        os.system("pip install -q openxlab")
        from openxlab.dataset import download
        print("✓ OpenDataLab SDK installed")

    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print("DOTA v1.5 Dataset Download")
    print(f"{'='*50}")
    print(f"Target: {target_path.absolute()}")
    print(f"Source: OpenDataLab/DOTA_V1_dot_5")
    print(f"Size: ~18-20GB (full dataset)")
    print(f"Time: ~30-60 minutes depending on network")
    print(f"{'='*50}\n")

    # Download dataset
    try:
        print("Starting download (this may take 30-60 minutes)...")
        download(
            dataset_repo='OpenDataLab/DOTA_V1_dot_5',
            source_path='',  # Empty string downloads entire dataset
            target_path=str(target_path)
        )
        print("\n✓ Download complete!")
        return True

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nAlternative download methods:")
        print("1. Baidu NetDisk: https://pan.baidu.com/s/1getIv5_RQR4mCi0yTOE0cA (code: DOTA)")
        print("2. Official site: https://captain-whu.github.io/DOTA/dataset.html")
        print("3. Academic Torrents: https://academictorrents.com/")
        return False

def verify_dataset(dataset_dir='dataset/DOTA'):
    """Verify downloaded dataset structure"""
    dataset_path = Path(dataset_dir)

    required_dirs = [
        'train/images',
        'train/labelTxt',
        'val/images',
        'val/labelTxt'
    ]

    print("\nVerifying dataset structure...")
    all_exist = True

    for req_dir in required_dirs:
        full_path = dataset_path / req_dir
        if full_path.exists():
            file_count = len(list(full_path.glob('*')))
            print(f"  ✓ {req_dir}: {file_count} files")
        else:
            print(f"  ✗ {req_dir}: NOT FOUND")
            all_exist = False

    return all_exist

if __name__ == '__main__':
    # Get target directory from command line or use default
    target = sys.argv[1] if len(sys.argv) > 1 else 'dataset/DOTA'

    # Download
    success = download_dota_sdk(target)

    if success:
        # Verify
        if verify_dataset(target):
            print("\n" + "="*50)
            print("✓ Dataset ready for splitting!")
            print("="*50)
            print("\nNext step:")
            print("  bash scripts/setupcloud.sh")
            sys.exit(0)
        else:
            print("\n✗ Dataset structure incomplete")
            sys.exit(1)
    else:
        sys.exit(1)
