#!/usr/bin/env python3
"""
Split datasets for unsupervised stereo training.

Requirements:
- vKITTI2: 1000 training, 200 validation (random sampling)
- KITTI 2012: 4:1 train/val split
- KITTI 2015: 4:1 train/val split
"""

import os
import sys
import json
import random
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

def split_vkitti2(vkitti2_root, num_train=1000, num_val=200):
    """
    Split vKITTI2 into train/val sets with random sampling.
    
    Args:
        vkitti2_root: Path to vKITTI2 dataset
        num_train: Number of training samples (1000)
        num_val: Number of validation samples (200)
    """
    vkitti2_root = Path(vkitti2_root)
    
    print("=" * 80)
    print("Splitting vKITTI2 Dataset")
    print("=" * 80)
    
    # Load metadata
    metadata_file = vkitti2_root / "vkitti2_metadata.json"
    if not metadata_file.exists():
        print(f"Error: Metadata not found at {metadata_file}")
        print("Please run convert_vkitti2_to_kitti.py first")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    total_samples = len(metadata)
    print(f"Total samples available: {total_samples}")
    
    if total_samples < num_train + num_val:
        print(f"Warning: Not enough samples. Requested {num_train + num_val}, available {total_samples}")
        num_train = int(total_samples * 0.833)  # ~5:1 ratio
        num_val = total_samples - num_train
        print(f"Adjusted to: {num_train} train, {num_val} val")
    
    # Random sampling
    random.seed(42)  # For reproducibility
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    
    train_indices = set(all_indices[:num_train])
    val_indices = set(all_indices[num_train:num_train + num_val])
    
    print(f"\nSplit:")
    print(f"  Training: {len(train_indices)} samples")
    print(f"  Validation: {len(val_indices)} samples")
    
    # Create split directories
    train_dir = vkitti2_root / "training"
    val_dir = vkitti2_root / "validation"
    
    for split_dir in [val_dir]:
        for subdir in ["image_0", "image_1", "disp_occ"]:
            (split_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Move validation samples
    print("\nMoving validation samples...")
    for idx in val_indices:
        sample = metadata[idx]
        filename = sample["filename"]
        
        for subdir in ["image_0", "image_1", "disp_occ"]:
            src = train_dir / subdir / filename
            dst = val_dir / subdir / filename
            if src.exists():
                shutil.move(str(src), str(dst))
    
    # Save split info
    split_info = {
        "train_indices": sorted(list(train_indices)),
        "val_indices": sorted(list(val_indices)),
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "total_count": total_samples,
        "split_ratio": f"{len(train_indices)}:{len(val_indices)}"
    }
    
    split_file = vkitti2_root / "split_info.json"
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ vKITTI2 split complete")
    print(f"  Training: {train_dir} ({len(train_indices)} samples)")
    print(f"  Validation: {val_dir} ({len(val_indices)} samples)")
    print(f"  Split info: {split_file}")
    
    return True

def split_kitti(kitti_root, dataset_name, train_ratio=0.8):
    """
    Split KITTI dataset into train/val (4:1 ratio).
    
    Args:
        kitti_root: Path to KITTI dataset
        dataset_name: "KITTI2012" or "KITTI2015"
        train_ratio: Ratio of training samples (0.8 = 4:1)
    """
    kitti_root = Path(kitti_root)
    
    print("=" * 80)
    print(f"Splitting {dataset_name} Dataset")
    print("=" * 80)
    
    train_dir = kitti_root / "training"
    val_dir = kitti_root / "validation"
    
    # Check if training directory exists
    if not train_dir.exists():
        print(f"Error: Training directory not found at {train_dir}")
        return False
    
    # Get all samples
    image_0_dir = train_dir / "image_0"
    if dataset_name == "KITTI2015":
        image_0_dir = train_dir / "image_2"  # KITTI 2015 uses image_2/image_3
    
    if not image_0_dir.exists():
        print(f"Error: Image directory not found at {image_0_dir}")
        return False
    
    all_images = sorted(image_0_dir.glob("*.png"))
    total_samples = len(all_images)
    
    print(f"Total samples: {total_samples}")
    
    # Split into train/val
    num_train = int(total_samples * train_ratio)
    num_val = total_samples - num_train
    
    # Random sampling with seed for reproducibility
    random.seed(42)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    train_indices = set(indices[:num_train])
    val_indices = set(indices[num_train:])
    
    print(f"\nSplit:")
    print(f"  Training: {num_train} samples ({train_ratio * 100:.0f}%)")
    print(f"  Validation: {num_val} samples ({(1 - train_ratio) * 100:.0f}%)")
    
    # Determine which subdirectories to copy
    if dataset_name == "KITTI2012":
        subdirs = ["image_0", "image_1", "disp_occ", "disp_noc"]
    else:  # KITTI2015
        subdirs = ["image_2", "image_3", "disp_occ_0", "disp_noc_0"]
    
    # Create validation directories
    for subdir in subdirs:
        src_dir = train_dir / subdir
        if src_dir.exists():
            (val_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Move validation samples
    print("\nMoving validation samples...")
    moved_count = 0
    
    for idx in val_indices:
        img_path = all_images[idx]
        filename = img_path.name
        
        for subdir in subdirs:
            src = train_dir / subdir / filename
            dst = val_dir / subdir / filename
            
            if src.exists():
                shutil.move(str(src), str(dst))
                moved_count += 1
    
    # Save split info
    split_info = {
        "dataset": dataset_name,
        "train_indices": sorted(list(train_indices)),
        "val_indices": sorted(list(val_indices)),
        "train_count": num_train,
        "val_count": num_val,
        "total_count": total_samples,
        "split_ratio": f"{num_train}:{num_val}"
    }
    
    split_file = kitti_root / "split_info.json"
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ {dataset_name} split complete")
    print(f"  Training: {train_dir} ({num_train} samples)")
    print(f"  Validation: {val_dir} ({num_val} samples)")
    print(f"  Split info: {split_file}")
    
    return True

def create_file_lists(base_dir, dataset_name):
    """
    Create file lists for easy loading during training.
    
    Creates:
        - train_files.txt: List of training sample IDs
        - val_files.txt: List of validation sample IDs
    """
    base_dir = Path(base_dir)
    
    for split in ["training", "validation"]:
        split_dir = base_dir / split
        
        if dataset_name == "vKITTI2":
            image_dir = split_dir / "image_0"
        elif dataset_name == "KITTI2012":
            image_dir = split_dir / "image_0"
        else:  # KITTI2015
            image_dir = split_dir / "image_2"
        
        if not image_dir.exists():
            continue
        
        images = sorted(image_dir.glob("*.png"))
        
        output_file = base_dir / f"{split}_files.txt"
        with open(output_file, 'w') as f:
            for img in images:
                # Write just the filename without extension
                f.write(img.stem + '\n')
        
        print(f"Created {output_file} ({len(images)} files)")

def main():
    datasets_root = Path("/openbayes/home/RAFT-Stereo/datasets")
    
    print("=" * 80)
    print("Dataset Splitting for Unsupervised Stereo Training")
    print("=" * 80)
    print()
    
    # 1. Split vKITTI2 (1000 train, 200 val)
    vkitti2_root = datasets_root / "vKITTI2"
    if vkitti2_root.exists():
        split_vkitti2(vkitti2_root, num_train=1000, num_val=200)
        create_file_lists(vkitti2_root, "vKITTI2")
        print()
    else:
        print(f"⚠ vKITTI2 not found at {vkitti2_root}, skipping...")
        print()
    
    # 2. Split KITTI 2012 (4:1 ratio)
    kitti2012_root = datasets_root / "KITTI2012"
    if kitti2012_root.exists():
        split_kitti(kitti2012_root, "KITTI2012", train_ratio=0.8)
        create_file_lists(kitti2012_root, "KITTI2012")
        print()
    else:
        print(f"⚠ KITTI2012 not found at {kitti2012_root}, skipping...")
        print()
    
    # 3. Split KITTI 2015 (4:1 ratio)
    kitti2015_root = datasets_root / "KITTI2015"
    if kitti2015_root.exists():
        split_kitti(kitti2015_root, "KITTI2015", train_ratio=0.8)
        create_file_lists(kitti2015_root, "KITTI2015")
        print()
    else:
        print(f"⚠ KITTI2015 not found at {kitti2015_root}, skipping...")
        print()
    
    print("=" * 80)
    print("All splits complete!")
    print("=" * 80)
    print("\nDataset structure:")
    print("  datasets/")
    print("    vKITTI2/")
    print("      training/     (1000 samples)")
    print("      validation/   (200 samples)")
    print("    KITTI2012/")
    print("      training/     (~80%)")
    print("      validation/   (~20%)")
    print("    KITTI2015/")
    print("      training/     (~80%)")
    print("      validation/   (~20%)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
