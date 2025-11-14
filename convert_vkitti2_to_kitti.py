#!/usr/bin/env python3
"""
Convert vKITTI2 dataset to KITTI format for RAFT-Stereo.

vKITTI2 structure:
vkitti_2.0.3/
    Scene01/
        clone/
            frames/
                rgb/
                    Camera_0/  # Left camera (15 degrees)
                        rgb_00001.jpg
                        ...
                    Camera_1/  # Right camera (15 degrees) 
                        rgb_00001.jpg
                        ...
                depth/
                    Camera_0/
                        depth_00001.png
                    Camera_1/
                        depth_00001.png

Target KITTI structure:
vKITTI2/
    training/
        image_0/  # Left images
            000000.png
            000001.png
            ...
        image_1/  # Right images
            000000.png
            000001.png
            ...
        disp_occ/  # Disparity from depth
            000000.png
            000001.png
            ...
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

def depth_to_disparity(depth_path, focal_length=725.0, baseline=0.532725):
    """
    Convert depth map to disparity map.
    
    Args:
        depth_path: Path to depth image (PNG, depth in cm)
        focal_length: Camera focal length in pixels
        baseline: Stereo baseline in meters
        
    Returns:
        disparity: Disparity map in pixels
    """
    # Load depth (stored as uint16, values in cm)
    depth_img = np.array(Image.open(depth_path))
    
    # Convert from cm to meters
    depth_m = depth_img.astype(np.float32) / 100.0
    
    # Avoid division by zero
    depth_m[depth_m < 0.01] = 0.01
    
    # Calculate disparity: d = (focal_length * baseline) / depth
    disparity = (focal_length * baseline) / depth_m
    
    # KITTI disparity format: stored as uint16, multiplied by 256
    disparity_uint = (disparity * 256.0).astype(np.uint16)
    
    return disparity_uint

def collect_vkitti2_samples(vkitti2_root):
    """
    Collect all stereo pairs from vKITTI2.
    
    Returns:
        List of sample dicts with paths to left/right images and depth
    """
    vkitti2_root = Path(vkitti2_root)
    samples = []
    
    # Scenes in vKITTI2
    scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
    
    # Variations
    variations = ["clone", "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right",
                  "fog", "morning", "overcast", "rain", "sunset"]
    
    for scene in scenes:
        scene_dir = vkitti2_root / scene
        if not scene_dir.exists():
            print(f"Warning: Scene {scene} not found, skipping...")
            continue
            
        for variation in variations:
            var_dir = scene_dir / variation
            if not var_dir.exists():
                continue
                
            # Paths to image and depth directories
            rgb_left_dir = var_dir / "frames" / "rgb" / "Camera_0"
            rgb_right_dir = var_dir / "frames" / "rgb" / "Camera_1"
            depth_left_dir = var_dir / "frames" / "depth" / "Camera_0"
            
            if not rgb_left_dir.exists():
                continue
            
            # Get all frames
            left_images = sorted(rgb_left_dir.glob("rgb_*.jpg"))
            
            for left_img in left_images:
                frame_id = left_img.stem.split("_")[1]  # e.g., "00001"
                
                right_img = rgb_right_dir / f"rgb_{frame_id}.jpg"
                depth_img = depth_left_dir / f"depth_{frame_id}.png" if depth_left_dir.exists() else None
                
                if right_img.exists():
                    samples.append({
                        "scene": scene,
                        "variation": variation,
                        "frame": frame_id,
                        "left_img": left_img,
                        "right_img": right_img,
                        "depth": depth_img
                    })
    
    return samples

def convert_vkitti2_to_kitti(vkitti2_root, output_root, max_samples=None):
    """
    Convert vKITTI2 to KITTI format.
    
    Args:
        vkitti2_root: Path to extracted vKITTI2 (raw/)
        output_root: Path to output directory
        max_samples: Maximum number of samples to convert (None = all)
    """
    vkitti2_root = Path(vkitti2_root)
    output_root = Path(output_root)
    
    print("=" * 80)
    print("Converting vKITTI2 to KITTI format")
    print("=" * 80)
    print(f"Input: {vkitti2_root}")
    print(f"Output: {output_root}")
    print()
    
    # Create output directories
    train_dir = output_root / "training"
    for subdir in ["image_0", "image_1", "disp_occ"]:
        (train_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Collect samples
    print("Collecting samples...")
    samples = collect_vkitti2_samples(vkitti2_root)
    print(f"Found {len(samples)} stereo pairs")
    
    if max_samples:
        samples = samples[:max_samples]
        print(f"Limiting to {max_samples} samples")
    
    # Convert samples
    print("\nConverting samples...")
    metadata = []
    
    for idx, sample in enumerate(tqdm(samples)):
        # Output filenames
        img_name = f"{idx:06d}.png"
        
        # Copy left image (convert JPG to PNG)
        left_out = train_dir / "image_0" / img_name
        Image.open(sample["left_img"]).save(left_out)
        
        # Copy right image (convert JPG to PNG)
        right_out = train_dir / "image_1" / img_name
        Image.open(sample["right_img"]).save(right_out)
        
        # Convert depth to disparity (optional for unsupervised training)
        if sample["depth"] and sample["depth"].exists():
            disp_out = train_dir / "disp_occ" / img_name
            disparity = depth_to_disparity(sample["depth"])
            Image.fromarray(disparity).save(disp_out)
        
        # Save metadata
        metadata.append({
            "id": idx,
            "filename": img_name,
            "scene": sample["scene"],
            "variation": sample["variation"],
            "frame": sample["frame"]
        })
    
    # Save metadata
    metadata_file = output_root / "vkitti2_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Count disparity maps
    disp_count = len(list((train_dir / "disp_occ").glob("*.png")))
    
    print("\n" + "=" * 80)
    print("Conversion complete!")
    print("=" * 80)
    print(f"Converted {len(samples)} samples")
    print(f"Output directory: {output_root}")
    print(f"  - training/image_0/ : {len(samples)} left images")
    print(f"  - training/image_1/ : {len(samples)} right images")
    if disp_count > 0:
        print(f"  - training/disp_occ/ : {disp_count} disparity maps")
    else:
        print(f"  - training/disp_occ/ : (skipped - no depth data, OK for unsupervised training)")
    print(f"  - vkitti2_metadata.json : metadata")
    
    return len(samples)

def main():
    # Paths
    vkitti2_raw = Path("/openbayes/home/RAFT-Stereo/datasets/vKITTI2/raw")
    output_root = Path("/openbayes/home/RAFT-Stereo/datasets/vKITTI2")
    
    if not vkitti2_raw.exists():
        print(f"Error: vKITTI2 raw data not found at: {vkitti2_raw}")
        print("Please extract the dataset first using: bash extract_vkitti2.sh")
        return 1
    
    # Check if Scene directories exist
    scene_dirs = list(vkitti2_raw.glob("Scene*"))
    if not scene_dirs:
        print(f"Error: No Scene directories found in {vkitti2_raw}")
        print("Please ensure the dataset is properly extracted")
        return 1
    
    # Convert all samples (no limit)
    convert_vkitti2_to_kitti(vkitti2_raw, output_root)
    
    print("\nNext step: Create train/val split")
    print("Run: python split_vkitti2_data.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
