#!/usr/bin/env python3
"""
Download and organize vKITTI2 dataset for RAFT-Stereo training.
vKITTI2 official download: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
"""

import os
import sys
import urllib.request
import tarfile
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar"""
    print(f"Downloading {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_tar(tar_path, extract_path):
    """Extract tar.gz file"""
    print(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted to {extract_path}")

def download_vkitti2():
    """
    Download vKITTI2 dataset.
    
    Note: vKITTI2 requires registration to download. 
    You need to:
    1. Go to https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
    2. Register and get download links
    3. Use the links below
    
    For this script, we'll provide the structure and you can manually download or 
    use wget/curl with authentication.
    """
    
    base_dir = Path("/openbayes/home/RAFT-Stereo/datasets/vKITTI2")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # vKITTI2 download URLs (you may need to update these with actual download links)
    # These are the typical scenes in vKITTI2
    scenes = [
        "Scene01",
        "Scene02", 
        "Scene06",
        "Scene18",
        "Scene20"
    ]
    
    # Variations available
    variations = ["clone", "fog", "morning", "overcast", "rain", "sunset"]
    
    print("=" * 80)
    print("vKITTI2 Dataset Download Instructions")
    print("=" * 80)
    print("\nvKITTI2 requires registration. Please follow these steps:")
    print("\n1. Visit: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/")
    print("2. Register and accept terms")
    print("3. Download the following files:")
    print("\n   Recommended downloads for stereo:")
    print("   - vkitti_2.0.3_rgb.tar (RGB images - left and right)")
    print("   - vkitti_2.0.3_depth.tar (Depth maps)")
    print("   - vkitti_2.0.3_textgt.tar (Camera parameters)")
    print("\n4. Place downloaded files in:", base_dir)
    print("\nOr use wget/curl with your credentials:")
    print(f"\n   cd {base_dir}")
    print("   wget --user=YOUR_EMAIL --password=YOUR_PASSWORD <download_link>")
    
    return base_dir

def organize_vkitti2_for_raft(vkitti2_root):
    """
    Organize vKITTI2 into KITTI-like structure for RAFT-Stereo.
    
    Original vKITTI2 structure:
    vkitti_2.0.3/
        Scene01/
            clone/
                frames/
                    rgb/
                        Camera_0/  # Left camera
                        Camera_1/  # Right camera
                    depth/
                        Camera_0/
                        Camera_1/
    
    Target KITTI-like structure:
    vKITTI2/
        training/
            image_0/  # Left images
            image_1/  # Right images  
            disp_occ/  # Disparity maps (from depth)
        testing/
            image_0/
            image_1/
    """
    
    vkitti2_root = Path(vkitti2_root)
    print(f"\nOrganizing vKITTI2 dataset at: {vkitti2_root}")
    
    # Create target directories
    train_dir = vkitti2_root / "training"
    test_dir = vkitti2_root / "testing"
    
    for split_dir in [train_dir, test_dir]:
        (split_dir / "image_0").mkdir(parents=True, exist_ok=True)
        (split_dir / "image_1").mkdir(parents=True, exist_ok=True)
        (split_dir / "disp_occ").mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created:")
    print(f"  {train_dir}/image_0/")
    print(f"  {train_dir}/image_1/")
    print(f"  {train_dir}/disp_occ/")
    print(f"  {test_dir}/image_0/")
    print(f"  {test_dir}/image_1/")
    print(f"  {test_dir}/disp_occ/")
    
    return train_dir, test_dir

def main():
    print("vKITTI2 Dataset Setup for RAFT-Stereo")
    print("=" * 80)
    
    # Step 1: Provide download instructions
    vkitti2_dir = download_vkitti2()
    
    # Step 2: Create organized structure
    print("\n" + "=" * 80)
    print("Creating KITTI-compatible directory structure...")
    print("=" * 80)
    train_dir, test_dir = organize_vkitti2_for_raft(vkitti2_dir)
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Download vKITTI2 files following the instructions above")
    print("2. Extract them to:", vkitti2_dir)
    print("3. Run the conversion script: python convert_vkitti2_to_kitti.py")
    print("4. Run data split script: python split_vkitti2_data.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
