# vKITTI2 Dataset Setup for RAFT-Stereo

This guide explains how to download and prepare the vKITTI2 dataset for unsupervised stereo training with RAFT-Stereo.

## Overview

**vKITTI2** (Virtual KITTI 2) is a synthetic dataset with perfect ground truth for stereo vision. It's ideal for pre-training before fine-tuning on real KITTI data.

- **Total frames**: ~21,260 stereo pairs
- **Scenes**: 5 scenes (Scene01, Scene02, Scene06, Scene18, Scene20)
- **Variations**: Multiple weather/lighting conditions
- **Resolution**: 1242 × 375 pixels
- **Ground truth**: Perfect depth and disparity maps

## Method 1: Direct Download (Recommended)

### Step 1: Register and Get Download Links

1. Visit: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
2. Click "Download" and fill in the registration form
3. You'll receive download links via email

### Step 2: Download Required Files

You need these files:
- `vkitti_2.0.3_rgb.tar` (28 GB) - RGB images for both cameras
- `vkitti_2.0.3_depth.tar` (6.4 GB) - Depth maps
- `vkitti_2.0.3_textgt.tar` (44 MB) - Camera parameters (optional)

**Option A: Manual Download**
```bash
# Download files from the links in your email and place them in:
mkdir -p /openbayes/home/RAFT-Stereo/datasets/vKITTI2/downloads
# Move downloaded files to the downloads directory
```

**Option B: Using wget (if you have direct links)**
```bash
cd /openbayes/home/RAFT-Stereo/datasets/vKITTI2/downloads

# Replace <DOWNLOAD_LINK> with your actual links from the email
wget -c <RGB_DOWNLOAD_LINK> -O vkitti_2.0.3_rgb.tar
wget -c <DEPTH_DOWNLOAD_LINK> -O vkitti_2.0.3_depth.tar
wget -c <TEXTGT_DOWNLOAD_LINK> -O vkitti_2.0.3_textgt.tar
```

**Option C: Using aria2c (faster, multi-threaded)**
```bash
# Install aria2
apt-get update && apt-get install -y aria2

cd /openbayes/home/RAFT-Stereo/datasets/vKITTI2/downloads

# Multi-threaded download with resume support
aria2c -x 4 -s 4 -k 1M -c --check-certificate=false <RGB_DOWNLOAD_LINK> -o vkitti_2.0.3_rgb.tar
aria2c -x 4 -s 4 -k 1M -c --check-certificate=false <DEPTH_DOWNLOAD_LINK> -o vkitti_2.0.3_depth.tar
aria2c -x 4 -s 4 -k 1M -c --check-certificate=false <TEXTGT_DOWNLOAD_LINK> -o vkitti_2.0.3_textgt.tar
```

### Step 3: Extract the Dataset

```bash
cd /openbayes/home/RAFT-Stereo
bash extract_vkitti2.sh
```

This will extract all files to `datasets/vKITTI2/raw/vkitti_2.0.3/`

### Step 4: Convert to KITTI Format

```bash
python convert_vkitti2_to_kitti.py
```

This converts vKITTI2 to KITTI-compatible structure:
```
datasets/vKITTI2/
  training/
    image_0/      # Left images (PNG)
    image_1/      # Right images (PNG)
    disp_occ/     # Disparity maps (PNG, uint16, scaled by 256)
```

### Step 5: Create Train/Val Split

```bash
python split_datasets.py
```

This creates:
- **vKITTI2**: 1000 training, 200 validation samples (random sampling)
- **KITTI2012**: 4:1 train/val split (~80%/20%)
- **KITTI2015**: 4:1 train/val split (~80%/20%)

## Method 2: Alternative Public Sources

Some users have shared vKITTI2 on public cloud storage. Here are some options:

### Google Drive (Community Shared)
If available, search for "vKITTI2 dataset Google Drive" or check computer vision forums.

### Kaggle Datasets
Check: https://www.kaggle.com/datasets?search=vkitti

### Academic Mirrors
Some universities host mirrors. Contact your institution.

## Quick Setup Script

For automated setup (after you have download links):

```bash
# 1. Edit the script with your download URLs
nano auto_download_vkitti2.sh

# 2. Run the automated setup
bash auto_download_vkitti2.sh

# 3. Extract and convert
bash extract_vkitti2.sh
python convert_vkitti2_to_kitti.py

# 4. Split datasets
python split_datasets.py
```

## Directory Structure After Setup

```
datasets/
├── vKITTI2/
│   ├── downloads/                    # Original tar files
│   ├── raw/                          # Extracted raw data
│   │   └── vkitti_2.0.3/
│   │       ├── Scene01/
│   │       ├── Scene02/
│   │       ├── Scene06/
│   │       ├── Scene18/
│   │       └── Scene20/
│   ├── training/                     # 1000 samples
│   │   ├── image_0/
│   │   ├── image_1/
│   │   └── disp_occ/
│   ├── validation/                   # 200 samples
│   │   ├── image_0/
│   │   ├── image_1/
│   │   └── disp_occ/
│   ├── vkitti2_metadata.json        # Sample metadata
│   ├── split_info.json              # Train/val split info
│   ├── training_files.txt           # Training sample IDs
│   └── validation_files.txt         # Validation sample IDs
├── KITTI2012/
│   ├── training/                     # ~80% samples
│   └── validation/                   # ~20% samples
└── KITTI2015/
    ├── training/                     # ~80% samples
    └── validation/                   # ~20% samples
```

## Dataset Statistics

After setup, you should have:

| Dataset   | Training | Validation | Total  |
|-----------|----------|------------|--------|
| vKITTI2   | 1,000    | 200        | 1,200  |
| KITTI2012 | ~160     | ~40        | ~200   |
| KITTI2015 | ~160     | ~40        | ~200   |

## Troubleshooting

### Download Issues

1. **Slow download**: Use `aria2c` for multi-threaded download
2. **Connection timeout**: Use `-c` flag for resume capability
3. **Access denied**: Make sure you completed registration

### Extraction Issues

```bash
# Check tar file integrity
tar -tzf vkitti_2.0.3_rgb.tar | head

# If corrupted, re-download
rm vkitti_2.0.3_rgb.tar
# Download again
```

### Conversion Issues

```bash
# Check if raw data exists
ls datasets/vKITTI2/raw/vkitti_2.0.3/

# Check Python dependencies
pip install numpy pillow tqdm
```

## Using vKITTI2 for Training

The dataset is now ready for the unsupervised training pipeline:

1. **Pre-training**: Train on vKITTI2 (1000 samples)
2. **Fine-tuning**: Fine-tune on KITTI 2012 & 2015
3. **Validation**: Evaluate on validation splits

Training command example:
```bash
python train_unsupervised.py \
    --dataset vkitti2 \
    --datapath datasets/vKITTI2 \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.0002
```

## References

- **vKITTI2 Paper**: "Virtual KITTI 2" (2020)
- **Official Site**: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
- **Citation**:
  ```bibtex
  @article{cabon2020virtual,
    title={Virtual KITTI 2},
    author={Cabon, Yohann and Murray, Naila and Humenberger, Martin},
    journal={arXiv preprint arXiv:2001.10773},
    year={2020}
  }
  ```

## Next Steps

After setting up the datasets, proceed with:

1. **Implement unsupervised losses** (photometric, smoothness, etc.)
2. **Create training script** for pre-training on vKITTI2
3. **Create fine-tuning script** for KITTI datasets
4. **Run ablation studies** on different loss combinations

See `unsupervised_training.md` for implementation details.
