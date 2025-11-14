#!/bin/bash
# Direct download script for vKITTI2 dataset
# vKITTI2 is available at: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

set -e

VKITTI2_DIR="/openbayes/home/RAFT-Stereo/datasets/vKITTI2"
DOWNLOAD_DIR="${VKITTI2_DIR}/downloads"

echo "========================================"
echo "vKITTI2 Dataset Download"
echo "========================================"

# Create directories
mkdir -p "${DOWNLOAD_DIR}"
cd "${DOWNLOAD_DIR}"

echo ""
echo "Dataset will be downloaded to: ${DOWNLOAD_DIR}"
echo ""

# vKITTI2 official links (requires registration)
# Alternative: You can use aria2c or wget if you have credentials

echo "Downloading vKITTI2 RGB images (left and right cameras)..."
# RGB images (both Camera_0 and Camera_1)
# File: vkitti_2.0.3_rgb.tar (28 GB)

echo "Downloading vKITTI2 Depth maps..."
# Depth images for both cameras  
# File: vkitti_2.0.3_depth.tar (6.4 GB)

echo "Downloading vKITTI2 Camera parameters..."
# Camera extrinsics and intrinsics
# File: vkitti_2.0.3_textgt.tar (44 MB)

echo ""
echo "========================================"
echo "IMPORTANT: vKITTI2 requires registration"
echo "========================================"
echo ""
echo "Please visit: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/"
echo "Register and download these files:"
echo ""
echo "  1. vkitti_2.0.3_rgb.tar (28 GB) - RGB images"
echo "  2. vkitti_2.0.3_depth.tar (6.4 GB) - Depth maps" 
echo "  3. vkitti_2.0.3_textgt.tar (44 MB) - Camera parameters"
echo ""
echo "After downloading, place files in: ${DOWNLOAD_DIR}"
echo "Then run: bash extract_vkitti2.sh"
echo ""
echo "========================================"
echo "Alternative: Use wget with credentials"
echo "========================================"
echo ""
echo "If you have registered, use:"
echo "  cd ${DOWNLOAD_DIR}"
echo '  wget --http-user=YOUR_EMAIL --http-password=YOUR_PASSWORD <download_link>'
echo ""

# Check if files already exist
if [ -f "vkitti_2.0.3_rgb.tar" ] && [ -f "vkitti_2.0.3_depth.tar" ]; then
    echo "Found downloaded files! Extracting..."
    bash /openbayes/home/RAFT-Stereo/extract_vkitti2.sh
else
    echo "Files not found. Please download manually as instructed above."
fi
