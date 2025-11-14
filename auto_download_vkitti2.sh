#!/bin/bash
# Automated vKITTI2 download using wget (requires registration credentials)
# 
# Usage: 
#   1. Register at https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
#   2. Get your download links from the email
#   3. Edit this script and add your download URLs
#   4. Run: bash auto_download_vkitti2.sh

set -e

VKITTI2_DIR="/openbayes/home/RAFT-Stereo/datasets/vKITTI2"
DOWNLOAD_DIR="${VKITTI2_DIR}/downloads"

echo "========================================"
echo "Automated vKITTI2 Download"
echo "========================================"

mkdir -p "${DOWNLOAD_DIR}"
cd "${DOWNLOAD_DIR}"

# ============================================
# EDIT THESE URLS WITH YOUR DOWNLOAD LINKS
# ============================================
# After registration, you'll receive download links via email
# Replace the URLs below with your actual links

RGB_URL="YOUR_RGB_DOWNLOAD_LINK_HERE"
DEPTH_URL="YOUR_DEPTH_DOWNLOAD_LINK_HERE"
TEXTGT_URL="YOUR_TEXTGT_DOWNLOAD_LINK_HERE"

# ============================================
# Alternative: Using aria2c (faster, supports resume)
# ============================================
# Install aria2: apt-get install -y aria2
# Then use: aria2c -x 16 -s 16 -k 1M "${RGB_URL}" -o vkitti_2.0.3_rgb.tar

echo "Checking download tools..."
if command -v aria2c &> /dev/null; then
    DOWNLOAD_CMD="aria2c -x 16 -s 16 -k 1M -c"
    echo "Using aria2c (multi-threaded download)"
elif command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -c"
    echo "Using wget"
else
    echo "Error: Neither wget nor aria2c found. Please install one of them."
    exit 1
fi

# Download RGB images (~28 GB)
if [ ! -f "vkitti_2.0.3_rgb.tar" ]; then
    echo ""
    echo "Downloading RGB images (28 GB)..."
    if [ "${RGB_URL}" != "YOUR_RGB_DOWNLOAD_LINK_HERE" ]; then
        ${DOWNLOAD_CMD} "${RGB_URL}" -O vkitti_2.0.3_rgb.tar
    else
        echo "⚠ RGB_URL not set. Please edit this script and add your download link."
    fi
else
    echo "✓ RGB images already downloaded"
fi

# Download depth maps (~6.4 GB)
if [ ! -f "vkitti_2.0.3_depth.tar" ]; then
    echo ""
    echo "Downloading depth maps (6.4 GB)..."
    if [ "${DEPTH_URL}" != "YOUR_DEPTH_DOWNLOAD_LINK_HERE" ]; then
        ${DOWNLOAD_CMD} "${DEPTH_URL}" -O vkitti_2.0.3_depth.tar
    else
        echo "⚠ DEPTH_URL not set. Please edit this script and add your download link."
    fi
else
    echo "✓ Depth maps already downloaded"
fi

# Download camera parameters (~44 MB)
if [ ! -f "vkitti_2.0.3_textgt.tar" ]; then
    echo ""
    echo "Downloading camera parameters (44 MB)..."
    if [ "${TEXTGT_URL}" != "YOUR_TEXTGT_DOWNLOAD_LINK_HERE" ]; then
        ${DOWNLOAD_CMD} "${TEXTGT_URL}" -O vkitti_2.0.3_textgt.tar
    else
        echo "⚠ TEXTGT_URL not set. Please edit this script and add your download link."
    fi
else
    echo "✓ Camera parameters already downloaded"
fi

echo ""
echo "========================================"
echo "Download Status"
echo "========================================"

# Check file sizes
if [ -f "vkitti_2.0.3_rgb.tar" ]; then
    RGB_SIZE=$(du -h vkitti_2.0.3_rgb.tar | cut -f1)
    echo "✓ RGB images: ${RGB_SIZE}"
fi

if [ -f "vkitti_2.0.3_depth.tar" ]; then
    DEPTH_SIZE=$(du -h vkitti_2.0.3_depth.tar | cut -f1)
    echo "✓ Depth maps: ${DEPTH_SIZE}"
fi

if [ -f "vkitti_2.0.3_textgt.tar" ]; then
    TEXTGT_SIZE=$(du -h vkitti_2.0.3_textgt.tar | cut -f1)
    echo "✓ Camera params: ${TEXTGT_SIZE}"
fi

echo ""
echo "Next steps:"
echo "  1. Extract: bash extract_vkitti2.sh"
echo "  2. Convert: python convert_vkitti2_to_kitti.py"
echo "  3. Split: python split_datasets.py"
