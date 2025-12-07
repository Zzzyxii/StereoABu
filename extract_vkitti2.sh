#!/bin/bash
# Extract vKITTI2 tar files

set -e

VKITTI2_DIR="/global_data/sft_intern/slz/zyx/CKPT/4bgrpocom50/vKITTI2"
DOWNLOAD_DIR="${VKITTI2_DIR}/downloads"
EXTRACT_DIR="${VKITTI2_DIR}/raw"

echo "========================================"
echo "Extracting vKITTI2 Dataset"
echo "========================================"

mkdir -p "${EXTRACT_DIR}"
cd "${DOWNLOAD_DIR}"

extract_archive() {
    local base_name="$1"
    local label="$2"
    local archive=""
    local marker="${EXTRACT_DIR}/.${base_name}_extracted"

    if [ -z "${FORCE_REEXTRACT}" ] && [ -f "${marker}" ]; then
        echo "• ${label} already extracted (skip). Set FORCE_REEXTRACT=1 to re-extract."
        return
    fi

    if [ -f "${base_name}.tar" ]; then
        archive="${base_name}.tar"
    elif [ -f "${base_name}.tar.gz" ]; then
        archive="${base_name}.tar.gz"
    fi

    if [ -n "${archive}" ]; then
        echo "Extracting ${label}..."
        tar -xf "${archive}" -C "${EXTRACT_DIR}"
        echo "✓ ${label} extracted"
        touch "${marker}"
    else
        echo "⚠ ${base_name}.tar(.gz) not found"
    fi
}

# Extract archives (supports .tar and .tar.gz)
extract_archive "vkitti_2.0.3_rgb" "RGB images"
extract_archive "vkitti_2.0.3_depth" "Depth maps"
extract_archive "vkitti_2.0.3_textgt" "Camera parameters"

echo ""
echo "========================================"
echo "Extraction complete!"
echo "========================================"
echo "Raw data location: ${EXTRACT_DIR}"
echo ""
echo "Next step: Convert to KITTI format"
echo "Run: python convert_vkitti2_to_kitti.py"
