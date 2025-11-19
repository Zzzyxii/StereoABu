# Unsupervised Stereo Matching Training Guide

## Overview
This guide describes how to train unsupervised stereo matching models using RAFT-Stereo framework with photometric consistency and geometric constraints.

## Prerequisites
- Completed supervised baseline training on KITTI2015, KITTI2012, and vKITTI2
- Clean runs directory: `rm -rf runs_unsup*`
- Ensure datasets are properly configured

## Training Strategy

### Stage 1: Pretraining on vKITTI2
Train on 1000 randomly sampled images from vKITTI2, validate on 200 images.

```bash
python train_unsupervised.py \
  --name raft-stereo-unsup \
  --stage pretrain \
  --batch_size 4 \
  --train_iters 16 \
  --num_steps 50000 \
  --image_size 320 720 \
  --spatial_scale -0.2 0.4 \
  --saturation_range 0 1.4 \
  --n_downsample 2 \
  --mixed_precision \
  --photo_weight 1.0 \
  --smooth_weight 0.1 \
  --lr_weight 0.1 \
  --smooth_order 1
```

**Output**: `checkpoints_unsup/pretrain/raft-stereo-unsup_final.pth`

### Stage 2: Finetuning on KITTI
Finetune on KITTI 2012 & 2015 with 80/20 train/val split.

```bash
python train_unsupervised.py \
  --name raft-stereo-unsup \
  --stage finetune \
  --restore_ckpt checkpoints_unsup/pretrain/raft-stereo-unsup_final.pth \
  --batch_size 4 \
  --train_iters 16 \
  --num_steps 30000 \
  --image_size 320 720 \
  --spatial_scale -0.2 0.4 \
  --saturation_range 0 1.4 \
  --n_downsample 2 \
  --mixed_precision \
  --photo_weight 1.0 \
  --smooth_weight 0.1 \
  --lr_weight 0.1 \
  --smooth_order 1
```

**Output**: `checkpoints_unsup/finetune/raft-stereo-unsup_final.pth`

## Loss Function Components

### 1. Photometric Consistency Loss
Reconstructs one view from the other using predicted disparity.
- **Weight**: `--photo_weight` (default: 1.0)
- Combines SSIM (85%) and L1 (15%)

### 2. Edge-Aware Smoothness Loss
Encourages smooth disparities except at image edges.
- **Weight**: `--smooth_weight` (default: 0.1)
- **Order**: `--smooth_order` (1 or 2)
  - Order 1: First-order gradients (faster, suitable for most cases)
  - Order 2: Second-order gradients (smoother but slower)

### 3. Left-Right Consistency Loss
Enforces that left and right disparity predictions are consistent.
- **Weight**: `--lr_weight` (default: 0.1)

## Ablation Study

To conduct ablation experiments, train with different loss combinations:

### Experiment 1: Photometric only
```bash
--photo_weight 1.0 --smooth_weight 0.0 --lr_weight 0.0
```

### Experiment 2: Photometric + Smoothness
```bash
--photo_weight 1.0 --smooth_weight 0.1 --lr_weight 0.0
```

### Experiment 3: Full loss (recommended)
```bash
--photo_weight 1.0 --smooth_weight 0.1 --lr_weight 0.1
```

### Experiment 4: Second-order smoothness
```bash
--photo_weight 1.0 --smooth_weight 0.1 --lr_weight 0.1 --smooth_order 2
```

## Evaluation

### Quantitative Evaluation
Compare supervised vs unsupervised on KITTI validation sets:

```bash
# Supervised baseline
python evaluate_stereo.py \
  --restore_ckpt checkpoints/200000_raft-stereo.pth \
  --dataset kitti

# Unsupervised model
python evaluate_stereo.py \
  --restore_ckpt checkpoints_unsup/finetune/raft-stereo-unsup_final.pth \
  --dataset kitti
```

**Metrics reported**:
- EPE (End-Point Error): Average pixel disparity error
- Bad-1: Percentage of pixels with error > 1 pixel
- Bad-3: Percentage of pixels with error > 3 pixels

### Visualization & Qualitative Analysis

Generate comparison visualizations:

```bash
python evaluate_unsupervised.py \
  --supervised_model checkpoints/200000_raft-stereo.pth \
  --unsupervised_model checkpoints_unsup/finetune/raft-stereo-unsup_final.pth \
  --dataset kitti2015 \
  --output_dir visualizations \
  --n_samples 10
```

**Output**: Side-by-side comparisons showing:
1. Input image
2. Supervised disparity map
3. Unsupervised disparity map
4. Ground truth disparity
5. Error maps (supervised vs unsupervised)
6. Difference visualization

**Analysis focus areas**:
- Texture-sparse regions (walls, sky)
- Reflective surfaces (glass, water)
- Object boundaries and edges
- Occluded regions

## Expected Results

### Supervised Baseline (after full training)
- KITTI 2015: EPE ~2-3, Bad-3 ~5-10%
- KITTI 2012: EPE ~1.5-2.5, Bad-3 ~4-8%

### Unsupervised Model
- KITTI 2015: EPE ~5-8, Bad-3 ~15-25%
- KITTI 2012: EPE ~4-6, Bad-3 ~12-20%

Performance gap is expected, especially in:
- Occluded regions (no direct supervision)
- Textureless areas (photometric loss weak)
- Thin structures (smoothness may over-regularize)

However, unsupervised should show:
- Better generalization to new domains
- No dependence on expensive ground truth
- Reasonable performance on edge regions with smoothness

## Troubleshooting

### NaN Loss
If training encounters NaN:
1. Reduce learning rate: `--lr 0.0001`
2. Reduce batch size: `--batch_size 2`
3. Check if images are properly normalized (auto-handled in script)
4. Enable gradient clipping (already enabled at 1.0)

### Poor Quality Disparities
- Increase smoothness weight: `--smooth_weight 0.2`
- Try second-order smoothness: `--smooth_order 2`
- Increase training steps: `--num_steps 100000`

### Overly Smooth Results
- Reduce smoothness weight: `--smooth_weight 0.05`
- Use first-order smoothness: `--smooth_order 1`

## Directory Structure
```
RAFT-Stereo/
├── core/
│   ├── unsupervised_loss.py      # Loss functions
│   ├── dataset_split.py           # Train/val splitting
│   └── stereo_datasets.py         # Modified for unsupervised
├── train_unsupervised.py          # Training script
├── evaluate_unsupervised.py       # Evaluation & visualization
├── checkpoints_unsup/             # Saved models
│   ├── pretrain/
│   └── finetune/
├── runs_unsup_pretrain/           # TensorBoard logs (pretrain)
├── runs_unsup_finetune/           # TensorBoard logs (finetune)
└── visualizations/                # Output images
```

## Monitoring Training

View TensorBoard logs:
```bash
tensorboard --logdir runs_unsup_pretrain --port 6006
tensorboard --logdir runs_unsup_finetune --port 6007
```

**Key metrics to watch**:
- `loss/photo`: Should decrease steadily
- `loss/smooth`: May increase initially then stabilize
- `loss/lr`: Should decrease
- `loss/total`: Overall trend should be downward
- `learning_rate`: Follows OneCycleLR schedule

## Report Contents

Your final report should include:

1. **Methodology**
   - Supervised baseline setup and training
   - Unsupervised loss formulation
   - Training strategy (pretrain → finetune)

2. **Ablation Study**
   - Impact of each loss component
   - First-order vs second-order smoothness
   - Optimal loss weight combinations

3. **Quantitative Results**
   - Tables comparing EPE, Bad-1, Bad-3
   - Supervised vs unsupervised on KITTI datasets

4. **Qualitative Analysis**
   - Visualization comparisons
   - Performance in challenging regions:
     * Textureless areas
     * Reflective surfaces
     * Object edges
     * Occluded regions

5. **Discussion**
   - Trade-offs between supervised and unsupervised
   - Generalization capabilities
   - Limitations and future work

## Next Steps

1. Complete supervised baseline training
2. Run unsupervised pretraining on vKITTI2
3. Finetune on KITTI
4. Conduct ablation experiments
5. Generate visualizations
6. Compile results for report

Good luck with your experiments!
