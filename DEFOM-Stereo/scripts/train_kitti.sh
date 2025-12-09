#!/usr/bin/env bash

# trained on 8 x 24GB 3090/4090 GPUs

CHECKPOINT_DIR=checkpoints/defomstereo_vitl_kitti && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=9992 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 2 3 4 5 6 7 \
--name defomstereo_vitl_kitti \
--batch_size 8  \
--num_workers 4  \
--train_datasets kitti12 kitti15 vkitti2 \
--train_folds 50 50 1 \
--num_steps 50000 \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder vitl \
--image_size 352 1216 \
--resume_ckpt checkpoints/defomstereo_vitl_sceneflow.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


