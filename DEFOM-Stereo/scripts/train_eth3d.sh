#!/usr/bin/env bash

# trained on 4 x 24GB 3090/4090 GPUs

CHECKPOINT_DIR=checkpoints/defomstereo_vitl_eth3d_pretrain && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9994 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 2 3 \
--name defomstereo_vitl_eth3d_pretrain \
--batch_size 8  \
--num_workers 8  \
--train_datasets tartan_air sceneflow sintel_stereo eth3d instereo2k crestereo \
--train_folds 1 1 50 1000 100 2 \
--num_steps 300000 \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder vitl \
--image_size 384 512 \
--resume_ckpt checkpoints/defomstereo_vitl_sceneflow.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log && \

CHECKPOINT_DIR=checkpoints/defomstereo_vitl_eth3d && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9993 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 2 3 \
--name defomstereo_vitl_eth3d \
--batch_size 8  \
--num_workers 8  \
--train_datasets eth3d instereo2k crestereo \
--train_folds 1000 10 1 \
--num_steps 90000 \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder vitl \
--image_size 384 512 \
--resume_ckpt checkpoints/defomstereo_vitl_eth3d_pretrain.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log 
