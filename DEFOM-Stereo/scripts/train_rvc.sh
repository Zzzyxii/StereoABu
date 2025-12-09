#!/usr/bin/env bash

# trained on 4 x 24GB 3090/4090 GPUs
CHECKPOINT_DIR=checkpoints/defomstereo_vits_rvc_pretrain && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9995 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 2 3 \
--name defomstereo_vits_rvc_pretrain \
--batch_size 8  \
--num_workers 8  \
--train_datasets tartan_air sceneflow irs 3dkenburns crestereo falling_things sintel_stereo vkitti2 carla_highres \
--train_folds 1 1 1 1 1 1 3 3 80 \
--num_steps 200000 \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder vits \
--image_size 384 768 \
--resume_ckpt checkpoints/defomstereo_vits_sceneflow.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log && \

# trained on 4 x 24GB 3090/4090 GPUs
CHECKPOINT_DIR=checkpoints/defomstereo_vits_rvc_pretrain2 && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9996 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 2 3 \
--name defomstereo_vits_rvc_pretrain2 \
--batch_size 8  \
--num_workers 8  \
--train_datasets tartan_air irs 3dkenburns crestereo vkitti2 carla_highres kitti12 kitti15 middlebury_2005 middlebury_2006 middlebury_2014 middlebury_2021 middlebury_Q middlebury_H eth3d instereo2k booster \
--train_folds 1 1 1 1 3 30 100 100 200 200 200 200 200 200 1000 20 10 \
--num_steps 100000 \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder vits \
--image_size 384 768 \
--resume_ckpt checkpoints/defomstereo_vits_rvc_pretrain.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log && \

# trained on 4 x 24GB 3090/4090 GPUs
CHECKPOINT_DIR=checkpoints/defomstereo_vits_rvc && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9997 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 2 3 \
--name defomstereo_vits_rvc \
--batch_size 8  \
--num_workers 8  \
--train_datasets tartan_air irs 3dkenburns crestereo vkitti2 carla_highres kitti12 kitti15 middlebury_2005 middlebury_2006 middlebury_2014 middlebury_2021 middlebury_Q middlebury_H eth3d instereo2k booster \
--train_folds 1 1 1 1 3 30 2500 2500 200 200 200 200 200 200 1000 20 10 \
--num_steps 20000 \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder vits \
--image_size 384 768 \
--resume_ckpt checkpoints/defomstereo_vits_rvc_pretrain2.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log 
