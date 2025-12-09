#!/usr/bin/env bash

# trained on 4 x 24GB 3090/4090 GPUs
CHECKPOINT_DIR=checkpoints/defomstereo_vitl_middlebury_pretrain && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9993 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 2 3 \
--name defomstereo_vitl_middlebury_pretrain \
--batch_size 8  \
--num_workers 8  \
--train_datasets tartan_air sceneflow falling_things instereo2k carla_highres crestereo middlebury_2014 middlebury_2021 middlebury_H \
--train_folds 1 1 1 50 50 1 200 200 200 \
--num_steps 200000 \
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

# trained on 8 x 24GB 3090/4090 GPUs
CHECKPOINT_DIR=checkpoints/defomstereo_vitl_middlebury && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=9993 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 2 3 4 5 6 7 \
--name defomstereo_vitl_middlebury \
--batch_size 8  \
--num_workers 4  \
--train_datasets crestereo instereo2k carla_highres middlebury_2014 middlebury_2021 middlebury_H middlebury_F falling_things \
--train_folds 1 50 50 200 200 200 200 5  \
--num_steps 100000 \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder vitl \
--image_size 512 768 \
--resume_ckpt checkpoints/defomstereo_vitl_middlebury_pretrain.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log 
