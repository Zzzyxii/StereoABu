#!/usr/bin/env bash

# make submission to stereo benchmarks

# make submission for  kitti12 and kitti15
python make_submission.py \
--restore_ckpt checkpoints/defomstereo_vitl_kitti.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets kitti12 kitti15 \
--dinov2_encoder vitl 

# make submission for eth3d
python make_submission.py \
--restore_ckpt checkpoints/defomstereo_vitl_eth3d.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets eth3d \
--dinov2_encoder vitl

# make submission for middlebury
python make_submission.py \
--restore_ckpt checkpoints/defomstereo_vitl_middlebury.pth  \
--method_name DEFOM-Stereo \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets middlebury_F \
--dinov2_encoder vitl

# make submission for kitti15, middlebury and eth3d using the RVC model
python make_submission.py \
--restore_ckpt checkpoints/defomstereo_vits_rvc.pth  \
--method_name DEFOM-Stereo_RVC \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets kitti15 middlebury_F eth3d \
--dinov2_encoder vits
