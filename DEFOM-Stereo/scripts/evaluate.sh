#!/usr/bin/env bash

# evalutate on scene flow
python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vitl_sceneflow.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets things \
--dinov2_encoder vitl 

python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets things \
--dinov2_encoder vits

# evalutate on kitti12, kitti15, and eth3d
python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vitl_sceneflow.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets kitti12 kitti15 eth3d \
--dinov2_encoder vitl 

python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets kitti12 kitti15 eth3d \
--dinov2_encoder vits


# evalutate on Middlebury; when evaluating defomstereo_vitl on Middlebury_F
python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vitl_sceneflow.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets middlebury_F middlebury_H middlebury_Q \
--dinov2_encoder vitl 

python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets middlebury_F middlebury_H middlebury_Q \
--dinov2_encoder vits

# evalutate on different region.
python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vitl_sceneflow.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets middlebury_F middlebury_H middlebury_Q \
--indetail \
--dinov2_encoder vitl 

python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets middlebury_F middlebury_H middlebury_Q \
--indetail \
--dinov2_encoder vits

