"""
Evaluation and Visualization for Supervised vs Unsupervised Stereo Matching
"""

import sys
sys.path.append('core')

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from raft_stereo import RAFTStereo
from utils import frame_utils
from utils.utils import InputPadder

import core.stereo_datasets as datasets


@torch.no_grad()
def validate(model, dataset_name='kitti2015', iters=32):
    """Evaluate model on validation set"""
    model.eval()
    
    if dataset_name == 'kitti2015':
        val_dataset = datasets.KITTI(root='datasets/KITTI2015', image_set='training', year=2015)
    elif dataset_name == 'kitti2012':
        val_dataset = datasets.KITTI(root='datasets/KITTI2012', image_set='training', year=2012)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    epe_list = []
    bad1_list = []
    bad3_list = []
    
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        # Compute metrics only on valid pixels
        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        epe = epe.view(-1)
        flow_gt = flow_gt.view(2, -1)
        valid_gt = valid_gt.view(-1)

        epe = epe[valid_gt >= 0.5]
        
        if len(epe) == 0:
            continue
            
        epe_list.append(epe.mean().item())
        bad1_list.append((epe > 1.0).float().mean().item())
        bad3_list.append((epe > 3.0).float().mean().item())

    epe_mean = np.mean(epe_list)
    bad1_mean = 100 * np.mean(bad1_list)
    bad3_mean = 100 * np.mean(bad3_list)
    
    print(f"{dataset_name} - EPE: {epe_mean:.3f}, Bad-1: {bad1_mean:.2f}%, Bad-3: {bad3_mean:.2f}%")
    
    return {
        'epe': epe_mean,
        'bad1': bad1_mean,
        'bad3': bad3_mean
    }


@torch.no_grad()
def visualize_comparison(model_sup, model_unsup, image1, image2, gt_disp=None, save_path=None):
    """
    Visualize disparity maps from supervised and unsupervised models
    
    Args:
        model_sup: supervised model
        model_unsup: unsupervised model
        image1: left image tensor [1, 3, H, W]
        image2: right image tensor [1, 3, H, W]
        gt_disp: ground truth disparity [1, 1, H, W] (optional)
        save_path: path to save visualization
    """
    model_sup.eval()
    model_unsup.eval()
    
    # Predict with both models
    padder = InputPadder(image1.shape, divis_by=32)
    image1_pad, image2_pad = padder.pad(image1, image2)
    
    _, flow_sup = model_sup(image1_pad, image2_pad, iters=32, test_mode=True)
    _, flow_unsup = model_unsup(image1_pad, image2_pad, iters=32, test_mode=True)
    
    flow_sup = padder.unpad(flow_sup[0]).cpu()
    flow_unsup = padder.unpad(flow_unsup[0]).cpu()
    
    disp_sup = -flow_sup[0].numpy()
    disp_unsup = -flow_unsup[0].numpy()
    
    # Create figure
    n_cols = 4 if gt_disp is not None else 3
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
    
    # Show input image
    img_vis = image1[0].permute(1, 2, 0).cpu().numpy() / 255.0
    axes[0, 0].imshow(img_vis)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Show supervised disparity
    im1 = axes[0, 1].imshow(disp_sup, cmap='magma', vmin=0, vmax=np.percentile(disp_sup, 95))
    axes[0, 1].set_title('Supervised Disparity')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Show unsupervised disparity
    im2 = axes[0, 2].imshow(disp_unsup, cmap='magma', vmin=0, vmax=np.percentile(disp_unsup, 95))
    axes[0, 2].set_title('Unsupervised Disparity')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    if gt_disp is not None:
        gt_vis = gt_disp[0, 0].cpu().numpy()
        im3 = axes[0, 3].imshow(gt_vis, cmap='magma', vmin=0, vmax=np.percentile(gt_vis, 95))
        axes[0, 3].set_title('Ground Truth')
        axes[0, 3].axis('off')
        plt.colorbar(im3, ax=axes[0, 3])
        
        # Error maps
        error_sup = np.abs(disp_sup - gt_vis)
        error_unsup = np.abs(disp_unsup - gt_vis)
        
        im4 = axes[1, 1].imshow(error_sup, cmap='hot', vmin=0, vmax=5)
        axes[1, 1].set_title('Supervised Error')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1])
        
        im5 = axes[1, 2].imshow(error_unsup, cmap='hot', vmin=0, vmax=5)
        axes[1, 2].set_title('Unsupervised Error')
        axes[1, 2].axis('off')
        plt.colorbar(im5, ax=axes[1, 2])
        
        # Difference map
        diff = error_unsup - error_sup
        im6 = axes[1, 3].imshow(diff, cmap='RdBu', vmin=-5, vmax=5)
        axes[1, 3].set_title('Error Difference (Unsup - Sup)')
        axes[1, 3].axis('off')
        plt.colorbar(im6, ax=axes[1, 3])
    else:
        # Just show disparity difference
        diff = disp_unsup - disp_sup
        im4 = axes[1, 1].imshow(diff, cmap='RdBu', vmin=-10, vmax=10)
        axes[1, 1].set_title('Disparity Difference (Unsup - Sup)')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1])
        
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def demo_visualization():
    """Generate comparison visualizations on test samples"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervised_model', required=True, help='path to supervised model checkpoint')
    parser.add_argument('--unsupervised_model', required=True, help='path to unsupervised model checkpoint')
    parser.add_argument('--dataset', default='kitti2015', choices=['kitti2015', 'kitti2012'])
    parser.add_argument('--output_dir', default='visualizations', help='output directory')
    parser.add_argument('--n_samples', type=int, default=5, help='number of samples to visualize')
    args = parser.parse_args()
    
    # Load models
    model_args = argparse.Namespace()
    model_args.corr_implementation = "reg"
    model_args.shared_backbone = False
    model_args.corr_levels = 4
    model_args.corr_radius = 4
    model_args.n_downsample = 2
    model_args.context_norm = "batch"
    model_args.slow_fast_gru = False
    model_args.n_gru_layers = 3
    model_args.hidden_dims = [128, 128, 128]
    model_args.mixed_precision = False
    
    model_sup = torch.nn.DataParallel(RAFTStereo(model_args))
    model_sup.load_state_dict(torch.load(args.supervised_model))
    model_sup.cuda()
    model_sup.eval()
    
    model_unsup = torch.nn.DataParallel(RAFTStereo(model_args))
    model_unsup.load_state_dict(torch.load(args.unsupervised_model))
    model_unsup.cuda()
    model_unsup.eval()
    
    # Load dataset
    if args.dataset == 'kitti2015':
        val_dataset = datasets.KITTI(root='datasets/KITTI2015', image_set='training', year=2015)
    elif args.dataset == 'kitti2012':
        val_dataset = datasets.KITTI(root='datasets/KITTI2012', image_set='training', year=2012)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Visualize samples
    indices = np.random.choice(len(val_dataset), min(args.n_samples, len(val_dataset)), replace=False)
    
    for idx in indices:
        _, image1, image2, flow_gt, valid_gt = val_dataset[int(idx)]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        # Convert flow to disparity
        disp_gt = -flow_gt[0:1, :, :].unsqueeze(0)
        
        save_path = output_dir / f'{args.dataset}_sample_{idx:04d}.png'
        visualize_comparison(model_sup, model_unsup, image1, image2, disp_gt, save_path)
    
    print(f"Generated {len(indices)} visualizations in {output_dir}")


if __name__ == '__main__':
    demo_visualization()
