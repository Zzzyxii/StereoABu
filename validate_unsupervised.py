from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.raft_stereo import RAFTStereo
from core.dataset_split import sample_vkitti2, split_dataset
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder


@torch.no_grad()
def compute_metrics(
    model: RAFTStereo,
    data_loader: DataLoader,
    iters: int,
    device: torch.device,
    use_amp: bool,
    save_dir: Optional[Path],
    max_visuals: int,
) -> Dict[str, float]:
    """Evaluate a RAFT-Stereo model on a validation loader and report disparity metrics."""
    model.eval()

    total_epe = 0.0
    total_bad1 = 0.0
    total_bad3 = 0.0
    total_px = 0
    visuals_saved = 0

    for _, img_l, img_r, flow_gt, valid in tqdm(data_loader, desc="Validating", leave=False):
        img_l = img_l.to(device)
        img_r = img_r.to(device)
        flow_gt = flow_gt.to(device)
        valid = valid.to(device)

        padder = InputPadder(img_l.shape, divis_by=32)
        img_l_pad, img_r_pad = padder.pad(img_l, img_r)

        autocast_enabled = torch.cuda.is_available() and use_amp
        with amp.autocast(enabled=autocast_enabled):
            _, flow_pred = model(img_l_pad, img_r_pad, iters=iters, test_mode=True)

        flow_pred = padder.unpad(flow_pred)

        disp_pred = -flow_pred[:, 0:1]
        disp_gt = -flow_gt[:, 0:1]

        valid_mask = valid.unsqueeze(1) >= 0.5
        if valid_mask.sum() == 0:
            continue

        diff = torch.abs(disp_pred - disp_gt)

        total_epe += diff[valid_mask].sum().item()
        total_bad1 += (diff > 1.0)[valid_mask].float().sum().item()
        total_bad3 += (diff > 3.0)[valid_mask].float().sum().item()
        total_px += valid_mask.sum().item()

        if save_dir is not None and visuals_saved < max_visuals:
            batch = disp_pred.shape[0]
            for b_idx in range(batch):
                if visuals_saved >= max_visuals:
                    break

                mask_np = valid_mask[b_idx].squeeze(0).detach().cpu().numpy()
                if mask_np.sum() == 0:
                    continue

                disp_pred_np = disp_pred[b_idx].squeeze(0).detach().cpu().numpy()
                disp_gt_np = disp_gt[b_idx].squeeze(0).detach().cpu().numpy()
                err_np = np.abs(disp_pred_np - disp_gt_np)

                err_np = np.where(mask_np, err_np, np.nan)
                disp_pred_masked = np.where(mask_np, disp_pred_np, np.nan)
                disp_gt_masked = np.where(mask_np, disp_gt_np, np.nan)

                vmax = np.nanpercentile(disp_gt_masked, 99)
                vmax = max(vmax, 1.0)
                err_max = np.nanpercentile(err_np, 99)
                err_max = max(err_max, 1.0)

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                axes[0].set_title("Predicted disparity")
                im0 = axes[0].imshow(disp_pred_masked, cmap="plasma", vmin=0, vmax=vmax)
                fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                axes[0].axis("off")

                axes[1].set_title("GT disparity")
                im1 = axes[1].imshow(disp_gt_masked, cmap="plasma", vmin=0, vmax=vmax)
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                axes[1].axis("off")

                axes[2].set_title("Error heatmap")
                im2 = axes[2].imshow(err_np, cmap="inferno", vmin=0, vmax=err_max)
                fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                axes[2].axis("off")

                fig.tight_layout()
                out_path = save_dir / f"sample_{visuals_saved:04d}.png"
                fig.savefig(out_path, dpi=200)
                plt.close(fig)

                visuals_saved += 1

    metrics = {
        "EPE": total_epe / total_px,
        "Bad-1": total_bad1 / total_px,
        "Bad-3": total_bad3 / total_px,
    }
    return metrics


def build_val_loader(args: argparse.Namespace) -> DataLoader:
    aug_params = {
        "crop_size": args.image_size,
        "min_scale": args.spatial_scale[0],
        "max_scale": args.spatial_scale[1],
        "do_flip": False,
        "yjitter": not args.noyjitter,
    }

    if args.stage == "pretrain":
        base_dataset = datasets.VKITTI(aug_params=None, root=args.vkitti2_path)
        _, val_dataset = sample_vkitti2(
            base_dataset,
            train_size=args.pretrain_train_size,
            val_size=args.pretrain_val_size,
            seed=args.split_seed,
        )
        val_dataset.unsupervised = False
    elif args.stage == "finetune":
        base_kitti15 = datasets.KITTI(aug_params=None, root=args.kitti2015_path, year=2015)
        base_kitti12 = datasets.KITTI(aug_params=None, root=args.kitti2012_path, year=2012)
        _, kitti15_val = split_dataset(base_kitti15, train_ratio=args.train_ratio, seed=args.split_seed)
        _, kitti12_val = split_dataset(base_kitti12, train_ratio=args.train_ratio, seed=args.split_seed)
        kitti15_val.unsupervised = False
        kitti12_val.unsupervised = False
        val_dataset = kitti15_val + kitti12_val
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate unsupervised RAFT-Stereo checkpoints")
    parser.add_argument("--restore_ckpt", required=True, help="path to checkpoint to evaluate")
    parser.add_argument("--stage", choices=["pretrain", "finetune"], required=True, help="dataset stage to evaluate")
    parser.add_argument("--mixed_precision", action="store_true", help="use autocast during evaluation")

    parser.add_argument("--vkitti2_path", type=str, default="datasets/vKITTI2")
    parser.add_argument("--kitti2015_path", type=str, default="datasets/KITTI2015")
    parser.add_argument("--kitti2012_path", type=str, default="datasets/KITTI2012")

    parser.add_argument("--pretrain_train_size", type=int, default=1000, help="number of vKITTI2 images for training split")
    parser.add_argument("--pretrain_val_size", type=int, default=200, help="number of vKITTI2 images for validation split")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train ratio for KITTI datasets")
    parser.add_argument("--split_seed", type=int, default=42, help="random seed for dataset splitting")

    parser.add_argument("--image_size", type=int, nargs=2, default=[320, 720], help="image size (unused for val aug, kept for compatibility)")
    parser.add_argument("--spatial_scale", type=float, nargs=2, default=[-0.2, 0.4])
    parser.add_argument("--noyjitter", action="store_true")

    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--valid_iters", type=int, default=32, help="number of RAFT iterations during evaluation")
    parser.add_argument("--save_visuals", type=str, default=None, help="directory to save disparity/error visualizations")
    parser.add_argument("--max_visuals", type=int, default=0, help="maximum number of samples to visualize")

    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128] * 3)
    parser.add_argument("--corr_implementation", choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=4)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--context_norm", type=str, default="batch", choices=['group', 'batch', 'instance', 'none'])
    parser.add_argument("--slow_fast_gru", action="store_true")
    parser.add_argument("--n_gru_layers", type=int, default=3)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.nn.DataParallel(RAFTStereo(args)).to(device)

    ckpt_path = Path(args.restore_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logging.info("Loading checkpoint %s", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    val_loader = build_val_loader(args)

    save_dir = None
    if args.save_visuals:
        save_dir = Path(args.save_visuals)
        save_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(
        model.module,
        val_loader,
        args.valid_iters,
        device,
        args.mixed_precision,
        save_dir,
        args.max_visuals,
    )
    logging.info("Validation results: EPE %.4f | Bad-1 %.4f | Bad-3 %.4f", metrics["EPE"], metrics["Bad-1"], metrics["Bad-3"])


if __name__ == "__main__":
    main()
