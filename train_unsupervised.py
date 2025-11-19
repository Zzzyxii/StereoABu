from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.raft_stereo import RAFTStereo
from core.unsupervised_loss import unsupervised_loss
from core.dataset_split import split_dataset, sample_vkitti2

import core.stereo_datasets as datasets

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, log_dir='runs_unsup'):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=log_dir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs_unsup')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs_unsup')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    model = nn.DataParallel(RAFTStereo(args))
    print("Parameter Count: %d" % count_parameters(model))

    # Load datasets
    aug_params = {
        'crop_size': args.image_size,
        'min_scale': args.spatial_scale[0],
        'max_scale': args.spatial_scale[1],
        'do_flip': False,
        'yjitter': not args.noyjitter
    }
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    # Create datasets with unsupervised flag
    if args.stage == 'pretrain':
        # vKITTI2 pretraining with sampling
        logging.info("Pretraining stage: using vKITTI2")
        full_dataset = datasets.VKITTI(aug_params, root=args.vkitti2_path)
        train_dataset, val_dataset = sample_vkitti2(full_dataset, train_size=1000, val_size=200, seed=42)
        train_dataset.unsupervised = True
        logging.info(f"vKITTI2 - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    elif args.stage == 'finetune':
        # KITTI finetuning with 4:1 split
        logging.info("Finetuning stage: using KITTI 2012 & 2015")
        kitti15_full = datasets.KITTI(aug_params, root=args.kitti2015_path, year=2015)
        kitti15_train, kitti15_val = split_dataset(kitti15_full, train_ratio=0.8, seed=42)
        kitti15_train.unsupervised = True
        
        kitti12_full = datasets.KITTI(aug_params, root=args.kitti2012_path, year=2012)
        kitti12_train, kitti12_val = split_dataset(kitti12_full, train_ratio=0.8, seed=42)
        kitti12_train.unsupervised = True
        
        train_dataset = kitti15_train + kitti12_train
        val_dataset = kitti15_val + kitti12_val
        logging.info(f"KITTI - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        pin_memory=True, 
        shuffle=True, 
        num_workers=4,
        drop_last=True
    )

    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler, log_dir=f'runs_unsup_{args.stage}')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.train()
    model.module.freeze_bn()

    validation_frequency = args.val_freq
    scaler = GradScaler(enabled=args.mixed_precision)

    # Loss weights
    loss_weights = {
        'photo': args.photo_weight,
        'smooth': args.smooth_weight,
        'lr': args.lr_weight
    }

    should_keep_training = True
    global_batch_num = 0
    
    while should_keep_training:
        for i_batch, (_, img1, img2, _, _) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            img1, img2 = img1.cuda(), img2.cuda()

            # Normalize images to [0, 1]
            img1_norm = img1 / 255.0
            img2_norm = img2 / 255.0

            assert model.training
            
            # Forward pass - predict disparity from both views
            flow_predictions_left = model(img1, img2, iters=args.train_iters)
            flow_predictions_right = model(img2, img1, iters=args.train_iters)
            
            # Get final predictions (negative of flow[0] is disparity)
            disp_left = -flow_predictions_left[-1][:, 0:1, :, :]
            disp_right = -flow_predictions_right[-1][:, 0:1, :, :]

            # Compute unsupervised loss
            loss, loss_dict = unsupervised_loss(
                disp_left, disp_right, img1_norm, img2_norm,
                loss_weights=loss_weights,
                smooth_order=args.smooth_order
            )

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss detected at step {total_steps}, skipping batch")
                continue

            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            for k, v in loss_dict.items():
                logger.writer.add_scalar(f'loss/{k}', v, global_batch_num)
            
            global_batch_num += 1
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(loss_dict)

            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path(f'checkpoints_unsup/{args.stage}') / f'{total_steps + 1}_{args.name}.pth'
                save_path.parent.mkdir(exist_ok=True, parents=True)
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    print("FINISHED TRAINING")
    logger.close()
    
    save_path = Path(f'checkpoints_unsup/{args.stage}') / f'{args.name}_final.pth'
    save_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), save_path)

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-stereo-unsup', help="name your experiment")
    parser.add_argument('--stage', required=True, choices=['pretrain', 'finetune'], help="training stage")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="image crop size")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to disparity field")
    parser.add_argument('--wdecay', type=float, default=.00001, help="weight decay")
    parser.add_argument('--val_freq', type=int, default=5000, help="validation/checkpoint frequency")

    # Architecture choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument('--shared_backbone', action='store_true')
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'])
    parser.add_argument('--slow_fast_gru', action='store_true')
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None)
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4])
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'])
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4])
    parser.add_argument('--noyjitter', action='store_true')
    
    # Dataset paths
    parser.add_argument('--kitti2015_path', type=str, default='datasets/KITTI2015')
    parser.add_argument('--kitti2012_path', type=str, default='datasets/KITTI2012')
    parser.add_argument('--vkitti2_path', type=str, default='datasets/vKITTI2')

    # Unsupervised loss weights
    parser.add_argument('--photo_weight', type=float, default=1.0, help="photometric loss weight")
    parser.add_argument('--smooth_weight', type=float, default=0.1, help="smoothness loss weight")
    parser.add_argument('--lr_weight', type=float, default=0.1, help="left-right consistency loss weight")
    parser.add_argument('--smooth_order', type=int, default=1, choices=[1, 2], help="smoothness order")

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints_unsup").mkdir(exist_ok=True, parents=True)

    train(args)
