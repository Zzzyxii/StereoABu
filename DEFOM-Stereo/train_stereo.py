from __future__ import print_function, division
import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
from utils.utils import *
from core.defom_stereo import DEFOMStereo

from evaluate_stereo import validate_things, count_parameters
import core.stereo_datasets as datasets

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
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


def train(args):

    seed_everything(args.seed)

    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True

        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))

        setup_for_distributed(args.local_rank == 0)

    model = DEFOMStereo(args).to(device)
    print("Parameter Count: %d, Trainable: %d" % count_parameters(model))

    if args.distributed:
        process_group = torch.distributed.new_group(list(range(len(args.gpu_ids))))
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
            model_without_ddp = model.module
        else:
            model_without_ddp = model

        model_without_ddp.freeze_bn()  # BatchNorm kept frozen if not distributed
        
    start_epoch = 0
    start_step = 0
    optimizer, scheduler = fetch_optimizer(args, model)

    if args.resume_ckpt:
        assert args.resume_ckpt.endswith(".pth")
        logging.info("Loading checkpoint: %s" % args.resume_ckpt)
        loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume_ckpt, map_location=loc)
        if 'model' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)
        else:
            model_without_ddp.load_state_dict(checkpoint, strict=args.strict_resume)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer:
            print('Load optimizer')
            start_step = checkpoint['step']
            start_epoch = checkpoint['epoch']
            del optimizer, scheduler
            optimizer, scheduler = fetch_optimizer(args, model, start_step, checkpoint)

    train_data = datasets.fetch_dataset(args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank
        )
    else:
        train_sampler = None
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=train_sampler is None,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              sampler=train_sampler)

    total_steps = start_step
    epoch = start_epoch
    logger = Logger(model, scheduler, args.name)
    logger.total_steps = total_steps

    model.train()
    scaler = GradScaler(enabled=args.mixed_precision)
    should_keep_training = True

    while should_keep_training:

        # mannually change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if total_steps == start_step:
            epoch_start_step = start_step - len(train_loader)*start_epoch
        else:
            epoch_start_step = 0

        for i_batch, data_blob in enumerate(tqdm(train_loader, initial=epoch_start_step)):
            optimizer.zero_grad()
            image1 = data_blob["img1"].cuda()
            image2 = data_blob["img2"].cuda()
            disp_gt = data_blob["disp"].cuda()
            valid = data_blob["valid"].cuda()

            assert model.training
            disp_predictions = model(image1, image2, iters=args.train_iters, scale_iters=args.scale_iters)
            assert model.training

            loss, metrics = sequence_loss(disp_predictions, disp_gt, valid)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            total_steps += 1

            if args.local_rank == 0:
                logger.writer.add_scalar("train/live_loss", loss.item(), total_steps)
                logger.writer.add_scalar(f'train/learning_rate', optimizer.param_groups[0]['lr'], total_steps)
                logger.push(metrics)

                if total_steps % args.save_latest_ckpt_freq == 0:
                    save_path = Path('checkpoints/%s/checkpoint_latest.pth' % (args.name))
                    logging.info(f"Saving file {save_path.absolute()}")
                    save_dict = { 'model': model_without_ddp.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'step': total_steps,
                                  'epoch': epoch}
                    torch.save(save_dict, save_path)

                if total_steps % args.save_ckpt_freq == 0:
                    save_path = Path('checkpoints/%s/%s_%6d.pth' % (args.name, args.name, total_steps))
                    logging.info(f"Saving file {save_path.absolute()}")
                    torch.save(model_without_ddp.state_dict(), save_path)

                if total_steps % args.val_freq == 0:

                    # visualizing training results with tensorboard
                    disp = disp_predictions[-1]

                    for j in range(min(4, args.batch_size)):  # write a maxmimum of four images
                        logger.writer.add_image("image1/{}".format(j), image1[j].data.type(torch.uint8), total_steps)
                        logger.writer.add_image("image2/{}".format(j), image2[j].data.type(torch.uint8), total_steps)
                        logger.writer.add_image("disp/{}".format(j),
                                                (disp[j]).data.type(torch.uint8), total_steps)
                        logger.writer.add_image("gt_disp/{}".format(j),
                                                (disp_gt[j]).data.type(torch.uint8), total_steps)

                    results  = validate_things(model_without_ddp, args.valid_iters, args.scale_iters)
                    logger.write_dict(results)
                    model.train()
                    if not args.distributed: model_without_ddp.freeze_bn()

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        epoch += 1

        if len(train_loader) >= 10000:
            save_path = Path('checkpoints/%s/%d_epoch_%s.pth.gz' % (args.name, total_steps, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model_without_ddp.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model_without_ddp.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='defom-stereo', help="name your experiment")

    # resume pretrained model or resume training
    parser.add_argument('--resume_ckpt', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--train_folds', type=int, nargs='+', default=[1], help="training datasets' folds.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=18, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--scale_iters', type=int, default=8, help="number of scaling updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--seed', default=1234, type=int)

    # log
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--save_ckpt_freq', default=10000, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
    parser.add_argument('--val_freq', default=10000, type=int, help='validation frequency in terms of training steps')

    # distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--launcher', default='none', type=str)
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of disparity field updates during validation forward pass')

    # Raft Architecure choices
    parser.add_argument('--dinov2_encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--idepth_scale', type=float, default=0.5, help="the scale of inverse depth to initialize disparity")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")

    parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                        help='the list of scaling factors of disparity')
    parser.add_argument('--scale_corr_radius', type=int, default=2, help="width of the correlation pyramid for scaled disparity")

    parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3], help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0.0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default='v', choices=['v', 'None'], help='flip the images vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    Path("checkpoints/"+args.name).mkdir(exist_ok=True, parents=True)

    train(args)
