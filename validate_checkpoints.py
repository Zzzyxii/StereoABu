from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from raft_stereo import RAFTStereo, autocast
import stereo_datasets as datasets
from utils.utils import InputPadder

@torch.no_grad()
def validate_kitti(model, root, year, use_noc=False, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, root=root, image_set='training', year=year, use_noc=use_noc)
    torch.backends.cudnn.benchmark = True

    out_list, epe_list = [], []
    
    print(f"Validating on KITTI {year} (use_noc={use_noc})...")
    
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation KITTI {year} (NOC={use_noc}): EPE {epe:.4f}, D1 {d1:.4f}")
    return {'epe': epe, 'd1': d1}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--kitti2012_path', type=str, default='datasets/KITTI2012')
    parser.add_argument('--kitti2015_path', type=str, default='datasets/KITTI2015')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices (must match training)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--context_norm', type=str, default="batch")
    parser.add_argument('--shared_backbone', action='store_true')
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--slow_fast_gru', action='store_true')
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)

    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.eval()

    validate_kitti(model, args.kitti2012_path, year=2012, use_noc=True, iters=args.valid_iters, mixed_prec=args.mixed_precision)
    validate_kitti(model, args.kitti2015_path, year=2015, use_noc=True, iters=args.valid_iters, mixed_prec=args.mixed_precision)
