from __future__ import print_function, division

import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
import time
import os
import cv2
import sys


from core.defom_stereo import DEFOMStereo, autocast

import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from core.utils.frame_utils import writePFM


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def StrToBytes(text):
    if sys.version_info[0] == 2:
        return text
    else:
        return bytes(text, 'UTF-8')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def test_eth3d(model, save_path, iters=32, scale_iters=3, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}

    test_dataset = datasets.ETH3D(aug_params, split='testing', is_test=True)
    training_dataset = datasets.ETH3D(aug_params, split='training', is_test=True)
    dataset = test_dataset + training_dataset
    torch.backends.cudnn.benchmark = True

    for test_id in tqdm(range(len(dataset))):
        img1, img2, imageL_file = dataset[test_id]
        image1 = img1[None].cuda()
        image2 = img2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            disp_pr = model(image1, image2, iters=iters, scale_iters=scale_iters, test_mode=True)
            end = time.time()
        runtime = end - start
        disp = padder.unpad(disp_pr).cpu().squeeze().numpy()
        disp[disp < 0] = 0
        disp[disp > 64] = 64

        names = imageL_file.split("/")
        save_sub_path = os.path.join(save_path, "low_res_"+names[-3])
        makedirs(save_sub_path)

        disp_path = os.path.join(save_sub_path, names[-2] + '.pfm')
        writePFM(disp_path, disp)

        txt_path = os.path.join(save_sub_path, names[-2] + '.txt')
        with open(txt_path, 'wb') as time_file:
            time_file.write(StrToBytes('runtime ' + str(runtime)))


@torch.no_grad()
def test_kitti(model, save_path, iters=32, scale_iters=3, split='15', mixed_prec=False):
    """ Peform testing on the KITTI-2015 (test) split """
    model.eval()
    aug_params = {}
    save_path = os.path.join(save_path, "disp_0")
    makedirs(save_path)

    test_dataset = datasets.KITTI(aug_params, split=split, image_set='testing', is_test=True)

    runtime_sum = 0.0
    runtime_count = 0

    for test_id in tqdm(range(len(test_dataset))):
        img1, img2, imageL_file = test_dataset[test_id]
        image1 = img1[None].cuda()
        image2 = img2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            disp_pr = model(image1, image2, iters=iters, scale_iters=scale_iters, test_mode=True)
            end = time.time()
        runtime = end - start
        runtime_sum += runtime
        runtime_count += 1

        disp = padder.unpad(disp_pr).cpu().squeeze().numpy()
        disp[disp < 0] = 0
        disp[disp > 240] = 240
        disp = np.uint16(disp*256)

        name = imageL_file.split('/')[-1]
        path = os.path.join(save_path, name)
        cv2.imwrite(path, disp, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    print('The average runtime on Kitti test images is (you will need this for the submission): '
          + str(runtime_sum / runtime_count) + " seconds")


@torch.no_grad()
def test_middlebury(model, save_path, iters=32, scale_iters=8, split='F', mixed_prec=False, method_name="DEFOM-Stereo"):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    test_dataset = datasets.Middlebury(aug_params, split=split, image_set='test', is_test=True)
    training_dataset = datasets.Middlebury(aug_params, split=split, image_set='training', is_test=True)
    dataset = test_dataset + training_dataset
    torch.backends.cudnn.benchmark = True

    for test_id in tqdm(range(len(dataset))):
        img1, img2, imageL_file = dataset[test_id]
        image1 = img1[None].cuda()
        image2 = img2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            disp_pr = model(image1, image2, iters=iters, scale_iters=scale_iters, test_mode=True)
            end = time.time()
        runtime = end - start
        disp = padder.unpad(disp_pr).cpu().squeeze().numpy()
        disp[disp < 0] = 0
        disp[disp > 800] = 800

        names = imageL_file.split("/")
        save_sub_path = os.path.join(save_path, names[-3], names[-2])
        makedirs(save_sub_path)

        disp_path = os.path.join(save_sub_path, 'disp0' + method_name + '.pfm')
        writePFM(disp_path, disp)

        txt_path = os.path.join(save_sub_path, 'time' + method_name + '.txt')
        with open(txt_path, 'wb') as time_file:
            time_file.write(StrToBytes(str(runtime)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--datasets', nargs='+', type=str, help="dataset for evaluation", default=["kitti12", "kitti15"],
                        choices=["eth3d", "kitti12", "kitti15"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of disparity field updates during forward pass')
    parser.add_argument('--scale_iters', type=int, default=8, help="number of scaling updates to the disparity field in each forward pass.")
    parser.add_argument('--method_name', default="DEFOM-Stereo", help="the method to test")

    # Architecure choices
    parser.add_argument('--dinov2_encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--idepth_scale', type=float, default=0.5,
                        help="the scale of inverse depth to initialize disparity")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                        help='the list of scaling factors of disparity')
    parser.add_argument('--scale_corr_radius', type=int, default=2,
                        help="width of the correlation pyramid for scaled disparity")

    parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3],
                        help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'],
                        help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    args = parser.parse_args()

    model = DEFOMStereo(args)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt, map_location='cuda')
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors.
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if 'eth3d' in args.datasets:
        save_path = os.path.abspath(args.restore_ckpt).split('.')[0] + '_' + "eth3d"
        makedirs(save_path)
        test_eth3d(model, save_path, iters=args.valid_iters, scale_iters=args.scale_iters, mixed_prec=use_mixed_precision)

    if 'kitti12' in args.datasets:
        save_path = os.path.abspath(args.restore_ckpt).split('.')[0] + '_' + "kitti12"
        makedirs(save_path)
        test_kitti(model, save_path, iters=args.valid_iters, scale_iters=args.scale_iters, mixed_prec=use_mixed_precision, split='12')

    if 'kitti15' in args.datasets:
        save_path = os.path.abspath(args.restore_ckpt).split('.')[0] + '_' + "kitti15"
        makedirs(save_path)
        test_kitti(model, save_path, iters=args.valid_iters, scale_iters=args.scale_iters, mixed_prec=use_mixed_precision, split='15')

    for s in 'FHQ':
        if f"middlebury_{s}" in args.datasets:
            save_path = os.path.abspath(args.restore_ckpt).split('.')[0] + '_' + f"middlebury_{s}"
            makedirs(save_path)
            test_middlebury(model, save_path, iters=args.valid_iters, scale_iters=args.scale_iters, split=s,
                            method_name=args.method_name, mixed_prec=use_mixed_precision)

