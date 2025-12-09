from __future__ import print_function, division
import sys

import argparse
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
torch.cuda.empty_cache()
from PIL import Image

from tqdm import tqdm
from core.defom_stereo import DEFOMStereo, autocast

import core.stereo_datasets as datasets
from core.utils.utils import InputPadder


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def validate_things(model, iters=32, scale_iters=8, mixed_prec=False, max_disp=192, bad_threshold=1.0):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in tqdm(range(len(val_dataset))):
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        valid = data_blob["valid"]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            disp_pr = model(image1, image2, iters=iters, scale_iters=scale_iters, test_mode=True)
            end = time.time()
        if val_id > 50:
            elapsed_list.append(end-start)

        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0)

        epe = epe.flatten()
        val = (valid.flatten() >= 0.5) & (disp_gt.abs().flatten() < max_disp)

        if np.isnan(epe[val].mean().item()):
            continue
        out = (epe > bad_threshold)
        image_out = out[val].float().mean().item()
        image_epe = epe[val].mean().item()
        if val_id < 9 or (val_id+1) % 10 == 0:
            logging.info(f"Fhythings3D Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} Out{bad_threshold} {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")

        epe_list.append(image_epe)
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    out = 100 * np.mean(out_list)
    avg_runtime = np.mean(elapsed_list)

    print(f"Validation FlyingThings: EPE {epe}, Out{bad_threshold} {out}, "
          f"{format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'things-epe': epe, 'things-out': out}


@torch.no_grad()
def validate_eth3d(model, iters=32, scale_iters=8, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params, is_eval=True)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        valid = data_blob["valid"]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            disp_pr = model(image1, image2, iters=iters, scale_iters=scale_iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0)

        epe_flattened = epe.flatten()
        val = valid.flatten() >= 0.5
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    out1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, Out1 %f" % (epe, out1))
    return {'eth3d-epe': epe, 'eth3d-out1': out1}


@torch.no_grad()
def validate_kitti(model, iters=32, scale_iters=8, split='15', mixed_prec=False):
    """ Peform validation using the KITTI-2015/2012 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, split=split, image_set='training', is_eval=True)
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        valid = data_blob["valid"]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            disp_pr = model(image1, image2, iters=iters, scale_iters=scale_iters, test_mode=True)
            end = time.time()
        if val_id > 50:
            elapsed_list.append(end-start)

        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0)

        epe_flattened = epe.flatten()
        val = valid.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1) % 10 == 0:
            logging.info(f"KITTI{split} Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} Out3 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    out3 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI{split}: EPE {epe}, Out3 {out3}, "
          f"{format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {f'kitti{split}-epe': epe, f'kitti{split}-out3': out3}


@torch.no_grad()
def validate_middlebury(model, iters=32, scale_iters=8, split='H', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split, is_eval=True)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        valid = data_blob["valid"]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            disp_pr = model(image1, image2, iters=iters, scale_iters=scale_iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0)

        epe_flattened = epe.flatten()
        val = (valid.reshape(-1) >= 0.5) & (disp_gt.reshape(-1) < 1000)

        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. "
                     f"EPE {round(image_epe,4)} Out2 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    out2 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}: EPE {epe}, Out2 {out2}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-out2': out2}


def compute_nontexture(x, weight=None, c1=0.01**2, c2=0.03**2, weight_epsilon=0.01, window=33, threshold=0.95, split="F"):

    if split=="H":
        scale = 2
        threshold += 0.02
    elif split=="Q":
        scale = 4
        threshold += 0.03
    else:
        scale = 1
    
    x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)
    
    if x.max()>1:
        x = x/x.max()
    
    y = F.pad(x, (1, 1, 1, 1), mode='replicate')
    _, _, h, w = y.shape
    #y = y[..., 0:h-2, 1:w-1] #(y[..., 0:h-2, 1:w-1] + y[..., 2:h, 1:w-1] + y[..., 1:h-1, 0:w-2] + y[..., 1:h-1, 2:w])/4.0

    x = F.pad(x, (window//2, window//2, window//2, window//2), mode='replicate')
    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError(
            'Both c1 and c2 are infinite, SSIM loss is zero. This is '
            'likely unintended.')
    _, _, H, W = x.shape

    if weight is None:
        weight = torch.ones((H, W)).to(x)
    else:
        assert weight.shape == (H, W), \
                f'image shape is {(H, W)}, but weight shape is {weight.shape}'
    weight = weight[None, None, ...]
    average_pooled_weight = F.avg_pool2d(weight, (window, window), stride=(1, 1))
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (
        average_pooled_weight + weight_epsilon)

    def weighted_avg_pool(z):
        weighted_avg = F.avg_pool2d(
            z * weight_plus_epsilon, (window, window), stride=(1, 1))
        return weighted_avg * inverse_average_pooled_weight
    
    mu_x = weighted_avg_pool(x)
    sigma_x = weighted_avg_pool(x**2) - mu_x**2

    def ssim(x, y):
        y = F.pad(y, (window//2, window//2, window//2, window//2), mode='replicate')
        mu_y = weighted_avg_pool(y)
        sigma_y = weighted_avg_pool(y**2) - mu_y**2
        sigma_xy = weighted_avg_pool(x * y) - mu_x * mu_y
        if c1 == float('inf'):
            ssim_n = (2 * sigma_xy + c2)
            ssim_d = (sigma_x + sigma_y + c2)
        elif c2 == float('inf'):
            ssim_n = 2 * mu_x * mu_y + c1
            ssim_d = mu_x**2 + mu_y**2 + c1
        else:
            ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)

            result = ssim_n / ssim_d

        result = F.avg_pool2d(result, (scale, scale), stride=(scale, scale))

        return result
    
    mask = (ssim(x, y[..., 0:h-2, 1:w-1])>threshold) & (ssim(x, y[..., 2:h, 1:w-1])>threshold) & (ssim(x, y[..., 1:h-1, 0:w-2])>threshold) & (ssim(x, y[..., 1:h-1, 2:w])>threshold)
    mask = mask[0, 0] & mask[0, 1] & mask[0, 2]
    
    return mask.cpu().numpy()


@torch.no_grad()
def validate_middlebury_indetail(model, iters=32, scale_iters=8, split='H', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split, is_eval=True)

    out_list, epe_list, portion_list = [[], [], [], []], [[], [], [], []], [[], [], [], []]
    for val_id in range(len(val_dataset)):
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        valid = data_blob["valid"]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            disp_pr = model(image1, image2, iters=iters, scale_iters=scale_iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0)
        
        epe_flattened = epe.flatten()
        
        occ_mask = Image.open(data_blob["imageL_file"].replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()
        val_all = (valid.reshape(-1) >= 0.5) & (disp_gt.reshape(-1) < 1000)
        val_occ = val_all & (occ_mask==128)
        val_nocc = val_all & (occ_mask==255)

        val_ntt = val_all & compute_nontexture(data_blob["img1"][None].cuda(), split=split).flatten()
        
        out = (epe_flattened > 2.0)
        image_out = out[val_all].float().mean().item()
        image_epe = epe_flattened[val_all].mean().item()

        image_out_occ = out[val_occ].float().mean().item()
        image_epe_occ = epe_flattened[val_occ].mean().item()

        image_out_nocc = out[val_nocc].float().mean().item()
        image_epe_nocc = epe_flattened[val_nocc].mean().item()

        image_out_ntt = out[val_ntt].float().mean().item()
        image_epe_ntt = epe_flattened[val_ntt].mean().item()

        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. "
                     f"All({round((val_all.sum()/val_all.sum()).item(),4)}): EPE {round(image_epe,4)} Out2 {round(image_out,4)}, \n "
                     f"Occ({round((val_occ.sum()/val_all.sum()).item(),4)}): EPE {round(image_epe_occ,4)} Out2 {round(image_out_occ,4)}, "
                     f"NOcc({round((val_nocc.sum()/val_all.sum()).item(),4)}): EPE {round(image_epe_nocc,4)} Out2 {round(image_out_nocc,4)}, "
                     f"NonTexture({round((val_ntt.sum()/val_all.sum()).item(),4)}): EPE {round(image_epe_ntt,4)} Out2 {round(image_out_ntt,4)}")
        
        epe_list[0].append(image_epe)
        out_list[0].append(image_out)
        portion_list[0].append((val_all.sum()/val_all.sum()).item())
        epe_list[1].append(image_epe_occ)
        out_list[1].append(image_out_occ)
        portion_list[1].append((val_occ.sum()/val_all.sum()).item())
        epe_list[2].append(image_epe_nocc)
        out_list[2].append(image_out_nocc)
        portion_list[2].append((val_nocc.sum()/val_all.sum()).item())
        epe_list[3].append(image_epe_ntt)
        out_list[3].append(image_out_ntt)
        portion_list[3].append((val_ntt.sum()/val_all.sum()).item())

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)
    portion_list = np.array(portion_list)

    epe = np.mean(epe_list, axis=1)
    out2 = 100 * np.mean(out_list, axis=1)
    portion = 100 * np.mean(portion_list, axis=1)

    print(f"Validation Middlebury{split}: All({round(portion[0],8)}%): EPE {round(epe[0],8)} Out2 {round(out2[0],8)}, \n"
                     f"Occ({round(portion[1],8)}%): EPE {round(epe[1],8)} Out2 {round(out2[1],8)}, "
                     f"NOcc({round(portion[2],8)}%): EPE {round(epe[2],8)} Out2 {round(out2[2],8)}, "
                     f"NonTexture({round(portion[3],8)}%): EPE {round(epe[3],8)} Out2 {round(out2[3],8)}")
    return {f'middlebury{split}-epe': epe[0], f'middlebury{split}-out2': out2[0]}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--datasets', nargs='+', type=str, help="dataset for evaluation", default=["things"],
                        choices=["things", "eth3d", "kitti12", "kitti15"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--indetail', action='store_true', help='evaluate middlebury in detail (for different regions)')
    
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of disparity field updates during forward pass')
    parser.add_argument('--scale_iters', type=int, default=8, help="number of scaling updates to the disparity field in each forward pass.")

    # Architecure choices
    parser.add_argument('--dinov2_encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--idepth_scale', type=float, default=0.5, help="the scale of inverse depth to initialize disparity")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                        help='the list of scaling factors of disparity')
    parser.add_argument('--scale_corr_radius', type=int, default=2,
                        help="width of the correlation pyramid for scaled disparity")

    parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3], help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    args = parser.parse_args()

    model = DEFOMStereo(args)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt, map_location='cuda')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)[1]/1e6, '.2f')}M learnable parameters.")

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if 'things' in args.datasets:
        validate_things(model, iters=args.valid_iters, scale_iters=args.scale_iters, mixed_prec=use_mixed_precision)

    if 'eth3d' in args.datasets:
        validate_eth3d(model, iters=args.valid_iters, scale_iters=args.scale_iters, mixed_prec=use_mixed_precision)

    if 'kitti12' in args.datasets:
        validate_kitti(model, iters=args.valid_iters, scale_iters=args.scale_iters, split='12', mixed_prec=use_mixed_precision)

    if 'kitti15' in args.datasets:
        validate_kitti(model, iters=args.valid_iters, scale_iters=args.scale_iters, split='15', mixed_prec=use_mixed_precision)

    for s in 'FHQ':
        if f"middlebury_{s}" in args.datasets:
            if args.indetail:
                validate_middlebury_indetail(model, iters=args.valid_iters, scale_iters=args.scale_iters, split=s, mixed_prec=use_mixed_precision)
            else:
                validate_middlebury(model, iters=args.valid_iters, scale_iters=args.scale_iters, split=s, mixed_prec=use_mixed_precision)

