import torch
import torch.nn.functional as F
import numpy as np
import argparse
from core.raft_stereo import RAFTStereo


# --------------------------------------
#  Utility: warp img2 to img1 using disp
# --------------------------------------
def warp_with_disp(img2, disp):
    """
    img2: (B, 3, H, W)
    disp: (B, 1, H, W) pixel displacement in x-direction
    """
    b, c, h, w = img2.shape

    # Build normalized coordinates [-1, 1]
    xs = torch.linspace(-1, 1, w, device=img2.device)
    ys = torch.linspace(-1, 1, h, device=img2.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)  # (B,H,W,2)

    # pixel disp -> normalized disp
    disp_norm = disp / ((w - 1) / 2.0)  # (pixel -> [-1,1])
    grid[:, :, :, 0] = grid[:, :, :, 0] - disp_norm.squeeze(1)

    warped = F.grid_sample(img2, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped


# --------------------------------------
#  compute photometric error
# --------------------------------------
def photometric_error(img1, warp_img):
    return torch.mean(torch.abs(img1 - warp_img))


# --------------------------------------
#  main synthetic test
# --------------------------------------
def run_test(args):

    print("\n====== Loading RAFTStereo ======\n")
    model = RAFTStereo(args).cuda().eval()

    # --------------------------------------
    # Synthetic Test Data
    # --------------------------------------
    print("====== Creating synthetic stereo pair ======\n")

    H, W = 320, 720
    shift_pixels = 8  # known ground truth shift

    img_left = torch.rand(1, 3, H, W).cuda()
    img_right = torch.roll(img_left, shifts=-shift_pixels, dims=3)  # shift right image 8px left

    print(f"Synthetic image pair created. True disparity = {shift_pixels} pixels\n")

    # --------------------------------------
    # Forward pass
    # --------------------------------------
    print("====== Running RAFTStereo forward ======\n")

    with torch.no_grad():
        flow_preds = model(img_left, img_right, iters=args.iters)

    flow = flow_preds[-1]
    print("flow_prediction shape:", flow.shape)
    print("flow x min/max/mean:", flow[:,0].min().item(), flow[:,0].max().item(), flow[:,0].mean().item())
    print("flow y min/max/mean:", flow[:,1].min().item(), flow[:,1].max().item(), flow[:,1].mean().item())

    # --------------------------------------
    # Check both sign conventions
    # --------------------------------------
    print("\n====== Testing disparity sign (+/-) ======\n")

    disp_pos = flow[:, 0:1]          # no negative
    disp_neg = -flow[:, 0:1]         # your original choice

    print("disp_pos  min/max/mean:", disp_pos.min().item(), disp_pos.max().item(), disp_pos.mean().item())
    print("disp_neg  min/max/mean:", disp_neg.min().item(), disp_neg.max().item(), disp_neg.mean().item())

    # --------------------------------------
    # Optional: upsample to full image size (if needed)
    # --------------------------------------
    h2, w2 = flow.shape[2:]
    if (h2 != H or w2 != W):
        scale_factor = W / w2
        print(f"\nFlow is low resolution ({w2}x{h2}), upsampling by {scale_factor:.2f}...")

        disp_pos_up = F.interpolate(disp_pos, size=(H, W), mode='bilinear', align_corners=True) * scale_factor
        disp_neg_up = F.interpolate(disp_neg, size=(H, W), mode='bilinear', align_corners=True) * scale_factor
    else:
        disp_pos_up = disp_pos
        disp_neg_up = disp_neg

    print("\nDisparity after upsampling & scaling (pos): min/max/mean =",
          disp_pos_up.min().item(), disp_pos_up.max().item(), disp_pos_up.mean().item())
    print("Disparity after upsampling & scaling (neg): min/max/mean =",
          disp_neg_up.min().item(), disp_neg_up.max().item(), disp_neg_up.mean().item())

    # --------------------------------------
    # Evaluate photometric reconstruction error
    # --------------------------------------
    print("\n====== Evaluating photometric consistency for +/- disp ======\n")

    warp_pos = warp_with_disp(img_right, disp_pos_up)
    warp_neg = warp_with_disp(img_right, disp_neg_up)

    pe_pos = photometric_error(img_left, warp_pos)
    pe_neg = photometric_error(img_left, warp_neg)

    print("Photometric Error (POS disp):", pe_pos.item())
    print("Photometric Error (NEG disp):", pe_neg.item())

    print("\n====== Final conclusion ======\n")
    if pe_pos < pe_neg:
        print("✔ The CORRECT disparity is **disp = +flow_x**")
    else:
        print("✔ The CORRECT disparity is **disp = -flow_x** (your original choice)")

    print("\n====== Test Completed ======\n")


# --------------------------------------
# Script entry
# --------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=16)

    # RAFTStereo architecture parameters with reasonable defaults
    parser.add_argument("--corr_implementation", default="reg")
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=4)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--context_norm", type=str, default="batch")
    parser.add_argument("--slow_fast_gru", action="store_true")
    parser.add_argument("--n_gru_layers", type=int, default=3)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128,128,128])

    args = parser.parse_args()
    args.mixed_precision = False


    run_test(args)
