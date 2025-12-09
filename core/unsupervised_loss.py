import torch
import torch.nn.functional as F
import torch.nn as nn

# ---------------------- Helper Functions ---------------------- #

def gradient_x(img):
    # [B,C,H,W-1]
    return img[:, :, :, 1:] - img[:, :, :, :-1]


def gradient_y(img):
    # [B,C,H-1,W]
    return img[:, :, 1:, :] - img[:, :, :-1, :]


def edge_aware_smoothness_loss(disp, img):
    """
    Edge-aware smoothness:
        |∂x disp| * exp(-|∂x img|)
        |∂y disp| * exp(-|∂y img|)
    disp: [B,1,H,W]
    img:  [B,3,H,W] (0~1 or normalized)
    """
    disp_dx = gradient_x(disp)
    disp_dy = gradient_y(disp)

    img_dx = gradient_x(img).abs().mean(1, keepdim=True)
    img_dy = gradient_y(img).abs().mean(1, keepdim=True)

    weight_x = torch.exp(-img_dx)
    weight_y = torch.exp(-img_dy)

    smooth_x = (disp_dx.abs() * weight_x).mean()
    smooth_y = (disp_dy.abs() * weight_y).mean()
    return smooth_x + smooth_y


def ssim(x, y):
    """
    SSIM approximation, returns [B,1,H,W], value in [0,1], smaller is better (more similar)
    x, y: [B,3,H,W]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)

    ssim_map = ssim_n / (ssim_d + 1e-6)
    # Map to [0,1], 0 is most similar
    return torch.clamp((1 - ssim_map) / 2, 0, 1)


def census_transform(img, patch_size=3):
    """
    Soft Census Transform:
      - Unfold local patch
      - Subtract center pixel, pass through Tanh
    img: [B,3,H,W]
    Returns: [B, 3*(patch_size*patch_size-1), H, W]
    """
    B, C, H, W = img.shape
    pad = patch_size // 2

    img_pad = F.pad(img, (pad, pad, pad, pad), mode='replicate')

    # unfold: [B, C*K*K, H*W]
    patches = F.unfold(img_pad, kernel_size=patch_size, stride=1)
    patches = patches.view(B, C, patch_size * patch_size, H, W)

    center = patches[:, :, patch_size * patch_size // 2:patch_size * patch_size // 2 + 1, :, :]
    diff = patches - center

    # Remove center pixel
    mask = torch.ones(patch_size * patch_size, device=img.device, dtype=torch.bool)
    mask[patch_size * patch_size // 2] = False
    diff = diff[:, :, mask, :, :]  # (B, C, K*K-1, H, W)

    census = torch.tanh(diff)
    census = census.view(B, C * (patch_size * patch_size - 1), H, W)
    return census


def census_loss(img1, img2, mask=None):
    """
    img1, img2: [B,3,H,W]
    Returns: scalar loss
    """
    c1 = census_transform(img1)
    c2 = census_transform(img2)
    
    # Compute Hamming distance (approximated by soft difference)
    # Instead of full diff tensor, we can compute sum of squared diffs directly to save memory
    # But here we stick to the original logic but optimize memory if possible
    
    diff = c1 - c2
    eps = 1e-3
    
    # dist = torch.sqrt(diff * diff + eps * eps)
    # Optimization: Compute mean over channels first to reduce memory before masking
    # The original code was: dist = torch.sqrt(diff * diff + eps * eps) -> [B, C_census, H, W]
    # Then (dist * mask).sum()
    
    # We can compute the charbonnier penalty per pixel
    dist = torch.sqrt(diff.pow(2) + eps**2).mean(dim=1, keepdim=True) # [B, 1, H, W]
    
    if mask is not None:
        loss = (dist * mask).sum() / (mask.sum() + 1e-6)
    else:
        loss = dist.mean()
    return loss


def build_base_grid(batch, height, width, device):
    xx = torch.arange(0, width, device=device).view(1, 1, 1, width).repeat(batch, 1, height, 1)
    yy = torch.arange(0, height, device=device).view(1, 1, height, 1).repeat(batch, 1, 1, width)
    return xx, yy


# ---------------------- Single Scale Unsupervised Loss ---------------------- #

def unsup_single_scale(
    imageL, imageR,
    flow_L, flow_R,
    w_photo=1.0, w_smooth=0.01, w_lr=0.1,
    census_weight=0.0,   # Removed Census
    ssim_weight=0.85,    # Increased SSIM weight
    l1_weight=0.15,      # Increased L1 weight
    smooth_mult=1.0      # extra scaling for smooth at different scales
):
    """
    Single scale unsupervised loss:
      - photometric (census + SSIM + L1)
      - edge-aware smoothness (with scale decay)
      - LR consistency (with occlusion check)
    imageL, imageR: [B,3,H,W]
    flow_L, flow_R: [B,2,H,W] (horizontal component is disparity)
    """

    B, _, H, W = imageL.shape
    device = imageL.device

    # Take only disp component (horizontal)
    disp_L = flow_L[:, 0:1]   # [B,1,H,W]
    disp_R = flow_R[:, 0:1]

    # ---------- Warp Right to Left View ---------- #
    xx, yy = build_base_grid(B, H, W, device)
    xR = xx - disp_L
    yR = yy

    xR_norm = 2 * (xR / max(W - 1, 1)) - 1
    yR_norm = 2 * (yR / max(H - 1, 1)) - 1
    grid_L2R = torch.cat((xR_norm, yR_norm), dim=1).permute(0, 2, 3, 1)  # [B,H,W,2]

    warped_R = F.grid_sample(
        imageR, grid_L2R,
        align_corners=True,
        padding_mode='border'
    )

    # Valid mask
    valid_x = (xR_norm >= -1) & (xR_norm <= 1)
    valid_y = (yR_norm >= -1) & (yR_norm <= 1)
    valid_mask = (valid_x & valid_y).float()

    # ---------- LR consistency + occlusion ---------- #
    # Warp right disp to left view
    disp_R_warp = F.grid_sample(
        disp_R, grid_L2R,
        align_corners=True,
        padding_mode='border'
    )

    lr_diff = (disp_L - disp_R_warp).abs()  # [B,1,H,W]

    # Adaptive threshold: max(1px, 0.05 * |disp|)
    thr_map = torch.maximum(
        torch.ones_like(lr_diff),
        0.05 * disp_L.abs()
    )

    # occ_mask: 1 if consistent (not occluded), 0 if occluded
    occ_mask = ((lr_diff < thr_map) & (valid_mask > 0.5)).float()
    
    # Stop gradient on occlusion mask to prevent trivial solutions
    occ_mask = occ_mask.detach()

    # Combine valid_mask (borders) and occ_mask (occlusions)
    final_mask = valid_mask * occ_mask

    # ---------- Photometric: Census + SSIM + L1 ---------- #
    # L1
    l1_map = (imageL - warped_R).abs().mean(1, keepdim=True)  # [B,1,H,W]
    l1_loss = (l1_map * final_mask).sum() / (final_mask.sum() + 1e-6)

    # SSIM
    ssim_map = ssim(imageL, warped_R)  # [B,1,H,W]
    ssim_loss = (ssim_map * final_mask).sum() / (final_mask.sum() + 1e-6)

    # Census
    # census_l = census_loss(imageL, warped_R, mask=final_mask)

    photo = ssim_weight * ssim_loss + l1_weight * l1_loss

    # ---------- Smoothness (with scale decay) ---------- #
    smooth = edge_aware_smoothness_loss(disp_L, imageL) * smooth_mult

    if occ_mask.sum() > 0:
        lr = (lr_diff * occ_mask).sum() / (occ_mask.sum() + 1e-6)
    else:
        lr = torch.zeros_like(photo)

    total = w_photo * photo + w_smooth * smooth + w_lr * lr

    metrics = {
        "photo": float(photo.detach().cpu().item()),
        "smooth": float(smooth.detach().cpu().item()),
        "lr": float(lr.detach().cpu().item()),
    }
    return total, metrics


# ---------------------- Multi-step + Multi-scale RAFT Unsupervised Loss ---------------------- #

def build_pyramid(img, num_scales=2):
    """
    img: [B,C,H,W]
    Returns list: [scale0, scale1, ...]
    scale0 = original, scale1 = 1/2, scale2 = 1/4 ...
    """
    pyr = [img]
    for i in range(1, num_scales):
        s = 2 ** i
        h = img.shape[2] // s
        w = img.shape[3] // s
        pyr.append(F.interpolate(img, size=(h, w), mode='bilinear', align_corners=True))
    return pyr


def unsupervised_loss(
    imageL, imageR,
    flow_L_preds, flow_R_preds,
    w_photo=1.0, w_smooth=0.01, w_lr=0.1,
    gamma=0.9,
    num_scales=2
):
    """
    RAFT Multi-step, Multi-scale Unsupervised Loss:
      - Calculate unsup_single_scale for each step and each scale
      - Step weights decay with gamma^k (later steps have higher weight)
    imageL, imageR: [B,3,H,W]
    flow_L_preds, flow_R_preds: list of [B,2,H,W], from coarse -> fine
    """

    n_preds = len(flow_L_preds)

    pyrL = build_pyramid(imageL, num_scales=num_scales)
    pyrR = build_pyramid(imageR, num_scales=num_scales)

    total_loss = 0.0
    last_metrics = None

    for i, (flow_L, flow_R) in enumerate(zip(flow_L_preds, flow_R_preds)):
        # Step weight: later steps have higher weight
        step_weight = gamma ** (n_preds - i - 1)

        step_loss = 0.0
        
        # Compute loss for both Left and Right views (Symmetric Loss)
        # Left View: reconstruct ImageL using ImageR and FlowL
        # Right View: reconstruct ImageR using ImageL and FlowR
        
        for s in range(num_scales):
            imgL_s = pyrL[s]
            imgR_s = pyrR[s]

            scale_factor = 1.0 / (2 ** s)
            if s > 0:
                h_s, w_s = imgL_s.shape[2], imgL_s.shape[3]
                flow_L_s = F.interpolate(flow_L, size=(h_s, w_s), mode='bilinear', align_corners=True) * scale_factor
                flow_R_s = F.interpolate(flow_R, size=(h_s, w_s), mode='bilinear', align_corners=True) * scale_factor
            else:
                flow_L_s = flow_L
                flow_R_s = flow_R

            # smooth_mult: smaller scale (lower resolution) -> smaller smooth weight
            smooth_mult = 1.0 / (2 ** s)

            # --- Left View Loss ---
            li_L, metrics_L = unsup_single_scale(
                imgL_s, imgR_s, flow_L_s, flow_R_s,
                w_photo=w_photo, w_smooth=w_smooth, w_lr=w_lr,
                smooth_mult=smooth_mult
            )
            
            # --- Right View Loss ---
            # Swap images: L <-> R
            # Negate flows: flow_L_s (pos) -> -flow_L_s (neg), flow_R_s (pos) -> -flow_R_s (neg)
            # We pass -flow_R_s as the primary flow (to warp L->R) and -flow_L_s as the secondary (for LR check)
            li_R, metrics_R = unsup_single_scale(
                imgR_s, imgL_s, -flow_R_s, -flow_L_s,
                w_photo=w_photo, w_smooth=w_smooth, w_lr=w_lr,
                smooth_mult=smooth_mult
            )

            step_loss += (li_L + li_R)

        step_loss = step_loss / num_scales
        total_loss += step_weight * step_loss
        
        # Average metrics from Left and Right
        last_metrics = {k: (metrics_L[k] + metrics_R[k]) / 2.0 for k in metrics_L}

    return total_loss, last_metrics
