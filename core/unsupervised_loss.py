"""
Unsupervised Loss Functions for Stereo Matching
Based on photometric consistency, smoothness, and left-right consistency
"""

import torch
import torch.nn.functional as F


def apply_disparity(img, disp):
    """
    Warp img using disparity map disp (horizontal shift)
    
    Args:
        img: [B, C, H, W] image tensor
        disp: [B, 1, H, W] disparity map (negative values for left camera)
    
    Returns:
        warped: [B, C, H, W] warped image
    """
    batch_size, _, height, width = img.size()
    
    # Create meshgrid
    mesh_x = torch.linspace(0, width - 1, width, device=img.device).view(1, 1, width).expand(batch_size, height, width)
    mesh_y = torch.linspace(0, height - 1, height, device=img.device).view(1, height, 1).expand(batch_size, height, width)
    
    # Apply horizontal shift
    grid_x = mesh_x + disp.squeeze(1)
    grid_y = mesh_y
    
    # Normalize to [-1, 1]
    grid_x = 2.0 * grid_x / (width - 1) - 1.0
    grid_y = 2.0 * grid_y / (height - 1) - 1.0
    
    # Stack and reshape for grid_sample
    grid = torch.stack([grid_x, grid_y], dim=3)

    # Validity mask to ignore samples that fall outside image bounds
    valid_x = (grid_x >= -1.0) & (grid_x <= 1.0)
    valid_y = (grid_y >= -1.0) & (grid_y <= 1.0)
    mask = (valid_x & valid_y).float().unsqueeze(1)
    
    # Warp image
    warped = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return warped, mask


def ssim(img1, img2, window_size=3):
    """
    Compute SSIM between two images
    
    Args:
        img1, img2: [B, C, H, W] images
        window_size: size of the gaussian window
    
    Returns:
        ssim_map: [B, C, H, W] SSIM values
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean(1, keepdim=True)


def census_transform(img):
    """
    Compute Census transform (3x3 window)
    """
    B, C, H, W = img.shape
    if C == 3:
        img = img.mean(1, keepdim=True)
    
    patches = F.unfold(img, kernel_size=3, padding=1) # [B, 9, H*W]
    patches = patches.view(B, 9, H, W)
    
    center = patches[:, 4:5, :, :]
    census = (patches > center).float()
    # Exclude center
    return torch.cat([census[:, :4, :, :], census[:, 5:, :, :]], dim=1)


def hamming_distance(c1, c2):
    """
    Compute Hamming distance between two census transforms
    """
    dist = torch.abs(c1 - c2).sum(dim=1, keepdim=True) / 8.0
    return dist


def photometric_loss(img_left, img_right, disp_left, disp_right=None, alpha=0.85):
    """
    Photometric reconstruction loss using SSIM + Census (instead of L1)
    
    Args:
        img_left: [B, C, H, W] left image
        img_right: [B, C, H, W] right image  
        disp_left: [B, 1, H, W] disparity predicted from left image
        disp_right: [B, 1, H, W] disparity predicted from right image (optional)
        alpha: weight for SSIM (1-alpha for Census)
    
    Returns:
        loss: photometric loss
    """
    # Warp right image to left using predicted disparity
    img_right_warped, valid_mask_left = apply_disparity(img_right, -disp_left)
    mask_left = valid_mask_left
    if disp_right is not None:
        occlusion_mask_left = compute_occlusion_mask(disp_left, disp_right)
        mask_left = mask_left * occlusion_mask_left
    eps = 1e-6
    
    # SSIM loss
    ssim_map_left = ssim(img_left, img_right_warped)
    
    # Census loss
    census_left = census_transform(img_left)
    census_right_warped = census_transform(img_right_warped)
    census_map_left = hamming_distance(census_left, census_right_warped)
    
    ssim_loss_left = ((1 - ssim_map_left) * mask_left).sum() / (mask_left.sum() + eps)
    census_loss_left = (census_map_left * mask_left).sum() / (mask_left.sum() + eps)
    
    photo_loss_left = alpha * ssim_loss_left + (1 - alpha) * census_loss_left
    
    if disp_right is not None:
        # Warp left image to right
        img_left_warped, valid_mask_right = apply_disparity(img_left, disp_right)
        mask_right = valid_mask_right * compute_occlusion_mask(disp_right, disp_left)
        
        ssim_map_right = ssim(img_right, img_left_warped)
        
        census_right = census_transform(img_right)
        census_left_warped = census_transform(img_left_warped)
        census_map_right = hamming_distance(census_right, census_left_warped)
        
        ssim_loss_right = ((1 - ssim_map_right) * mask_right).sum() / (mask_right.sum() + eps)
        census_loss_right = (census_map_right * mask_right).sum() / (mask_right.sum() + eps)
        
        photo_loss_right = alpha * ssim_loss_right + (1 - alpha) * census_loss_right
        photo_loss = (photo_loss_left + photo_loss_right) / 2
    else:
        photo_loss = photo_loss_left
    
    return photo_loss


def smoothness_loss(disp, img, order=1):
    """
    Edge-aware smoothness loss for disparity
    
    Args:
        disp: [B, 1, H, W] disparity map
        img: [B, C, H, W] reference image for edge detection
        order: 1 for first-order, 2 for second-order smoothness
    
    Returns:
        loss: smoothness loss
    """
    # Compute image gradients
    img_grad_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    img_grad_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    
    # Average across channels for edge weight
    weight_x = torch.exp(-img_grad_x.mean(1, keepdim=True))
    weight_y = torch.exp(-img_grad_y.mean(1, keepdim=True))
    
    if order == 1:
        # First-order gradients
        disp_grad_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        disp_grad_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        
        smooth_loss = (disp_grad_x * weight_x).mean() + (disp_grad_y * weight_y).mean()
    
    elif order == 2:
        # Second-order gradients
        disp_grad_xx = torch.abs(disp[:, :, :, :-2] - 2 * disp[:, :, :, 1:-1] + disp[:, :, :, 2:])
        disp_grad_yy = torch.abs(disp[:, :, :-2, :] - 2 * disp[:, :, 1:-1, :] + disp[:, :, 2:, :])
        
        weight_xx = weight_x[:, :, :, :-1]
        weight_yy = weight_y[:, :, :-1, :]
        
        smooth_loss = (disp_grad_xx * weight_xx).mean() + (disp_grad_yy * weight_yy).mean()
    
    else:
        raise ValueError(f"Unsupported smoothness order: {order}")
    
    return smooth_loss


def lr_consistency_loss(disp_left, disp_right):
    """
    Left-right consistency loss
    
    Args:
        disp_left: [B, 1, H, W] disparity from left image
        disp_right: [B, 1, H, W] disparity from right image
    
    Returns:
        loss: left-right consistency loss
    """
    # Warp right disparity to left view
    disp_right_warped, valid_mask = apply_disparity(disp_right, -disp_left)
    occlusion_mask = compute_occlusion_mask(disp_left, disp_right)
    combined_mask = valid_mask * occlusion_mask
    lr_diff = torch.abs(disp_left - disp_right_warped)
    eps = 1e-6
    return (lr_diff * combined_mask).sum() / (combined_mask.sum() + eps)


def compute_occlusion_mask(disp_left, disp_right, threshold=1.0):
    """
    Compute occlusion mask based on left-right consistency
    
    Args:
        disp_left: [B, 1, H, W] disparity from left
        disp_right: [B, 1, H, W] disparity from right
        threshold: consistency threshold
    
    Returns:
        mask: [B, 1, H, W] binary mask (1 = non-occluded, 0 = occluded)
    """
    disp_right_warped, valid_mask = apply_disparity(disp_right, -disp_left)
    lr_diff = torch.abs(disp_left - disp_right_warped)
    mask = (lr_diff < threshold).float() * valid_mask
    
    return mask


def unsupervised_loss(disp_left, disp_right, img_left, img_right, 
                      loss_weights={'photo': 1.0, 'smooth': 0.1, 'lr': 0.1},
                      smooth_order=1):
    """
    Combined unsupervised loss
    
    Args:
        disp_left: [B, 1, H, W] predicted disparity from left
        disp_right: [B, 1, H, W] predicted disparity from right
        img_left: [B, 3, H, W] left image
        img_right: [B, 3, H, W] right image
        loss_weights: dict with keys 'photo', 'smooth', 'lr'
        smooth_order: 1 or 2 for smoothness order
    
    Returns:
        total_loss: scalar tensor
        loss_dict: dict with individual loss components
    """
    # Photometric loss
    photo_loss = photometric_loss(img_left, img_right, disp_left, disp_right)
    
    # Smoothness loss
    smooth_loss_left = smoothness_loss(disp_left, img_left, order=smooth_order)
    smooth_loss_right = smoothness_loss(disp_right, img_right, order=smooth_order)
    smooth_loss = (smooth_loss_left + smooth_loss_right) / 2
    
    # Left-right consistency loss
    lr_loss = lr_consistency_loss(disp_left, disp_right)
    
    # Combine losses
    total_loss = (loss_weights['photo'] * photo_loss + 
                  loss_weights['smooth'] * smooth_loss + 
                  loss_weights['lr'] * lr_loss)
    
    loss_dict = {
        'photo': photo_loss.item(),
        'smooth': smooth_loss.item(),
        'lr': lr_loss.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict
