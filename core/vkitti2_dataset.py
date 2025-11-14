"""
Virtual KITTI 2 Dataset for Unsupervised Stereo Training

vKITTI2 structure:
    Scene01/
        clone/
            frames/
                rgb/
                    Camera_0/  # Left camera
                        rgb_00001.jpg
                    Camera_1/  # Right camera
                        rgb_00001.jpg
                depth/
                    Camera_0/
                        depth_00001.png
"""

import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path
from PIL import Image
import random


class VirtualKITTI2(data.Dataset):
    """Virtual KITTI 2 dataset for unsupervised stereo training"""
    
    def __init__(self, root, split='training', augmentor=None):
        """
        Args:
            root: Path to vKITTI2/raw directory
            split: 'training' or 'validation' 
            augmentor: Data augmentation object
        """
        self.root = Path(root)
        self.split = split
        self.augmentor = augmentor
        self.init_seed = False
        
        # Collect all stereo pairs
        self.samples = self._collect_samples()
        
        print(f"VirtualKITTI2 {split}: {len(self.samples)} stereo pairs")
    
    def _collect_samples(self):
        """Collect all available stereo pairs from vKITTI2"""
        samples = []
        
        # Scenes in vKITTI2
        scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
        
        # Variations
        variations = ["clone", "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right",
                      "fog", "morning", "overcast", "rain", "sunset"]
        
        for scene in scenes:
            scene_dir = self.root / scene
            if not scene_dir.exists():
                continue
            
            for variation in variations:
                var_dir = scene_dir / variation
                if not var_dir.exists():
                    continue
                
                # Check if rgb frames exist
                rgb_left_dir = var_dir / "frames" / "rgb" / "Camera_0"
                rgb_right_dir = var_dir / "frames" / "rgb" / "Camera_1"
                
                if not rgb_left_dir.exists() or not rgb_right_dir.exists():
                    continue
                
                # Get all left images
                left_images = sorted(rgb_left_dir.glob("rgb_*.jpg"))
                
                for left_img in left_images:
                    frame_id = left_img.stem.split("_")[1]  # e.g., "00001"
                    right_img = rgb_right_dir / f"rgb_{frame_id}.jpg"
                    
                    if right_img.exists():
                        samples.append({
                            'left': left_img,
                            'right': right_img,
                            'scene': scene,
                            'variation': variation,
                            'frame': frame_id
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        
        sample = self.samples[idx]
        
        # Load images
        img_left = np.array(Image.open(sample['left'])).astype(np.uint8)
        img_right = np.array(Image.open(sample['right'])).astype(np.uint8)
        
        # Ensure RGB (not RGBA)
        if img_left.shape[-1] == 4:
            img_left = img_left[..., :3]
        if img_right.shape[-1] == 4:
            img_right = img_right[..., :3]
        
        # Data augmentation
        if self.augmentor is not None:
            img_left, img_right = self.augmentor(img_left, img_right)
        
        # Convert to torch tensors
        img_left = torch.from_numpy(img_left).permute(2, 0, 1).float()
        img_right = torch.from_numpy(img_right).permute(2, 0, 1).float()
        
        return img_left, img_right
    
    def split_train_val(self, train_size=1000, val_size=200, seed=42):
        """
        Split dataset into train and validation sets
        
        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        
        # Random sampling
        total_samples = len(self.samples)
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        
        # Create train and val datasets
        train_dataset = VirtualKITTI2Subset(self, train_indices)
        val_dataset = VirtualKITTI2Subset(self, val_indices)
        
        return train_dataset, val_dataset


class VirtualKITTI2Subset(data.Dataset):
    """Subset of VirtualKITTI2 dataset"""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def fetch_dataloader(args):
    """Create train and validation dataloaders for vKITTI2"""
    
    # Create full dataset
    vkitti2_root = Path(args.vkitti2_path) / "raw"
    full_dataset = VirtualKITTI2(
        root=vkitti2_root,
        split='training',
        augmentor=None  # Add augmentor if needed
    )
    
    # Split into train and val
    train_dataset, val_dataset = full_dataset.split_train_val(
        train_size=1000,
        val_size=200,
        seed=42
    )
    
    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    vkitti2_root = Path("/openbayes/home/RAFT-Stereo/datasets/vKITTI2/raw")
    
    dataset = VirtualKITTI2(root=vkitti2_root, split='training')
    print(f"Total samples: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test loading one sample
        img_left, img_right = dataset[0]
        print(f"Left image shape: {img_left.shape}")
        print(f"Right image shape: {img_right.shape}")
        
        # Test train/val split
        train_ds, val_ds = dataset.split_train_val(train_size=1000, val_size=200)
        print(f"Training set: {len(train_ds)} samples")
        print(f"Validation set: {len(val_ds)} samples")
