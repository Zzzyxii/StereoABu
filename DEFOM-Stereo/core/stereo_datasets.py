# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
from numpy import linalg as LA
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp

from core.utils import frame_utils
from core.utils.augmentor import DispAugmentor, SparseDispAugmentor


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, is_eval=False, is_test=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseDispAugmentor(**aug_params)
            else:
                self.augmentor = DispAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.is_eval = is_eval
        self.is_test = is_test
        self.init_seed = False
        self.disparity_list = []
        self.image_list = []

        # number of copies of the datasets
        self.v = 1

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            if len(img1.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
                img2 = np.tile(img2[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.image_list[index][0]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % (len(self.image_list)*self.v)
        index = index % len(self.image_list)

        if not self.is_eval and len(self.disparity_list[index]) > 1 and np.random.rand() > 0.5:
            disp = self.disparity_reader(self.disparity_list[index][1])
            if isinstance(disp, tuple):
                disp, valid = disp
            else:
                valid = disp < 1024
            img1 = frame_utils.read_gen(self.image_list[index][1])
            img2 = frame_utils.read_gen(self.image_list[index][0])

            img1 = np.array(img1).astype(np.uint8)[:, ::-1]
            img2 = np.array(img2).astype(np.uint8)[:, ::-1]
            disp = np.array(disp).astype(np.float32)[:, ::-1]
            valid = np.array(valid).astype(np.bool_)[:, ::-1]

        else:
            disp = self.disparity_reader(self.disparity_list[index][0])
            if isinstance(disp, tuple):
                disp, valid = disp
            else:
                valid = disp < 1024

            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            disp = np.array(disp).astype(np.float32)
            valid = np.array(valid).astype(np.bool_)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, disp, valid = self.augmentor(img1, img2, disp, valid)
            else:
                img1, img2, disp = self.augmentor(img1, img2, disp)

        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2.copy()).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp[..., np.newaxis].copy()).permute(2, 0, 1).float()
        if self.sparse:
            valid = torch.from_numpy(valid[..., np.newaxis].astype(np.bool_).copy()).permute(2, 0, 1)
        else:
            valid = disp < 512

        return {"img1": img1, "img2": img2, "disp": disp, "valid": valid, "imageL_file": self.image_list[index][0], "disp_file": self.disparity_list[index][0]}

    def __mul__(self, v):
        self.v = v
        return self
        
    def __len__(self):
        return len(self.image_list)*self.v


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/SceneFlow/', dstype='frames_cleanpass', things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params, is_eval=things_test)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'FlyingThings3D')
        left_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/left/*.png')))
        right_images = [im.replace('left', 'right') for im in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(left_images))[:400])
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [[img1, img2]]
                self.disparity_list += [[disp, disp.replace('left', 'right')]]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted(glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp, disp.replace('left', 'right')]]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Driving')
        left_images = sorted(glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp, disp.replace('left', 'right')]]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/ETH3D', split='training', is_eval=False, is_test=False):
        super(ETH3D, self).__init__(aug_params, sparse=True, is_eval=is_eval, is_test=is_test)

        image1_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im0.png')))
        image2_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im1.png')))
        disp_list = sorted(glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm'))) if split == 'training'\
            else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp]]


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/KITTI', split='15', image_set='training', is_eval=False, is_test=False):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, is_eval=is_eval, is_test=is_test)
        assert split in ["12", "15"]
        root = root + split
        assert os.path.exists(root)

        if split == '15':
            image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_10.png')))
            image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_10.png')))
            disp_list = sorted(
                glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(
                root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)
        else:
            image1_list = sorted(glob(os.path.join(root, image_set, 'colored_0/*_10.png')))
            image2_list = sorted(glob(os.path.join(root, image_set, 'colored_1/*_10.png')))
            disp_list = sorted(
                glob(os.path.join(root, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(
                root, 'training/disp_occ/000085_10.png')] * len(image1_list)
                
        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp]]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/Middlebury', split='F', image_set='training', is_eval=False, is_test=False):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury, is_eval=is_eval, is_test=is_test)
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2005", "2006", "2014", "2021"]
        assert image_set in ["training", "test"]

        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [[str(scene / "disp1.png"), str(scene / "disp5.png")]]    
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [[str(scene / "disp1.png"), str(scene / "disp5.png")]]       
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [[str(scene / "disp1.png"), str(scene / "disp5.png")]]   
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [[str(scene / "disp1.png"), str(scene / "disp5.png")]]  
        elif split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.image_list += [[str(scene / "im0.png"), str(scene / f"im1{s}.png")]]
                    self.disparity_list += [[str(scene / "disp0.pfm"), str(scene / "disp1.pfm")]]
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "im0.png"), str(scene / "im1.png")]]
                self.disparity_list += [[str(scene / "disp0.pfm"), str(scene / "disp1.pfm")]]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [[str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")]]
                        self.disparity_list += [[str(scene / "disp0.pfm"), str(scene / "disp1.pfm")]]
        else:
            if image_set == 'training':
                lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
                if is_eval:
                    lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "MiddEval3/official_train.txt")).read_text().splitlines()), lines))
            else:
                lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/testF/*"))))

            image1_list = sorted([os.path.join(root, "MiddEval3", f'{image_set}{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'{image_set}{split}', f'{name}/im1.png') for name in lines])

            disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines]) \
                if image_set == 'training' else [os.path.join(root, "MiddEval3", f'training{split}', 'Adirondack/disp0GT.pfm')]*len(image1_list)

            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [[img1, img2]]
                self.disparity_list += [[disp]]


class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/SintelStereo'):
        super().__init__(aug_params, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted(glob(osp.join(root, 'training/*_left/*/frame_*.png')))
        image2_list = sorted(glob(osp.join(root, 'training/*_right/*/frame_*.png')))
        disp_list = sorted(glob(osp.join(root, 'training/disparities/*/frame_*.png'))) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp]]


class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/FallingThings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        image1_list = sorted(glob(osp.join(root, 'fat/single/*/*/*.left.jpg'))) + \
                      sorted(glob(osp.join(root, 'fat/mixed/*/*.left.jpg')))
        image2_list = [e.replace('left.jpg', 'right.jpg') for e in image1_list]
        disp_list = [e.replace('left.jpg', 'left.depth.png') for e in image1_list]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp, disp.replace('left', 'right')]]


class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/TartanAir'):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        image1_list = sorted(glob(osp.join(root, '*/*/*/*/image_left/*_left.png')))
        image2_list = [e.replace('_left', '_right') for e in image1_list]
        disp_list = [e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy') for e in image1_list]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp, disp.replace('left', 'right')]]


class CarlaHighres(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/HRVS/carla-highres'):
        super().__init__(aug_params)
        assert os.path.exists(root)

        image1_list = sorted(glob(osp.join(root, 'trainingF/*/im0.png')))
        image2_list = [e.replace('im0', 'im1') for e in image1_list]
        disp1_list = [e.replace('im0.png', 'disp0GT.pfm') for e in image1_list]
        disp2_list = [e.replace('im1.png', 'disp1GT.pfm') for e in image2_list]

        for img1, img2, disp1, disp2 in zip(image1_list, image2_list, disp1_list, disp2_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp1, disp2]]


class InStereo2K(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/InStereo2K', split='training'):
        super(InStereo2K, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispInStereo2K, is_eval=split!="training")
        if split == "training":
            image1_list = sorted(glob(osp.join(root, 'part*/*/left.png')))
        else:
            image1_list = sorted(glob(osp.join(root, 'test/*/left.png')))

        image2_list = [e.replace('left', 'right') for e in image1_list]
        disp_list = [e.replace('left', 'left_disp') for e in image1_list]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp, disp.replace('left', 'right')]]


class CreStereo(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/CreStereo'):
        super(CreStereo, self).__init__(aug_params, reader=frame_utils.readDispCreStereo)

        image1_list = sorted(glob(osp.join(root, '*/*_left.jpg')))
        image2_list = [e.replace('left', 'right') for e in image1_list]
        disp_list = [e.replace('_left.jpg', '_left.disp.png') for e in image1_list]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp, disp.replace('left', 'right')]]


class IRS(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/IRSDataset'):
        super().__init__(aug_params)
        image1_list = sorted(glob(osp.join(root, '*/*/l_*.png')))
        image2_list = sorted(glob(osp.join(root, '*/*/r_*.png')))
        disp_list = sorted(glob(osp.join(root, '*/*/d_*.pfm')))
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2] == disp.split('/')[-2]
            assert img1.split('.')[0].split('_')[-1] == disp.split('.')[0].split('_')[-1]
            if 'QAOfficeAndSecurityRoom2_Night' in img1: # bad scenes
                continue
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp]]


class Booster(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/Booster_Dataset', split='train', is_eval=False, is_test=False):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispBooster, is_eval=is_eval, is_test=is_test)
        assert os.path.exists(root)

        folder_list = sorted(glob(osp.join(root, split+'/balanced/*')))
        for folder in folder_list:
            image1_list = sorted(glob(osp.join(folder, 'camera_00/im*.png')))
            image2_list = sorted(glob(osp.join(folder, 'camera_02/im*.png')))
            if split=="train":
                for img1 in image1_list:
                    for img2 in image2_list:
                        self.image_list += [[img1, img2]]
                        self.disparity_list += [[osp.join(folder, 'disp_00.npy'), osp.join(folder, 'disp_02.npy')]]
            else:
                for img1, img2 in zip(image1_list, image2_list):
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [[osp.join(folder, 'disp_00.npy'), osp.join(folder, 'disp_02.npy')]]


class ThreeDKenBurns(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/3dkenburns'):
        super().__init__(aug_params, reader=frame_utils.readDisp3DKenBurns)

        image1_list = sorted(glob(osp.join(root, '*/*l-image.png')))
        image2_list = sorted(glob(osp.join(root, '*/*r-image.png')))

        disp1_list = sorted(glob(osp.join(root, '*/*l-depth.exr')))
        disp2_list = sorted(glob(osp.join(root, '*/*r-depth.exr')))

        for img1, img2, disp1, disp2 in zip(image1_list, image2_list, disp1_list, disp2_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp1, disp2]]


class VKITTI2(StereoDataset):
    def __init__(self, aug_params=None, root='./datasets/VKITTI2'):
        super().__init__(aug_params, reader=frame_utils.readDispVKITTI2)

        image1_list = sorted(glob(osp.join(root, 'Scene*/*/frames/rgb/Camera_0/rgb_*.jpg')))
        image2_list = sorted(glob(osp.join(root, 'Scene*/*/frames/rgb/Camera_1/rgb_*.jpg')))

        disp1_list = sorted(glob(osp.join(root, 'Scene*/*/frames/depth/Camera_0/depth_*.png')))
        disp2_list = sorted(glob(osp.join(root, 'Scene*/*/frames/depth/Camera_1/depth_*.png')))

        for img1, img2, disp1, disp2 in zip(image1_list, image2_list, disp1_list, disp2_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp1, disp2]]


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0],
                  'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    assert len(args.train_datasets)  == len(args.train_folds)

    train_dataset = None
    for fold, dataset_name in zip(args.train_folds, args.train_datasets):
        if dataset_name.startswith("middlebury_"):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_','')) * fold
        elif dataset_name == 'sceneflow':
            clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass')
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            new_dataset = clean_dataset*fold+final_dataset*fold
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif 'kitti1' in dataset_name:
            new_dataset = KITTI(aug_params, split=dataset_name[-2:], image_set='training') * fold
            logging.info(f"Adding {len(new_dataset)} samples from KITTI"+dataset_name[-2:])
        elif 'eth3d' in dataset_name:
            new_dataset = ETH3D(aug_params, split='training') * fold
            logging.info(f"Adding {len(new_dataset)} samples from ETH3D")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params)*fold
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        elif dataset_name.startswith('carla_highres'):
            new_dataset = CarlaHighres(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from Carla Highres")
        elif dataset_name.startswith('irs'):
            new_dataset = IRS(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from IRS")
        elif dataset_name.startswith('crestereo'):
            new_dataset = CreStereo(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from CreStereo")
        elif dataset_name.startswith('instereo2k'):
            new_dataset = InStereo2K(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from InStereo2K")
        elif dataset_name.startswith('booster'):
            new_dataset = Booster(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from Booster")
        elif dataset_name.startswith('3dkenburns'):
            new_dataset = ThreeDKenBurns(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from 3D Ken Burns")
        elif dataset_name.startswith('vkitti2'):
            new_dataset = VKITTI2(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from VKITTI2")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader


def fetch_dataset(args):
    """ Create the dataset for the corresponding training set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 
                  'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    assert len(args.train_datasets)  == len(args.train_folds)

    train_dataset = None
    for fold, dataset_name in zip(args.train_folds, args.train_datasets):
        if dataset_name.startswith("middlebury_"):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_', '')) * fold
            logging.info(f"Adding {len(new_dataset)} samples from {dataset_name}")
        elif 'eth3d' in dataset_name:
            new_dataset = ETH3D(aug_params, split='training') * fold
            logging.info(f"Adding {len(new_dataset)} samples from ETH3D")
        elif 'kitti1' in dataset_name:
            new_dataset = KITTI(aug_params, split=dataset_name[-2:], image_set='training') * fold
            logging.info(f"Adding {len(new_dataset)} samples from KITTI"+dataset_name[-2:])
        elif dataset_name == 'sceneflow':
            clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass')
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            new_dataset = clean_dataset*fold+final_dataset*fold
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        elif dataset_name.startswith('carla_highres'):
            new_dataset = CarlaHighres(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from Carla Highres")
        elif dataset_name.startswith('irs'):
            new_dataset = IRS(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from IRS")
        elif dataset_name.startswith('crestereo'):
            new_dataset = CreStereo(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from CreStereo")
        elif dataset_name.startswith('instereo2k'):
            new_dataset = InStereo2K(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from InStereo2K")
        elif dataset_name.startswith('booster'):
            new_dataset = Booster(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from Booster")
        elif dataset_name.startswith('3dkenburns'):
            new_dataset = ThreeDKenBurns(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from 3D Ken Burns")
        elif dataset_name.startswith('vkitti2'):
            new_dataset = VKITTI2(aug_params) * fold
            logging.info(f"Adding {len(new_dataset)} samples from VKITTI2")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_dataset

