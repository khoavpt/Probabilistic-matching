import h5py
# import kornia.augmentation as K
import math
import numpy as np
import os
import random
import time
import torch
from torchvision import transforms
import torchvision.transforms.functional as tvf
import cv2
from PIL import Image
import albumentations as A
from datasets.utils import *

DEBUG_MODE=False

class MegadepthScene:
    def __init__(
            self,
            data_root,
            scene_info,
            RANK,
            ht=384,
            wt=512,
            min_overlap=0.0,
            max_overlap=1.0,
            shake_t=0,
            rot_prob=0.0,
            max_num_pairs=100_000,
            scene_name=None,
            use_horizontal_flip_aug=False,
            use_single_horizontal_flip_aug=False,
            colorjiggle_params=None,
            random_eraser=None,
            use_randaug=False,
            randaug_params=None,
            randomize_size=False,
            img_padding=False,
            depth_padding=False,
            modality_list=None,
            reverse_mode=None,
    ) -> None:
        self.data_root = data_root
        self.scene_name = os.path.splitext(scene_name)[0] + f"_{min_overlap}_{max_overlap}"
        self.image_paths = scene_info["image_paths"]
        self.depth_paths = scene_info["depth_paths"]
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]
        self.pairs = scene_info["pairs"]
        self.overlaps = scene_info["overlaps"]
        threshold = (self.overlaps > min_overlap) & (self.overlaps < max_overlap)
        self.pairs = self.pairs[threshold]
        self.overlaps = self.overlaps[threshold]

        if len(self.pairs) > max_num_pairs:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), max_num_pairs, replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]
        if randomize_size:
            area = ht * wt
            s = int(16 * (math.sqrt(area) // 16))
            sizes = ((ht, wt), (s, s), (wt, ht))
            choice = RANK % 3
            ht, wt = sizes[choice]
            # counts, bins = np.histogram(self.overlaps,20)
        # print(counts)
        self.wt, self.ht = wt, ht
        self.resize = (ht, wt)
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None
        self.to_tensor = transforms.ToTensor()
        self.shake_t = shake_t
        self.random_eraser = random_eraser
        if use_horizontal_flip_aug and use_single_horizontal_flip_aug:
            raise ValueError("Can't both flip both images and only flip one")
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.use_single_horizontal_flip_aug = use_single_horizontal_flip_aug
        self.use_randaug = use_randaug
        if modality_list is None:
            self.modal_options = ['visible']
        else:
            self.modal_options = modality_list
        self.modality_list = modality_list
        self.infrared_root = os.path.join(data_root, 'infrared')
        self.depth_root = 'data/megadepth/depth/'
        self.normal_root = 'data/megadepth/normal/'
        self.paint_root = 'data/megadepth/paint/'
        self.sketch_root = 'data/megadepth/sketch/'
        self.event_root = 'data/megadepth/event/'
        self.modality_to_root = {
            'visible': self.data_root,
            'infrared': self.infrared_root,
            'depth': self.depth_root,
            'normal': self.normal_root,
            'event': self.event_root,
            'sketch': self.sketch_root,
            'paint': self.paint_root,
        }
        self.local_random_1 = random.Random(time.time())
        self.local_random_2 = random.Random(time.time() + 666)
        self.reverse_mode = reverse_mode
        if self.use_randaug:
            self.photometric_aug = A.Compose([ 
                A.HueSaturationValue(hue_shift_limit=[-10, 10], sat_shift_limit=[-30, 30], val_shift_limit=[-30, 30], p=0.3),
                # A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3), 
                A.OneOf([
                    A.Blur(blur_limit=(3,5), p=1.0), 
                    A.MotionBlur(blur_limit=(3, 7), allow_shifted=False, angle_range=[0, 360], p=1.0), 
                    A.Defocus(radius=(2, 4), p=1.0), 
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0), 
                ], p=0.3), 
                A.OneOf([
                    A.GaussNoise(std_range=[0.01, 0.07], noise_scale_factor=0.5, p=1.0), 
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=1.0), 
                ], p=0.5),
                A.ImageCompression(quality_range=[40, 100], p=0.3), 
            ], p=0.5)

    def load_im(self, im_path, to_gray=False, padding=False):
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        h, w, _ = im.shape
        h_new, w_new = self.resize
        im = cv2.resize(im, (w_new, h_new))
        scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)
        if self.use_randaug:
            im = self.photometric_aug(image=im)['image']
        if to_gray:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)[None]

        if padding:  # padding
            pad_to = max(h_new, w_new)
            im, mask = pad_bottom_right(im, pad_to, ret_mask=True)
        else:
            mask = None
        im = (torch.from_numpy(im)/255.).float()
        mask = torch.from_numpy(mask) if mask is not None else None
        return im, mask, scale

    def horizontal_flip(self, im_A, im_B, depth_A, depth_B, K_A, K_B):
        im_A = im_A.flip(-1)
        im_B = im_B.flip(-1)
        depth_A, depth_B = depth_A.flip(-1), depth_B.flip(-1)
        flip_mat = torch.tensor([[-1, 0, self.wt], [0, 1, 0], [0, 0, 1.]]).to(K_A.device)
        K_A = flip_mat @ K_A
        K_B = flip_mat @ K_B

        return im_A, im_B, depth_A, depth_B, K_A, K_B

    def load_depth(self, depth_ref, pad_to=None):
        depth = np.array(h5py.File(depth_ref, "r")["depth"])
        h_new, w_new = self.resize
        depth = cv2.resize(depth, (w_new, h_new))
        if pad_to is not None:
            depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
        return torch.from_numpy(depth).float()

    def __len__(self):
        return len(self.pairs)

    def scale_intrinsic(self, K, scale):
        sx, sy = (1/scale[0], 1/scale[1])
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return sK @ K

    def rand_shake(self, *things):
        t = np.random.choice(range(-self.shake_t, self.shake_t + 1), size=2)
        return [
            tvf.affine(thing, angle=0.0, translate=list(t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ], t

    def __getitem__(self, pair_idx):
        # read intrinsics of original size
        if self.reverse_mode is None:
            reverse_mode = self.local_random_1.randint(0, 1)  # 0: visible-infrared, 1: infrared-visible
        else:
            reverse_mode = self.reverse_mode
        self.multi_model = self.local_random_2.choice(self.modal_options)
        self.multi_model = str(self.multi_model).strip().strip('"').strip("'")
        try:
            self.multi_model_root = self.modality_to_root[self.multi_model]
        except KeyError:
            raise ValueError(f"Unknown multi_model: {self.multi_model}")
        idx1, idx2 = self.pairs[pair_idx]
        K_0 = torch.tensor(self.intrinsics[idx1].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.intrinsics[idx2].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T0 = torch.tensor(self.poses[idx1].copy(), dtype=torch.float)
        T1 = torch.tensor(self.poses[idx2].copy(), dtype=torch.float)
        T_0to1 = (T1 @ torch.linalg.inv(T0))[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()
        
        # Load positive pair data
        im_A, im_B = self.image_paths[idx1], self.image_paths[idx2]
        depth1, depth2 = self.depth_paths[idx1], self.depth_paths[idx2]
        im_A_ref = os.path.join(self.data_root, im_A)
        im_B_ref = os.path.join(self.data_root, im_B)
        im_A_ref_x_modality = os.path.join(self.multi_model_root, im_A)
        im_B_ref_x_modality = os.path.join(self.multi_model_root, im_B)
        depth_A_ref = os.path.join(self.data_root, depth1)
        depth_B_ref = os.path.join(self.data_root, depth2)
        # print('multi_model:', self.multi_model, 'reverse_mode:', reverse_mode)
        if reverse_mode:

            if self.multi_model in ['infrared', 'depth', 'normal', 'sketch']:
                filename_wo_ext, _ = os.path.splitext(im_A_ref_x_modality)
                im_A_ref = filename_wo_ext + '.jpg'
                # assert os.path.exists(img_name1), f"img not found: {img_name1}"
                # assert os.path.exists(img_name0_x_modality), f"img not found: {img_name0_x_modality}"

            elif self.multi_model in ['event']:
                filename_wo_ext, _ = os.path.splitext(im_A_ref_x_modality)
                im_A_ref = filename_wo_ext + '.png'
                # assert os.path.exists(img_name1), f"img not found: {img_name1}"
                # assert os.path.exists(img_name0_x_modality), f"img not found: {img_name0_x_modality}"
            elif self.multi_model == 'visible':
                pass
            else:
                raise ValueError(f"Unknown multi_model: {self.multi_model}")
        else:
            if self.multi_model in ['infrared', 'depth', 'normal', 'sketch']:
                filename_wo_ext, _ = os.path.splitext(im_B_ref_x_modality)
                im_B_ref = filename_wo_ext + '.jpg'

                # assert os.path.exists(img_name0), f"img not found: {img_name0}"
                # assert os.path.exists(img_name1_x_modality), f"img not found: {img_name1_x_modality}"

            elif self.multi_model in ['event']:
                filename_wo_ext, _ = os.path.splitext(im_B_ref_x_modality)
                im_B_ref = filename_wo_ext + '.png'
                # assert os.path.exists(img_name0), f"img not found: {img_name0}"
                # assert os.path.exists(img_name1_x_modality), f"img not found: {img_name1_x_modality}"

            elif self.multi_model == 'visible':
                pass

            else:
                raise ValueError(f"Unknown multi_model: {self.multi_model}")

        im_A, mask_A, scale_A = self.load_im(im_A_ref, padding=self.img_padding, to_gray=True)
        im_B, mask_B, scale_B = self.load_im(im_B_ref, padding=self.img_padding, to_gray=True)
        K_0 = self.scale_intrinsic(K_0, scale_A)
        K_1 = self.scale_intrinsic(K_1, scale_B)

        depth_A = self.load_depth(depth_A_ref, pad_to=self.depth_max_size)
        depth_B = self.load_depth(depth_B_ref, pad_to=self.depth_max_size)

        depth_A, depth_B = (depth_A[None, None], depth_B[None, None])

        [im_A, im_B, depth_A, depth_B], t = self.rand_shake(im_A, im_B, depth_A, depth_B)
        K_0[:2, 2] += t
        K_1[:2, 2] += t
        im_A, im_B = im_A[None], im_B[None]
        if self.random_eraser is not None:
            im_A, depth_A = self.random_eraser(im_A, depth_A)
            im_B, depth_B = self.random_eraser(im_B, depth_B)

        if self.use_horizontal_flip_aug:
            if np.random.rand() > 0.5:
                im_A, im_B, depth_A, depth_B, K_0, K_1 = self.horizontal_flip(im_A, im_B, depth_A, depth_B, K_0, K_1)
        if self.use_single_horizontal_flip_aug:
            if np.random.rand() > 0.5:
                im_B, depth_B, K_1 = self.single_horizontal_flip(im_B, depth_B, K_1)

        if DEBUG_MODE:
            tensor_to_pil(im_A[0], unnormalize=True).save(
                f"vis/im_A.jpg")
            tensor_to_pil(im_B[0], unnormalize=True).save(
                f"vis/im_B.jpg")
        data_dict = {
            "image0": im_A[0],
            "image0_identifier": self.image_paths[idx1].split("/")[-1].split(".jpg")[0],
            "image1": im_B[0],
            "image1_identifier": self.image_paths[idx2].split("/")[-1].split(".jpg")[0],
            "depth0": depth_A[0, 0],
            "depth1": depth_B[0, 0],
            "K0": K_0,
            "K1": K_1,
            "scale0": scale_A,  # [scale_w, scale_h]
            "scale1": scale_B,
            "T_0to1": T_0to1,  # (4, 4)
            "T_1to0": T_1to0,
            "T0": T0,
            "T1": T1,
            "im_A_path": im_A_ref,
            "im_B_path": im_B_ref,

            "scene_name": self.scene_name,
        }
        return data_dict


class MegadepthBuilder:
    def __init__(self, data_root="data/megadepth", loftr_ignore=True, imc21_ignore=True, modality_list=None) -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, "pre_scene_info_no_sfm")
        self.all_scenes = os.listdir(self.scene_info_root)
        self.test_scenes = ["0017.npy", "0004.npy", "0048.npy", "0013.npy"]
        if modality_list is None:
            self.modality_list = ['visible']
        else:
            self.modality_list = modality_list
        # LoFTR did the D2-net preprocessing differently than we did and got more ignore scenes, can optionially ignore those
        self.loftr_ignore_scenes = set(
            ['0121.npy', '0133.npy', '0168.npy', '0178.npy', '0229.npy', '0349.npy', '0412.npy', '0430.npy', '0443.npy',
             '1001.npy', '5014.npy', '5015.npy', '5016.npy'])
        self.imc21_scenes = set(
            ['0008.npy', '0019.npy', '0021.npy', '0024.npy', '0025.npy', '0032.npy', '0063.npy', '1589.npy'])
        self.test_scenes_loftr = ["0015.npy", "0022.npy"]
        self.loftr_ignore = loftr_ignore
        self.imc21_ignore = imc21_ignore

    def build_scenes(self, RANK, split="train", min_overlap=0.0, scene_names=None, **kwargs):
        if split == "train":
            scene_names = set(self.all_scenes) - set(self.test_scenes)
        elif split == "train_loftr":
            scene_names = set(self.all_scenes) - set(self.test_scenes_loftr)
        elif split == "test":
            scene_names = self.test_scenes
        elif split == "test_loftr":
            scene_names = self.test_scenes_loftr
        elif split == "custom":
            scene_names = scene_names
        else:
            raise ValueError(f"Split {split} not available")
        scenes = []
        for scene_name in scene_names:
            if self.loftr_ignore and scene_name in self.loftr_ignore_scenes:
                continue
            if self.imc21_ignore and scene_name in self.imc21_scenes:
                continue
            if ".npy" not in scene_name:
                continue
            scene_info = np.load(
                os.path.join(self.scene_info_root, scene_name), allow_pickle=True
            ).item()
            if "test" in split:
                scenes.append(
                    MegadepthScene(
                        self.data_root, scene_info, RANK, min_overlap=min_overlap, scene_name=scene_name,
                        modality_list=self.modality_list, reverse_mode=False, **kwargs
                    )
                )
                scenes.append(
                    MegadepthScene(
                        self.data_root, scene_info, RANK, min_overlap=min_overlap, scene_name=scene_name,
                        modality_list=self.modality_list, reverse_mode=True, **kwargs
                    )
                )
            else:
                scenes.append(
                    MegadepthScene(
                        self.data_root, scene_info, RANK, min_overlap=min_overlap, scene_name=scene_name,
                        modality_list=self.modality_list, **kwargs
                    )
            )
        return scenes

    def weight_scenes(self, concat_dataset, alpha=0.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n) / n ** alpha for n in ns])
        return ws