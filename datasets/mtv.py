import h5py
# import kornia.augmentation as K
import math
import numpy as np
import os
import os.path as osp
from scipy.spatial.transform import Rotation as R

import random
import time
import torch
from torchvision import transforms
import torchvision.transforms.functional as tvf

import cv2
from PIL import Image
from datasets.utils import *

import albumentations as A

DEBUG_MODE=True

RGB_CAM_INFO = {
    2: [9087.72, 9087.72, 3003.5, 2000.5],
    1: [9130.54, 9130.54, 3003.5, 2000.5],
    3: [9306.22, 9306.22, 3002.0, 2000.0],
    4: [9094.4,  9094.4,  3003.0, 2000.5],
    6: [7044.84, 7044.84, 2484.5, 1863.5],
    5: [9119.63, 9119.63, 3001.5, 2000.0],
}
def make_K(fx, fy, cx, cy):
    return np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=float)

RGB_INTRINSICS_QUERY_DICT = {cid: make_K(*params) for cid, params in RGB_CAM_INFO.items()}

class MTVScene:
    def __init__(
            self,
            data_root,
            # scene_info,
            RANK,
            ht=640,
            wt=960,
            min_overlap=0.0,
            max_overlap=1.0,
            shake_t=0,
            rot_prob=0.0,
            normalize=False,
            max_num_pairs=100_000,
            scene_name=None,
            # pose_file=None,
            npz_file=None,
            use_horizontal_flip_aug=False,
            use_single_horizontal_flip_aug=False,
            colorjiggle_params=None,
            random_eraser=None,
            use_randaug=False,
            randaug_params=None,
            randomize_size=False,
            img_padding=False,
            depth_padding=False,
            # modality_list=None,
            split="train",                   #### NEW
    ) -> None:
        self.data_root = osp.join(data_root, scene_name)
        self.scene_name = scene_name
        # self.image_paths = scene_info["image_paths"]
        # self.depth_paths = scene_info["depth_paths"]
        # self.intrinsics = scene_info["intrinsics"]
        # self.poses = scene_info["poses"]
        # self.pairs = scene_info["pairs"]
        # self.overlaps = scene_info["overlaps"]
        # threshold = (self.overlaps > min_overlap) & (self.overlaps < max_overlap)
        # self.pairs = self.pairs[threshold]
        # self.overlaps = self.overlaps[threshold]

        self.DEPTH_DIR = osp.join(self.data_root, 'depths')
        self.IMAGE_DIR = osp.join(self.data_root, 'images')
        # THERMAL_INTRINSIC_FILE = osp.join(self.data_root, 'T_Pose', 'intrinsics.txt') # e.g: query/20211204164547_0039/20211204164547_0039_T.JPG SIMPLE_RADIAL 640 512 1108.81 320 256 -0.05475
        # RGB_IMAGE_FILE = osp.join(self.data_root, 'sparse', 'images.txt')
        # POSE_FILE = osp.join(self.data_root, 'T_Pose', f"{scene_name}.txt") # DJI_20211204164547_0039_T.JPG -0.07013761932220364 0.018802470060566182 0.06553912869954459 0.9952044031678818 -2.0469158552986317 -0.023389970016583543 -0.1501977856089987
        
        pair_data = np.load(osp.join(self.data_root, npz_file), allow_pickle=True)

        # breakpoint()

        self.image_paths = [f.split("/")[-1] for f in pair_data['image_paths'][:-1]]
        self.depth_paths = [f.split("/")[-1] for f in pair_data['depth_paths'][:-1]]
        self.intrinsics = pair_data['intrinsics'][:-1]
        self.poses = pair_data['poses'][:-1] 
        self.pair_infos = pair_data['pair_infos'][:-1]  

        self.idx_pairs = []
        for idx_pair, overlap, size in self.pair_infos:
            if overlap < min_overlap or overlap > max_overlap:
                continue
            self.idx_pairs.append(idx_pair)

        del pair_data
        del self.pair_infos

        if randomize_size:
            area = ht * wt
            s = int(16 * (math.sqrt(area) // 16))
            sizes = ((ht, wt), (s, s), (wt, ht))
            choice = RANK % 3
            ht, wt = sizes[choice]
            # counts, bins = np.histogram(self.overlaps,20)
        
        # print(counts)
        self.wt, self.ht = (wt, ht) if split == "train" else (1200, 800)
        self.resize = (ht, wt) # (384, 512)
        self.normalize = normalize
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

        self.local_random_1 = random.Random(time.time())
        self.local_random_2 = random.Random(time.time() + 666)

    def load_im(self, im_path, normalize=False, to_gray=False, padding=False):
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, _ = im.shape
        h_new, w_new = self.resize
        im = cv2.resize(im, (w_new, h_new))
        if self.use_randaug:
            im = self.photometric_aug(image=im)['image']
        scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)
        if to_gray:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)[None]
            
        if padding:  # padding to square
            pad_to = max(h_new, w_new)
            im, mask = pad_bottom_right(im, pad_to, ret_mask=True)
        else:
            mask = None
        im = (torch.from_numpy(im)/ 255).float()
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
            # pad_to = max(h_new, w_new)
            depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
        return torch.from_numpy(depth).float()

    def __len__(self):
        return len(self.idx_pairs)

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
        reverse_mode = self.local_random_1.randint(0, 1)  # 0: visible-infrared, 1: infrared-visible

        idx1, idx2 = self.idx_pairs[pair_idx]
        
        K_0 = torch.tensor(self.intrinsics[idx1].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.intrinsics[idx2].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T0 = torch.tensor(self.poses[idx1].copy(), dtype=torch.float)
        T1 = torch.tensor(self.poses[idx2].copy(), dtype=torch.float)

        T_0to1 = torch.as_tensor(
            np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float
        )[:4, :4]
        T_1to0 = T_0to1.inverse()

        im_A_ref = osp.join(self.IMAGE_DIR, self.image_paths[idx1])
        im_B_ref = osp.join(self.IMAGE_DIR, self.image_paths[idx2])

        depth_A_ref =  osp.join(self.DEPTH_DIR, self.depth_paths[idx1])
        depth_B_ref = osp.join(self.DEPTH_DIR, self.depth_paths[idx2])

        im_A, mask_A, scale_A = self.load_im(im_A_ref, normalize=self.normalize, padding=self.img_padding, to_gray=True)
        im_B, mask_B, scale_B = self.load_im(im_B_ref, normalize=self.normalize, padding=self.img_padding, to_gray=True)
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
        # if DEBUG_MODE:
        #     tensor_to_pil(im_A[0], unnormalize=True).save(
        #         f"vis/im_A.jpg")
        #     tensor_to_pil(im_B[0], unnormalize=True).save(
        #         f"vis/im_B.jpg")
        data_dict = {
            "image0": im_A[0],
            "image0_identifier": im_A_ref.split(".jpg")[0],
            "image1": im_B[0],
            "image1_identifier": im_B_ref.split(".jpg")[0],
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


class MTVBuilder:
    def __init__(self, data_root="MTV", loftr_ignore=True, imc21_ignore=True) -> None:
        self.data_root = data_root
        self.all_scenes = [ name for name in os.listdir(data_root) if osp.isdir(osp.join(data_root, name)) ]
        print(f" ALL SCENES: {self.all_scenes}")

    def build_scenes(self, RANK, split="train", min_overlap=0.0, scene_names=['countryroad', 'racetrack', 'villa', 'park'], **kwargs):
    # def build_scenes(self, RANK, split="train", min_overlap=0.0, scene_names=['racetrack2'], **kwargs):
        if split == "train":
            npz_files = [osp.join(self.data_root, scene_name, f"{scene_name}_train.npz") for scene_name in scene_names]
        elif split == "val":
            npz_files = [osp.join(self.data_root, scene_name, f"{scene_name}_val.npz") for scene_name in scene_names]
        elif split == "test":
            npz_files = [osp.join(self.data_root, scene_name, f"{scene_name}_test.npz") for scene_name in scene_names]
        else:
            raise ValueError(f"Split {split} not available")

        scenes = []
        for i in range(len(npz_files)):
            scenes.append(
                MTVScene(
                    self.data_root, RANK, min_overlap=min_overlap, scene_name=scene_names[i], npz_file=npz_files[i], split=split,
                    **kwargs
                )
            )

        return scenes

    def weight_scenes(self, concat_dataset, alpha=0.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n) / n ** alpha for n in ns])
        return ws
