import math
import os
import json
import re
from dataclasses import dataclass, field

import random
import pickle
import imageio
import fpsample
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
from io import BytesIO
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from posemaster import register
from posemaster.utils.typing import *
from posemaster.utils.config import parse_structured

def get_normalized_info(skeleton: np.ndarray):
    bbox = np.min(skeleton, axis=0), np.max(skeleton, axis=0)
    center = (bbox[1] + bbox[0]) / 2
    scale = (bbox[1] - bbox[0]).max()
    return center, scale

def interpolate_bone_points(bone_start, bone_end, bone_point_interval=0.02):
    """
    在骨骼的起点和终点之间插入点
    
    Args:
        bone_start: 骨骼起点坐标, shape (1, 3)
        bone_end: 骨骼终点坐标, shape (1, 3)
        bone_point_interval: 插入点的间隔距离
    
    Returns:
        interpolated_points: 插值后的所有点，包括起点和终点
    """
    # 计算骨骼长度
    bone_vector = bone_end - bone_start
    bone_length = np.linalg.norm(bone_vector)
    
    # 计算需要插入的点数
    interpolation_num = int(bone_length / bone_point_interval)
    
    # 如果骨骼长度小于分割距离，直接返回起点和终点
    if interpolation_num <= 0:
        return np.array([bone_start, bone_end])
    
    # 生成插值参数 t (从0到1)
    t_values = np.linspace(0, 1, interpolation_num + 2)  # +2 包含起点和终点
    
    # 线性插值
    interpolated_points = bone_start + np.outer(t_values, bone_vector)
    
    return np.array(interpolated_points)

def load_skeleton(skeleton_data_path, num_bone_points=256, normalize_bone=True, bone_point_interval=0.005, translation=np.array([0.0, 0.0, 0.0]), aug_scale=1.0):
    data = np.load(skeleton_data_path, allow_pickle=True)
    skeleton = data['body']
    if "lefthand" in data.keys():
        lefthand = data["lefthand"]
        skeleton += lefthand
    if "righthand" in data.keys():
        righthand = data["righthand"]
        skeleton += righthand
     
    # obtain the bone data
    bone_points = np.empty([0, 6])
    bone_starts = []
    bone_ends = []
    for idx, (keypoint, name) in enumerate(skeleton):
        bone_starts.append(keypoint[0])
        bone_ends.append(keypoint[1])

    bone_starts = np.array(bone_starts)
    bone_ends = np.array(bone_ends)

    if normalize_bone:
        # normalize the bone to align surface
        # if os.path.exists(skeleton_data_path.replace("bone_points.pkl", "geometry.npz")):
        #     bone_norm_info = np.load(skeleton_data_path.replace("bone_points.pkl", "geometry.npz"), allow_pickle=True)
        #     center, scale = bone_norm_info['center'], bone_norm_info['scale']
        #     bone_starts = (bone_starts - center) * scale
        #     bone_ends = (bone_ends - center) * scale
        # else:
            print(f"norm info not found for {skeleton_data_path}, using skeleton bounding box")
            skeleton_points = np.concatenate([bone_starts, bone_ends], axis=0)
            center, scale = get_normalized_info(skeleton_points)
            skeleton_points = (skeleton_points - center) * 1 / scale * 2 * 0.80
            bone_starts = (bone_starts - center) * 1 / scale * 2 * 0.80
            bone_ends = (bone_ends - center) * 1 / scale * 2 * 0.80
            # move the bone to the bottom of the y-axis for y-up coordinate system
            offset = [0, -0.98 + abs(skeleton_points[:, 1].min()), 0]
            bone_starts = bone_starts + np.array(offset).reshape(1, 3)
            bone_ends = bone_ends + np.array(offset).reshape(1, 3)


    # skeleton augmentation
    bone_starts = bone_starts + translation
    bone_ends = bone_ends + translation
    bone_starts = bone_starts * aug_scale
    bone_ends = bone_ends * aug_scale

    # obtain the dense bone points
    for idx in range(len(bone_starts)):
        bone_start = bone_starts[idx]
        bone_end = bone_ends[idx]
        bone_point = interpolate_bone_points(bone_start, bone_end, bone_point_interval)
        bone_point_direction = (bone_start - bone_end).reshape(1,3).repeat(bone_point.shape[0], 0)
        bone_point = np.concatenate([bone_point, bone_point_direction], axis=-1)
        bone_points = np.concatenate([bone_points, bone_point], axis=0)

    # sample the dense bone points
    if bone_points.shape[0] < num_bone_points:
        idx = np.random.choice(bone_points.shape[0], num_bone_points - bone_points.shape[0])
        bone_points = np.concatenate([bone_points, bone_points[idx]], axis=0)
    else:
        idx = fpsample.bucket_fps_kdline_sampling(bone_points[:, :3], num_bone_points, h=5)
        bone_points = bone_points[idx]

    bone_points = torch.from_numpy(bone_points).to(torch.float32)
    ret = {"bone_points": bone_points[:, :6]}
    return ret

def process_img(image, image_size, background_color=(255,255,255), foreground_ratio=0.95):
    alpha = image.split()[-1]
    background = Image.new("RGBA", image.size, (*background_color, 255))
    image = Image.alpha_composite(background, image)
    image = image.crop(alpha.getbbox())
    new_size = tuple(int(dim * foreground_ratio) for dim in image.size)
    resized_image = image.resize(new_size)
    padded_image = Image.new("RGBA", image.size, (*background_color, 255))
    paste_position = ((image.width - resized_image.width) // 2, (image.height - resized_image.height) // 2)
    padded_image.paste(resized_image, paste_position)

    # expand image to 1:1
    width, height = padded_image.size
    if width == height:
        image = padded_image.resize((image_size, image_size))
        return image.convert("RGB"), alpha
    new_size = (max(width, height), max(width, height))
    image = Image.new("RGBA", new_size, (*background_color, 255))
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    image.paste(padded_image, paste_position)
    image = image.resize((image_size, image_size))
    return image.convert("RGB"), alpha

def tsdf_func(sdf_value, sdf_res):
    '''
    sdf_res = 128
    '''
    if sdf_value >= 1 / sdf_res:
        ret = 1 / sdf_res
    elif sdf_value <= -1 / sdf_res:
        ret = -1 / sdf_res
    else:
        ret = sdf_value
    return ret * sdf_res

@dataclass
class CharacterDataModuleConfig:
    root_dir: List[str] = None

    ################################# Geometry part #################################
    load_geometry: bool = True           # whether to load geometry data
    geo_data_type: str = "occupancy"     # occupancy, sdf
    # for occupancy and sdf data
    n_samples: int = 4096                # number of points in input point cloud
    n_sharp_samples: int = 16384         # number of points in sharp point cloud
    upsample_ratio: int = 1              # upsample ratio for input point cloud
    sampling_strategy: Optional[str] = None    # sampling strategy for input point cloud
    scale: float = 1.0                   # scale of the input point cloud and target supervision
    noise_sigma: float = 0.0             # noise level of the input point cloud
    rotate_points: bool = False          # whether to rotate the input point cloud and the supervision
    load_supervision: bool = True        # whether to load supervision
    supervision_type: str = "occupancy"  # occupancy, sdf, tsdf
    tsdf_threshold: float = 0.05         # threshold for truncating sdf values, used when input is sdf
    sdf_res: int = 256                   # resolution of the sdf grid
    n_supervision: int = 10000           # number of points in supervision

    ################################# Image part #################################
    load_image: bool = False             # whether to load images 
    random_color_jitter: bool = False     # whether to
    random_rotate: bool = True           # whether to 
    crop_image: bool = True              # whether to crop the input image
    image_data_path: str = ""            # path to the image data
    image_type: str = "rgb"              # rgb, normal
    background_color: Tuple[int, int, int] = field(
            default_factory=lambda: (255, 255, 255)
        )
    idx: Optional[List[int]] = None      # index of the image to load
    n_views: int = 1                     # number of views
    image_size: int = 224               # size of the image
    foreground_ratio: float = 0.95      # foreground ratio of the image

    ################################# Skeleton part #################################
    load_skeleton: bool = True           # whether to load skeleton data
    num_bone_points: int = 256          # number of bone query points
    aug_scale_range: List[float] = field(
            default_factory=lambda: [0.9, 1]
        )
    bone_point_interval: float = 0.02
    data_aug_prob: float = 0.15
    batch_size: int = 1 # batch size
    num_workers: int = 8 # number of

# cos_helper = CosHelper()
class CharacterDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: CharacterDataModuleConfig = cfg
        self.split = split
        self.data = []
        
        for data in self.cfg.root_dir:
            self.data += json.load(open(f'{data}/{split}.json'))
        
        print(f"Loaded {len(self.data)} {split} uids")

        if split != "train":
            self.data_aug_prob = -1.0

        # add ColorJitter transforms for input images
        if self.cfg.random_color_jitter:
            self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
        
        # add RandomRotation transforms for input images
        if self.cfg.random_rotate:
            self.rotate = transforms.RandomRotation(degrees=30, fill=(*self.cfg.background_color, 0.0)) # by default 30 deg

        self.tsdf_func = np.vectorize(tsdf_func, otypes=[float], excluded=['sdf_res'])

    
    def __len__(self):
        return len(self.data)


    def _load_surface_from_geometry(self, data) -> Dict[str, Any]:
        surface = data['surface']
        if "sharp_surface" in data:
            sharp_surface = data['sharp_surface']
            surface = np.concatenate((surface, sharp_surface), axis=0)
        # padding to 81920 points
        if surface.shape[0] < self.cfg.n_samples:
            choice = np.random.permutation(self.cfg.n_samples - surface.shape[0])
            surface = np.concatenate((surface, surface[choice]))
        else:
            choice = np.random.permutation(surface.shape[0])[:self.cfg.n_samples]
            surface = surface[choice]
        
        if random.random() < self.cfg.data_aug_prob:
            points = surface[:, :3]  # [N, 3]

            # Global scaling augmentation
            scale = random.uniform(*self.cfg.aug_scale_range)
            surface[:, :3] = surface[:, :3] * scale

            points = surface[:, :3]

            # Translation augmentation
            axis_max = np.argmax(np.ptp(points, axis=0))
            shift_axis = random.choice([i for i in range(3) if i != axis_max])
            points_axis_range = [points[:, shift_axis].max(), points[:, shift_axis].min()]
            shift_range = [-0.98 + abs(points_axis_range[1]), 0.98 - abs(points_axis_range[0])]
            translation = random.uniform(*shift_range)
            translation = np.array([[translation if i == shift_axis else 0 for i in range(3)]], dtype=np.float32)

            surface[:, :3] = surface[:, :3] + translation

            # angle = np.random.uniform(0, 180) * np.pi / 180
            
            # # 随机选择旋转轴
            # axis = np.random.choice(['x', 'y', 'z'])
            
            # if axis == 'x':
            #     rotation_matrix = np.array([
            #         [1, 0, 0],
            #         [0, np.cos(angle), -np.sin(angle)],
            #         [0, np.sin(angle), np.cos(angle)]
            #     ])
            # elif axis == 'y':
            #     rotation_matrix = np.array([
            #         [np.cos(angle), 0, np.sin(angle)],
            #         [0, 1, 0],
            #         [-np.sin(angle), 0, np.cos(angle)]
            #     ])
            # else:  # z轴
            #     rotation_matrix = np.array([
            #         [np.cos(angle), -np.sin(angle), 0],
            #         [np.sin(angle), np.cos(angle), 0],
            #         [0, 0, 1]
            #     ])
        else:
            translation = np.array([0.0, 0.0, 0.0])
            scale = 1.0
            # rotation_matrix = np.eye(3)
        surface = torch.from_numpy(surface).to(torch.float32)
        return surface, translation, scale

    def _load_shape_supervision_occupancy_or_sdf(self, data) -> Dict[str, Any]:
        # for supervision
        ret = {}
        if self.cfg.geo_data_type == "occupancy":
            rand_points = np.asarray(data['points']) * 2 # range from -1.1 to 1.1
            occupancies = np.asarray(data['occupancies'])
            occupancies = np.unpackbits(occupancies)
        elif self.cfg.geo_data_type == "sdf":
            rand_points = data['rand_points']
            sdfs = data['sdf']
            # sharp_rand_points = data['sharp_rand_points']
            # sharp_sdfs = data['sharp_sdf']
        else:
            raise NotImplementedError(f"Data type {self.cfg.geo_data_type} not implemented")

        
        # random sampling
        rng = np.random.default_rng()
        ind = rng.choice(rand_points.shape[0], self.cfg.n_supervision, replace=False)
        rand_points = rand_points[ind]
        rand_points = rand_points * self.cfg.scale
        # sharp_ind = rng.choice(sharp_rand_points.shape[0], self.cfg.n_supervision, replace=False)
        # sharp_rand_points = sharp_rand_points[sharp_ind]
        # sharp_rand_points = sharp_rand_points * self.cfg.scale

        # detect inf and nan for rand_points
        if np.any(np.isinf(rand_points)) or np.any(np.isnan(rand_points)):
            raise ValueError(f"Inf or nan in rand_points")
        # detect inf and nan for sdfs
        if self.cfg.geo_data_type == "sdf":
            if np.any(np.isinf(sdfs)) or np.any(np.isnan(sdfs)):
                raise ValueError(f"Inf or nan in sdfs")
        ret["rand_points"] = rand_points.astype(np.float32)
        # ret["sharp_rand_points"] = sharp_rand_points.astype(np.float32)

        if self.cfg.geo_data_type == "occupancy":
            assert self.cfg.supervision_type == "occupancy", "Only occupancy supervision is supported for occupancy data"
            occupancies = occupancies[ind]
            ret["occupancies"] = occupancies.astype(np.float32)
        elif self.cfg.geo_data_type == "sdf":
            if self.cfg.supervision_type == "sdf":
                ret["sdf"] = sdfs[ind].flatten().astype(np.float32)
            elif self.cfg.supervision_type == "occupancy":
                ret["occupancies"] = np.where(sdfs[ind].flatten() < 1e-3, 0, 1).astype(np.float32)
            elif self.cfg.supervision_type == "tsdf":
                # ret["sdf"] = sdfs[ind].flatten().astype(np.float32).clip(-self.cfg.tsdf_threshold, self.cfg.tsdf_threshold) / self.cfg.tsdf_threshold
                ret["sdf"] = self.tsdf_func(sdfs[ind], self.cfg.sdf_res).flatten().astype(np.float32)
                # ret["sharp_sdf"] = sharp_sdfs[sharp_ind].flatten().astype(np.float32).clip(-self.cfg.tsdf_threshold, self.cfg.tsdf_threshold) / self.cfg.tsdf_threshold
            else:
                raise NotImplementedError(f"Supervision type {self.cfg.supervision_type} not implemented")

        return ret


    def _load_image(self, image) -> Dict[str, Any]:
        ret = {}
        if self.cfg.background_color == [-1, -1, -1]:
            background_color = torch.randint(0, 256, (3,))
        else:
            background_color = torch.as_tensor(self.cfg.background_color)
        if self.cfg.image_type == "rgb" or self.cfg.image_type == "normal":
            assert self.cfg.n_views == 1, "Only single view is supported for single image"
            # img = torch.from_numpy(np.asarray(data[f"{str(idx).zfill(4)}_{self.cfg.image_type}"]).copy()) / 255.0
            if self.cfg.random_color_jitter and random.random() < self.cfg.data_aug_prob and self.split == "train":
                rgb = self.color_jitter(image.convert("RGB"))
                image = Image.merge("RGBA", (*rgb.split(), image.getchannel('A')))
    
            # add random rotation
            if self.cfg.random_rotate and random.random() < self.cfg.data_aug_prob and self.split == "train":
                rgb = self.rotate(image)
            
            if self.cfg.crop_image:
                image, alpha = process_img(image, self.cfg.image_size, background_color, self.cfg.foreground_ratio)
            else:
                alpha = image.getchannel('A') 
                background = Image.new("RGBA", image.size, (*background_color, 255))
                image = Image.alpha_composite(background, image).convert("RGB")

            ret["image"] = torch.from_numpy(np.array(image) / 255)
        else:
            raise NotImplementedError(f"Image type {self.cfg.image_type} not implemented")
        
        return ret

    def _get_data(self, index):
        data = self.data[index]
        uid = list(data.keys())[0]
        ret = {"uid": uid}
        frame_length = len(data[uid])
        random_img_idx = np.random.randint(0, frame_length)
        random_pose_idx = np.random.randint(0, frame_length)

        # load image
        if self.cfg.load_image:
            img_data_path = data[uid][random_img_idx]['image']
            if img_data_path is None:
                raise ValueError(f"Image data not found for {uid}")
            img_data = Image.open(img_data_path).copy()
            ret.update(self._load_image(img_data))

        # load geometry
        if self.cfg.load_geometry:
            geo_data_path = data[uid][random_pose_idx]['geometry']
            geo_data = np.load(geo_data_path, allow_pickle=True)
            surface, translation, scale = self._load_surface_from_geometry(geo_data)
            ret.update({'surface': surface})

        # load supervision for shape vae
        if self.cfg.load_supervision:
            ret.update(self._load_shape_supervision_occupancy_or_sdf(geo_data))

        
        
        # load skeleton
        if self.cfg.load_skeleton:
            skeleton_data_path = data[uid][random_pose_idx]['skeleton']
            if skeleton_data_path is None:
                raise ValueError(f"Skeleton data not found for {uid}")
            ret.update(load_skeleton(skeleton_data_path, num_bone_points=self.cfg.num_bone_points, bone_point_interval=self.cfg.bone_point_interval, translation=translation, aug_scale=scale))

        return ret
        
    def __getitem__(self, index):
        try:
            return self._get_data(index)
        except Exception as e:
            print(f"Error in {list(self.data[index].keys())[0]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))
        # return self._get_data(index)

    def collate(self, batch):
        return torch.utils.data.default_collate(batch)


@register("Character-direction-datamodule")
class CharacterDataModule(pl.LightningDataModule):
    cfg: CharacterDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(CharacterDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = CharacterDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = CharacterDataset(self.cfg, "test")
        if stage in [None, "test", "predict"]:
            self.test_dataset = CharacterDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)