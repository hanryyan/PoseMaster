import random
import torch
from torch import nn
import numpy as np
from einops import rearrange
from PIL import Image
from dataclasses import dataclass
from torchvision import transforms
import pytorch_lightning as pl

from transformers import AutoImageProcessor, AutoModel
from transformers.utils import ModelOutput
from typing import Iterable, Optional, Union, List

import posemaster
from posemaster.utils.typing import *
from posemaster.models.autoencoders.hunyuan3d.modules.transformer_blocks import ResidualAttentionBlock
from posemaster.models.autoencoders.hunyuan3d.modules.embedder import FourierEmbedder
from posemaster.utils.base import BaseModule

ImageType = Union[np.ndarray, torch.Tensor, Image.Image]

@posemaster.register("dinov2-bone-point-embedder")
class Dinov2BonePointEncoder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: Optional[str] = None # the pretrained model name or path for condition model
        pretrained_dino_name_or_path: Optional[str] = None # the pretrained model name or path for dino
        image_size_dino: int = 224
        empty_embeds_ratio: float = 0.1 # the ratio of empty embeds
        dino_type: Optional[str] = None

        ############### skeleton #############
        num_freqs: int = 8
        include_pi: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # Load the DINOV2 model and processor
        self.dino_type = self.cfg.pretrained_dino_name_or_path
        self.dino_model = AutoModel.from_pretrained(self.cfg.pretrained_dino_name_or_path)

        self.image_preprocess_dino = AutoImageProcessor.from_pretrained(self.dino_type)
        self.transform_dino = transforms.Compose(
            [   
                transforms.Resize(self.cfg.image_size_dino, transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(self.cfg.image_size_dino),  # crop a (image_size_dino, image_size_dino) square
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.bone_points_embedder = FourierEmbedder(num_freqs=self.cfg.num_freqs, include_pi=self.cfg.include_pi)

        self.bone_linear = nn.Linear(self.bone_points_embedder.get_dims(3) + 3, 1024)

        self.self_attn_1 = ResidualAttentionBlock(
            n_ctx=None,
            device=None,
            dtype=None,
            width=1024,
            heads=16,
            init_scale=0.02,
            flash=True
        )

        self.self_attn_2 = ResidualAttentionBlock(
            n_ctx=None,
            device=None,
            dtype=None,
            width=1024,
            heads=16,
            init_scale=0.02,
            flash=True
        )

        # freeze the dino model parameters
        self.dino_model.eval()
        self.dino_model.requires_grad_(False)

        # load pretrained_model_name_or_path
        if self.cfg.pretrained_model_name_or_path:
            print(f"####### Loading condition model from {self.cfg.pretrained_model_name_or_path} #######")
            ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            condition_model_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith('condition.'):
                    if k.startswith('condition.dino_model'):
                        continue
                    else:
                        condition_model_ckpt[k.replace('condition.', '')] = v
            self.load_state_dict(condition_model_ckpt, strict=False)
 
    def encode_image_dino(self, images: Iterable[Optional[ImageType]], **kwargs) -> torch.FloatTensor:
        if isinstance(images, (np.ndarray, torch.Tensor)): # for training process
            assert images.min() >= 0.0 and images.max() <= 1.0, "The pixel values should be in the range of [0, 1]"
            pixel_values = self.transform_dino(images.permute(0, 3, 1, 2))
        else: # for inference process
            pixel_values = self.image_preprocess_dino.preprocess(images, return_tensors='pt', \
                    do_rescale=True, do_resize=True, size=self.cfg.image_size_dino, crop_size=self.cfg.image_size_dino).pixel_values


        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)

        vision_outputs = self.dino_model(
            rearrange(pixel_values.to(self.dino_model.device), "B N C H W -> (B N) C H W"), 
        )

        return vision_outputs.last_hidden_state

    def encode_skeleton(self, bone_points: torch.Tensor = None) -> torch.FloatTensor:
        bone_points_pcd = bone_points[:, :, :3]
        bone_points_feat = bone_points[:, :, 3:]
        bone_points_embedd = self.bone_points_embedder(bone_points_pcd)
        bone_points_feat = self.bone_linear(torch.cat([bone_points_embedd, bone_points_feat], dim=-1))
        bone_points_feat = self.self_attn_1(bone_points_feat)
        bone_points_feat = self.self_attn_2(bone_points_feat)
        return bone_points_feat

    @torch.no_grad()
    def encode_image(self, images: Iterable[Optional[ImageType]], **kwargs) -> torch.FloatTensor:
        dino_embeds = self.encode_image_dino(images, **kwargs)
        return dino_embeds

    def forward(self, batch):
        bs = batch["surface"].shape[0]
        dtype = batch["surface"].dtype
        dino_embeds = None
        skeleton_embeds = None
        
        dino_embeds = self.encode_image(batch["image"])
        if random.random() < self.cfg.empty_embeds_ratio:
            dino_embeds = torch.zeros_like(dino_embeds).to(self.device).to(dtype)

        skeleton_embeds= self.encode_skeleton(batch['bone_points'])
        
        return torch.cat([dino_embeds, skeleton_embeds], dim=1)
