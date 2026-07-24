import torch
import torch.nn as nn
import numpy as np

import posemaster
from dataclasses import dataclass
from typing import Optional

from posemaster.utils.base import BaseModule
from posemaster.models.denoisers.utils import *

@posemaster.register("hunyuan3d-denoiser")
class Hunyuan3DDenoiser(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: Optional[str] = None
        input_size: int = 1024
        in_channels: int = 4
        hidden_size: int = 2048
        context_dim: int = 1024
        depth: int = 24
        num_heads: int = 16
        # mlp_ratio: float = 4.0
        norm_type: str ='layer'
        qk_norm_type: str ='rms'
        qk_norm: bool = False
        text_len: int=257
        with_decoupled_ca: bool =False
        additional_cond_hidden_state: int=768
        decoupled_ca_dim: int =16
        decoupled_ca_weight: float =1.0
        use_pos_emb: bool=False
        use_attention_pooling: bool=True
        guidance_cond_proj_dim: Optional[str] = None
        qkv_bias: bool =True
        num_moe_layers: int = 6
        num_experts: int = 8
        moe_top_k: int = 2

    cfg: Config

    def configure(self) -> None:
        self.input_size = self.cfg.input_size
        self.depth = self.cfg.depth
        self.in_channels = self.cfg.in_channels
        self.out_channels = self.cfg.in_channels
        self.num_heads = self.cfg.num_heads

        self.hidden_size = self.cfg.hidden_size
        self.norm = nn.LayerNorm if self.cfg.norm_type == 'layer' else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if self.cfg.qk_norm_type == 'rms' else nn.LayerNorm
        self.context_dim = self.cfg.context_dim

        self.with_decoupled_ca = self.cfg.with_decoupled_ca
        self.decoupled_ca_dim = self.cfg.decoupled_ca_dim
        self.decoupled_ca_weight = self.cfg.decoupled_ca_weight
        self.use_pos_emb = self.cfg.use_pos_emb
        self.use_attention_pooling = self.cfg.use_attention_pooling
        self.guidance_cond_proj_dim = self.cfg.guidance_cond_proj_dim

        self.text_len = self.cfg.text_len

        self.x_embedder = nn.Linear(self.cfg.in_channels, self.cfg.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(self.cfg.hidden_size, self.cfg.hidden_size * 4, cond_proj_dim=self.cfg.guidance_cond_proj_dim)

        # Will use fixed sin-cos embedding:
        if self.use_pos_emb:
            self.register_buffer("pos_embed", torch.zeros(1, self.cfg.input_size, self.cfg.hidden_size))
            pos = np.arange(self.input_size, dtype=np.float32)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.use_attention_pooling = self.cfg.use_attention_pooling
        if self.cfg.use_attention_pooling:
            self.pooler = AttentionPool(self.text_len, self.cfg.context_dim, num_heads=8, output_dim=1024)
            self.extra_embedder = nn.Sequential(
                nn.Linear(1024, self.cfg.hidden_size * 4),
                nn.SiLU(),
                nn.Linear(self.cfg.hidden_size * 4, self.cfg.hidden_size, bias=True),
            )

        if self.cfg.with_decoupled_ca:
            self.additional_cond_hidden_state = self.cfg.additional_cond_hidden_state
            self.additional_cond_proj = nn.Sequential(
                nn.Linear(self.cfg.additional_cond_hidden_state, self.cfg.hidden_size * 4),
                nn.SiLU(),
                nn.Linear(self.cfg.hidden_size * 4, 1024, bias=True),
            )

        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=self.cfg.hidden_size,
                            c_emb_size=self.cfg.hidden_size,
                            num_heads=self.cfg.num_heads,
                            # mlp_ratio=self.cfg.mlp_ratio,
                            text_states_dim=self.cfg.context_dim,
                            qk_norm=self.cfg.qk_norm,
                            norm_layer=self.norm,
                            qk_norm_layer=self.qk_norm,
                            skip_connection=layer > self.cfg.depth // 2,
                            with_decoupled_ca=self.cfg.with_decoupled_ca,
                            decoupled_ca_dim=self.cfg.decoupled_ca_dim,
                            decoupled_ca_weight=self.cfg.decoupled_ca_weight,
                            qkv_bias=self.cfg.qkv_bias,
                            use_moe=True if self.cfg.depth - layer <= self.cfg.num_moe_layers else False,
                            num_experts=self.cfg.num_experts,
                            moe_top_k=self.cfg.moe_top_k
                            )
            for layer in range(self.cfg.depth)
        ])
        self.depth = self.cfg.depth

        self.final_layer = FinalLayer(self.cfg.hidden_size, self.out_channels)

        if self.cfg.pretrained_model_name_or_path:
            print(f"####### Loading pretrained DiT model from {self.cfg.pretrained_model_name_or_path} #######")
            ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            denoiser_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith('denoiser_model.'):
                    denoiser_ckpt[k.replace('denoiser_model.', '')] = v
            self.load_state_dict(denoiser_ckpt, strict=True)

    def forward(self,
                model_input: torch.FloatTensor,
                timestep: torch.LongTensor,
                context: torch.FloatTensor):
        # print('contexts', contexts)
        cond = context.to(model_input.dtype)
        # print(f"cond dtype: {cond.dtype}, model_input dtype: {model_input.dtype}")


        t = self.t_embedder(timestep, condition=None)
        x = self.x_embedder(model_input)

        if self.use_pos_emb:
            pos_embed = self.pos_embed.to(x.dtype)
            x = x + pos_embed

        if self.use_attention_pooling:
            extra_vec = self.pooler(cond, None)
            c = t + self.extra_embedder(extra_vec)  # [B, D]
        else:
            c = t

        x = torch.cat([c, x], dim=1)

        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(x, c, cond, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(x)

        x = self.final_layer(x)
        return x