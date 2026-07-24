# -*- coding: utf-8 -*-
"""
Tencent is pleased to support the open source community by making Tencent Hunyuan 3D Omni available.

Copyright (C) 2025 Tencent.  All rights reserved. The below software and/or models in this 
distribution may have been modified by Tencent ("Tencent Modifications"). All Tencent Modifications 
are Copyright (C) Tencent.

Tencent Hunyuan 3D Omni is licensed under the TENCENT HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT 
except for the third-party components listed below, which is licensed under different terms. 
Tencent Hunyuan 3D Omni does not impose any additional limitations beyond what is outlined in the 
respective licenses of these third-party components. Users must comply with all terms and conditions 
of original licenses of these third-party components and must ensure that the usage of the third party 
components adheres to all relevant laws and regulations. 

For avoidance of doubts, Tencent Hunyuan 3D Omni means training code, inference-enabling code, parameters, 
and/or weights of this Model, which are made publicly available by Tencent in accordance with TENCENT 
HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT.
"""

import math
import random
from functools import partial
from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .inference_utils import extract_geometry_vanilla, extract_geometry_fast
from .modules.checkpoint import checkpoint
from .modules.distributions import DiagonalGaussianDistribution
from .modules.embedder import FourierEmbedder, TriplaneLearnedFourierEmbedder
from .modules.transformer_blocks import ResidualCrossAttentionBlock, Transformer


def fps(
    src: torch.Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
):
    from torch_cluster import fps as fps_fn
    return fps_fn(src.float(), batch, ratio, random_start, batch_size, ptr)


class PointCrossAttentionEncoder(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 train_num_latents_range: list,
                 train_num_latents_probs: list,
                 downsample_ratio: float,
                 pc_size: int,
                 pc_sharpedge_size: int,
                 fourier_embedder: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 normal_pe: bool = False,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 qk_norm: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.downsample_ratio = downsample_ratio
        self.point_feats = point_feats
        self.normal_pe = normal_pe

        self.surface_pts_range = [int(self.downsample_ratio * _) for _ in train_num_latents_range]
        self.surface_pts_probs = train_num_latents_probs

        if pc_sharpedge_size == 0:
            print(f'PointCrossAttentionEncoder INFO: pc_sharpedge_size is not given, using pc_size as pc_sharpedge_size')
        else:
            print(f'PointCrossAttentionEncoder INFO: pc_sharpedge_size is given, using pc_size={pc_size}, pc_sharpedge_size={pc_sharpedge_size}')

        assert pc_size + pc_sharpedge_size >= max(
            self.surface_pts_range), f"Sum of pc_size {pc_size} and pc_sharpedge_size {pc_sharpedge_size} must be greater than the maximum surface points range {self.surface_pts_range}"
        assert sum(self.surface_pts_probs) == 1, "Sum of surface_pts_probs must be 1"
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size

        self.fourier_embedder = fourier_embedder

        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width, device=device, dtype=dtype)
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            qk_norm=qk_norm
        )

        self.self_attn = None
        if layers > 0:
            self.self_attn = Transformer(
                device=device,
                dtype=dtype,
                n_ctx=num_latents,
                width=width,
                layers=layers,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                flash=flash,
                use_checkpoint=use_checkpoint,
                qk_norm=qk_norm
            )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def sample_points_and_latents(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):

        B, N, D = pc.shape
        assert N >= max(
            self.surface_pts_range), "Number of points must be greater than the maximum surface points range"

        if self.training:
            # Select suitable surface points
            num_pts = random.choices(self.surface_pts_range, weights=self.surface_pts_probs)[0]
        else:
            # infer with corresponding surface points
            num_pts = self.num_latents * self.downsample_ratio

        # Compute number of latents
        num_latents = int(num_pts / self.downsample_ratio)

        # Compute the number of random and sharpedge latentst
        num_random_query = self.pc_size / (self.pc_size + self.pc_sharpedge_size) * num_latents
        num_sharpedge_query = num_latents - num_random_query

        # Split random and sharpedge surface points
        random_pc, sharpedge_pc = torch.split(pc, [self.pc_size, self.pc_sharpedge_size], dim=1)
        assert random_pc.shape[1] <= self.pc_size, "Random surface points size must be less than or equal to pc_size"
        assert sharpedge_pc.shape[
                   1] <= self.pc_sharpedge_size, "Sharpedge surface points size must be less than or equal to pc_sharpedge_size"

        # Randomly select random surface points and random query points
        input_random_pc_size = int(num_random_query * self.downsample_ratio)
        random_query_ratio = num_random_query / input_random_pc_size
        #idx_random_pc = torch.randperm(random_pc.shape[1])[:input_random_pc_size]
        idx_random_pc = torch.randperm(random_pc.shape[1], device = random_pc.device)[:input_random_pc_size] #加速
        
        input_random_pc = random_pc[:, idx_random_pc, :]
        flatten_input_random_pc = input_random_pc.view(B * input_random_pc_size, D)
        N_down = int(flatten_input_random_pc.shape[0] / B)
        batch_down = torch.arange(B).to(pc.device)
        batch_down = torch.repeat_interleave(batch_down, N_down)
        idx_query_random = fps(flatten_input_random_pc, batch_down, ratio=random_query_ratio)
        query_random_pc = flatten_input_random_pc[idx_query_random].view(B, -1, D)

        # flatten_input_random_pc = input_random_pc[:, 0: int(num_random_query*4), :].contiguous() #speed up
        # flatten_input_random_pc = flatten_input_random_pc.view(B * int(num_random_query*4), D)
        # N_down = int(flatten_input_random_pc.shape[0] / B)
        # batch_down = torch.arange(B).to(pc.device)
        # batch_down = torch.repeat_interleave(batch_down, N_down)
        # idx_query_random = fps(flatten_input_random_pc, batch_down, ratio=0.25)
        # query_random_pc = flatten_input_random_pc[idx_query_random].view(B, -1, D)
        # import pdb
        # pdb.set_trace()
        

        # Randomly select sharpedge surface points and sharpedge query points
        input_sharpedge_pc_size = int(num_sharpedge_query * self.downsample_ratio)
        if input_sharpedge_pc_size == 0:
            input_sharpedge_pc = torch.zeros(B, 0, D, dtype=input_random_pc.dtype).to(pc.device)
            query_sharpedge_pc = torch.zeros(B, 0, D, dtype=query_random_pc.dtype).to(pc.device)
        else:
            sharpedge_query_ratio = num_sharpedge_query / input_sharpedge_pc_size
            #idx_sharpedge_pc = torch.randperm(sharpedge_pc.shape[1])[:input_sharpedge_pc_size]
            idx_sharpedge_pc = torch.randperm(sharpedge_pc.shape[1],device = sharpedge_pc.device)[:input_sharpedge_pc_size] #加速
            input_sharpedge_pc = sharpedge_pc[:, idx_sharpedge_pc, :]
            flatten_input_sharpedge_surface_points = input_sharpedge_pc.view(B * input_sharpedge_pc_size, D)
            N_down = int(flatten_input_sharpedge_surface_points.shape[0] / B)
            batch_down = torch.arange(B).to(pc.device)
            batch_down = torch.repeat_interleave(batch_down, N_down)
            idx_query_sharpedge = fps(flatten_input_sharpedge_surface_points, batch_down, ratio=sharpedge_query_ratio)
            query_sharpedge_pc = flatten_input_sharpedge_surface_points[idx_query_sharpedge].view(B, -1, D)

        # Concatenate random and sharpedge surface points and query points
        query_pc = torch.cat([query_random_pc, query_sharpedge_pc], dim=1)
        input_pc = torch.cat([input_random_pc, input_sharpedge_pc], dim=1)

        # PE
        query = self.fourier_embedder(query_pc)
        data = self.fourier_embedder(input_pc)

        # Concat normal if given 
        if self.point_feats != 0:

            random_surface_feats, sharpedge_surface_feats = torch.split(feats, [self.pc_size, self.pc_sharpedge_size],
                                                                        dim=1)
            input_random_surface_feats = random_surface_feats[:, idx_random_pc, :]
            flatten_input_random_surface_feats = input_random_surface_feats.view(B * input_random_pc_size, -1)
            query_random_feats = flatten_input_random_surface_feats[idx_query_random].view(B, -1,
                                                                                           flatten_input_random_surface_feats.shape[-1])

            if input_sharpedge_pc_size == 0:
                input_sharpedge_surface_feats = torch.zeros(B, 0, self.point_feats,
                                                            dtype=input_random_surface_feats.dtype).to(pc.device)
                query_sharpedge_feats = torch.zeros(B, 0, self.point_feats, dtype=query_random_feats.dtype).to(
                    pc.device)
            else:
                input_sharpedge_surface_feats = sharpedge_surface_feats[:, idx_sharpedge_pc, :]
                flatten_input_sharpedge_surface_feats = input_sharpedge_surface_feats.view(B * input_sharpedge_pc_size,
                                                                                           -1)
                query_sharpedge_feats = flatten_input_sharpedge_surface_feats[idx_query_sharpedge].view(B, -1,
                                                                                                        flatten_input_sharpedge_surface_feats.shape[
                                                                                                            -1])

            query_feats = torch.cat([query_random_feats, query_sharpedge_feats], dim=1)
            input_feats = torch.cat([input_random_surface_feats, input_sharpedge_surface_feats], dim=1)

            if self.normal_pe:
                query_normal_pe = self.fourier_embedder(query_feats[..., :3])
                input_normal_pe = self.fourier_embedder(input_feats[..., :3])
                query_feats = torch.cat([query_normal_pe, query_feats[..., 3:]], dim=-1)
                input_feats = torch.cat([input_normal_pe, input_feats[..., 3:]], dim=-1)

            query = torch.cat([query, query_feats], dim=-1)
            data = torch.cat([data, input_feats], dim=-1)

        if input_sharpedge_pc_size == 0:
            query_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)
            input_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)

        # print(f'query_pc: {query_pc.shape}')
        # print(f'input_pc: {input_pc.shape}')
        # print(f'query_random_pc: {query_random_pc.shape}')
        # print(f'input_random_pc: {input_random_pc.shape}')
        # print(f'query_sharpedge_pc: {query_sharpedge_pc.shape}')
        # print(f'input_sharpedge_pc: {input_sharpedge_pc.shape}')

        return query.view(B, -1, query.shape[-1]), data.view(B, -1, data.shape[-1]), [query_pc, input_pc,
                                                                                      query_random_pc, input_random_pc,
                                                                                      query_sharpedge_pc,
                                                                                      input_sharpedge_pc]

    def _forward(self, pc, feats, query_points = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """
        if query_points is None:
            query, data, pc_infos = self.sample_points_and_latents(pc, feats)
        else:
            B = query_points.shape[0]
            query = self.fourier_embedder(query_points[..., :3])
            if self.normal_pe:
                query_feats = self.fourier_embedder(query_points[..., :3])
            else:
                query_feats = query_points[..., 3:]
            query_sharp_feats = torch.zeros(B, query_points.shape[1], 1, dtype=query.dtype).to(query.device)
            query = torch.cat([query, query_feats, query_sharp_feats], dim=-1)
            data = self.fourier_embedder(pc)
            if self.normal_pe:
                data_feats = self.fourier_embedder(pc[..., :3])
            else:
                data_feats = feats
            data_sharp_feats = torch.zeros(B, pc.shape[1], 1, dtype=data.dtype).to(data.device)
            data = torch.cat([data, data_feats, data_sharp_feats], dim=-1)
            pc_infos = [query_points, pc]

        query = self.input_proj(query)
        query = query
        data = self.input_proj(data)
        data = data

        latents = self.cross_attn(query, data)
        if self.self_attn is not None:
            latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc_infos

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None, query_points: Optional[torch.FloatTensor] = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """

        return checkpoint(self._forward, (pc, feats, query_points), self.parameters(), self.use_checkpoint)


class CrossAttentionDecoder(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 out_channels: int,
                 fourier_embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_checkpoint: bool = False,
                 qk_norm: bool = False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.fourier_embedder = fourier_embedder

        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            qk_norm=qk_norm
        )

        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        return checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)


class ShapeVAE(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 train_num_latents_range: list,
                 train_num_latents_probs: list,
                 downsample_ratio: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 normal_pe: bool = False,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 qk_norm: bool = False,
                 drop_path_rate: float = 0.0,
                 pc_size: int = 0,
                 pc_sharpedge_size: int = 0,
                 sharpedge_label: bool = False
                 ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.sharpedge_label = sharpedge_label
        self.num_latents = num_latents
        self.downsample_ratio = downsample_ratio

        print(f'*' * 100)
        if num_freqs >= 0:
            self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
            print(f'FourierEmbedder: num_freqs={num_freqs}, include_pi={include_pi}')
        else:
            self.fourier_embedder = TriplaneLearnedFourierEmbedder(in_channels=3, dim=width)
            print(f'TriplaneLearnedFourierEmbedder: input_dim=3, out_dim={width}')
        print(f'*' * 100)

        if normal_pe:
            # ! Add more channels for the positional embedding normal
            point_feats = self.fourier_embedder.out_dim
            print(f'*' * 100)
            print(
                f'PointCrossAttentionEncoder INFO: normal_pe is True, adding {self.fourier_embedder.out_dim} channels for the positional embedding normal')
            print(f'*' * 100)

        if self.sharpedge_label:
            # ! Add one more channel for the sharpedge label
            point_feats = point_feats + 1

        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = PointCrossAttentionEncoder(
            device=device,
            dtype=dtype,
            pc_size=pc_size,
            pc_sharpedge_size=pc_sharpedge_size,
            fourier_embedder=self.fourier_embedder,
            normal_pe=normal_pe,
            num_latents=num_latents,
            train_num_latents_range=train_num_latents_range,
            train_num_latents_probs=train_num_latents_probs,
            downsample_ratio=self.downsample_ratio,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint,
            qk_norm=qk_norm
        )

        self.embed_dim = embed_dim
        if embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(width, embed_dim * 2, device=device, dtype=dtype)
            self.post_kl = nn.Linear(embed_dim, width, device=device, dtype=dtype)
            self.latent_shape = (num_latents, embed_dim)
        else:
            self.latent_shape = (num_latents, width)

        self.transformer = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )
        # geometry decoder
        self.geo_decoder = CrossAttentionDecoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint,
            qk_norm=qk_norm
        )

        print(f'*' * 100)
        print(f'ShapeVae INFO')
        print(f'num_latents: {num_latents}')
        print(f'train_num_latents_range: {train_num_latents_range}')
        print(f'train_num_latents_probs: {train_num_latents_probs}')
        print(f'downsample_ratio: {downsample_ratio}')
        print(f'*' * 100)

    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               query_points: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            sample_posterior (bool):

        Returns:
            latents (torch.FloatTensor)
            center_pos (torch.FloatTensor or None):
            posterior (DiagonalGaussianDistribution or None):
        """

        latents, center_pos = self.encoder(pc, feats, query_points = query_points)

        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                latents = posterior.sample()
            else:
                latents = posterior.mode()

        return latents, center_pos, posterior

    def decode(self, latents: torch.FloatTensor):
        latents = self.post_kl(latents)
        return self.transformer(latents)

    def query_geometry(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = self.geo_decoder(queries.to(latents.dtype), latents).squeeze(-1)
        return logits

    def latent2mesh(
        self,
        latents: torch.FloatTensor,
        bounds: Union[Tuple[float], List[float], float] = 1.1,
        octree_depth: int = 7,
        num_chunks: int = 10000,
        mc_level: float = -1 / 512,
        octree_resolution: int = None,
        mc_mode: str = 'mc',
    ):

        # latents: [bs, num_latents, dim]

        outputs = []

        geometric_func = partial(self.query_geometry, latents=latents)

        # 2. decode geometry
        device = latents.device

        SurfaceExtractor = extract_geometry_vanilla

        mesh_v_f, has_surface = SurfaceExtractor(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable_tqdm=True,
            mc_level=mc_level,
            octree_resolution=octree_resolution,
            mc_mode=mc_mode,
        )

        # 3. decode texture
        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            outputs.append((mesh_v, mesh_f))

        return outputs

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            volume_queries (torch.FloatTensor): [B, P, 3]
            sample_posterior (bool):

        Returns:
            logits (torch.FloatTensor): [B, P]
            center_pos (torch.FloatTensor): [B, M, 3]
            posterior (DiagonalGaussianDistribution or None).

        """

        latents, center_pos, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)

        latents = self.decode(latents)
        logits = self.query_geometry(volume_queries, latents)

        return logits, center_pos, posterior

