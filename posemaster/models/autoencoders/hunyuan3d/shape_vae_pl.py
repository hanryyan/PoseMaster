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

from typing import List, Tuple, Dict, Optional, Union
from functools import partial
from omegaconf import DictConfig
import torch
from torch.optim import lr_scheduler
import pytorch_lightning as pl

from .misc import instantiate_from_config
from .inference_utils import extract_geometry_vanilla, extract_geometry_fast
from .shape_vae import ShapeVAE


class Latent2MeshOutput(object):
    def __init__(self):
        self.mesh_v = None
        self.mesh_f = None


class ShapeVAEPL(pl.LightningModule):
    def __init__(
        self,
        *,
        module_cfg,
        loss_cfg,
        optimizer_cfg: Optional[DictConfig] = None,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = [],
        train_geo_decoder_only: bool = False,
        train_decoder_only: bool = False,
        fast_decode: bool= False
    ):
        super().__init__()

        self.sal: ShapeVAE = instantiate_from_config(module_cfg, device=None, dtype=None)
        self.loss = instantiate_from_config(loss_cfg)
        self.optimizer_cfg = optimizer_cfg
        self.train_geo_decoder_only = train_geo_decoder_only
        self.train_decoder_only = train_decoder_only

        if self.train_decoder_only:
            self.sal.requires_grad_(False)
            self.sal.geo_decoder.requires_grad_(True)
            self.sal.transformer.requires_grad_(True)

        if self.train_geo_decoder_only:
            self.sal.requires_grad_(False)
            self.sal.geo_decoder.requires_grad_(True)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.save_hyperparameters()
        self.fast_decode = fast_decode

    @property
    def latent_shape(self):
        return self.sal.latent_shape

    @property
    def zero_rank(self):
        return (self.trainer.local_rank==0) if self._trainer else True

    def set_shape_model_only(self):
        self.clip_model = None

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        if self.optimizer_cfg is None:
            optimizers = [torch.optim.AdamW(self.sal.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-3)]
            schedulers = []
        else:
            optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=self.sal.parameters(), lr=lr)
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            optimizers = [optimizer]
            schedulers = [scheduler]

        return optimizers, schedulers

    def forward(self, pc: torch.FloatTensor, feats: torch.FloatTensor, volume_queries: torch.FloatTensor):
        logits, center_pos, posterior = self.sal(pc, feats, volume_queries)
        return posterior, logits

    def encode(self, surface: torch.FloatTensor, sample_posterior=True):
        pc, feats = surface[..., 0:3], surface[..., 3:]
        latents, center_pos, posterior = self.sal.encode(pc=pc, feats=feats, sample_posterior=sample_posterior)
        return latents

    def encode_feat(self, surface, sample_posterior=True):
        pc, feats = surface[..., 0:3], surface[..., 3:]
        features, _ = self.sal.encoder(pc, feats)
        return features
    
    def decode(
        self,
        z_q,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        octree_depth: int = 8,
        num_chunks: int = 250000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        mc_mode: str = 'mc',
        **kwargs,
    ) -> List[Latent2MeshOutput]:
        latents = self.sal.decode(z_q)  # latents: [bs, num_latents, dim]
        
        outputs = self.latent2mesh(latents,
                                   bounds=bounds,
                                   octree_depth=octree_depth,
                                   num_chunks=num_chunks,
                                   mc_level=mc_level,
                                   octree_resolution=octree_resolution,
                                   mc_mode=mc_mode)
        return outputs

    def training_step(self, batch: Dict[str, torch.FloatTensor],
                      batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        """
        Args:
            batch (dict): the batch sample, and it contains:
                - surface (torch.FloatTensor): [bs, n_surface, (3 + input_dim)]
                - geo_points (torch.FloatTensor): [bs, n_pts, (3 + 1)]
            batch_idx (int):
            optimizer_idx (int):
        Returns:
            loss (torch.FloatTensor):
        """

        pc = batch["surface"][..., 0:3]
        feats = batch["surface"][..., 3:]

        volume_queries = batch["geo_points"][..., 0:3]
        volume_labels = batch["geo_points"][..., -1]

        posterior, logits = self(
            pc=pc, feats=feats, volume_queries=volume_queries
        )
        aeloss, log_dict_ae = self.loss(posterior, logits, volume_labels, split="train")

        lr = self.optimizers().param_groups[0]['lr']
        # print(f'Based Learning rate: {self.learning_rate}')
        log_dict_ae['lr_abs'] = lr

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=logits.shape[0],
                      sync_dist=False, rank_zero_only=True)

        return aeloss

    def validation_step(self, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:

        pc = batch["surface"][..., 0:3]
        feats = batch["surface"][..., 3:]

        volume_queries = batch["geo_points"][..., 0:3]
        volume_labels = batch["geo_points"][..., -1]

        posterior, logits = self(
            pc=pc, feats=feats, volume_queries=volume_queries,
        )
        aeloss, log_dict_ae = self.loss(posterior, logits, volume_labels, split="val")

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=logits.shape[0],
                      sync_dist=False, rank_zero_only=True)

        return aeloss

    def latent2mesh(
        self,
        latents: torch.FloatTensor,
        bounds: Union[Tuple[float], List[float], float] = 1.1,
        octree_depth: int = 7,
        num_chunks: int = 10000,
        mc_level: float = -1 / 512,
        octree_resolution: int = None,
        mc_mode: str = 'mc',
    ) -> List[Latent2MeshOutput]:

        # latents: [bs, num_latents, dim]

        outputs = []

        geometric_func = partial(self.sal.query_geometry, latents=latents)

        # 2. decode geometry
        device = latents.device

        SurfaceExtractor = extract_geometry_vanilla
        if self.fast_decode:
            SurfaceExtractor = extract_geometry_fast

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

            out = Latent2MeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f

            outputs.append(out)

        return outputs
