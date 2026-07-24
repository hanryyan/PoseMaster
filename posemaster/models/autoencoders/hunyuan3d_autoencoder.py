from dataclasses import dataclass, field
import torch
import torch.nn as nn

import posemaster
from posemaster.utils.base import BaseModule
from posemaster.utils.typing import *
from posemaster.models.autoencoders.hunyuan3d.shape_vae import ShapeVAE

@posemaster.register("hunyuan3d-autoencoder")
class Hunyuan3DAutoencoder(BaseModule):
    r"""
    A VAE model for encoding shapes into latents and decoding latent representations into shapes.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "" # the pretrained model name or path for condition model
        # pretrained_miche_config_path: str = 'miche/shapevae-256.yaml'
        num_latents: int = 4096
        train_num_latents_range: List[int] = field(
            default_factory=lambda: [4096]
        )
        train_num_latents_probs: List[float] = field(
            default_factory=lambda: [1.0]
        )
        downsample_ratio: int = 4
        embed_dim: int = 64
        point_feats: int = 3 
        sharpedge_label: bool = True
        pc_size: int = 81920
        pc_sharpedge_size: int = 0
        num_freqs: int = 8
        include_pi: bool = False
        heads: int = 16
        width: int = 1024
        num_encoder_layers: int = 8
        num_decoder_layers: int = 16
        use_ln_post: bool = True
        init_scale: float = 0.25
        qkv_bias: bool = False
        flash: bool = True
        qk_norm: bool = True
        use_checkpoint: bool = False
        sample_posterior: bool = True # whether to sample from posterior distribution during training

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.vae_model = ShapeVAE(
            device=None, 
            dtype=None,
            num_latents=self.cfg.num_latents,
            train_num_latents_range=self.cfg.train_num_latents_range,
            train_num_latents_probs=self.cfg.train_num_latents_probs,
            downsample_ratio=self.cfg.downsample_ratio,
            embed_dim=self.cfg.embed_dim,
            point_feats=self.cfg.point_feats,
            sharpedge_label=self.cfg.sharpedge_label,
            pc_size=self.cfg.pc_size,
            pc_sharpedge_size=self.cfg.pc_sharpedge_size,
            num_freqs=self.cfg.num_freqs,
            include_pi=self.cfg.include_pi,
            heads=self.cfg.heads,
            width=self.cfg.width,
            num_encoder_layers=self.cfg.num_encoder_layers,
            num_decoder_layers=self.cfg.num_decoder_layers,
            use_ln_post=self.cfg.use_ln_post,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            flash=self.cfg.flash,
            qk_norm=self.cfg.qk_norm,
            use_checkpoint=self.cfg.use_checkpoint,
        )
        self.latent_shape = (self.cfg.num_latents, self.cfg.embed_dim)
        if self.cfg.pretrained_model_name_or_path:
            print(f"####### Loading pretrained VAE model from {self.cfg.pretrained_model_name_or_path} #######")
            ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            self.load_state_dict(ckpt, strict=True)            
    
    def embed_pc(self, pc_normal):
        # encode point cloud to embeddings
        latents, center_pos, posterior = self.vae_model.encode(pc_normal[...,:3], pc_normal[...,3:], sample_posterior=self.cfg.sample_posterior)

        latents = self.vae_model.decode(latents)
        return latents

    def encode(self,
               surface: torch.FloatTensor,
               sample_posterior: bool = True,
               sharp_surface: torch.FloatTensor = None):
        if surface.shape[-1] < 7:
            surface = torch.cat([surface, torch.zeros_like(surface[...,:7-surface.shape[-1]])], dim=-1)
        latents, center_pos, posterior = self.vae_model.encode(surface[...,:3], surface[...,3:], sample_posterior=self.cfg.sample_posterior)
        return latents, center_pos, posterior

    @torch.no_grad()
    def extract_geometry(self,
                         latents: torch.FloatTensor,
                         extract_mesh_func: str = "mc",
                         bounds: Union[Tuple[float], List[float], float] = (-1.01, -1.01, -1.01, 1.01, 1.01, 1.01),
                         octree_depth: int = 9,
                         num_chunks: int = 100000,
                         ): 
        outputs = self.vae_model.latent2mesh(latents=latents,
                                   bounds=bounds,
                                    octree_depth=octree_depth,
                                    num_chunks=num_chunks,
                                    mc_level=0.0,
                                    octree_resolution=None,
                                    mc_mode=extract_mesh_func)

        return outputs

    def forward(self,
        surface: torch.FloatTensor,
        queries: torch.FloatTensor,
        sample_posterior: bool = True,
        sharp_surface: torch.FloatTensor = None
    ):
        if surface is None:
            if surface.shape[-1] < 7:
                surface = torch.cat([surface, torch.zeros_like(surface[...,:7-surface.shape[-1]])], dim=-1)
            latent = self.embed_pc(surface)
            
        # concat zeros for pc_embeds to 7 channels
        
        assert not torch.any(torch.isnan(latent)), 'NAN values in pc embedings'
        
        return latent
