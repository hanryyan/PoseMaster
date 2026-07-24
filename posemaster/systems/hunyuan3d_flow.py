from dataclasses import dataclass, field

import numpy as np
import json
import copy
import pickle
import torch
import torch.nn.functional as F
from skimage import measure
from einops import repeat
from tqdm import tqdm
from PIL import Image


import posemaster
from posemaster.systems.base import BaseSystem
from posemaster.utils.misc import get_rank, instantiate_from_config
from posemaster.utils.typing import *
from diffusers import DDIMScheduler
from posemaster.utils.structure import get_render
from posemaster.data.character_dataset_direction import load_skeleton, process_img


@posemaster.register("hunyuan3d-flow-system")
class Hunyuan3DFlowSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        compute_metric: bool = True
        visualize_mesh: bool = True
        extract_mesh_func: str = "mc"
        vis_training: bool = False
        remove_bg: bool = False
        background_color: Tuple[int, int, int] = field(
            default_factory=lambda: (255, 255, 255)
        )
        octree_depth: int = 7
        resize_img: bool = False
        num_bone_points: int = 256

        # diffusion config
        z_scale_factor: float = 1.0
        guidance_scale_img_joints: float = 7.5
        guidance_scale_img: float = 7.5
        guidance_scale_joints: float = 7.5
        cfg_type: str = "double_seperate_discard"

        num_inference_steps: int = 50
        eta: float = 0.0
        snr_gamma: float = 5.0
        # flow
        weighting_scheme: str = "logit_normal"
        logit_mean: float = 0
        logit_std: float = 1.0
        mode_scale: float = 1.29
        precondition_outputs: bool = True
        precondition_t: int = 1000

        # shape vae model
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)

        # condition model
        condition_model_type: str = None
        condition_model: dict = field(default_factory=dict)

        # diffusion model
        denoiser_model_type: str = None
        denoiser_model: dict = field(default_factory=dict)

        # noise scheduler
        noise_scheduler_type: str = None
        noise_scheduler: dict = field(default_factory=dict)

        # denoise scheduler
        denoise_scheduler_type: str = None
        denoise_scheduler: dict = field(default_factory=dict)

        scheduler_cfg: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = posemaster.find(self.cfg.shape_model_type)(self.cfg.shape_model)
        self.shape_model.eval()
        self.shape_model.requires_grad_(False)

        self.condition = posemaster.find(self.cfg.condition_model_type)(self.cfg.condition_model)
        
        self.denoiser_model = posemaster.find(self.cfg.denoiser_model_type)(self.cfg.denoiser_model)

        # self.noise_scheduler = posemaster.find(self.cfg.noise_scheduler_type)(**self.cfg.noise_scheduler)
        # self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        self.denoise_scheduler = posemaster.find(self.cfg.denoise_scheduler_type)(**self.cfg.denoise_scheduler)

        self.scheduler_cfg = self.cfg.scheduler_cfg
        self.sampler = None
        if 'transport' in self.cfg.scheduler_cfg:
            self.transport = instantiate_from_config(self.cfg.scheduler_cfg.transport)
            self.sampler = instantiate_from_config(self.cfg.scheduler_cfg.sampler, transport=self.transport)
            self.sample_fn = self.sampler.sample_ode(**self.cfg.scheduler_cfg.sampler.ode_params)

    def forward(self, batch: Dict[str, Any], skip_noise=False) -> Dict[str, Any]:
        # 1. encode shape latents
        latents, _, _ = self.shape_model.encode(
            batch["surface"][..., :3 + self.cfg.shape_model.point_feats], 
            sample_posterior=True
        )
        latents = latents * self.cfg.z_scale_factor

        # 2. gain condition. assert not (text_cond and image_cond), "Only one of text or image condition must be provided."
        if "image" in batch and batch['image'].dim() == 5:
            if self.training:
                bs, n_images = batch['image'].shape[:2]
                batch['image'] = batch['image'].view(bs*n_images, *batch['image'].shape[-3:])
            else:
                batch['image'] = batch['image'][:, 0, ...]
                n_images = 1
                bs = batch['image'].shape[0]
            cond_latents = self.condition(batch)
            latents = latents.unsqueeze(1).repeat(1, n_images, 1, 1)
            latents = latents.view(bs*n_images, *latents.shape[-2:])
        else:
            n_images = 1
            cond_latents = self.condition(batch).to(self.dtype)

        loss = self.transport.training_losses(self.denoiser_model, latents, cond_latents)["loss"].mean()

        return {
            "loss_diffusion": loss,
            "latents": latents,
            }

    def training_step(self, batch, batch_idx):
        self.train()
        out = self(batch)

        loss = 0.
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            if name.startswith("lambda_"):
                self.log(f"train_params/{name}", self.C(value))

        if self.cfg.vis_training and get_rank() == 0:
            latents = self.shape_model.decode(out["latents"])
            mesh_v_f, has_surface = self.shape_model.extract_geometry(latents=latents, extract_mesh_func=self.cfg.extract_mesh_func, octree_depth=self.cfg.octree_depth)
            normals = []
            uids = []
            gt_rgbs = []
            for idx in range(latents.shape[0]):
                self.save_mesh(
                    f"it{self.true_global_step}_train/{batch['uid'][idx]}.obj",
                    mesh_v_f[idx][0], mesh_v_f[idx][1]
                )
                with torch.cuda.amp.autocast(False):
                    normal, depth = get_render(mesh_v_f[idx][0],  mesh_v_f[idx][1], self.device, img_size=2048)
                    normals.append(normal)
                    uids.append(batch['uid'][idx])
                gt_rgbs.append(batch['image'][idx])
            normals = torch.stack(normals)
            gt_rgbs = torch.stack(gt_rgbs)
            normals = torch.cat([gt_rgbs * 2 - 1, normals], dim=2)
            self.save_rgb_image(f"it{self.true_global_step}_train/rgb.png", normals.view(-1, normals.shape[-2], normals.shape[-1]), data_format="HWC", data_range=(-1,1))
            self.save_txt(f"it{self.true_global_step}_train/uids.txt", "\n".join(uids))
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        out = self(batch)

        # vae recon mesh
        latents = self.shape_model.vae_model.decode(out["latents"])
        mesh_v_f = self.shape_model.extract_geometry(latents=latents, octree_depth=self.cfg.octree_depth)
        # print(f"mesh_v_f: {mesh_v_f}")
        self.save_mesh(
                f"it{self.true_global_step}/{batch['uid'][0]}_recon.glb",
                mesh_v_f[0][0], mesh_v_f[0][1][:, ::-1]
        )

        # generated mesh
        pred_latents, bone_points, imgs = self.sample(batch)
        mesh_v_f = self.shape_model.extract_geometry(latents=pred_latents, octree_depth=self.cfg.octree_depth)
        # print(f"mesh_v_f: {mesh_v_f}")
        self.save_mesh(
                f"it{self.true_global_step}/{batch['uid'][0]}_gen.glb",
                mesh_v_f[0][0], mesh_v_f[0][1][:, ::-1]
        )

        self.save_ply_points(f"it{self.true_global_step}/{batch['uid'][0]}_bone_points.ply", bone_points[0].to(torch.float32).cpu().numpy())
        self.save_ply_points(f"it{self.true_global_step}/{batch['uid'][0]}_surface.ply", batch['surface'][0].to(torch.float32).cpu().numpy())
        self.save_rgb_image(f"it{self.true_global_step}/{batch['uid'][0]}.png", np.array(imgs[0]), data_format="HWC", data_range=(0, 1))

        return {"val/loss": out["loss_diffusion"]}
        
    def prepare_latents(self, timestep, batch_size, dtype, device, generator, latents=None):
        from diffusers.utils.torch_utils import randn_tensor
        shape = (batch_size, *self.shape_model.vae_model.latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.sampler, 'init_noise_sigma', 1.0)
        return latents
    
    @torch.no_grad()
    def sample(self,
               sample_inputs: Dict[str, Union[torch.FloatTensor, List[str]]],
               steps: Optional[int] = None,
               guidance_scale: Optional[float] = 4.5,
               eta: float = 0.0,
               seed: Optional[int] = None,
               skeleton_trans: Optional[bool] = True,
               **kwargs):

        if steps is None:
            steps = self.cfg.num_inference_steps
        do_classifier_free_guidance = True
        # dtype = self.dtype
        # conditional encode
        if "image" in sample_inputs and "bone_points" in sample_inputs:
            # print(f"sample_inputs['image']: {sample_inputs['image'].shape}")
            imgs = [img.to(torch.float32).cpu().numpy() for img in sample_inputs['image']]
            img_cond = self.condition.encode_image(sample_inputs['image'])
            bone_points = sample_inputs['bone_points']
            skeleton_cond = self.condition.encode_skeleton(bone_points)
            cond = torch.cat([img_cond, skeleton_cond], dim=1)
            if do_classifier_free_guidance:
                un_img_cond = torch.zeros_like(img_cond).to(cond)
                uncond = torch.cat([un_img_cond, skeleton_cond], dim=1)
        elif "image" in sample_inputs and "skeleton" in sample_inputs:
            imgs = []
            for img in sample_inputs["image"]:
                imgs.append(Image.open(img))
            if self.cfg.remove_bg:
                from posemaster.utils.bg_processing import RMBG
                rmbg = RMBG()
                imgs = [rmbg.run("REMBG", img, 518, 0.95, "Remove", background_color=self.cfg.background_color) for img in imgs]
            if self.cfg.resize_img:
                imgs = [process_img(img, 518, self.cfg.background_color)[0] for img in imgs]
            img_cond = self.condition.encode_image(imgs)
            bone_points = []
            
            for idx, skeleton in enumerate(sample_inputs["skeleton"]):
                data = load_skeleton(skeleton, num_bone_points=self.cfg.num_bone_points, is_transform=skeleton_trans)
                # print(f"idx: {idx}, bone_points shape: {data['bone_points'].shape}")
                bone_points.append(data['bone_points'])
            bone_points = torch.from_numpy(np.stack(bone_points, axis=0)).to(img_cond)
            skeleton_cond = self.condition.encode_skeleton(bone_points)
            # print(f"img_cond shape: {img_cond.shape} skeleton_cond shape: {skeleton_cond.shape}")
            cond = torch.cat([img_cond, skeleton_cond], dim=1)
            if do_classifier_free_guidance:
                un_img_cond = torch.zeros_like(img_cond).to(cond)
                uncond = torch.cat([un_img_cond, skeleton_cond], dim=1)
        elif "image" in sample_inputs:
            imgs = []
            for img in sample_inputs["image"]:
                imgs.append(Image.open(img))
            if self.cfg.remove_bg:
                from posemaster.utils.bg_processing import RMBG
                rmbg = RMBG()
                imgs = [rmbg.run("REMBG", img, 518, 0.95, "Remove", background_color=self.cfg.background_color) for img in imgs]
            if self.cfg.resize_img:
                imgs = [process_img(img, 518, self.cfg.background_color)[0] for img in imgs]
            img_cond = self.condition.encode_image(imgs)
            cond = img_cond
            bone_points = None
            if do_classifier_free_guidance:
                uncond = torch.zeros_like(cond).to(cond)
        else:
            raise NotImplementedError

        latents = None
        # sample_fn = self.scheduler.sample_ode(num_steps=num_inference_steps, sampling_method=sampling_method)

        if seed != None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None
        batch_size = img_cond.shape[0]

        cond = torch.cat([uncond, cond], dim=0)
        dtype = img_cond.dtype
        device = img_cond.device
        latents = self.prepare_latents(None, batch_size, dtype, device, generator)

        def denoise_with_cfg(inputs, ts, contexts):
            noise_pred = self.denoiser_model(inputs, ts, contexts)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                return torch.cat([noise_pred, noise_pred], dim=0)
            return noise_pred
        if do_classifier_free_guidance:
            latents = torch.cat([latents, latents], dim=0)
        
        latents = self.sample_fn(latents, denoise_with_cfg, contexts=cond)[-1]
        if do_classifier_free_guidance:
            latents = latents.chunk(2, dim=0)[0]

        latents = 1. / self.cfg.z_scale_factor * latents
        print("converting latents dtype to ", next(self.shape_model.vae_model.parameters()).dtype)
        latents = latents.to(next(self.shape_model.vae_model.parameters()).dtype)
        latents = self.shape_model.vae_model.decode(latents)
        return latents, bone_points, imgs

    def on_validation_epoch_end(self):
        pass
