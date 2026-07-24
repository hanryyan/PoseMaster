import os
import warnings
from typing import Callable, List, Optional, Union, Dict, Any
import PIL.Image
import trimesh
import rembg
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from diffusers.utils import BaseOutput

import posemaster
from posemaster.utils.config import ExperimentConfig, load_config

class MeshPipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[trimesh.Trimesh]` or `np.ndarray`)
            List of denoised trimesh meshes of length `batch_size` or a tuple of NumPy array with shape `((vertices, 3), (faces, 3)) of length `batch_size``.
    """

    meshes: Union[List[trimesh.Trimesh], np.ndarray]


class PoseMasterPipeline():
    """
    Pipeline for text-guided image to image generation using posemaster(https://github.com/wyysf-98/posemaster).

    Args:
        feature_extractor ([`CLIPFeatureExtractor`]):
            Feature extractor for image pre-processing before being encoded.
    """
    def __init__(
        self,
        cfg: ExperimentConfig,
        system,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.cfg = cfg
        self.system = system
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, config_path: Optional[Union[str, os.PathLike]], 
                        ckpt_path: Optional[Union[str, os.PathLike]], 
                        repo_id: Optional[Union[str, os.PathLike]] = None,
                        dtype=torch.dtype, 
                        device=torch.device,
                        **kwargs):
        r"""
        A simpler version that instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.
        The pipeline is set in evaluation mode (`model.eval()`) by default.
        """
        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if repo_id is not None:
            ckpt_path = hf_hub_download(repo_id=repo_id, filename="posemaster_dit.ckpt", repo_type="model")
            config_path = hf_hub_download(repo_id=repo_id, filename="shape-diff-hunyuan3d-flow-freeze-skeleton.yaml", repo_type="model")

        # 2. Load the model
        cfg = load_config(config_path)
        system = posemaster.find(cfg.system_type)(cfg.system)
        print(f"Restoring states from the checkpoint path at {ckpt_path} with config {cfg}")
        system = system.to(device).eval()

        system.shape_model = system.shape_model.to(dtype)
        system.condition = system.condition.to(dtype)
        system.denoiser_model = system.denoiser_model.to(dtype)

        return cls(
            device=device,
            dtype=dtype,
            cfg=cfg,
            system=system
        )

    def check_inputs(
        self,
        image,
    ):
        r"""
        Check if the inputs are valid. Raise an error if not.
        """
        if isinstance(image, str):
            assert os.path.isfile(image) or image.startswith("http"), "Input image must be a valid URL or a file path."
        elif not isinstance(image, (torch.Tensor, PIL.Image.Image)):
            raise ValueError("Input image must be a `torch.Tensor` or `PIL.Image.Image`.")
        
    def preprocess_image(
        self,
        image: PIL.Image.Image,
        force: bool = False,
        background_color: List[int] = [255, 255, 255],
        foreground_ratio: float = 1.0,
    ):
        r"""
        Crop and remote the background of the input image
        Args:
            image_pil (`List[PIL.Image.Image]`):
                List of `PIL.Image.Image` objects representing the input image.
            force (`bool`, *optional*, defaults to `False`):
                Whether to force remove the background even if the image has an alpha channel.
        Returns:
            `List[PIL.Image.Image]`: List of `PIL.Image.Image` objects representing the preprocessed image.
        """
        width, height, size = image.width, image.height, image.size
        do_remove = True
        if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
            # explain why current do not rm bg
            print("alhpa channl not empty, skip remove background, using alpha channel as mask")
            do_remove = False
        do_remove = do_remove or force
        if do_remove:
            image = rembg.remove(image)

        # calculate the min bbox of the image
        alpha = image.split()[-1]
        bboxs = alpha.getbbox()
        x1, y1, x2, y2 = bboxs
        dy, dx = y2 - y1, x2 - x1
        s = min(height * foreground_ratio / dy, width * foreground_ratio / dx)
        Ht, Wt = int(dy * s), int(dx * s)
        
        background = PIL.Image.new("RGBA", image.size, (*background_color, 255))
        image = PIL.Image.alpha_composite(background, image)
        image = image.crop(alpha.getbbox())
        alpha = alpha.crop(alpha.getbbox())

        # Calculate the new size after rescaling
        new_size = tuple(int(dim * foreground_ratio) for dim in size)
        # Resize the image while maintaining the aspect ratio
        resized_image = image.resize((Wt, Ht))
        resized_alpha = alpha.resize((Wt, Ht))
        # Create a new image with the original size and white background
        padded_image = PIL.Image.new("RGB", size, tuple(background_color))
        padded_alpha = PIL.Image.new("L", size, (0))
        paste_position = ((width - resized_image.width) // 2, (height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        padded_alpha.paste(resized_alpha, paste_position)

        # expand image to 1:1
        width, height = padded_image.size
        if width == height:
            padded_image = padded_image.resize((518, 518))
            padded_alpha = padded_alpha.resize((518, 518))
            padded_image.putalpha(padded_alpha)
            return padded_image.convert("RGB")
        new_size = (max(width, height), max(width, height))
        new_image = PIL.Image.new("RGB", new_size, tuple(background_color))
        new_alpha = PIL.Image.new("L", new_size, (0))
        paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
        new_image.paste(padded_image, paste_position)
        new_alpha.paste(padded_alpha, paste_position)
        new_image.putalpha(new_alpha)
        new_image = new_image.resize((518, 518))
        new_alpha = new_alpha.resize((518, 518))

        return new_image.convert("RGB")


    @torch.no_grad()
    def __call__(
        self,
        image: str,
        skeleton: str = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_type: Optional[str] = "trimesh",
        seed: Optional[int] = None,
        force_remove_background: bool = False,
        background_color: List[int] = [255, 255, 255],
        foreground_ratio: float = 0.95,
        octree_depth: int = 8,
        sampling_method='euler',
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch. The image will be encoded to its CLIP/DINO-v2 embedding 
                which the DiT will be conditioned on. 
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter as defined in [DDIM](https://arxiv.org/abs/2010.02502). `eta` is a parameter that
                controls the amount of noise added to the latent space. It is only used with the DDIM scheduler and
                will be ignored for other schedulers. `eta` should be between [0, 1].
            num_meshes_per_prompt (`int`, *optional*, defaults to 1):
                The number of meshes to generate per prompt.
            output_type (`str`, *optional*, defaults to `"trimesh"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image`, `latents` or `np.array of v and f`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            seed (`int`, *optional*, defaults to `None`):
                Seed for the random number generator. Setting a seed will ensure reproducibility.
            force_remove_background (`bool`, *optional*, defaults to `False`):
                Whether to force remove the background even if the image has an alpha channel.
            foreground_ratio (`float`, *optional*, defaults to 1.0):
                The ratio of the foreground in the image. The foreground is the part of the image that is not the
                background. The foreground is resized to the size of the background image while maintaining the aspect
                ratio. The background is filled with black color. The foreground ratio should be between [0, 1].
            mc_depth (`int`, *optional*, defaults to 8):
                The resolution of the Marching Cubes algorithm. The resolution is the number of cubes in the x, y, and z.
                8 means 2^8 = 256 cubes in each dimension. The higher the resolution, the more detailed the mesh will be.
            only_max_component (`bool`, *optional*, defaults to `False`):
                Whether to only keep the largest connected component of the mesh. This is useful when the mesh has
                multiple components and only the largest one is needed.
        Examples:

        Returns:
            [`~MeshPipelineOutput`] or `tuple`: [`~MeshPipelineOutput`] if `return_dict` is True, otherwise a `tuple`. 
            When returning a tuple, the first element is a list with the generated meshes.
        """
        # 0. Check inputs. Raise error if not correct
        self.check_inputs(
            image=image,
        )
        
        do_classifier_free_guidance = guidance_scale >= 0
        # 2. Preprocess image cond
        images_pil = PIL.Image.open(image)
        images_pil = self.preprocess_image(
            images_pil, 
            force=force_remove_background,
            background_color=background_color,
            foreground_ratio=foreground_ratio
            )
        img_cond = self.system.condition.encode_image([images_pil])

        if skeleton is not None:
            # 3. Prepare skeleton cond
            skeleton_cond = skeleton.to(self.device).unsqueeze(0).to(self.dtype)
            skeleton_cond = self.system.condition.encode_skeleton(skeleton_cond)

            # 4. Prepare the joint cond
            cond = torch.cat([img_cond, skeleton_cond], dim=1)
            if do_classifier_free_guidance:
                uncond = torch.cat([torch.zeros_like(img_cond).to(cond), skeleton_cond], dim=1)
                cond = torch.cat([uncond, cond], dim=0)
        else:
            if do_classifier_free_guidance:
                uncond = torch.zeros_like(img_cond).to(img_cond)
                cond = torch.cat([uncond, img_cond], dim=0)

        # 5. Prepare latents
        if seed != None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None
        batch_size = img_cond.shape[0]
        latents = self.system.prepare_latents(None, batch_size, self.dtype, self.device, generator)
        
        # 6. Denoising loop
        def denoise_with_cfg(inputs, ts, contexts):
            noise_pred = self.system.denoiser_model(inputs, ts, contexts)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                return torch.cat([noise_pred, noise_pred], dim=0)
            return noise_pred
        
        if do_classifier_free_guidance:
            latents = torch.cat([latents, latents], dim=0)
        
        sample_fn = self.system.sampler.sample_ode(num_steps=num_inference_steps, sampling_method=sampling_method)
        latents = sample_fn(latents, denoise_with_cfg, contexts=cond)[-1]
        if do_classifier_free_guidance:
            latents = latents.chunk(2, dim=0)[0]

        latents = 1. / self.cfg.system.z_scale_factor * latents
        print("converting latents dtype to ", next(self.system.shape_model.vae_model.parameters()).dtype)
        latents = latents.to(next(self.system.shape_model.vae_model.parameters()).dtype)
        latents = self.system.shape_model.vae_model.decode(latents)

        # 7. Post-processing
        
        if output_type == "trimesh":
            mesh_v_f = self.system.shape_model.extract_geometry(latents=latents, octree_depth=octree_depth)
            cur_mesh = trimesh.Trimesh(vertices=mesh_v_f[0][0], faces=mesh_v_f[0][1][:, ::-1])
            return {'image': images_pil, 'mesh': cur_mesh}
        else:
            return {'image': images_pil, 'latents': latents}