import argparse
import os
from typing import Any, cast

import torch
from diffusers import StableDiffusion3Pipeline  # type: ignore
from tqdm import trange
from typing import Union, List, Optional, Dict, Callable
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift, retrieve_timesteps
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.utils.import_utils import is_torch_xla_available
import numpy as np
import PIL.Image
PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # pyright: ignore[reportMissingImports]
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
from ..models.components.model import ProbHead
from ..data_gen.utils import ActivationGrabber_SD3
from ..data_gen.evaluator.ImgReward_evaluator import load as load_imgreward

# Load ImageReward BLIP text encoder at module level
_imgreward_model = load_imgreward(
    name="/mnt/sharedata/ssd_large/users/guohl/pretrained_models/THUDM/ImageReward/ImageReward.pt",
    device='cpu',
    med_config="/mnt/sharedata/ssd_large/users/guohl/pretrained_models/THUDM/ImageReward/med_config.json"
)
BLIP_TEXT_ENCODER = _imgreward_model.blip
BLIP_TOKENIZER = _imgreward_model.blip.tokenizer

class ProbeSelect_SD3(StableDiffusion3Pipeline):  
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], 
            probe_path: str="workdir/SD3L_IR_ema.ckpt", **kwargs):
        model = super(ProbeSelect_SD3, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        probe_head = ProbHead(
            latent_channels=48, latent_height=48, num_layers=1,
            latent_width=48, hidden_dims=[128, 256], hidden_dims_mlp=[512], use_batch_norm=True, 
            activation="relu", dropout_rate=0.5,
            resnet_eps=1e-5, resnet_act_fn="silu",
            resnet_groups=16, attention_head_dim=5,
            output_scale_factor=1.0, downsample_type="conv", 
            temb_channels=512, resnet_time_scale_shift="default", 
            num_channels_time=64, text_dim=768, has_text_embed=True
        )
        probe_head.load_state_dict(torch.load(probe_path, weights_only=False))
        probe_head.eval()
        torch_dtype = kwargs.get("torch_dtype", torch.float32)
        probe_head = probe_head.to(torch_dtype)
        model.probe_head = probe_head
        return model

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,  # pyright: ignore[reportArgumentType]
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        enable_probe_select: bool = False,
        num_select: int = 1,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # Initialize activation grabber for probe selection
        grabber = None
        if enable_probe_select:
            grabber = ActivationGrabber_SD3(self.transformer)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,  # pyright: ignore[reportArgumentType]
            prompt_3=prompt_3,  # pyright: ignore[reportArgumentType]
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,  # pyright: ignore[reportArgumentType]
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,  # pyright: ignore[reportArgumentType]
        )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)    # pyright: ignore[reportArgumentType, reportCallIssue]
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)  # pyright: ignore[reportArgumentType, reportCallIssue]

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents,  # pyright: ignore[reportAssignmentType, reportOperatorIssue]
            height, width, prompt_embeds.dtype, device, generator, latents)

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(  # pyright: ignore[reportAssignmentType]
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        # 7. Denoising loop
        try:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        should_skip_layers = (
                            True
                            if i > num_inference_steps * skip_layer_guidance_start
                            and i < num_inference_steps * skip_layer_guidance_stop
                            else False
                        )
                        if skip_guidance_layers is not None and should_skip_layers:
                            timestep = t.expand(latents.shape[0])
                            latent_model_input = latents
                            noise_pred_skip_layers = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=original_prompt_embeds,
                                pooled_projections=original_pooled_prompt_embeds,
                                joint_attention_kwargs=self.joint_attention_kwargs,
                                return_dict=False,
                                skip_layers=skip_guidance_layers,
                            )[0]
                            noise_pred = (
                                noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                            )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()
                    #########################################################
                    # Probe-based early selection at ~20% of denoising steps
                    if enable_probe_select and (i + 1) / num_inference_steps >= 0.2:
                        enable_probe_select = False  # Only run once
                        
                        # Get PCA features from grabber
                        pca_features = grabber.get_pca_feature().to(device, dtype=latents.dtype)  # [N, 48, 48, 48]
                        
                        # Per-prompt top-k selection
                        num_prompts = batch_size  # Number of distinct prompts
                        imgs_per_prompt = num_images_per_prompt
                        
                        # Encode prompt with BLIP to get text embedding
                        if isinstance(prompt, str):
                            prompts_list = [prompt] * (num_prompts * imgs_per_prompt)
                        else:
                            # Expand prompts to match num_images_per_prompt
                            prompts_list = []
                            for p in prompt:
                                prompts_list.extend([p] * imgs_per_prompt)
                        
                        # Get text embeddings via module-level BLIP
                        text_input = BLIP_TOKENIZER(
                            prompts_list, padding='max_length', truncation=True, 
                            max_length=35, return_tensors="pt"
                        )
                        
                        text_emb = BLIP_TEXT_ENCODER.text_encoder(
                            text_input.input_ids,
                            attention_mask=text_input.attention_mask,
                            encoder_hidden_states=None,
                            encoder_attention_mask=None,
                            return_dict=True,
                            mode='textonly',
                        ).last_hidden_state[:,0,:].to(latents)  # [N, 768]
                        
                        # Predict scores with text_emb
                        self.probe_head.to(device)
                        t_normalized = torch.full((pca_features.shape[0],), (i + 1) / num_inference_steps, device=device, dtype=latents.dtype)
                        scores, _ = self.probe_head(pca_features, t_normalized, text_emb=text_emb)
                        
                        selected_indices = []
                        
                        for p in range(num_prompts):
                            start_idx = p * imgs_per_prompt
                            end_idx = start_idx + imgs_per_prompt
                            prompt_scores = scores[start_idx:end_idx]
                            _, top_k_local = torch.topk(prompt_scores, min(num_select, imgs_per_prompt))
                            selected_indices.extend((start_idx + top_k_local).tolist())
                        
                        selected_indices = torch.tensor(selected_indices, device=device, dtype=torch.long)
                        
                        # Slice latents
                        latents = latents[selected_indices]
                        
                        # Slice embeddings (handle CFG: first half negative, second half positive)
                        if self.do_classifier_free_guidance:
                            total_imgs = batch_size * imgs_per_prompt
                            neg_embeds = prompt_embeds[:total_imgs]
                            pos_embeds = prompt_embeds[total_imgs:]
                            prompt_embeds = torch.cat([neg_embeds[selected_indices], pos_embeds[selected_indices]], dim=0)
                            
                            neg_pooled = pooled_prompt_embeds[:total_imgs]
                            pos_pooled = pooled_prompt_embeds[total_imgs:]
                            pooled_prompt_embeds = torch.cat([neg_pooled[selected_indices], pos_pooled[selected_indices]], dim=0)
                        else:
                            prompt_embeds = prompt_embeds[selected_indices]
                            pooled_prompt_embeds = pooled_prompt_embeds[selected_indices]
                    #########################################################
        finally:
            # Cleanup grabber if it wasn't closed during selection
            if grabber is not None:
                grabber.close()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
