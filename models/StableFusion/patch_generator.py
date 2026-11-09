"""Text-prompt-conditioned Stable Diffusion patch initialisation."""

from __future__ import annotations

import json
import os
import random
from typing import List

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights


DEFAULT_STABLE_DIFFUSION_CHECKPOINT = "stable-diffusion-v1-5/stable-diffusion-v1-5"


class PatchGenerator:
    PROMPT_TEMPLATES = [
        "A {target_class}-shaped {patch_type} on a {region_name}{color_desc}",
        "A pattern resembling a {target_class} on a {region_name}{color_desc}",
        "A silhouette of a {target_class} blended into the {region_name}{color_desc}",
        "A distorted {target_class} shape over the {region_name}{color_desc}",
        "An artistic shape of {target_class} subtly appearing on the {region_name}{color_desc}",
    ]

    def __init__(self, model_path=DEFAULT_STABLE_DIFFUSION_CHECKPOINT,
                 json_path="data/ADE20K_patches.json", device="cuda",
                 local_files_only=False, disable_safety_checker=False, seed=42,
                 num_inference_steps=20, guidance_scale=7.5):
        self.device = torch.device(device)
        self.seed = int(seed)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.idx_to_label = dict(enumerate(ResNet50_Weights.IMAGENET1K_V1.meta["categories"]))
        from diffusers import StableDiffusionPipeline

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=bool(local_files_only)
        ).to(self.device)
        if disable_safety_checker:
            self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Patch prompt definition file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as handle:
            region_patch_map = json.load(handle)
        self.region2patch = {item["name"]: item["possible_patches"] for item in region_patch_map}
        self._rng = random.Random(self.seed)

    def get_label_name(self, idx):
        if isinstance(idx, torch.Tensor):
            if idx.numel() != 1:
                raise ValueError("A target class tensor must contain exactly one value per prompt")
            idx = idx.item()
        return self.idx_to_label.get(int(idx), "unknown object")

    def _run_pipe(self, prompts, generator):
        autocast_enabled = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
            return self.pipe(
                prompt=prompts,
                height=512,
                width=512,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).images

    def generate_patch(self, prompt, patch_size=(64, 64), num_inference_steps=None, guidance_scale=None):
        if num_inference_steps is not None or guidance_scale is not None:
            old_steps, old_guidance = self.num_inference_steps, self.guidance_scale
            self.num_inference_steps = num_inference_steps or old_steps
            self.guidance_scale = guidance_scale if guidance_scale is not None else old_guidance
        else:
            old_steps = old_guidance = None
        try:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            image = self._run_pipe(prompt, generator)[0]
            return image.resize(patch_size, Image.Resampling.LANCZOS)
        finally:
            if old_steps is not None:
                self.num_inference_steps, self.guidance_scale = old_steps, old_guidance

    def generate_patch_by_region(self, region_name, color_names=None, patch_size=(24, 24), k=1,
                                 target_class=None, **kwargs):
        targets = None if target_class is None else [target_class]
        patches, _prompts = self.generate_patch_by_region_batch(
            [region_name], [color_names or []], patch_size=patch_size,
            per_region_k=k, target_classes=targets, **kwargs
        )
        return patches[0]

    def generate_patch_by_region_batch(self, region_names, color_names_list=None, patch_size=(24, 24),
                                       num_inference_steps=None, guidance_scale=None, per_region_k=1,
                                       target_classes=None):
        """Condition diffusion through text prompts containing palette colour names.

        This is prompt conditioning only; no claim is made that diffusion directly
        receives or optimises a Lab tensor. Lab colours are used later by the loss.
        """
        batch_size = len(region_names)
        if color_names_list is not None and len(color_names_list) != batch_size:
            raise ValueError("color_names_list must match region_names")
        if target_classes is not None and len(target_classes) != batch_size:
            raise ValueError("target_classes must match region_names")
        if num_inference_steps is not None:
            self.num_inference_steps = int(num_inference_steps)
        if guidance_scale is not None:
            self.guidance_scale = float(guidance_scale)
        prompts = []
        for index, region_name in enumerate(region_names):
            colours = color_names_list[index] if color_names_list is not None else []
            colour_desc = " in " + " and ".join(colours) if colours else ""
            target = target_classes[index] if target_classes is not None else None
            if isinstance(target, (int, torch.Tensor)):
                target = self.get_label_name(target)
            for _ in range(per_region_k):
                patch_type = self._rng.choice(self.region2patch.get(region_name, ["texture"]))
                if target is not None:
                    prompt = self._rng.choice(self.PROMPT_TEMPLATES).format(
                        target_class=target, patch_type=patch_type,
                        region_name=region_name, color_desc=colour_desc,
                    )
                else:
                    prompt = f"A realistic photo of {patch_type} on a {region_name}{colour_desc}"
                prompts.append(prompt)
        try:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            images = self._run_pipe(prompts, generator)
        except TypeError as exc:
            raise RuntimeError("Stable Diffusion pipeline does not accept the configured generator/device") from exc
        resized = [image.resize(patch_size, Image.Resampling.LANCZOS) for image in images]
        return (
            [resized[i * per_region_k:(i + 1) * per_region_k] for i in range(batch_size)],
            [prompts[i * per_region_k:(i + 1) * per_region_k] for i in range(batch_size)],
        )

    def visualize_patch_batch_with_prompts(self, patch_batch: List[List[Image.Image]],
                                           prompts_batch: List[List[str]], save_dir=None,
                                           timestamp_str="default_time", show=False):
        for index, (patches, prompts) in enumerate(zip(patch_batch, prompts_batch)):
            fig, axes = plt.subplots(1, len(patches), figsize=(3 * len(patches), 3.2), squeeze=False)
            for axis, patch, prompt in zip(axes[0], patches, prompts):
                axis.imshow(patch)
                axis.axis("off")
                axis.set_title(prompt, fontsize=8, wrap=True)
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f"patches_with_prompts_{timestamp_str}_{index}.png"),
                            dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
