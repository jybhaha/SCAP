"""Unified end-to-end SCAP attack pipeline.

``MyPatchAttack`` returns an adversarial tensor. ``MyPatchAttack_Eva`` additionally
returns per-step predictions. ``attack.attacker`` re-exports these classes.
"""

from __future__ import annotations

import copy
import hashlib
import os
import random
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image

from configs.experiment_configs import DEFAULT_CONFIG
from models.Mask2Former.mask2former import SemanticSegmentor
from models.imagenet_model import ensure_imagenet_normalized
from models.patch_optimizer import PatchOptimizer
from models.StableFusion.patch_generator import PatchGenerator
from utils.cam_generator import CAMGenerator
from utils.color_extractor import ColorExtractor
from utils.color_space import rgb_to_lab_numpy
from utils.patch_blender import PatchBlender
from utils.patch_selector import PatchSelectorBatch, UnresolvedPlacementError
from utils.profiling import StageProfiler, save_diagnostics
from utils.result_saver import ResultManager


class MyPatchAttack:
    _evaluation = False

    def __init__(self, model, device="cuda", config=None, debug=False,
                 save_imgs=False, save_attack_process=False, components=None):
        self.device = torch.device(device)
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config.update(copy.deepcopy(config or {}))
        if isinstance(self.config.get("patch_size"), list):
            self.config["patch_size"] = tuple(self.config["patch_size"])
        classifier_size = (299, 299) if self.config.get("cam_model_name") == "inception_v3" else (224, 224)
        self.model = ensure_imagenet_normalized(model, input_size=classifier_size).to(self.device).eval()
        self.debug = bool(debug)
        self.save_imgs = bool(save_imgs)
        self.save_attack_process = bool(save_attack_process)
        self.target = bool(self.config["targeted_attack"])
        self.rm = ResultManager.get_instance() if (self.debug or self.save_imgs or self.save_attack_process) else None
        self._timestamp = datetime.now().strftime("%Y%m%dT%H%M%S") if self.save_imgs else None
        self.last_diagnostics = {}
        self._init_modules(components or {})
        self._log(f"Attack mode: {'targeted' if self.target else 'untargeted'}")

    def _init_modules(self, components):
        scene_aware = self.config.get("scene_aware_placement", True)
        diffusion_init = self.config.get("diffusion_init", True)
        self.segmentor = components.get("segmentor")
        if self.segmentor is None and scene_aware:
            self.segmentor = SemanticSegmentor(
                device=self.device,
                checkpoint=self.config.get("mask2former_checkpoint", "facebook/mask2former-swin-base-ade-panoptic"),
                local_files_only=self.config.get("mask2former_local_files_only", False),
            )
        self.patch_gen = components.get("patch_gen")
        if self.patch_gen is None and diffusion_init:
            self.patch_gen = PatchGenerator(
                model_path=self.config.get("stable_diffusion_checkpoint", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
                device=self.device,
                local_files_only=self.config.get("stable_diffusion_local_files_only", False),
                disable_safety_checker=self.config.get("disable_safety_checker", False),
                seed=self.config.get("seed", 42),
                num_inference_steps=self.config.get("diffusion_steps", 20),
                guidance_scale=self.config.get("diffusion_guidance_scale", 7.5),
            )
        self.color_extractor = components.get("color_extractor") or ColorExtractor(
            n_colors=self.config.get("n_colors", 3), random_state=self.config.get("seed", 42)
        )
        self.patch_blender = components.get("patch_blender") or PatchBlender(
            blend_width=self.config["blend_width"]
        )
        self.optimizer = components.get("optimizer") or PatchOptimizer(
            device=self.device, steps=self.config["steps"], eps=self.config["eps"],
            alpha=self.config["alpha"], decay=self.config.get("decay", 1.0), model=self.model,
            lambda_adv=self.config.get("lambda_adv", 1.0),
            lambda_smooth=self.config.get("lambda_smooth", 0.1),
            lambda_color=self.config.get("lambda_color", 0.05),
            use_momentum=self.config.get("use_momentum", True),
        )
        self.cam_gen = components.get("cam_gen")
        if self.cam_gen is None and scene_aware:
            self.cam_gen = CAMGenerator(
                model_name=self.config["cam_model_name"], device=self.device, model=self.model,
                input_normalized=self.config.get("input_normalized", False),
                input_size=self.config.get("cam_input_size"),
            )

    def _log(self, message):
        if self.debug and self.rm:
            self.rm.log(f"[Debug] {message}")

    def _save_img(self, vis_func, *args, **kwargs):
        if self.save_imgs:
            vis_func(*args, save_dir=self.rm.img_dir, timestamp_str=self._timestamp, show=False, **kwargs)

    def _save_attack_process(self, step_name, images_dict, sample_idx=0):
        if not self.save_attack_process:
            return
        sample_dir = os.path.join(self.rm.img_dir, "attack_process", f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        for name, image in images_dict.items():
            if image is None:
                continue
            if isinstance(image, Image.Image):
                tensor = TF.to_tensor(image)
            else:
                tensor = torch.as_tensor(image).detach().cpu()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
            tensor = tensor.float()
            if tensor.numel() and tensor.max() > 1:
                tensor = tensor / 255.0
            vutils.save_image(tensor.clamp(0, 1), os.path.join(sample_dir, f"{step_name}_{name}.png"))

    def _to_pil(self, x):
        return [TF.to_pil_image(image.detach().cpu()) for image in x]

    def _extract_colors(self, pil_imgs, coords_list=None):
        palettes = []
        for index, image in enumerate(pil_imgs):
            pixel_mask = None
            if coords_list is not None:
                pixel_mask = np.zeros((image.height, image.width), dtype=bool)
                for x1, y1, x2, y2 in coords_list[index]:
                    pixel_mask[y1:y2, x1:x2] = True
            palettes.append(self.color_extractor.get_dominant_palette(image, pixel_mask=pixel_mask))
        self._palette_rgb = [palette["rgb"] for palette in palettes]
        self._palette_lab = [palette["lab"] for palette in palettes]
        names = [palette["names"] for palette in palettes]
        if not self.config.get("use_color_info", True):
            return [[] for _ in names]
        return names

    def _segment_semantics(self, x):
        if self.segmentor is None:
            raise RuntimeError("Semantic segmentation is required but no segmentor is configured")
        seg_maps = self.segmentor.segment_batch_panoptic(x)
        try:
            names, masks = self.segmentor.get_largest_stuff_region_batch(seg_maps)
        except ValueError as exc:
            if "no non-empty stuff" not in str(exc):
                raise
            self._selection_diagnostics = [{
                "fallback_triggered": False,
                "relaxation_steps": 0,
                "unresolved_reason": "no_non_empty_stuff",
            } for _ in range(x.shape[0])]
            raise UnresolvedPlacementError("No non-empty panoptic stuff region is available") from exc
        if self.save_imgs:
            self._save_img(self.segmentor.visualize_panoptic_and_largest_stuff, x, seg_maps, masks, names)
        return seg_maps, names, masks

    def _generate_cam_and_masks(self, x, label=None):
        if self.cam_gen is None:
            raise RuntimeError("CAM is required but no CAM generator is configured")
        cams = self.cam_gen.generate_cam_batch(x, target_classes=label)
        if cams.shape[-2:] != x.shape[-2:]:
            cams = torch.nn.functional.interpolate(
                cams.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False
            )[:, 0]
        masks = self.cam_gen.cam_to_mask_batch(cams, percentile=self.config["cam_percentile"])
        if self.save_imgs:
            self._save_img(self.cam_gen.save_cam_composite_4in1_batch,
                           images_tensor=x, cams=cams, masks_batch=masks)
        return cams, masks

    def _select_patch_positions(self, x, region_masks, cam_masks):
        selector = PatchSelectorBatch(
            binary_masks_batch=region_masks,
            rgb_images_batch=x,
            patch_size=self.config["patch_size"],
            cam_masks_batch=cam_masks,
            cam_thresh_ratio=self.config["cam_thresh_ratio"],
            mask_thresh_ratio=self.config["mask_thresh_ratio"],
            relaxation_steps=self.config.get("selection_relaxation_steps", 4),
            fallback_policy=self.config.get("selection_fallback_policy", "best"),
        )
        coords = selector.get_best_patch_positions(k=self.config["patch_k"], stride_ratio=0.5)
        self._selection_diagnostics = selector.get_diagnostics()
        if self.save_imgs:
            self._save_img(selector.visualize_patch_selection, x, coords)
        return coords

    def _sample_seed(self, sample_idx, offset=0):
        payload = f"{self.config.get('seed', 42)}:{sample_idx}:{offset}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2 ** 63)

    def _random_patch_positions(self, x, sample_idx):
        """Seeded random Stage-2 ablation without changing later stages."""
        rng = random.Random(self._sample_seed(sample_idx, offset=17))
        height, width = x.shape[-2:]
        ph, pw = self.config["patch_size"]
        if ph > height or pw > width:
            raise ValueError(f"Patch size {(ph, pw)} does not fit image {(height, width)}")
        coords = []
        for _batch_index in range(x.shape[0]):
            sample = []
            for _ in range(self.config["patch_k"]):
                x1 = rng.randint(0, width - pw)
                y1 = rng.randint(0, height - ph)
                sample.append((x1, y1, x1 + pw, y1 + ph))
            coords.append(sample)
        self._selection_diagnostics = [{
            "fallback_triggered": False,
            "relaxation_steps": 0,
            "placement_strategy": "seeded_random",
            "random_seed": self._sample_seed(sample_idx, offset=17),
        } for _ in range(x.shape[0])]
        return coords

    def _validate_external_coords(self, coords, x):
        if not isinstance(coords, (list, tuple)) or len(coords) != x.shape[0]:
            raise ValueError("coords must contain one coordinate list per input image")
        validated = []
        height, width = x.shape[-2:]
        for sample_coords in coords:
            if not sample_coords:
                raise ValueError("External coords cannot contain an empty sample")
            sample = []
            for coord in sample_coords:
                if len(coord) != 4:
                    raise ValueError(f"Invalid coordinate {coord!r}")
                x1, y1, x2, y2 = (int(value) for value in coord)
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Coordinate {coord!r} is outside image {(height, width)}")
                sample.append((x1, y1, x2, y2))
            validated.append(sample)
        return validated

    def _generate_patches(self, region_names, color_names, target_classes=None):
        if self.patch_gen is None:
            raise RuntimeError("Diffusion initialisation is enabled but no patch generator is configured")
        patches, prompts = self.patch_gen.generate_patch_by_region_batch(
            region_names=region_names,
            color_names_list=color_names,
            patch_size=self.config["patch_size"],
            per_region_k=self.config["patch_k"],
            target_classes=target_classes,
            num_inference_steps=self.config.get("diffusion_steps", 20),
            guidance_scale=self.config.get("diffusion_guidance_scale", 7.5),
        )
        self._last_prompts = prompts
        if self.save_imgs:
            self._save_img(self.patch_gen.visualize_patch_batch_with_prompts,
                           patch_batch=patches, prompts_batch=prompts)
        return torch.stack([
            torch.stack([TF.to_tensor(patch) for patch in sample]) for sample in patches
        ]).to(self.device)

    def _generate_noise_patches(self, batch_size, sample_idx):
        """Seeded uniform-noise initialisation for the no-diffusion ablation."""
        height, width = self.config["patch_size"]
        k = self.config["patch_k"]
        generator = torch.Generator(device=self.device).manual_seed(self._sample_seed(sample_idx, offset=29))
        patches = torch.rand((batch_size, k, 3, height, width), generator=generator, device=self.device)
        self._last_prompts = [["seeded_noise_initialisation"] * k for _ in range(batch_size)]
        return patches

    def _blend_patches(self, x, patch_tensor, coords):
        blended = self.patch_blender.blend_patch_tensor_batch(
            backgrounds=x, patches=patch_tensor, positions=coords,
            blend_width=self.config["blend_width"],
        )
        if self.save_imgs:
            self._save_img(self.patch_blender.visualize_patch_blending_batch, x, patch_tensor, coords, blended)
        return blended

    def _optimize(self, x, patched_x, coords, y_or_target):
        palettes = getattr(self, "_palette_lab", None)
        result = self.optimizer.optimize_batch(
            x, patched_x, coords, y_or_target,
            palette_lab=palettes,
            targeted=self.target,
            return_step_preds=self._evaluation,
            use_momentum=self.config.get("use_momentum", True),
        )
        if self.save_imgs:
            self._save_img(self.optimizer.visualize_adversarial_comparison,
                           x=x, x_adv=result[0] if isinstance(result, tuple) else result)
        return result

    def run(self, x, y, target=None, sample_idx=0, coords=None,
            return_info=False, diagnostics=False, diagnostics_path=None):
        """Run the pipeline without suppressing stage failures.

        ``coords`` strictly reuses Stage-2 ``(x1,y1,x2,y2)`` patch coordinates.
        ``return_info`` (or ``diagnostics=True``) adds a JSON-compatible info dict.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("x must have shape [B,3,H,W]")
        x = x.to(self.device).float()
        if x.numel() and (x.min() < 0 or x.max() > 1):
            raise ValueError("x must contain unnormalised RGB values in [0,1]")
        y = torch.as_tensor(y, dtype=torch.long, device=self.device).reshape(-1)
        if y.numel() == 1 and x.shape[0] > 1:
            y = y.expand(x.shape[0])
        if y.numel() != x.shape[0]:
            raise ValueError("y must contain one label per input image")
        if self.target and target is None:
            if self.config.get("target_class") is None:
                raise ValueError("target is required when targeted_attack=True and target_class is not configured")
            target = torch.full((x.shape[0],), int(self.config["target_class"]), dtype=torch.long, device=self.device)
        attack_id = target if self.target else y
        attack_id = torch.as_tensor(attack_id, dtype=torch.long, device=self.device).reshape(-1)
        if attack_id.numel() == 1 and x.shape[0] > 1:
            attack_id = attack_id.expand(x.shape[0])
        if attack_id.numel() != x.shape[0]:
            raise ValueError("target must contain one class per input image")

        profiler = StageProfiler(self.device)
        profiler.start()
        info = {
            "status": "running",
            "sample_idx": sample_idx,
            "seed": self.config.get("seed", 42),
            "coords_source": "external" if coords is not None else "selector",
            "config": copy.deepcopy(self.config),
            "failure_reason": None,
        }
        self.last_diagnostics = info
        self._selection_diagnostics = None
        self._last_prompts = None
        self._palette_rgb = [[] for _ in range(x.shape[0])]
        self._palette_lab = [[] for _ in range(x.shape[0])]
        try:
            pil_images = self._to_pil(x)
            if self.config.get("scene_aware_placement", True):
                with profiler.stage("segmentation"):
                    _seg_maps, region_names, region_masks = self._segment_semantics(x)
                with profiler.stage("cam"):
                    _cams, cam_masks = self._generate_cam_and_masks(x, label=attack_id)
            else:
                region_names = ["uniform_scene"] * x.shape[0]
                region_masks = [torch.ones(x.shape[-2:], dtype=torch.uint8, device=self.device) for _ in range(x.shape[0])]
                cam_masks = [torch.zeros(x.shape[-2:], dtype=torch.uint8, device=self.device) for _ in range(x.shape[0])]
            with profiler.stage("selection"):
                if coords is not None:
                    coords_list = self._validate_external_coords(coords, x)
                elif self.config.get("placement_strategy", "scene_aware") == "random":
                    coords_list = self._random_patch_positions(x, sample_idx)
                else:
                    coords_list = self._select_patch_positions(x, region_masks, cam_masks)
            with profiler.stage("color"):
                # The palette and prompt colours come from the actual Stage-2 window.
                color_names = self._extract_colors(pil_images, coords_list=coords_list)
            with profiler.stage("diffusion"):
                if self.config.get("diffusion_init", True):
                    patch_tensor = self._generate_patches(
                        region_names, color_names, attack_id if self.target else None
                    )
                else:
                    patch_tensor = self._generate_noise_patches(x.shape[0], sample_idx)
            with profiler.stage("blending"):
                patched = self._blend_patches(x, patch_tensor, coords_list)
            with profiler.stage("optimization"):
                optimized = self._optimize(x, patched, coords_list, attack_id)
            if isinstance(optimized, tuple):
                x_adv, step_preds = optimized
            else:
                x_adv, step_preds = optimized, []
            if self._evaluation:
                with torch.no_grad():
                    original_preds = self.model(x).argmax(dim=1).detach().cpu()
                step_preds = [original_preds] + list(step_preds)
            info.update({
                "status": "success",
                "coords": coords_list,
                "selection": self._selection_diagnostics,
                "palette_rgb": self._palette_rgb,
                "palette_lab": self._palette_lab,
                "color_names": color_names,
                "prompts": self._last_prompts,
                "optimizer": self.optimizer.last_diagnostics,
            })
            if self.save_attack_process:
                self._save_attack_process("adversarial", {"final": x_adv[0]}, sample_idx=sample_idx)
        except Exception as exc:
            info.update({
                "status": "unresolved" if isinstance(exc, UnresolvedPlacementError) else "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "coords": locals().get("coords_list"),
                "selection": self._selection_diagnostics,
            })
            raise
        finally:
            info["profiling"] = profiler.summary()
            self.last_diagnostics = info
            if diagnostics_path is not None:
                save_diagnostics(info, diagnostics_path)

        wants_info = return_info or diagnostics
        if self._evaluation:
            return (x_adv, step_preds, info) if wants_info else (x_adv, step_preds)
        return (x_adv, info) if wants_info else x_adv


class MyPatchAttack_Eva(MyPatchAttack):
    """Evaluation variant preserving ``(tensor, step_preds)`` by default."""

    _evaluation = True
