"""Seven-row leave-one-component-out protocol from the revised manuscript."""

from __future__ import annotations

import copy

import torch

from configs.experiment_configs import load_config, set_global_seed
from experiments.protocol_utils import class_name, info_fields, perceptual_values, target_batch
from models.my_patch_attack import MyPatchAttack
from utils.coordinates import save_stage2_coords
from utils.experiment_stats import make_sample_record, save_statistics
from utils.patch_selector import UnresolvedPlacementError


COMPONENT_VARIANTS = {
    "full": {},
    "no_scene_aware_placement": {"placement_strategy": "random"},
    "no_diffusion_init": {"diffusion_init": False},
    "no_mifgsm": {"steps": 0},
    "no_total_variation": {"lambda_smooth": 0.0},
    "no_cielab_color": {"lambda_color": 0.0},
    "no_boundary_fusion": {"blend_width": 0},
}


def run_component_ablation(model, dataloader, config=None, variants=None, output_path=None,
                           coords_output_path=None, max_samples=None, per_class_csv_path=None,
                           perceptual_evaluator=None):
    """Run all variants on fixed samples; only the placement row relocates."""
    base_config = load_config() if config is None else copy.deepcopy(config)
    variant_names = list(variants or COMPONENT_VARIANTS)
    unknown = set(variant_names) - set(COMPONENT_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown component variants: {sorted(unknown)}")
    if "full" not in variant_names:
        raise ValueError("The 'full' variant is required to establish shared coordinates")
    variant_names.remove("full")
    variant_names.insert(0, "full")
    sample_limit = int(max_samples or base_config.get("reference_num_samples", 1000))
    cached = []
    for images, labels in dataloader:
        cached.append((images, labels))
        if sum(batch[0].shape[0] for batch in cached) >= sample_limit:
            break
    seeds = list(base_config.get("seeds", [base_config.get("seed", 42)]))
    records = []
    generated_coords = {}
    for seed in seeds:
        shared_coords = {}
        for variant_name in variant_names:
            config_variant = copy.deepcopy(base_config)
            config_variant.update(COMPONENT_VARIANTS[variant_name])
            config_variant["seed"] = int(seed)
            set_global_seed(seed)
            attack = MyPatchAttack(model=model, device=next(model.parameters()).device, config=config_variant)
            for batch_idx, (images, labels) in enumerate(cached):
                images = images.to(attack.device)
                labels = labels.to(attack.device)
                for index in range(images.shape[0]):
                    sample_number = sum(batch[0].shape[0] for batch in cached[:batch_idx]) + index
                    if sample_number >= sample_limit:
                        break
                    sample_id = str(sample_number)
                    target = target_batch(config_variant, 1, attack.device)
                    coords = None if variant_name in ("full", "no_scene_aware_placement") else shared_coords.get(sample_id)
                    if coords is None and variant_name not in ("full", "no_scene_aware_placement"):
                        records.append(make_sample_record(
                            sample_id, seed, int(labels[index]), False, targeted=target is not None,
                            target_label=int(target[0]) if target is not None else None,
                            class_name=class_name(dataloader, labels[index]), variant=variant_name,
                            status="unresolved", SSIM=None, PSNR=None, LPIPS=None,
                            failure_reason="Full SCAP placement unresolved", coords=None,
                        ))
                        continue
                    try:
                        adv, info = attack.run(
                            images[index:index + 1], labels[index:index + 1], target=target,
                            sample_idx=sample_id, coords=coords, return_info=True,
                        )
                        if variant_name == "full":
                            shared_coords[sample_id] = info["coords"]
                            if seed == seeds[0]:
                                generated_coords[sample_id] = info["coords"]
                        elif variant_name != "no_scene_aware_placement" and info["coords"] != shared_coords[sample_id]:
                            raise RuntimeError(f"Shared coordinate mismatch for {sample_id}/{variant_name}")
                        with torch.no_grad():
                            prediction = attack.model(adv).argmax(dim=1)
                        success = (bool((prediction == target).all()) if target is not None
                                   else bool((prediction != labels[index:index + 1]).all()))
                        perceptual_evaluator, metrics = perceptual_values(
                            perceptual_evaluator, images[index:index + 1], adv
                        )
                        records.append(make_sample_record(
                            sample_id, seed, int(labels[index]), success, targeted=target is not None,
                            target_label=int(target[0]) if target is not None else None,
                            class_name=class_name(dataloader, labels[index]), variant=variant_name,
                            status="success", iterations=config_variant["steps"], step_size=config_variant["alpha"],
                            **info_fields(info), **metrics,
                        ))
                    except UnresolvedPlacementError:
                        records.append(make_sample_record(
                            sample_id, seed, int(labels[index]), False, targeted=target is not None,
                            target_label=int(target[0]) if target is not None else None,
                            class_name=class_name(dataloader, labels[index]), variant=variant_name,
                            status="unresolved", SSIM=None, PSNR=None, LPIPS=None,
                            **info_fields(attack.last_diagnostics),
                        ))
    if coords_output_path:
        save_stage2_coords(coords_output_path, generated_coords)
    if output_path:
        return save_statistics(
            output_path, records, n_boot=base_config.get("n_boot", 10000),
            bootstrap_seed=base_config.get("data_seed", 42), config=base_config,
            protocol={"name": "leave_one_component_out", "variants": variant_names,
                      "seeds": seeds, "independent_images_per_seed": sample_limit,
                      "total_runs_per_variant": sample_limit * len(seeds),
                      "shared_coordinates_except_placement": True},
            per_class_csv_path=per_class_csv_path,
        )
    return {"records": records}
