"""Reference/fallback/profiling protocol with explicit unresolved records."""

from __future__ import annotations

import argparse
import copy

import torch

from configs.experiment_configs import load_config, set_global_seed
from data.dataset import get_dataloader
from experiments.protocol_utils import class_name, info_fields, perceptual_values, target_batch
from models.model_registry import build_model
from models.my_patch_attack import MyPatchAttack
from utils.coordinates import load_stage2_coords, save_stage2_coords
from utils.experiment_stats import make_sample_record, save_statistics
from utils.patch_selector import UnresolvedPlacementError


def run_profiling(config_path=None, data_root=None, output_path="Results/profiling.json",
                  model_name="resnet50", weights="IMAGENET1K_V1", device=None, num_samples=None,
                  coords_path=None, save_coords_path=None, config_overrides=None,
                  per_class_csv_path=None, perceptual_evaluator=None, seeds=None):
    if not data_root:
        raise ValueError("data_root is required; profiling never uses synthetic experiment data implicitly")
    config = load_config(config_path)
    config.update(copy.deepcopy(config_overrides or {}))
    seeds = [int(value) for value in (seeds or config.get("seeds", [config.get("seed", 42)]))]
    sample_limit = int(num_samples or config.get("reference_num_samples", 1000))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(model_name, weights=weights, device=device)
    loader = get_dataloader(data_root=data_root, batch_size=1, total_samples=sample_limit,
                            seed=config.get("data_seed", 42), shuffle=False)
    cached = [(image.to(device), label.to(device)) for image, label in loader]
    shared_coords = load_stage2_coords(coords_path) if coords_path else None
    generated_coords = {}
    records = []
    for seed in seeds:
        config_seed = copy.deepcopy(config)
        config_seed["seed"] = int(seed)
        set_global_seed(seed)
        attack = MyPatchAttack(model=model, device=device, config=config_seed)
        for sample_idx, (image, label) in enumerate(cached[:sample_limit]):
            sample_id = str(sample_idx)
            coords = shared_coords[sample_id] if shared_coords is not None else generated_coords.get(sample_id)
            target = target_batch(config_seed, 1, device)
            try:
                adv, info = attack.run(image, label, target=target, sample_idx=sample_id,
                                       coords=coords, return_info=True)
                if shared_coords is None and seed == seeds[0]:
                    generated_coords[sample_id] = info["coords"]
                with torch.no_grad():
                    prediction = attack.model(adv).argmax(dim=1)
                success = bool((prediction == target).all()) if target is not None else bool((prediction != label).all())
                perceptual_evaluator, metrics = perceptual_values(perceptual_evaluator, image, adv)
                record = make_sample_record(
                    sample_id, seed, int(label), success, targeted=target is not None,
                    target_label=int(target[0]) if target is not None else None,
                    class_name=class_name(loader, label), method="SCAP", status="success",
                    **info_fields(info), **metrics,
                )
            except UnresolvedPlacementError:
                info = attack.last_diagnostics
                record = make_sample_record(
                    sample_id, seed, int(label), False, targeted=target is not None,
                    target_label=int(target[0]) if target is not None else None,
                    class_name=class_name(loader, label), method="SCAP", status="unresolved",
                    SSIM=None, PSNR=None, LPIPS=None, **info_fields(info),
                )
            records.append(record)
    if save_coords_path:
        save_stage2_coords(save_coords_path, generated_coords or shared_coords,
                           image_size=cached[0][0].shape[-2:])
    return save_statistics(
        output_path, records, n_boot=config.get("n_boot", 10000),
        bootstrap_seed=config.get("data_seed", 42), config=config,
        protocol={"name": "reference_and_fallback", "seeds": seeds,
                  "independent_images_per_seed": sample_limit, "total_runs": sample_limit * len(seeds),
                  "target_strategy": config.get("target_strategy"), "target_class": config.get("target_class"),
                  "unresolved_asr_policy": "failure", "perceptual_denominator": "resolved"},
        per_class_csv_path=per_class_csv_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Run the SCAP reference/fallback protocol")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="Results/profiling.json")
    parser.add_argument("--per-class-csv", default=None)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--weights", default="IMAGENET1K_V1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Use --seeds 42 for the single-pass fallback audit; default is the configured three-seed protocol")
    parser.add_argument("--coords", default=None)
    parser.add_argument("--save-coords", default=None)
    args = parser.parse_args()
    weights = None if args.weights == "None" else args.weights
    run_profiling(args.config, args.data_root, args.output, args.model, weights,
                  args.device, args.num_samples, args.coords, args.save_coords,
                  per_class_csv_path=args.per_class_csv, seeds=args.seeds)


if __name__ == "__main__":
    main()
