"""Matched-budget AdvPatch/LaVAN/SCAP comparison at shared Stage-2 boxes."""

from __future__ import annotations

import copy

import torch

from configs.experiment_configs import load_config, set_global_seed
from data.dataset import get_dataloader
from experiments.protocol_utils import class_name, info_fields, perceptual_values, target_batch
from models.base_line import PatchAttackBaseline
from models.model_registry import build_model
from models.my_patch_attack import MyPatchAttack
from utils.coordinates import load_stage2_coords, save_stage2_coords
from utils.experiment_stats import make_sample_record, save_statistics
from utils.patch_selector import UnresolvedPlacementError
from utils.profiling import StageProfiler


def _profile_baseline(call, device):
    profiler = StageProfiler(device)
    profiler.start()
    with profiler.stage("end_to_end"):
        adversarial = call()
    return adversarial, profiler.summary()


def run_controlled_comparison(config_path=None, data_root=None, output_path="Results/controlled_comparison.json",
                              model_name="resnet50", weights="IMAGENET1K_V1", device=None, num_samples=None,
                              coords_path=None, save_coords_path=None, per_class_csv_path=None,
                              perceptual_evaluator=None):
    if not data_root:
        raise ValueError("data_root is required for the controlled comparison")
    config = load_config(config_path)
    seeds = list(config.get("seeds", [config.get("seed", 42)]))
    sample_limit = int(num_samples or config.get("reference_num_samples", 1000))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(model_name, weights=weights, device=device)
    loader = get_dataloader(data_root, batch_size=1, total_samples=sample_limit,
                            seed=config.get("data_seed", 42), shuffle=False)
    cached = [(image.to(device), label.to(device)) for image, label in loader]
    shared_coords = load_stage2_coords(coords_path) if coords_path else None
    generated_coords = {}
    records = []
    for seed in seeds:
        config_seed = copy.deepcopy(config)
        config_seed["seed"] = int(seed)
        set_global_seed(seed)
        scap = MyPatchAttack(model=model, device=device, config=config_seed)
        baseline = PatchAttackBaseline(model=model, device=device)
        for sample_idx, (image, label) in enumerate(cached[:sample_limit]):
            sample_id = str(sample_idx)
            coords = shared_coords[sample_id] if shared_coords is not None else generated_coords.get(sample_id)
            target = target_batch(config_seed, 1, device)
            try:
                adv, info = scap.run(image, label, target=target, sample_idx=sample_id,
                                     coords=coords, return_info=True)
                coords = info["coords"]
                if shared_coords is None and seed == seeds[0]:
                    generated_coords[sample_id] = coords
                outputs = {"SCAP": (adv, info)}
                advpatch_adv, advpatch_profile = _profile_baseline(lambda: baseline.advpatch(
                    image, label, target=target, coords=coords,
                    step_size=config.get("advpatch_step_size", 0.01),
                    iterations=config["steps"], eps=config["eps"],
                ), device)
                lavan_adv, lavan_profile = _profile_baseline(lambda: baseline.lavan(
                    image, label, target=target, coords=coords,
                    epsilon=config.get("lavan_step_size", 0.01),
                    iterations=config["steps"], eps=config["eps"],
                ), device)
                outputs["AdvPatch"] = (advpatch_adv, {"coords": coords, "profiling": advpatch_profile})
                outputs["LAVAN"] = (lavan_adv, {"coords": coords, "profiling": lavan_profile})
                for method, (method_adv, method_info) in outputs.items():
                    with torch.no_grad():
                        prediction = scap.model(method_adv).argmax(dim=1)
                    success = bool((prediction == target).all()) if target is not None else bool((prediction != label).all())
                    perceptual_evaluator, metrics = perceptual_values(perceptual_evaluator, image, method_adv)
                    step_size = (config["alpha"] if method == "SCAP" else
                                 config.get("advpatch_step_size" if method == "AdvPatch" else "lavan_step_size", 0.01))
                    fields = info_fields(method_info) if method_info is not None else {
                        "coords": coords, "selection": None, "fallback_triggered": False,
                        "relaxation_steps": 0, "profiling": None, "failure_reason": None,
                    }
                    records.append(make_sample_record(
                        sample_id, seed, int(label), success, targeted=target is not None,
                        target_label=int(target[0]) if target is not None else None,
                        class_name=class_name(loader, label), method=method, status="success",
                        iterations=config["steps"], step_size=step_size, **fields, **metrics,
                    ))
            except UnresolvedPlacementError:
                info = scap.last_diagnostics
                for method in ("AdvPatch", "LAVAN", "SCAP"):
                    records.append(make_sample_record(
                        sample_id, seed, int(label), False, targeted=target is not None,
                        target_label=int(target[0]) if target is not None else None,
                        class_name=class_name(loader, label), method=method, status="unresolved",
                        iterations=config["steps"], step_size=None, SSIM=None, PSNR=None, LPIPS=None,
                        **info_fields(info),
                    ))
    if save_coords_path:
        save_stage2_coords(save_coords_path, generated_coords or shared_coords,
                           image_size=cached[0][0].shape[-2:])
    return save_statistics(
        output_path, records, n_boot=config.get("n_boot", 10000),
        bootstrap_seed=config.get("data_seed", 42), config=config,
        protocol={"name": "controlled_comparison", "methods": ["AdvPatch", "LAVAN", "SCAP"],
                  "seeds": seeds, "independent_images_per_seed": sample_limit,
                  "total_runs_per_method": sample_limit * len(seeds), "iterations": config["steps"],
                  "step_sizes": {"SCAP": config["alpha"],
                                 "AdvPatch": config.get("advpatch_step_size", 0.01),
                                 "LAVAN": config.get("lavan_step_size", 0.01)},
                  "shared_stage2_coordinates": True, "targeted": config["targeted_attack"],
                  "target_class": config.get("target_class"), "epsilon": config["eps"]},
        per_class_csv_path=per_class_csv_path,
    )
