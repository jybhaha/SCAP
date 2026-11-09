"""Shared, side-effect-free helpers for the paper experiment runners."""

from __future__ import annotations

import torch


def target_batch(config, batch_size, device):
    if not config.get("targeted_attack"):
        return None
    if config.get("target_strategy", "fixed") != "fixed":
        raise ValueError("The paper protocol currently defines target_strategy=fixed")
    if config.get("target_class") is None:
        raise ValueError("targeted_attack=True requires an explicit target_class")
    return torch.full((batch_size,), int(config["target_class"]), dtype=torch.long, device=device)


def class_name(dataloader, label):
    names = getattr(dataloader, "class_names", None)
    return names[int(label)] if names is not None and int(label) < len(names) else str(int(label))


def perceptual_values(evaluator, clean, adversarial):
    if evaluator is None:
        from utils.perceptual_evaluator import PerceptualEvaluator
        evaluator = PerceptualEvaluator(device=clean.device)
    return evaluator, {
        "SSIM": float(evaluator.calc_ssim(clean, adversarial)),
        "PSNR": float(evaluator.calc_psnr(clean, adversarial)),
        "LPIPS": float(evaluator.calc_lpips(clean, adversarial)),
    }


def info_fields(info, sample_position=0):
    selection = info.get("selection")
    if isinstance(selection, list) and selection:
        selection = selection[min(sample_position, len(selection) - 1)]
    return {
        "coords": info.get("coords"),
        "selection": selection,
        "fallback_triggered": bool(selection and selection.get("fallback_triggered")),
        "relaxation_steps": int(selection.get("relaxation_steps", 0)) if selection else 0,
        "profiling": info.get("profiling"),
        "failure_reason": info.get("failure_reason"),
    }
