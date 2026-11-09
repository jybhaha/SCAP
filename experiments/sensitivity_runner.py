"""Hardware/settings sensitivity wrapper using the reference three-seed protocol."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.profiling_runner import run_profiling


SENSITIVITY_SETTINGS = {
    "reference": {},
    "patch_32": {"patch_size": [32, 32]},
    "patch_96": {"patch_size": [96, 96]},
    "epsilon_8": {"eps": 8 / 255},
}


def run_sensitivity(config_path=None, data_root=None, output_dir="Results/sensitivity",
                    model_name="resnet50", weights="IMAGENET1K_V1", device=None, num_samples=None,
                    hardware_name=None):
    if not data_root:
        raise ValueError("data_root is required for sensitivity evaluation")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    documents = {}
    for name, overrides in SENSITIVITY_SETTINGS.items():
        documents[name] = run_profiling(
            config_path=config_path, data_root=data_root,
            output_path=destination / f"{name}.json", model_name=model_name,
            weights=weights, device=device, num_samples=num_samples,
            config_overrides=overrides,
        )
    aggregate = {
        "hardware": hardware_name or str(device or "auto"),
        "settings": {name: overrides for name, overrides in SENSITIVITY_SETTINGS.items()},
        "summaries": {name: document["summary"] for name, document in documents.items()},
        "seed_aggregates": {
            name: {
                "seeds": document["summary"]["seeds"],
                "independent_images_per_seed": document["protocol"]["independent_images_per_seed"],
                "ASR_mean": document["summary"]["success_rate"],
                "ASR_seed_std": document["summary"]["success_rate_seed_std"],
                "runtime_seconds_mean": document["summary"]["profiling_summary"]["runtime_seconds_mean"],
                "runtime_seconds_std": document["summary"]["profiling_summary"]["runtime_seconds_std"],
                "cuda_peak_memory_mib_max": document["summary"]["profiling_summary"]["cuda_peak_memory_mib_max"],
            }
            for name, document in documents.items()
        },
        "note": "ASR mean±SD is computed over seed-level estimates on the same independent images; runtime statistics use per-image runs.",
    }
    (destination / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate
