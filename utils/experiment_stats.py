"""Image-clustered, seed-aware experiment statistics and CSV exports."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np


def make_sample_record(sample_id, seed, true_label, success, targeted=False, target_label=None, **extra):
    record = {
        "sample_id": str(sample_id),
        "seed": int(seed),
        "true_label": int(true_label),
        "success": bool(success),
        "targeted": bool(targeted),
        "target_label": int(target_label) if target_label is not None else None,
        "status": "success",
    }
    record.update(extra)
    return record


def _image_clusters(records):
    clusters = defaultdict(list)
    for record in records:
        clusters[str(record["sample_id"])].append(record)
    return [clusters[key] for key in sorted(clusters)]


def _seed_point_estimate(records):
    """Equal-weight seeds, then equal-weight images within each seed."""
    per_seed = defaultdict(list)
    for record in records:
        per_seed[int(record["seed"])].append(bool(record["success"]))
    return float(np.mean([np.mean(values) for _, values in sorted(per_seed.items())])) if per_seed else None


def bootstrap_success_ci(records, n_boot=10000, confidence=0.95, seed=42, cluster_by_image=True):
    """Percentile bootstrap of ASR.

    With repeated seeds, all outcomes for an image form one resampling cluster.
    The statistic first averages seed outcomes within an image and then averages
    image means. With one outcome per image this reduces to the ordinary image
    bootstrap. The number of runs therefore never masquerades as the number of
    independent images.
    """
    if not records:
        return {"lower": None, "upper": None, "n_boot": 0, "confidence": confidence}
    if cluster_by_image:
        values = np.asarray([
            np.mean([bool(record["success"]) for record in cluster])
            for cluster in _image_clusters(records)
        ], dtype=np.float64)
        unit = "image_cluster_with_nested_seed_mean"
    else:
        values = np.asarray([bool(record["success"]) for record in records], dtype=np.float64)
        unit = "run"
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "n_boot": int(n_boot),
        "confidence": float(confidence),
        "unit": unit,
        "n_clusters": int(len(values)),
        "n_runs": int(len(records)),
        "estimate": _seed_point_estimate(records),
    }


def wilson_success_ci(records, confidence=0.95):
    """Wilson interval, restricted to one independent run per image."""
    n = len(records)
    if n == 0:
        return {"lower": None, "upper": None, "confidence": confidence}
    if confidence != 0.95:
        raise ValueError("The reference per-class export defines a 95% Wilson interval")
    z = 1.959963984540054
    successes = sum(bool(record["success"]) for record in records)
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return {"lower": max(0.0, centre - radius), "upper": min(1.0, centre + radius),
            "confidence": confidence}


def _metric_values(records, name):
    return [float(record[name]) for record in records if record.get(name) is not None]


def _metric_mean(records, name):
    values = _metric_values(records, name)
    return float(np.mean(values)) if values else None


def _metric_std(records, name):
    values = _metric_values(records, name)
    return float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)


def _profiling_summary(records):
    totals = []
    peaks = []
    stages = defaultdict(lambda: {"times": [], "peaks": []})
    for record in records:
        profiling = record.get("profiling") or {}
        total = profiling.get("total_wall_time_seconds")
        peak = profiling.get("cuda_peak_memory_bytes")
        if total is not None:
            totals.append(float(total))
        if peak is not None:
            peaks.append(int(peak))
        for name, values in (profiling.get("stages") or {}).items():
            if values.get("wall_time_seconds") is not None:
                stages[name]["times"].append(float(values["wall_time_seconds"]))
            if values.get("cuda_peak_memory_bytes") is not None:
                stages[name]["peaks"].append(int(values["cuda_peak_memory_bytes"]))
    mib = 1024.0 * 1024.0
    return {
        "runtime_seconds_mean": float(np.mean(totals)) if totals else None,
        "runtime_seconds_std": float(np.std(totals, ddof=1)) if len(totals) > 1 else (0.0 if totals else None),
        "cuda_peak_memory_mib_mean": float(np.mean(peaks) / mib) if peaks else None,
        "cuda_peak_memory_mib_max": float(max(peaks) / mib) if peaks else None,
        "stages": {
            name: {
                "wall_time_seconds_mean": float(np.mean(values["times"])) if values["times"] else None,
                "wall_time_seconds_std": float(np.std(values["times"], ddof=1)) if len(values["times"]) > 1 else (0.0 if values["times"] else None),
                "cuda_peak_memory_mib_mean": float(np.mean(values["peaks"]) / mib) if values["peaks"] else None,
                "cuda_peak_memory_mib_max": float(max(values["peaks"]) / mib) if values["peaks"] else None,
            }
            for name, values in sorted(stages.items())
        },
    }


def _group_summary(records, n_boot, bootstrap_seed):
    seeds = sorted({int(record["seed"]) for record in records})
    per_seed_rates = []
    for seed_value in seeds:
        subset = [record for record in records if int(record["seed"]) == seed_value]
        per_seed_rates.append(float(np.mean([bool(record["success"]) for record in subset])))
    num_images = len({str(record["sample_id"]) for record in records})
    one_run_per_image = num_images == len(records)
    return {
        "num_samples": len(records),
        "num_runs": len(records),
        "num_images": num_images,
        "num_seeds": len(seeds),
        "resolved_samples": sum(record.get("status", "success") == "success" for record in records),
        "unresolved_samples": sum(record.get("status") == "unresolved" for record in records),
        "fallback_triggered_samples": sum(bool(record.get("fallback_triggered")) for record in records),
        "success_rate": _seed_point_estimate(records),
        "success_rate_seed_std": float(np.std(per_seed_rates, ddof=1)) if len(per_seed_rates) > 1 else 0.0,
        "success_rate_bootstrap_ci": bootstrap_success_ci(records, n_boot=n_boot, seed=bootstrap_seed),
        "success_rate_wilson_ci": wilson_success_ci(records) if one_run_per_image else None,
        "SSIM": _metric_mean(records, "SSIM"),
        "SSIM_std": _metric_std(records, "SSIM"),
        "PSNR": _metric_mean(records, "PSNR"),
        "PSNR_std": _metric_std(records, "PSNR"),
        "LPIPS": _metric_mean(records, "LPIPS"),
        "LPIPS_std": _metric_std(records, "LPIPS"),
        "profiling_summary": _profiling_summary(records),
    }


def summarize_records(records, n_boot=10000, bootstrap_seed=42):
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise TypeError("records must be a list of dictionaries")
    per_class = defaultdict(list)
    for record in records:
        if "success" not in record or "true_label" not in record or "seed" not in record:
            raise ValueError("Every record needs success, true_label and seed")
        per_class[int(record["true_label"])].append(record)
    summary = _group_summary(records, n_boot, bootstrap_seed)
    summary["seeds"] = sorted({int(record["seed"]) for record in records})
    summary["per_class"] = {}
    for label, class_records in sorted(per_class.items()):
        summary["per_class"][str(label)] = _group_summary(
            class_records, n_boot=n_boot, bootstrap_seed=bootstrap_seed + label
        )
        summary["per_class"][str(label)]["class_name"] = next(
            (record.get("class_name") for record in class_records if record.get("class_name")), None
        )
    variants = sorted({str(record["variant"]) for record in records if "variant" in record})
    if variants:
        summary["per_variant"] = {}
        for variant in variants:
            subset = [record for record in records if str(record.get("variant")) == variant]
            summary["per_variant"][variant] = _group_summary(subset, n_boot, bootstrap_seed)
    summary["per_seed"] = {}
    for seed_value in summary["seeds"]:
        subset = [record for record in records if int(record["seed"]) == seed_value]
        summary["per_seed"][str(seed_value)] = _group_summary(subset, n_boot, bootstrap_seed + seed_value)
    methods = sorted({str(record["method"]) for record in records if "method" in record})
    if methods:
        summary["per_method"] = {}
        for method in methods:
            subset = [record for record in records if str(record.get("method")) == method]
            summary["per_method"][method] = _group_summary(subset, n_boot, bootstrap_seed)
    return summary


def export_per_class_csv(path, records, n_boot=10000, bootstrap_seed=42):
    """Export one row per class (and method/variant when present).

    ``n`` is the number of independent images; ``n_runs`` includes seed repeats.
    Single-seed classes use Wilson intervals. Multi-seed classes use the same
    image-cluster bootstrap as the aggregate tables, avoiding pseudoreplication.
    """
    groups = defaultdict(list)
    multiple_methods = len({record.get("method") for record in records if record.get("method") is not None}) > 1
    multiple_variants = len({record.get("variant") for record in records if record.get("variant") is not None}) > 1
    include_method = multiple_methods or any("method" in record for record in records)
    include_variant = multiple_variants or any("variant" in record for record in records)
    for record in records:
        key = (record.get("method"), record.get("variant"), int(record["true_label"]))
        groups[key].append(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    if include_method:
        fieldnames.append("method")
    if include_variant:
        fieldnames.append("variant")
    fieldnames += ["class_id", "class_name", "n", "n_runs", "n_seeds", "ASR",
                   "CI_low", "CI_high", "CI_method", "SSIM", "LPIPS"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (method, variant, class_id), class_records in sorted(
                groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), item[0][2])):
            values = _group_summary(class_records, n_boot=n_boot, bootstrap_seed=bootstrap_seed + class_id)
            one_run_per_image = values["num_runs"] == values["num_images"]
            ci = values["success_rate_wilson_ci"] if one_run_per_image else values["success_rate_bootstrap_ci"]
            row = {
                "class_id": class_id,
                "class_name": next((record.get("class_name") for record in class_records if record.get("class_name")), str(class_id)),
                "n": values["num_images"],
                "n_runs": values["num_runs"],
                "n_seeds": values["num_seeds"],
                "ASR": 100 * values["success_rate"],
                "CI_low": 100 * ci["lower"],
                "CI_high": 100 * ci["upper"],
                "CI_method": "Wilson" if one_run_per_image else "image_cluster_percentile_bootstrap",
                "SSIM": values["SSIM"] if values["SSIM"] is not None else "",
                "LPIPS": values["LPIPS"] if values["LPIPS"] is not None else "",
            }
            if include_method:
                row["method"] = method or "SCAP"
            if include_variant:
                row["variant"] = variant or "full"
            writer.writerow(row)
    return destination


def save_statistics(path, records, n_boot=10000, bootstrap_seed=42, config=None,
                    protocol=None, per_class_csv_path=None):
    statistical_protocol = {
        "independent_unit": "image",
        "seed_aggregation": "equal_weight_seed_means",
        "aggregate_ci": "10,000-resample image-cluster percentile bootstrap with nested seed mean",
        "n_boot": int(n_boot),
        "single_seed_per_class_ci": "Wilson score",
        "multi_seed_per_class_ci": "image-cluster percentile bootstrap",
        "unresolved_asr_policy": "failure",
        "perceptual_denominator": "resolved runs only",
    }
    statistical_protocol.update((protocol or {}).get("statistics", {}))
    document = {
        "protocol": protocol or {},
        "statistics": statistical_protocol,
        "config": config or {},
        "records": records,
        "summary": summarize_records(records, n_boot=n_boot, bootstrap_seed=bootstrap_seed),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if per_class_csv_path:
        export_per_class_csv(per_class_csv_path, records, n_boot=n_boot, bootstrap_seed=bootstrap_seed)
    return document
