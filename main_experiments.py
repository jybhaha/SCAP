"""Unified CLI for the existing experiments and paper-protocol runners."""

from __future__ import annotations

import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="Run SCAP experiments (weights/data are never bundled)")
    parser.add_argument("--experiment", required=True, choices=[
        "baseline", "controlled_comparison", "ablation", "component_ablation", "cost", "profiling", "sensitivity",
        "transferability", "real_world_effectiveness", "real_world_api_test", "all",
    ])
    parser.add_argument("--dataloader_mode", default="small", choices=["small", "medium", "large", "default"])
    parser.add_argument("--comparison_mode", default="all", choices=["pixel", "patch", "all"])
    parser.add_argument("--config", default="default", choices=["default", "optimized", "paper"],
                        help="Named configuration for the legacy hyperparameter ablation")
    parser.add_argument("--config-path", default=None, help="YAML used by profiling/component_ablation")
    parser.add_argument("--categories", default="household")
    parser.add_argument("--data-root", default=None, help="ImageFolder root required by profiling/component_ablation")
    parser.add_argument("--output", default=None, help="JSON output path for paper protocols")
    parser.add_argument("--per-class-csv", default=None, help="Wilson-CI per-class CSV export")
    parser.add_argument("--coords", default=None, help="Shared Stage-2 coordinate JSON for profiling")
    parser.add_argument("--save-coords", default=None)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Defaults to reference_num_samples=1000 from the paper YAML")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--weights", default="IMAGENET1K_V1",
                        help="Paper default IMAGENET1K_V1; also accepts DEFAULT or None")
    parser.add_argument("--device", default=None)
    parser.add_argument("--hardware", default=None, help="Hardware label recorded by sensitivity runs")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of full/no_scene_aware_placement/no_diffusion_init/no_mifgsm/"
                             "no_total_variation/no_cielab_color/no_boundary_fusion")
    parser.add_argument("--execute-api-tests", action="store_true",
                        help="Enable network calls; also requires SCAP_AUTHORIZED_API_TEST=1 and env credentials")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.experiment == "baseline":
        from experiments.experiment_01_02 import run_baseline_comparison_experiment
        return run_baseline_comparison_experiment(args.comparison_mode, args.dataloader_mode)
    if args.experiment == "ablation":
        from experiments.experiment_03_hyperparameter_ablation import run_hyperparameter_ablation_experiment
        return run_hyperparameter_ablation_experiment(args.dataloader_mode, args.config)
    if args.experiment == "transferability":
        from experiments.experiment_05_transferability import run_transferability_experiment
        return run_transferability_experiment(args.dataloader_mode)
    if args.experiment == "real_world_effectiveness":
        from experiments.experiment_06_real_world_effectiveness import run_real_world_effectiveness_experiment
        return run_real_world_effectiveness_experiment(args.dataloader_mode)
    if args.experiment == "real_world_api_test":
        from experiments.experiment_08_real_world_api_test import run_real_world_api_test_experiment
        return run_real_world_api_test_experiment(
            args.dataloader_mode, args.categories, dry_run=not args.execute_api_tests
        )
    if args.experiment == "cost":
        from experiments.experiment_attack_cost_comparison import run_attack_cost_comparison
        return run_attack_cost_comparison(args.dataloader_mode, args.num_samples if args.num_samples is not None else 1)
    if args.experiment == "profiling":
        if not args.data_root:
            raise SystemExit("--data-root is required for profiling")
        from experiments.profiling_runner import run_profiling
        weights = None if args.weights == "None" else args.weights
        return run_profiling(
            args.config_path, args.data_root, args.output or "Results/profiling.json",
            args.model, weights, args.device, args.num_samples, args.coords, args.save_coords,
            per_class_csv_path=args.per_class_csv,
        )
    if args.experiment == "controlled_comparison":
        if not args.data_root:
            raise SystemExit("--data-root is required for controlled_comparison")
        from experiments.controlled_comparison_runner import run_controlled_comparison
        weights = None if args.weights == "None" else args.weights
        return run_controlled_comparison(
            args.config_path, args.data_root, args.output or "Results/controlled_comparison.json",
            args.model, weights, args.device, args.num_samples, args.coords, args.save_coords,
            per_class_csv_path=args.per_class_csv,
        )
    if args.experiment == "sensitivity":
        if not args.data_root:
            raise SystemExit("--data-root is required for sensitivity")
        from experiments.sensitivity_runner import run_sensitivity
        weights = None if args.weights == "None" else args.weights
        return run_sensitivity(
            args.config_path, args.data_root, args.output or "Results/sensitivity",
            args.model, weights, args.device, args.num_samples, args.hardware,
        )
    if args.experiment == "component_ablation":
        if not args.data_root:
            raise SystemExit("--data-root is required for component_ablation")
        import torch
        from configs.experiment_configs import load_config
        from data.dataset import get_dataloader
        from experiments.component_ablation_runner import run_component_ablation
        from models.model_registry import build_model
        config = load_config(args.config_path)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = None if args.weights == "None" else args.weights
        model = build_model(args.model, weights=weights, device=device)
        sample_limit = args.num_samples or config.get("reference_num_samples", 1000)
        loader = get_dataloader(args.data_root, batch_size=1, total_samples=sample_limit,
                                seed=config.get("data_seed", 42), shuffle=False)
        return run_component_ablation(
            model, loader, config=config, variants=args.variants,
            output_path=args.output or "Results/component_ablation.json",
            coords_output_path=args.save_coords, max_samples=sample_limit,
            per_class_csv_path=args.per_class_csv,
        )
    if args.experiment == "all":
        results = {}
        results["baseline"] = main(["--experiment", "baseline", "--dataloader_mode", args.dataloader_mode,
                                    "--comparison_mode", args.comparison_mode])
        results["ablation"] = main(["--experiment", "ablation", "--dataloader_mode", args.dataloader_mode,
                                    "--config", args.config])
        results["transferability"] = main(["--experiment", "transferability",
                                           "--dataloader_mode", args.dataloader_mode])
        results["real_world_effectiveness"] = main([
            "--experiment", "real_world_effectiveness", "--dataloader_mode", args.dataloader_mode
        ])
        # The commercial API stage remains a dry-run unless explicitly enabled.
        api_args = ["--experiment", "real_world_api_test", "--dataloader_mode", args.dataloader_mode,
                    "--categories", args.categories]
        if args.execute_api_tests:
            api_args.append("--execute-api-tests")
        results["real_world_api_test"] = main(api_args)
        return results
    raise ValueError(f"Unhandled experiment {args.experiment}")


if __name__ == "__main__":
    main()
