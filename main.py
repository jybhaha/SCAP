"""Compatibility wrapper for ``main_experiments.py``.

Real examples (require local data and checkpoints; not run by unit tests)::

    python main.py --experiment profiling --data-root /path/to/imagenet/val \
        --output Results/profiling.json --save-coords Results/stage2_coords.json
    python main.py --experiment component_ablation --data-root /path/to/imagenet/val \
        --num-samples 100 --save-coords Results/stage2_coords.json
    python main.py --experiment real_world_api_test  # dry-run, no request

The legacy experiment names and options are shown by ``python main.py --help``.
"""

from main_experiments import main


if __name__ == "__main__":
    main()
