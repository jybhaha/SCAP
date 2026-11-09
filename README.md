# SCAP

**Semantic-guided Contextual Adversarial Patch** research toolbox. The pipeline combines Mask2Former stuff masks, Grad-CAM-aware placement, text-prompt-conditioned Stable Diffusion initialisation, and MI-FGSM refinement with total-variation and CIELab DeltaE76 regularisation.

> For defensive research only. Test solely systems you own or have explicit authorisation to assess. No accuracy or experiment-result reproduction is claimed by this repository snapshot.

## Installation

The v1.1.0 reference environment is Python 3.10, PyTorch 2.2.1, torchvision 0.17.1 and CUDA 11.8 (the Docker image pins this combination).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatives:

```bash
conda env create -f environment.yml
conda activate scap
# or
docker build -t scap .
```

`requirements.txt` contains direct Python dependencies only. Model weights, datasets, results and notebooks are not bundled.

## Checkpoints and offline use

The exact defaults are:

- Mask2Former: `facebook/mask2former-swin-base-ade-panoptic`
- Stable Diffusion: `stable-diffusion-v1-5/stable-diffusion-v1-5`

Both accept a local path or Hugging Face ID through YAML. The paper configuration sets `local_files_only: true`, so populate the Hugging Face cache beforehand or change the paths to local checkpoints. Quick-start `DEFAULT_CONFIG` permits hub download on first use. No test downloads weights. Stable Diffusion's safety checker remains enabled unless `disable_safety_checker: true` is explicitly configured. No Hugging Face mirror is forced.

## Unified API

`attack.attacker.MyPatchAttack` and `models.my_patch_attack.MyPatchAttack` are the same implementation. The target classifier is reused for CAM, optimisation and evaluation through one ImageNet-normalising adapter; no secondary ResNet-50 is loaded and no stage receives differently preprocessed inputs.

```python
from attack.attacker import MyPatchAttack, MyPatchAttack_Eva
from configs.experiment_configs import PAPER_CONFIG
from models.model_registry import build_model

model = build_model("resnet50", weights="IMAGENET1K_V1", device="cuda")
attack = MyPatchAttack(model, device="cuda", config=PAPER_CONFIG)
x_adv = attack.run(x, y)                       # x: unnormalised RGB in [0,1]
x_adv, info = attack.run(x, y, return_info=True)

evaluation_attack = MyPatchAttack_Eva(model, device="cuda", config=PAPER_CONFIG)
x_adv, step_preds = evaluation_attack.run(x, y)
```

The paper configuration is targeted at fixed class 207, so `target=...` may be omitted and is filled from `target_class`. Custom targeted configurations must provide either `target=...` or `target_class`. `coords=[[(x1,y1,x2,y2), ...]]` strictly reuses Stage-2 coordinates. `diagnostics_path="run.json"` writes diagnostics even when a stage raises; unresolved placement is explicit and is never replaced by a clean image or undocumented random location.

The model registry supports exactly nine torchvision interfaces: `resnet50`, `vgg16`, `densenet121`, `inception_v3`, `efficientnet_b0`, `mobilenet_v3_large`, `convnext_tiny`, `vit_b_16`, and `swin_t`. `build_model(name, weights=None)` is available for architecture-only construction. ViT token reshape and Swin channels-last activations are handled explicitly; unsupported CAM architectures fail loudly.

## Paper YAML and optimisation

`configs/paper_config.yaml` records the reference implementation settings: one 64×64 patch, targeted class 207, `eps=16/255`, 100 steps, `alpha=8/255`, `lambda_adv=1.0`, `lambda_smooth=0.1`, `lambda_color=0.05`, seeds 42/43/44, independent `data_seed=42`, 10,000 bootstrap resamples, blend width 5, bounded placement relaxation with explicit unresolved failures, and exact checkpoint IDs. The default protocol uses 1,000 images per seed; use `--num-samples 5000` for the extended-scale run.

The optimiser minimises CE for targeted attacks and negative CE for untargeted attacks using momentum gradient descent. Its L-inf projection is centred on the **patched initialisation**, and non-patch pixels retain their initial values. Colour palettes are clustered in CIELab from the actual Stage-2 window with a fixed random state. Diffusion receives palette colour **names in text prompts**; the optimiser separately uses the actual Lab palette through differentiable RGB→Lab conversion and nearest-colour DeltaE76. This is not direct Lab-tensor control of diffusion.

## Protocol commands

All commands below execute real models/data when invoked; they are examples, not commands run by the test suite.

```bash
# Reference run: 1,000 independent images per seed, seeds 42/43/44.
python main_experiments.py --experiment profiling \
  --data-root /path/to/imagenet/val \
  --output Results/profiling.json --per-class-csv Results/per_class.csv \
  --save-coords Results/stage2_coords.json

# Single-pass fallback audit: exactly 1,000 images/1,000 runs with seed 42.
python experiments/profiling_runner.py \
  --data-root /path/to/imagenet/val --seeds 42 \
  --output Results/fallback_seed42.json \
  --per-class-csv Results/fallback_seed42_per_class.csv

# Extended scale: the same protocol on 5,000 images per seed.
python main_experiments.py --experiment profiling \
  --data-root /path/to/imagenet/val --num-samples 5000 \
  --output Results/profiling_5000.json --per-class-csv Results/per_class_5000.csv

# AdvPatch, LaVAN and SCAP: targeted, 100 steps, identical Stage-2 boxes.
python main_experiments.py --experiment controlled_comparison \
  --data-root /path/to/imagenet/val --coords Results/stage2_coords.json \
  --output Results/controlled_comparison.json

# Seven independent leave-one-out rows from the manuscript.
python main_experiments.py --experiment component_ablation \
  --data-root /path/to/imagenet/val \
  --variants full no_scene_aware_placement no_diffusion_init no_mifgsm \
    no_total_variation no_cielab_color no_boundary_fusion \
  --save-coords Results/ablation_coords.json --output Results/component_ablation.json

# Reference, patch 32/96 and epsilon 8/255 on the current hardware.
python main_experiments.py --experiment sensitivity \
  --data-root /path/to/imagenet/val --device cuda --hardware "RTX 3090" \
  --output Results/sensitivity_3090

# Legacy battery and ART optional integration remain available separately.
python main_experiments.py --experiment baseline --comparison_mode all
python main_experiments.py --experiment cost --dataloader_mode small --num-samples 100
```

Component switches cover seeded random placement, noise initialisation, zero-step no-refinement, independent TV/CIELab losses and zero-width boundary fusion. Statistics preserve seeds, per-sample true/target labels, success, unresolved status, coordinates, runtime/memory and SSIM/PSNR/LPIPS. Point estimates weight seed-level ASRs equally. Overall and variant intervals use 10,000-resample image-cluster percentile bootstrap: every resampled image retains all of its seed outcomes, so 1,000 images × three seeds remain 1,000 independent clusters rather than 3,000 independent images. Per-class CSV columns are `n`, `n_runs`, `n_seeds`, ASR, CI limits/method, SSIM and LPIPS; Wilson intervals are used only for single-seed exports, while multi-seed exports use image-cluster bootstrap. Unresolved cases count as ASR failures and are excluded only from perceptual means. There are no preset experiment numbers.

Diagnostics include wall time for colour, segmentation, CAM, selection, diffusion, blending and optimisation. CUDA peak statistics are reset immediately before each stage, yielding independent stage peaks; the end-to-end peak is their maximum. Controlled summaries include per-method runtime mean/SD, memory mean/maximum, fallback counts and unresolved counts. Coordinate files include checksums; palette/prompt data, failure reasons, final thresholds and deterministic-fallback markers are retained.

## Tests

Core tests use toy tensors and no pretrained weights or network:

```bash
python -m compileall -q attack configs models utils experiments tests
python -m pytest -q
```

The six default test files currently report **32 passed**. Tests marked `pretrained` are excluded by default. A separate cached-weight smoke test can be run explicitly; neither check is presented as experiment reproduction.

## Commercial APIs

`python main_experiments.py --experiment real_world_api_test` is a **dry-run** and performs no data loading or network request. Live calls require all of the following:

1. documented authorisation from the API provider and compliance with its terms;
2. CLI flag `--execute-api-tests`;
3. environment variable `SCAP_AUTHORIZED_API_TEST=1`;
4. `SCAP_API_KEY` or `SCAP_API_KEY_<PLATFORM>` from the environment;
5. optional `SCAP_API_ENDPOINT_<PLATFORM>` endpoint overrides.

Requests are client-side rate limited. No credential is stored in source or configuration.

## Licence

BSD 3-Clause; see `LICENSE.txt`. The vendored `torchattacks/` directory retains its upstream MIT licence.
