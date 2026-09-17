# SCAR

**S**emantic-guided **C**ontextual **A**dversarial patch attack — an open-source PyTorch
toolbox for generating contextually camouflaged adversarial patches.

SCAR couples **where** a patch is placed with **how** it is synthesised: panoptic
semantic masks and Grad-CAM saliency delimit admissible placement regions, a latent
diffusion prior conditioned on scene semantics and a CIELab colour seed synthesises a
natural-looking initial patch, and an MI-FGSM-style optimiser refines it under an
L-infinity budget with smoothness and colour regularisers. The result is a patch that
degrades the target model while remaining inconspicuous in the scene.

> ⚠️ **This toolbox is released for defensive research**: auditing the robustness of
> computer-vision systems is a prerequisite for making them trustworthy. Users are
> responsible for complying with the laws, institutional policies and terms of service
> that apply to their deployment context. Do not use this toolbox against systems you
> do not own or lack authorisation to test.

---

## Repository layout

```
SCAR_pub/
├── attack/
│   └── attacker.py               # MyPatchAttack: end-to-end attack pipeline
├── models/
│   ├── my_patch_attack.py        # attack orchestration variants
│   ├── patch_optimizer.py        # MI-FGSM refinement + smoothness/colour regularisers
│   ├── patch_optimizer_changeloss.py           # alternative loss ablations
│   ├── patch_optimizer_changebehindloss.py    # alternative loss ablations
│   ├── base_line.py              # patch baselines
│   ├── Mask2Former/mask2former.py    # semantic segmentation wrapper
│   └── StableFusion/patch_generator.py  # diffusion-based patch synthesis
├── utils/
│   ├── patch_selector.py         # semantic placement region selection
│   ├── cam_generator.py          # Grad-CAM saliency extraction
│   ├── color_extractor.py        # CIELab k-means colour seeds
│   ├── patch_blender.py          # patch blending into the scene
│   ├── perceptual_evaluator.py   # perceptual quality metrics
│   ├── asr_evaluator.py          # attack success rate evaluation
│   ├── success_function.py       # success criteria
│   ├── real_world_visualizer.py  # visualisation utilities
│   └── result_saver.py           # experiment output management
├── experiments/                  # experiment scripts (01-09) + configs
├── examples/
│   └── real_world_evaluation_example.py
├── torchattacks/                 # vendored, extended copy of Harry24k/torchattacks (MIT)
│                                 # `*_mine.py` modules implement the paper's patch baselines
├── configs/                      # experiment / real-world configuration
├── data/                         # dataloaders and transforms
├── main_experiments.py           # unified experiment entry point
├── requirements.txt
├── README.md
└── LICENSE.txt
```

---

## Installation

```bash
git clone https://github.com/jybhaha/SCAR_pub.git
cd SCAR_pub
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Main dependencies (pinned in `requirements.txt`): PyTorch, torchvision, OpenCV,
NumPy, scikit-learn, diffusers (patch generator), transformers (Mask2Former backend).
A CUDA-capable GPU is strongly recommended.

Pretrained weights (Mask2Former checkpoints) are **not** bundled; the code downloads
or loads them from configurable paths.

---

## Quick start

The core attack is exposed through `MyPatchAttack`:

```python
import torch
from attack.attacker import MyPatchAttack
from data.dataset import get_dataloader

device = torch.device("cuda")
dataloader = get_dataloader(data_root="/path/to/data", batch_size=4, image_size=224)

attack = MyPatchAttack(model=target_model, device=device, config=None, debug=False)

for x, y in dataloader:
    result = attack.run(x, y)   # returns the adversarial image and attack diagnostics
    break
```

Run the experiment battery through the unified entry point:

```bash
python main_experiments.py --experiment baseline        # baseline comparison
python main_experiments.py --experiment ablation        # hyperparameter ablation
python main_experiments.py --experiment transferability # cross-architecture transfer
python main_experiments.py --experiment real_world_effectiveness
python main_experiments.py --experiment real_world_api_test
```

Individual experiment scripts live in `experiments/`; configuration in `configs/`.

---

## Experiments

| Script | Protocol |
|---|---|
| `experiments/experiment_01_02.py` | baseline comparison |
| `experiments/experiment_03_hyperparameter_ablation*.py` | hyperparameter / loss ablations |
| `experiments/experiment_04_robustness*.py` | robustness under degraded conditions |
| `experiments/experiment_05_transferability.py` | cross-architecture transferability |
| `experiments/experiment_06_real_world_effectiveness.py` | simulated real-world conditions |
| `experiments/experiment_08_real_world_api_test.py` | black-box commercial vision API probing |
| `experiments/experiment_09_physical_world_attack.py` | print-and-capture physical attack |

`torchattacks/` is a vendored, extended copy of
[torchattacks](https://github.com/Harry24k/torchattacks) (MIT licence). The `*_mine.py`
modules implement the patch-oriented baseline attacks used in the paper; all other
modules are unchanged upstream code.

---

## Licence

Released under the BSD 3-Clause Licence — see [`LICENSE.txt`](LICENSE.txt).
The vendored `torchattacks/` directory remains under its upstream MIT licence.

## Contact

Yibo Jiao — jiaoyibo@besti.edu.cn
