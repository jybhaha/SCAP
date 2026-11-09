import csv

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from experiments.component_ablation_runner import COMPONENT_VARIANTS
from models.base_line import PatchAttackBaseline
from models.imagenet_model import ensure_imagenet_normalized
from models.my_patch_attack import MyPatchAttack
from models.patch_optimizer import PatchOptimizer
from utils.color_extractor import ColorExtractor
from utils.experiment_stats import export_per_class_csv, make_sample_record, summarize_records
from utils.patch_blender import PatchBlender


def test_selected_window_palette_excludes_outside_colours():
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array[:, :4] = (255, 0, 0)
    array[:, 4:] = (0, 0, 255)
    mask = np.zeros((8, 8), dtype=bool)
    mask[:, :4] = True
    palette = ColorExtractor(n_colors=1).get_dominant_palette(Image.fromarray(array), pixel_mask=mask)
    assert palette["rgb"][0][0] > 240
    assert palette["rgb"][0][1] < 30 and palette["rgb"][0][2] < 30


def test_all_seven_manuscript_ablations_have_distinct_real_switches():
    assert set(COMPONENT_VARIANTS) == {
        "full", "no_scene_aware_placement", "no_diffusion_init", "no_mifgsm",
        "no_total_variation", "no_cielab_color", "no_boundary_fusion",
    }
    assert COMPONENT_VARIANTS["no_scene_aware_placement"] == {"placement_strategy": "random"}
    assert COMPONENT_VARIANTS["no_diffusion_init"] == {"diffusion_init": False}
    assert COMPONENT_VARIANTS["no_mifgsm"] == {"steps": 0}
    assert COMPONENT_VARIANTS["no_total_variation"] == {"lambda_smooth": 0.0}
    assert COMPONENT_VARIANTS["no_cielab_color"] == {"lambda_color": 0.0}
    assert COMPONENT_VARIANTS["no_boundary_fusion"] == {"blend_width": 0}


def test_zero_step_refinement_and_zero_boundary_are_exact():
    clean = torch.rand(1, 3, 4, 4)
    patched = clean.clone()
    patched[:, :, 1:3, 1:3] = 0.8
    optimizer = PatchOptimizer("cpu", model=torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2)),
                               steps=0, lambda_color=0)
    adv = optimizer.optimize_batch(clean, patched, [[(1, 1, 3, 3)]], torch.tensor([0]))
    assert torch.equal(adv, patched)
    blended = PatchBlender(blend_width=0).blend_patch_tensor_batch(
        clean, torch.rand(1, 1, 3, 2, 2), [[(1, 1, 3, 3)]], blend_width=0
    )
    assert torch.isfinite(blended).all()
    assert torch.equal(blended[:, :, 0, 0], clean[:, :, 0, 0])


def test_random_placement_and_noise_are_seeded_and_in_bounds():
    images = torch.rand(1, 3, 8, 8)
    outputs = []
    for _ in range(2):
        attack = MyPatchAttack.__new__(MyPatchAttack)
        attack.device = torch.device("cpu")
        attack.config = {"seed": 42, "patch_size": (2, 2), "patch_k": 1}
        coords = attack._random_patch_positions(images, sample_idx="sample-3")
        noise = attack._generate_noise_patches(1, sample_idx="sample-3")
        outputs.append((coords, noise))
    assert outputs[0][0] == outputs[1][0]
    assert torch.equal(outputs[0][1], outputs[1][1])
    x1, y1, x2, y2 = outputs[0][0][0][0]
    assert 0 <= x1 < x2 <= 8 and 0 <= y1 < y2 <= 8
    assert outputs[0][1].min() >= 0 and outputs[0][1].max() <= 1


def test_unresolved_counts_as_asr_failure_but_not_perceptual_mean(tmp_path):
    records = [
        make_sample_record("a", 42, 0, True, class_name="class-a", SSIM=0.8, LPIPS=0.1),
        make_sample_record("b", 42, 0, False, class_name="class-a", status="unresolved",
                           SSIM=None, LPIPS=None),
    ]
    summary = summarize_records(records, n_boot=20)
    assert summary["num_samples"] == 2 and summary["unresolved_samples"] == 1
    assert summary["success_rate"] == 0.5 and summary["SSIM"] == 0.8
    path = export_per_class_csv(tmp_path / "per_class.csv", records, n_boot=20)
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert list(row) == ["class_id", "class_name", "n", "n_runs", "n_seeds", "ASR",
                         "CI_low", "CI_high", "CI_method", "SSIM", "LPIPS"]
    assert row["n"] == "2" and row["n_runs"] == "2" and row["n_seeds"] == "1"
    assert row["CI_method"] == "Wilson"
    assert float(row["ASR"]) == 50.0 and float(row["SSIM"]) == 0.8

    # Repeated seeds are clustered by image rather than treated as independent runs.
    records = []
    for sample_id in ("a", "b"):
        for seed, a_success, b_success in ((42, True, False), (43, False, True), (44, True, True)):
            success = a_success if sample_id == "a" else b_success
            records.append(make_sample_record(sample_id, seed, 0, success, class_name="class-a"))
    summary = summarize_records(records, n_boot=100)
    ci = summary["success_rate_bootstrap_ci"]
    assert summary["num_images"] == 2 and summary["num_runs"] == 6 and summary["num_seeds"] == 3
    assert ci["n_clusters"] == 2 and ci["n_runs"] == 6
    assert ci["unit"] == "image_cluster_with_nested_seed_mean"
    path = export_per_class_csv(tmp_path / "clustered.csv", records, n_boot=100)
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["n"] == "2" and row["n_runs"] == "6" and row["n_seeds"] == "3"
    assert row["CI_method"] == "image_cluster_percentile_bootstrap"


def _baseline(model):
    baseline = PatchAttackBaseline.__new__(PatchAttackBaseline)
    baseline.model = ensure_imagenet_normalized(model).eval()
    baseline.device = torch.device("cpu")
    return baseline


def test_lavan_and_advpatch_update_in_the_attack_direction():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2)).eval()
    images = 0.2 + 0.6 * torch.rand(1, 3, 4, 4)
    labels = torch.tensor([0])
    target = torch.tensor([1])
    coords = [[(1, 1, 3, 3)]]
    for method in ("lavan", "advpatch"):
        baseline = _baseline(model)
        kwargs = {"coords": coords, "iterations": 0, "eps": 0.2}
        if method == "lavan":
            kwargs["epsilon"] = 0.05
        else:
            kwargs["step_size"] = 0.01
        torch.manual_seed(19)
        initial = getattr(baseline, method)(images, labels, target=target, **kwargs)
        before_target = F.cross_entropy(baseline.model(initial), target)
        kwargs["iterations"] = 3
        torch.manual_seed(19)
        targeted = getattr(baseline, method)(images, labels, target=target, **kwargs)
        after_target = F.cross_entropy(baseline.model(targeted), target)
        assert after_target < before_target

        kwargs["iterations"] = 0
        torch.manual_seed(19)
        initial = getattr(baseline, method)(images, labels, target=None, **kwargs)
        before_untargeted = F.cross_entropy(baseline.model(initial), labels)
        kwargs["iterations"] = 3
        torch.manual_seed(19)
        untargeted = getattr(baseline, method)(images, labels, target=None, **kwargs)
        after_untargeted = F.cross_entropy(baseline.model(untargeted), labels)
        assert after_untargeted > before_untargeted


def test_shared_model_adapter_normalises_once_and_preserves_raw_identity():
    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received = None

        def forward(self, images):
            self.received = images.detach().clone()
            return torch.zeros(images.shape[0], 2)

    raw = Capture()
    wrapped = ensure_imagenet_normalized(raw)
    assert ensure_imagenet_normalized(wrapped) is wrapped
    images = torch.rand(1, 3, 2, 2)
    wrapped(images)
    expected = (images - wrapped.imagenet_mean) / wrapped.imagenet_std
    assert torch.allclose(raw.received, expected)
    assert wrapped.raw_model is raw

    # Repeated stage names must not overwrite independent profiling records.
    from utils.profiling import StageProfiler

    profiler = StageProfiler("cpu")
    profiler.start()
    with profiler.stage("selection"):
        pass
    try:
        with profiler.stage("selection"):
            pass
    except ValueError as exc:
        assert "already recorded" in str(exc)
    else:
        raise AssertionError("Repeated stage names would silently overwrite independent measurements")
    assert profiler.summary()["stages"]["selection"]["memory_measurement"] == "cpu_no_cuda_memory"
