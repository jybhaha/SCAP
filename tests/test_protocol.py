import json
from types import SimpleNamespace

import pytest
import torch

from configs.experiment_configs import PAPER_CONFIG, load_config, save_config, set_global_seed
from models.Mask2Former.mask2former import (
    DEFAULT_MASK2FORMER_CHECKPOINT, SemanticSegmentor,
)
from models.model_registry import MODEL_NAMES, default_weight_identifier
from models.StableFusion.patch_generator import DEFAULT_STABLE_DIFFUSION_CHECKPOINT
from utils.coordinates import load_stage2_coords, save_stage2_coords
from utils.experiment_stats import make_sample_record, summarize_records


def test_paper_config_and_nine_model_registry(tmp_path):
    assert PAPER_CONFIG["patch_size"] == (64, 64)
    assert PAPER_CONFIG["steps"] == 100
    assert PAPER_CONFIG["eps"] == pytest.approx(16 / 255)
    assert PAPER_CONFIG["alpha"] == pytest.approx(8 / 255)
    assert PAPER_CONFIG["seed"] == 42
    assert PAPER_CONFIG["seeds"] == [42, 43, 44]
    assert PAPER_CONFIG["patch_k"] == 1
    assert PAPER_CONFIG["targeted_attack"] is True
    assert PAPER_CONFIG["target_class"] == 207
    assert PAPER_CONFIG["n_boot"] == 10000
    assert PAPER_CONFIG["selection_fallback_policy"] == "raise"
    assert PAPER_CONFIG["blend_width"] == 5
    assert len(MODEL_NAMES) == 9 and len(PAPER_CONFIG["models"]) == 9
    assert tuple(PAPER_CONFIG["models"]) == MODEL_NAMES
    assert default_weight_identifier("resnet50") == "ResNet50_Weights.IMAGENET1K_V2"
    assert default_weight_identifier("mobilenet_v3_large") == "MobileNet_V3_Large_Weights.IMAGENET1K_V2"
    assert all(default_weight_identifier(name).endswith("IMAGENET1K_V1")
               for name in MODEL_NAMES if name not in {"resnet50", "mobilenet_v3_large"})
    assert DEFAULT_MASK2FORMER_CHECKPOINT == "facebook/mask2former-swin-base-ade-panoptic"
    assert DEFAULT_STABLE_DIFFUSION_CHECKPOINT == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    path = tmp_path / "config.yaml"
    save_config(PAPER_CONFIG, path)
    assert load_config(path)["patch_size"] == (64, 64)
    set_global_seed(42)


def test_shared_coordinate_checksum_and_tamper_detection(tmp_path):
    path = tmp_path / "coords.json"
    coords = {"0": [[(1, 2, 5, 6)]]}
    save_stage2_coords(path, coords, image_size=(8, 8))
    assert load_stage2_coords(path) == {"0": [[[1, 2, 5, 6]]]}
    document = json.loads(path.read_text(encoding="utf-8"))
    document["samples"][0]["coords"][0][0][0] = 2
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_stage2_coords(path)


def test_seed_per_class_variant_and_bootstrap_statistics():
    records = [
        make_sample_record("a", 42, 0, True, variant="full"),
        make_sample_record("b", 42, 0, False, variant="full"),
        make_sample_record("c", 43, 1, True, variant="no_mifgsm"),
    ]
    summary = summarize_records(records, n_boot=50, bootstrap_seed=42)
    assert summary["seeds"] == [42, 43]
    assert summary["num_samples"] == 3
    assert summary["per_class"]["0"]["num_samples"] == 2
    assert set(summary["per_variant"]) == {"full", "no_mifgsm"}
    interval = summary["success_rate_bootstrap_ci"]
    assert 0 <= interval["lower"] <= interval["upper"] <= 1


def test_panoptic_selection_filters_things():
    segmentor = SemanticSegmentor.__new__(SemanticSegmentor)
    segmentor.model = SimpleNamespace(config=SimpleNamespace(id2label={0: "sky", 12: "person"}))
    output = {
        "segmentation": torch.tensor([[1, 1], [2, 2]]),
        "segments_info": [
            {"id": 1, "label_id": 12, "isthing": True},
            {"id": 2, "label_id": 0, "isthing": False},
        ],
    }
    names, masks = segmentor.get_largest_stuff_region_batch([output])
    assert names == ["sky"]
    assert masks[0].tolist() == [[0, 0], [1, 1]]
    output["segments_info"] = output["segments_info"][:1]
    with pytest.raises(ValueError, match="no non-empty stuff"):
        segmentor.get_largest_stuff_region_batch([output])


def test_patch_baseline_coordinate_validation_without_model_initialisation():
    from models.base_line import PatchAttackBaseline
    images = torch.rand(2, 3, 8, 8)
    coords = [[(0, 0, 4, 4)], [(2, 2, 6, 6)]]
    resolved = PatchAttackBaseline._resolve_patch_coords(images, coords=coords)
    assert resolved[1] == {"x_min": 2, "y_min": 2, "x_max": 6, "y_max": 6}
    mask = PatchAttackBaseline._mask_from_coords(images, resolved)
    assert mask[0, :, :4, :4].sum() == 3 * 4 * 4
    assert mask[0].sum() + mask[1].sum() == 2 * 3 * 4 * 4
    with pytest.raises(ValueError, match="exactly one"):
        PatchAttackBaseline._resolve_patch_coords(images[:1], coords=[[(0, 0, 2, 2), (2, 2, 4, 4)]])


def test_commercial_api_defaults_to_dry_run_without_network(monkeypatch):
    monkeypatch.delenv("SCAP_AUTHORIZED_API_TEST", raising=False)
    from experiments.experiment_08_real_world_api_test import RealWorldAPITester
    tester = RealWorldAPITester(device="cpu")
    plan = tester.dry_run_plan("household")
    assert plan["dry_run"] is True and plan["network_calls"] == 0
    response = tester._call_api("ecommerce", "jd", "not-sent")
    assert response["dry_run"] is True
    with pytest.raises(PermissionError, match="SCAP_AUTHORIZED_API_TEST"):
        RealWorldAPITester(device="cpu", dry_run=False)
