"""
实验配置管理模块
包含默认配置和超参数实验配置
"""

# 默认配置
DEFAULT_CONFIG = {
    "patch_size": (64, 64),
    "blend_width": 5,
    "patch_k": 1,
    "placement_strategy": "scene_aware",
    "cam_model_name": "resnet50",
    "cam_percentile": 30,
    "topk_cam_regions": 1,
    "eval_topk": 1,
    "classifier_weights": "IMAGENET1K_V1",
    "targeted_attack": True,
    "target_class": 207,
    "target_strategy": "fixed",
    "cam_thresh_ratio": 0.3,
    "mask_thresh_ratio": 0.2,
    "steps": 100,
    "eps": 16/255,
    "alpha": 8/255,
    "advpatch_step_size": 0.01,
    "lavan_step_size": 0.01,
    "decay": 1.0,
    "lambda_adv": 1.0,
    "lambda_smooth": 0.1,
    "lambda_color": 0.05,
    "use_momentum": True,
    "seed": 42,
    "seeds": [42, 43, 44],
    "data_seed": 42,
    "n_boot": 10000,
    "reference_num_samples": 1000,
    "extended_num_samples": 5000,
    "n_colors": 3,
    "use_color_info": True,
    "scene_aware_placement": True,
    "diffusion_init": True,
    "selection_relaxation_steps": 4,
    "selection_fallback_policy": "raise",
    "mask2former_checkpoint": "facebook/mask2former-swin-base-ade-panoptic",
    "mask2former_local_files_only": False,
    "stable_diffusion_checkpoint": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "stable_diffusion_local_files_only": False,
    "disable_safety_checker": False,
    "diffusion_steps": 20,
    "diffusion_guidance_scale": 7.5
}

# 优化后的默认配置（基于超参数实验）
OPTIMIZED_CONFIG = {
    "patch_size": (48, 48),
    "blend_width": 5,
    "patch_k": 3,
    "cam_model_name": "resnet50",
    "cam_percentile": 30,
    "topk_cam_regions": 1,
    "eval_topk": 1,
    "targeted_attack": True,
    "cam_thresh_ratio": 0.3,
    "mask_thresh_ratio": 0.2,
    "steps": 25,
    "eps": 12/255,
    "alpha": 8/255
}

# Pixel模式配置
PIXEL_CONFIGS = {
    "targeted": {
        "patch_size": (224, 224),
        "blend_width": 5,
        "patch_k": 1,
        "cam_model_name": "resnet50",
        "cam_percentile": 30,
        "topk_cam_regions": 1,
        "eval_topk": 1,
        "targeted_attack": True,
        "cam_thresh_ratio": 0.3,
        "mask_thresh_ratio": 0.2,
        "steps": 10,
        "eps": 4/255,
        "alpha": 12/255
    },
    "untargeted": {
        "patch_size": (224, 224),
        "blend_width": 5,
        "patch_k": 1,
        "cam_model_name": "resnet50",
        "cam_percentile": 30,
        "topk_cam_regions": 1,
        "eval_topk": 1,
        "targeted_attack": False,
        "cam_thresh_ratio": 0.3,
        "mask_thresh_ratio": 0.2,
        "steps": 10,
        "eps": 1/255,
        "alpha": 32/255
    }
}

# Patch模式配置
PATCH_CONFIGS = {
    "targeted": {
        "patch_size": (48, 48),
        "blend_width": 5,
        "patch_k": 3,
        "cam_model_name": "resnet50",
        "cam_percentile": 30,
        "topk_cam_regions": 1,
        "eval_topk": 1,
        "targeted_attack": True,
        "cam_thresh_ratio": 0.3,
        "mask_thresh_ratio": 0.2,
        "steps": 25,
        "eps": 16/255,
        "alpha": 8/255
    },
    "untargeted": {
        "patch_size": (32, 32),
        "blend_width": 5,
        "patch_k": 3,
        "cam_model_name": "resnet50",
        "cam_percentile": 30,
        "topk_cam_regions": 1,
        "eval_topk": 1,
        "targeted_attack": False,
        "cam_thresh_ratio": 0.3,
        "mask_thresh_ratio": 0.2,
        "steps": 25,
        "eps": 16/255,
        "alpha": 8/255
    }
}

# 超参数消融实验配置
ABLATION_PARAMS = {
    "patch_size": [(32, 32), (48, 48), (64, 64)],
    "blend_width": [7, 3, 5],
    "patch_k": [1, 3, 5],
    "cam_percentile": [20, 30, 40],
    "topk_cam_regions": [1, 2],
    "cam_thresh_ratio": [0.2, 0.3, 0.4],
    "mask_thresh_ratio": [0.1, 0.2, 0.3],
    "steps": [50, 10, 25],
    "eps": [12/255, 8/255, 16/255],
    "alpha": [16/255, 5/255, 8/255],
    "targeted_attack": [True, False],
}

# 攻击方法配置
ATTACK_METHODS_CONFIG = {
    "pixel_attacks": ["FGSM", "MIFGSM", "DIFGSM", "NIFGSM", "VNIFGSM", "PGD"],
    "patch_attacks": ["LAVAN", "ART_AdvPatch"],
    "my_methods": ["MyPatchMethod"]
}

# 实验目标类别
TARGET_CLASSES = {
    "default": 207,
    "alternatives": [100, 200, 300, 400, 500]
}

def get_config(config_name="default"):
    """获取指定配置"""
    configs = {
        "default": DEFAULT_CONFIG,
        "optimized": OPTIMIZED_CONFIG,
        "paper": PAPER_CONFIG,
    }
    return configs.get(config_name, DEFAULT_CONFIG)

def get_mode_config(comparison_mode, attack_type):
    """获取指定模式和攻击类型的配置
    
    Args:
        comparison_mode: 对比模式，"pixel"或"patch"
        attack_type: 攻击类型，"targeted"或"untargeted"
    """
    if comparison_mode == "pixel":
        return PIXEL_CONFIGS.get(attack_type, PIXEL_CONFIGS["targeted"])
    elif comparison_mode == "patch":
        return PATCH_CONFIGS.get(attack_type, PATCH_CONFIGS["targeted"])
    else:
        return DEFAULT_CONFIG

def get_ablation_params():
    """获取消融实验参数"""
    return ABLATION_PARAMS

def get_attack_methods():
    """获取攻击方法配置"""
    return ATTACK_METHODS_CONFIG

def get_target_class(target_name="default"):
    """获取目标类别"""
    return TARGET_CLASSES.get(target_name, TARGET_CLASSES["default"])


def load_config(path=None):
    """Load a YAML configuration and normalise tuple-valued fields."""
    from copy import deepcopy
    from pathlib import Path
    import yaml

    config_path = Path(path) if path is not None else Path(__file__).with_name("paper_config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    config = deepcopy(DEFAULT_CONFIG)
    config.update(loaded)
    if isinstance(config.get("patch_size"), list):
        config["patch_size"] = tuple(config["patch_size"])
    return config


def save_config(config, path):
    """Save a configuration as deterministic, human-readable YAML."""
    from pathlib import Path
    import yaml

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True, allow_unicode=True)


def set_global_seed(seed=42, deterministic=False):
    """Seed Python, NumPy and PyTorch; deterministic kernels are opt-in."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


PAPER_CONFIG = load_config()