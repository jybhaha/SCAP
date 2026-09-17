"""
实验配置管理模块
包含默认配置和超参数实验配置
"""

# 默认配置
DEFAULT_CONFIG = {
    "patch_size": (64, 64),
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
        "optimized": OPTIMIZED_CONFIG
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