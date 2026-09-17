"""
真实世界评估配置文件
包含各种物理条件的参数设置和评估策略
"""

# 物理条件模拟参数
PHYSICAL_CONDITIONS = {
    "lighting": {
        "brightness_range": (0.6, 1.4),  # 亮度变化范围
        "contrast_range": (0.7, 1.3),    # 对比度变化范围
        "saturation_range": (0.8, 1.2),  # 饱和度变化范围
    },
    "viewpoint": {
        "max_rotation": 20,               # 最大旋转角度
        "max_scale": 0.85,               # 最大缩放比例
        "max_translation": 0.1,          # 最大平移比例
    },
    "noise_blur": {
        "noise_std_range": (0.01, 0.05), # 噪声标准差范围
        "blur_radius_range": (0.5, 2.0), # 模糊半径范围
        "noise_probability": 0.7,         # 添加噪声的概率
        "blur_probability": 0.4,          # 添加模糊的概率
    },
    "compression": {
        "quality_range": (50, 90),        # JPEG质量范围
        "compression_probability": 0.6,   # 压缩概率
    },
    "weather": {
        "rain_intensity": (0.1, 0.8),    # 雨滴强度
        "fog_density": (0.05, 0.3),      # 雾霾密度
        "weather_probability": 0.3,       # 天气效果概率
    }
}

# 评估参数
EVALUATION_CONFIG = {
    "num_samples": 200,                   # 评估样本数量
    "num_variations": 30,                 # 物理条件变化次数
    "batch_size": 16,                     # 批处理大小
    "device": "cuda",                     # 计算设备
    
    # 感知质量评估参数
    "perceptual_metrics": {
        "ssim": True,                     # 是否计算SSIM
        "psnr": True,                     # 是否计算PSNR
        "lpips": True,                    # 是否计算LPIPS
        "fid": False,                     # 是否计算FID（可选）
    },
    
    # 攻击成功率评估参数
    "attack_evaluation": {
        "topk": 1,                        # Top-K准确率评估
        "confidence_threshold": 0.5,      # 置信度阈值
        "targeted_attack": False,         # 是否为目标攻击
    }
}

# 物理条件组合策略
PHYSICAL_STRATEGIES = {
    "random_single": {
        "description": "随机选择单一物理条件",
        "conditions": ["lighting", "viewpoint", "noise_blur", "compression"],
        "selection_method": "random_single"
    },
    "random_combined": {
        "description": "随机组合多种物理条件",
        "conditions": ["lighting", "viewpoint", "noise_blur", "compression"],
        "selection_method": "random_combined",
        "max_conditions": 3
    },
    "systematic": {
        "description": "系统性测试所有条件组合",
        "conditions": ["lighting", "viewpoint", "noise_blur", "compression"],
        "selection_method": "systematic"
    },
    "realistic": {
        "description": "模拟真实场景条件",
        "conditions": {
            "indoor": ["lighting", "noise_blur"],
            "outdoor": ["lighting", "viewpoint", "compression"],
            "mobile": ["viewpoint", "compression", "noise_blur"],
            "surveillance": ["lighting", "compression"]
        },
        "selection_method": "realistic"
    }
}

# 攻击方法配置
ATTACK_CONFIGS = {
    "patch_attack": {
        "patch_size": 32,
        "blend_width": 4,
        "patch_k": 5,
        "cam_percentile": 0.8,
        "cam_thresh_ratio": 0.3,
        "mask_thresh_ratio": 0.5,
        "steps": 100,
        "eps": 0.1,
        "alpha": 0.01,
        "targeted_attack": False
    },
    "pixel_attack": {
        "eps": 0.1,
        "steps": 100,
        "alpha": 0.01,
        "targeted_attack": False
    }
}

# 报告生成配置
REPORT_CONFIG = {
    "save_detailed_results": True,        # 是否保存详细结果
    "generate_visualizations": True,      # 是否生成可视化图表
    "save_processed_images": False,       # 是否保存处理后的图像
    "report_format": "json",              # 报告格式
    "include_per_condition_analysis": True, # 是否包含每种条件的分析
}

def get_real_world_config(config_name="default"):
    """获取真实世界评估配置"""
    configs = {
        "default": {
            "physical_conditions": PHYSICAL_CONDITIONS,
            "evaluation": EVALUATION_CONFIG,
            "strategies": PHYSICAL_STRATEGIES,
            "attack_configs": ATTACK_CONFIGS,
            "report": REPORT_CONFIG
        },
        "lightweight": {
            "physical_conditions": PHYSICAL_CONDITIONS,
            "evaluation": {
                **EVALUATION_CONFIG,
                "num_samples": 50,
                "num_variations": 10
            },
            "strategies": PHYSICAL_STRATEGIES,
            "attack_configs": ATTACK_CONFIGS,
            "report": REPORT_CONFIG
        },
        "comprehensive": {
            "physical_conditions": PHYSICAL_CONDITIONS,
            "evaluation": {
                **EVALUATION_CONFIG,
                "num_samples": 500,
                "num_variations": 50
            },
            "strategies": PHYSICAL_STRATEGIES,
            "attack_configs": ATTACK_CONFIGS,
            "report": {
                **REPORT_CONFIG,
                "save_processed_images": True,
                "include_per_condition_analysis": True
            }
        }
    }
    
    return configs.get(config_name, configs["default"])

def get_physical_condition_params(condition_type, intensity="medium"):
    """获取特定物理条件的参数"""
    intensity_levels = {
        "light": {
            "brightness_range": (0.8, 1.2),
            "contrast_range": (0.9, 1.1),
            "noise_std_range": (0.005, 0.02),
            "blur_radius_range": (0.3, 1.0),
            "max_rotation": 10,
            "max_scale": 0.9,
            "quality_range": (70, 95)
        },
        "medium": {
            "brightness_range": (0.6, 1.4),
            "contrast_range": (0.7, 1.3),
            "noise_std_range": (0.01, 0.05),
            "blur_radius_range": (0.5, 2.0),
            "max_rotation": 20,
            "max_scale": 0.85,
            "quality_range": (50, 90)
        },
        "heavy": {
            "brightness_range": (0.4, 1.6),
            "contrast_range": (0.5, 1.5),
            "noise_std_range": (0.02, 0.08),
            "blur_radius_range": (1.0, 3.0),
            "max_rotation": 30,
            "max_scale": 0.8,
            "quality_range": (30, 80)
        }
    }
    
    base_params = PHYSICAL_CONDITIONS.get(condition_type, {})
    intensity_params = intensity_levels.get(intensity, intensity_levels["medium"])
    
    # 合并参数
    if condition_type == "lighting":
        return {
            **base_params,
            "brightness_range": intensity_params["brightness_range"],
            "contrast_range": intensity_params["contrast_range"]
        }
    elif condition_type == "viewpoint":
        return {
            **base_params,
            "max_rotation": intensity_params["max_rotation"],
            "max_scale": intensity_params["max_scale"]
        }
    elif condition_type == "noise_blur":
        return {
            **base_params,
            "noise_std_range": intensity_params["noise_std_range"],
            "blur_radius_range": intensity_params["blur_radius_range"]
        }
    elif condition_type == "compression":
        return {
            **base_params,
            "quality_range": intensity_params["quality_range"]
        }
    else:
        return base_params 