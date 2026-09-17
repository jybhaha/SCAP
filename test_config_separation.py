#!/usr/bin/env python3
"""
测试配置分离功能的脚本
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.experiment_01_baseline_comparison import run_baseline_comparison_experiment
from utils.result_saver import ResultManager

def test_config_separation():
    """测试配置分离功能"""
    print("=== 测试配置分离功能 ===")
    
    # 测试pixel模式
    print("\n1. 测试pixel模式 - 使用不同配置")
    run_baseline_comparison_experiment(
        comparison_mode='pixel',
        config_pixel_target='optimized',
        config_pixel_untarget='default'
    )
    
    # 测试patch模式
    print("\n2. 测试patch模式 - 使用不同配置")
    run_baseline_comparison_experiment(
        comparison_mode='patch',
        config_patch_target='default',
        config_patch_untarget='optimized'
    )

if __name__ == "__main__":
    test_config_separation() 