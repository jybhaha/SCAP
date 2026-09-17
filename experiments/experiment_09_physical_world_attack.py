#!/usr/bin/env python3
"""
实验09：物理世界攻击
端到端的物理世界对抗样本生成与测试流程

主要功能：
1. 加载真实世界拍摄的照片
2. 物体检测与定位
3. 生成针对物理世界的对抗样本
4. 物理世界优化（打印、显示友好）
5. 导出可在手机/打印机使用的格式
6. 集成移动设备API进行验证
7. 生成完整的攻击报告

使用流程：
1. 用手机拍摄目标物体照片，放入 input_photos/ 目录
2. 运行脚本生成对抗样本
3. 导出的对抗样本保存在 Results/YYYYMMDD/09_physical_world_attack/ 目录
4. 将生成的图片传回手机进行验证

作者: SCAR Team
日期: 2025-10-09
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import json
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from io import BytesIO
import base64
from pathlib import Path

from utils.result_saver import ResultManager
from experiments.experiment_runner import BaseExperimentRunner
from models.my_patch_attack import MyPatchAttack
from torchvision import models, transforms
import matplotlib.pyplot as plt


class PhysicalWorldAttackGenerator:
    """物理世界攻击生成器"""
    
    def __init__(self, device='cuda', attack_config=None):
        """
        初始化物理世界攻击生成器
        
        Args:
            device: 设备 ('cuda' 或 'cpu')
            attack_config: 攻击配置字典，如果为None则使用默认配置
        """
        self.device = torch.device(device)
        self.rm = ResultManager.get_instance()
        self.attack_config = attack_config  # 保存配置供后续使用
        self._init_models()
        self._init_transforms()
        self._init_class_labels()
        
    def _init_models(self):
        """初始化模型"""
        self.rm.log("Initializing models for physical world attack...")
        
        # 分类器
        self.classifier = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        ).eval().to(self.device)
        
        # 攻击器
        try:
            # 如果没有提供config，使用默认配置
            if self.attack_config is None:
                # 默认配置
                # 注意：patch_size 必须 < 224（模型输入尺寸）
                self.attack_config = {
                    "patch_size": (60, 60),
                    "patch_k": 5,
                    "patch_type": "square",
                    "steps": 1000,  # 优化步数
                    "eps": 16/255,  # 最大扰动
                    "alpha": 8/255,  # 步长
                    "blend_width": 6,  # 边缘融合宽度
                    "targeted_attack": False,
                    "cam_model_name": "resnet50",
                    "cam_thresh_ratio": 0.5,  # CAM阈值比例
                    "mask_thresh_ratio": 0.3,  # 掩码阈值比例
                    "debug": False,
                    "save_imgs": False,
                    "save_attack_process": False,
                }
            
            self.attacker = MyPatchAttack(
                model=self.classifier,
                device=self.device,
                config=self.attack_config
            )
            self.rm.log(f"Attacker initialized with config: patch_size={self.attack_config['patch_size']}, "
                       f"patch_k={self.attack_config['patch_k']}, eps={self.attack_config['eps']:.4f}")
        except Exception as e:
            self.rm.log(f"Failed to initialize attacker: {e}")
            self.attacker = None
        
        self.rm.log("Models initialized successfully")
    
    def _init_transforms(self):
        """初始化图像变换"""
        # 直接resize到224x224，不要center crop
        # 这样对抗样本会覆盖整个裁剪区域
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # 直接resize到224x224
            transforms.ToTensor(),
        ])
    
    def _init_class_labels(self):
        """初始化ImageNet类别标签"""
        # 加载ImageNet类别名称
        try:
            # 简化版ImageNet类别（常见物体）
            self.class_names = {
                0: "tench", 1: "goldfish", 2: "great_white_shark", 3: "tiger_shark",
                4: "hammerhead", 5: "electric_ray", 6: "stingray", 7: "cock",
                8: "hen", 9: "ostrich", 281: "tabby_cat", 282: "tiger_cat",
                283: "persian_cat", 285: "egyptian_cat", 207: "golden_retriever",
                151: "chihuahua", 950: "espresso", 967: "cup", 504: "coffee_mug",
                924: "guacamole", 963: "pizza", 925: "trifle", 567: "iPod"
            }
        except Exception as e:
            self.rm.log(f"Failed to load class names: {e}")
            self.class_names = {}
    
    def load_real_world_photo(self, photo_path):
        """加载真实世界照片"""
        try:
            self.rm.log(f"Loading photo from: {photo_path}")
            
            # 加载图像
            image = Image.open(photo_path).convert('RGB')
            original_size = image.size
            
            self.rm.log(f"Photo loaded successfully, size: {original_size}")
            
            return image, original_size
            
        except Exception as e:
            self.rm.log(f"Failed to load photo: {e}")
            return None, None
    
    def detect_and_crop_object(self, image, detection_method='large_center'):
        """检测并裁剪物体 - 尽可能保留大的区域
        
        Args:
            image: PIL Image
            detection_method: 'large_center' (大范围中心裁剪，保留85%区域)
        """
        try:
            # 使用大范围中心裁剪，保留图片的大部分区域（85%）
            width, height = image.size
            
            # 计算裁剪尺寸（取短边的85%，确保是正方形）
            crop_ratio = 0.90  # 保留90%的区域
            size = int(min(width, height) * crop_ratio)
            
            # 居中裁剪
            left = (width - size) // 2
            top = (height - size) // 2
            
            cropped = image.crop((left, top, left + size, top + size))
            bbox = (left, top, left + size, top + size)
            
            self.rm.log(f"Large area cropped, bbox: {bbox}, size: {size}x{size} (ratio: {crop_ratio:.0%})")
            return cropped, bbox
            
        except Exception as e:
            self.rm.log(f"Crop failed: {e}")
            return image, (0, 0, image.size[0], image.size[1])
    
    
    def classify_image(self, image_tensor):
        """分类图像"""
        try:
            with torch.no_grad():
                logits = self.classifier(image_tensor.to(self.device))
                probs = F.softmax(logits, dim=1)
                top5_prob, top5_class = torch.topk(probs, 5)
                
            results = {
                'top1_class': top5_class[0][0].item(),
                'top1_prob': top5_prob[0][0].item(),
                'top5_classes': top5_class[0].cpu().numpy().tolist(),
                'top5_probs': top5_prob[0].cpu().numpy().tolist()
            }
            
            self.rm.log(f"Classification result: class {results['top1_class']}, prob {results['top1_prob']:.3f}")
            return results
            
        except Exception as e:
            self.rm.log(f"Classification failed: {e}")
            return None
    
    def generate_physical_adversarial(self, image_tensor, original_class, 
                                     physical_optimization=True):
        """生成针对物理世界的对抗样本
        
        Args:
            image_tensor: 输入图像张量
            original_class: 原始类别
            physical_optimization: 是否进行物理世界优化
        """
        try:
            self.rm.log("Generating physical world adversarial example...")
            
            # 确保张量需要梯度
            image_tensor = image_tensor.to(self.device)
            image_tensor.requires_grad_(True)
            
            # 生成对抗样本
            if self.attacker is not None:
                result = self.attacker.run(image_tensor, torch.tensor([original_class]).to(self.device))
                
                # 处理返回值：可能是元组 (adv_tensor, step_preds) 或单个张量
                if isinstance(result, tuple):
                    adv_tensor, step_preds = result
                    self.rm.log(f"Attack completed with {len(step_preds)} steps")
                else:
                    adv_tensor = result
                    self.rm.log("Attack completed")
            else:
                self.rm.log("Attacker not available, using input image")
                adv_tensor = image_tensor
            
            # 物理世界优化
            if physical_optimization:
                adv_tensor = self._apply_physical_optimization(adv_tensor)
            
            self.rm.log("Adversarial example generated successfully")
            return adv_tensor
            
        except Exception as e:
            self.rm.log(f"Failed to generate adversarial example: {e}")
            import traceback
            self.rm.log(traceback.format_exc())
            return image_tensor
    
    def _apply_physical_optimization(self, adv_tensor):
        """应用物理世界优化
        
        优化内容：
        1. 颜色校正（适应打印机/显示器）
        2. 对比度增强
        3. 锐化（抵抗打印模糊）
        4. 去噪
        """
        try:
            # 转换为PIL图像
            adv_pil = self._tensor_to_pil(adv_tensor)
            
            # 1. 颜色校正 - 增加饱和度
            enhancer = ImageEnhance.Color(adv_pil)
            adv_pil = enhancer.enhance(1.1)
            
            # 2. 对比度增强
            enhancer = ImageEnhance.Contrast(adv_pil)
            adv_pil = enhancer.enhance(1.05)
            
            # 3. 锐化
            enhancer = ImageEnhance.Sharpness(adv_pil)
            adv_pil = enhancer.enhance(1.2)
            
            # 转换回张量
            optimized_tensor = self._pil_to_tensor(adv_pil).to(self.device)
            
            self.rm.log("Physical optimization applied")
            return optimized_tensor
            
        except Exception as e:
            self.rm.log(f"Physical optimization failed: {e}")
            return adv_tensor
    
    def _tensor_to_pil(self, tensor):
        """张量转PIL图像"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = torch.clamp(tensor, 0, 1)
        img_array = tensor.permute(1, 2, 0).cpu().detach().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _pil_to_tensor(self, pil_img):
        """PIL图像转张量"""
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return tensor
    
    def export_for_mobile(self, adv_tensor, original_image, classification_results,
                         output_dir, filename_prefix, bbox=None,
                         original_full_image=None, attack_config=None):
        """导出适合移动设备的格式 - 简化版，保存3张图片
        
        Args:
            adv_tensor: 对抗样本张量（224x224）
            original_image: 裁剪后的原始图像
            classification_results: 分类结果
            output_dir: 输出目录（使用 ResultManager 的 img_dir）
            filename_prefix: 文件名前缀
            bbox: 裁剪区域的边界框 (x, y, x+w, y+h)
            original_full_image: 完整的原始图片（未裁剪）
            attack_config: 攻击配置（用于文件名）
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            exported_files = []
            
            # 生成包含参数的文件名后缀
            if attack_config is not None:
                patch_h, patch_w = attack_config.get("patch_size", (0, 0))
                patch_k = attack_config.get("patch_k", 1)
                eps_val = int(attack_config.get("eps", 0) * 255)
                alpha_val = int(attack_config.get("alpha", 0) * 255)
                param_suffix = f"_p{patch_h}x{patch_w}x{patch_k}_e{eps_val}_a{alpha_val}"
            else:
                param_suffix = ""
            
            # 转换为PIL图像（224x224的对抗样本）
            adv_pil_small = self._tensor_to_pil(adv_tensor)
            
            # 将对抗样本放回原图
            if original_full_image is not None and bbox is not None:
                adv_pil_full = self._paste_adversarial_to_original(
                    adv_pil_small, original_full_image, bbox
                )
                self.rm.log(f"Adversarial patch pasted back to original image, size: {adv_pil_full.size}")
            else:
                adv_pil_full = adv_pil_small
                self.rm.log("Using cropped adversarial image (no full image restoration)")
            
            # 只保存PNG格式（高质量）
            # 1. 保存原始图片
            if original_full_image is not None:
                original_path = os.path.join(output_dir, f"{filename_prefix}_original.png")
                original_full_image.save(original_path, optimize=True)
                exported_files.append(original_path)
                self.rm.log(f"✓ Saved original: {original_path}")
            
            # 2. 保存对抗样本（完整尺寸，文件名包含参数）
            adversarial_path = os.path.join(output_dir, f"{filename_prefix}_adversarial{param_suffix}.png")
            adv_pil_full.save(adversarial_path, optimize=True)
            exported_files.append(adversarial_path)
            self.rm.log(f"✓ Saved adversarial: {adversarial_path}")
            
            # 3. 生成并保存对比图（文件名包含参数）
            if original_full_image is not None:
                comparison_path = os.path.join(output_dir, f"{filename_prefix}_comparison{param_suffix}.png")
                self._create_comparison_image(original_full_image, adv_pil_full, comparison_path)
                exported_files.append(comparison_path)
                self.rm.log(f"✓ Saved comparison: {comparison_path}")
            
            # 4. 保存元数据（包含攻击参数）
            metadata_path = os.path.join(output_dir, f"{filename_prefix}_info{param_suffix}.json")
            self._save_metadata(classification_results, metadata_path, bbox=bbox, attack_config=attack_config)
            exported_files.append(metadata_path)
            
            self.rm.log(f"✓ Exported {len(exported_files)} files (3 images + 1 metadata)")
            return exported_files
            
        except Exception as e:
            self.rm.log(f"Failed to export: {e}")
            return []
    
    def _paste_adversarial_to_original(self, adv_patch, original_full_image, bbox):
        """将对抗样本粘贴回原始图像的对应位置
        
        Args:
            adv_patch: 对抗样本（224x224）
            original_full_image: 完整的原始图像
            bbox: 裁剪区域 (x1, y1, x2, y2)
        
        Returns:
            完整尺寸的对抗图像
        """
        try:
            # 复制原始图像
            result_image = original_full_image.copy()
            
            # 计算裁剪区域的尺寸
            x1, y1, x2, y2 = bbox
            crop_width = x2 - x1
            crop_height = y2 - y1
            
            self.rm.log(f"Pasting adversarial patch:")
            self.rm.log(f"  - Original image size: {original_full_image.size}")
            self.rm.log(f"  - Adversarial patch size: {adv_patch.size}")
            self.rm.log(f"  - Target bbox: ({x1}, {y1}, {x2}, {y2})")
            self.rm.log(f"  - Target size: {crop_width}x{crop_height}")
            
            # 将对抗样本调整回裁剪区域的原始尺寸
            adv_resized = adv_patch.resize((crop_width, crop_height), Image.LANCZOS)
            self.rm.log(f"  - Resized adversarial to: {adv_resized.size}")
            
            # 粘贴到原图的对应位置
            result_image.paste(adv_resized, (x1, y1))
            
            self.rm.log(f"✓ Successfully pasted adversarial patch back to original position")
            
            return result_image
            
        except Exception as e:
            self.rm.log(f"✗ Failed to paste adversarial to original: {e}")
            import traceback
            self.rm.log(traceback.format_exc())
            return original_full_image
    
    def _create_comparison_image(self, original_pil, adversarial_pil, save_path):
        """创建对比图像 - 原图 vs 对抗样本并排显示
        
        Args:
            original_pil: 原始图片
            adversarial_pil: 对抗样本
            save_path: 保存路径
        """
        try:
            # 获取图片尺寸
            width, height = original_pil.size
            
            # 如果图片太大，缩小以便对比展示
            max_display_width = 800
            if width > max_display_width:
                scale = max_display_width / width
                new_width = max_display_width
                new_height = int(height * scale)
                orig_resized = original_pil.resize((new_width, new_height), Image.LANCZOS)
                adv_resized = adversarial_pil.resize((new_width, new_height), Image.LANCZOS)
            else:
                orig_resized = original_pil
                adv_resized = adversarial_pil
                new_width, new_height = width, height
            
            # 创建并排对比图（左右排列）
            gap = 20  # 两张图之间的间隙
            title_height = 40  # 标题高度
            
            comparison = Image.new('RGB', 
                                 (new_width * 2 + gap, new_height + title_height), 
                                 (255, 255, 255))
            
            # 粘贴两张图
            comparison.paste(orig_resized, (0, title_height))
            comparison.paste(adv_resized, (new_width + gap, title_height))
            
            # 添加标签
            draw = ImageDraw.Draw(comparison)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            if font:
                # 绘制标题
                draw.text((new_width//2 - 60, 5), "Original", fill=(0, 0, 0), font=font)
                draw.text((new_width + gap + new_width//2 - 80, 5), "Adversarial", fill=(255, 0, 0), font=font)
                
                # 绘制分隔线
                draw.line([(new_width + gap//2, title_height), 
                          (new_width + gap//2, new_height + title_height)], 
                         fill=(200, 200, 200), width=2)
            
            # 保存对比图
            comparison.save(save_path, quality=95, optimize=True)
            self.rm.log(f"Comparison image saved: {save_path}, size: {comparison.size}")
            
        except Exception as e:
            self.rm.log(f"Failed to create comparison image: {e}")
            import traceback
            self.rm.log(traceback.format_exc())
    
    def _save_metadata(self, classification_results, save_path, bbox=None, attack_config=None):
        """保存元数据"""
        try:
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'classification': classification_results,
                'model': 'ResNet50',
                'attack_method': 'SCAR'
            }
            
            # 添加bbox信息
            if bbox is not None:
                metadata['bbox'] = {
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                    'width': bbox[2] - bbox[0],
                    'height': bbox[3] - bbox[1]
                }
            
            # 添加攻击配置参数
            if attack_config is not None:
                metadata['attack_params'] = {
                    'patch_size': attack_config.get('patch_size'),
                    'patch_k': attack_config.get('patch_k'),
                    'eps': attack_config.get('eps'),
                    'alpha': attack_config.get('alpha'),
                    'steps': attack_config.get('steps'),
                    'blend_width': attack_config.get('blend_width'),
                }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.rm.log(f"Metadata saved: {save_path}")
            
        except Exception as e:
            self.rm.log(f"Failed to save metadata: {e}")
    


def process_photo_batch(input_dir, output_dir, detection_method='large_center', 
                       physical_optimization=True, attack_config=None, test_name=None):
    """批量处理照片 - 简化版
    
    Args:
        input_dir: 输入照片目录
        output_dir: 输出目录（会被 ResultManager.img_dir 覆盖）
        detection_method: 裁剪方法（默认large_center，保留90%区域）
        physical_optimization: 是否进行物理优化
        attack_config: 攻击配置字典（可选）
        test_name: 测试名称（用于区分不同配置的结果）
    """
    rm = ResultManager.get_instance()
    rm.set_experiment("PhysicalWorldAttack")
    
    # 如果提供了test_name，使用它；否则使用默认名称
    if test_name is None:
        test_name = "photo_batch_processing"
    rm.set_test(test_name)
    
    # 使用 ResultManager 的 img_dir 作为输出目录
    output_dir = rm.img_dir
    rm.log(f"Images will be saved to: {output_dir}")
    
    # 初始化生成器（传入attack_config）
    generator = PhysicalWorldAttackGenerator(attack_config=attack_config)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    photo_files = []
    for ext in image_extensions:
        photo_files.extend(Path(input_dir).glob(f'*{ext}'))
        photo_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    rm.log(f"Found {len(photo_files)} photos in {input_dir}")
    
    results = {
        'total_photos': len(photo_files),
        'successful_attacks': 0,
        'failed_attacks': 0,
        'photo_results': []
    }
    
    # 处理每张照片
    for idx, photo_path in enumerate(photo_files):
        rm.log(f"\n=== Processing photo {idx+1}/{len(photo_files)}: {photo_path.name} ===")
        
        try:
            # 1. 加载照片（保存完整原图）
            original_full_image, original_size = generator.load_real_world_photo(str(photo_path))
            if original_full_image is None:
                results['failed_attacks'] += 1
                continue
            
            rm.log(f"Original image size: {original_size}")
            
            # 2. 检测并裁剪物体
            cropped_image, bbox = generator.detect_and_crop_object(
                original_full_image, detection_method
            )
            
            rm.log(f"Cropped region: bbox={bbox}, cropped_size={cropped_image.size}")
            
            # 3. 预处理（直接resize到224x224，覆盖整个裁剪区域）
            image_tensor = generator.preprocess(cropped_image).unsqueeze(0)
            rm.log(f"Preprocessed tensor shape: {image_tensor.shape}")
            
            # 4. 分类原始图像
            original_results = generator.classify_image(image_tensor)
            if original_results is None:
                results['failed_attacks'] += 1
                continue
            
            rm.log(f"Original classification: class {original_results['top1_class']}, "
                  f"confidence {original_results['top1_prob']:.2%}")
            
            # 5. 生成对抗样本（224x224）
            adv_tensor = generator.generate_physical_adversarial(
                image_tensor, 
                original_results['top1_class'],
                physical_optimization=physical_optimization
            )
            
            # 6. 分类对抗样本
            adv_results = generator.classify_image(adv_tensor)
            if adv_results is None:
                results['failed_attacks'] += 1
                continue
            
            rm.log(f"Adversarial classification: class {adv_results['top1_class']}, "
                  f"confidence {adv_results['top1_prob']:.2%}")
            
            # 判断攻击是否成功
            attack_success = (original_results['top1_class'] != adv_results['top1_class'])
            if attack_success:
                results['successful_attacks'] += 1
            else:
                results['failed_attacks'] += 1
            
            # 7. 导出（包含完整原图和bbox信息，以及攻击配置）
            filename_prefix = f"{idx+1:03d}_{photo_path.stem}"
            # 获取攻击器配置
            attack_config = None
            if hasattr(generator.attacker, 'config'):
                attack_config = generator.attacker.config
            
            exported_files = generator.export_for_mobile(
                adv_tensor,
                cropped_image,
                adv_results,
                output_dir,
                filename_prefix,
                bbox=bbox,
                original_full_image=original_full_image,
                attack_config=attack_config
            )
            
            # 记录结果
            photo_result = {
                'photo_name': photo_path.name,
                'original_size': original_size,
                'original_class': original_results['top1_class'],
                'original_confidence': original_results['top1_prob'],
                'adversarial_class': adv_results['top1_class'],
                'adversarial_confidence': adv_results['top1_prob'],
                'attack_success': attack_success,
                'bbox': bbox,
                'exported_files': exported_files
            }
            results['photo_results'].append(photo_result)
            
            rm.log(f"Photo processed: {'SUCCESS' if attack_success else 'FAILED'}")
            
        except Exception as e:
            rm.log(f"Failed to process photo {photo_path.name}: {e}")
            results['failed_attacks'] += 1
            continue
    
    # 计算统计
    if results['total_photos'] > 0:
        results['attack_success_rate'] = results['successful_attacks'] / results['total_photos']
    else:
        results['attack_success_rate'] = 0.0
    
    # 保存结果
    results_path = os.path.join(output_dir, 'attack_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成报告
    generate_attack_report(results, output_dir)
    
    rm.log("\n=== Batch Processing Complete ===")
    rm.log(f"Total photos: {results['total_photos']}")
    rm.log(f"Successful attacks: {results['successful_attacks']}")
    rm.log(f"Failed attacks: {results['failed_attacks']}")
    rm.log(f"Attack success rate: {results['attack_success_rate']:.2%}")
    rm.log(f"Results saved to: {output_dir}")
    
    return results


def generate_attack_report(results, output_dir):
    """生成攻击报告"""
    try:
        report_path = os.path.join(output_dir, 'attack_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Physical World Attack Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Photos:** {results['total_photos']}\n")
            f.write(f"- **Successful Attacks:** {results['successful_attacks']}\n")
            f.write(f"- **Failed Attacks:** {results['failed_attacks']}\n")
            f.write(f"- **Attack Success Rate:** {results['attack_success_rate']:.2%}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| Photo | Original Class | Original Conf | Adversarial Class | Adversarial Conf | Attack Success |\n")
            f.write("|-------|---------------|---------------|-------------------|------------------|----------------|\n")
            
            for photo_result in results['photo_results']:
                f.write(f"| {photo_result['photo_name']} | ")
                f.write(f"{photo_result['original_class']} | ")
                f.write(f"{photo_result['original_confidence']:.2%} | ")
                f.write(f"{photo_result['adversarial_class']} | ")
                f.write(f"{photo_result['adversarial_confidence']:.2%} | ")
                f.write(f"{'✅' if photo_result['attack_success'] else '❌'} |\n")
            
            f.write("\n## Output Files (Simplified)\n\n")
            f.write("每张照片生成4个文件：\n\n")
            f.write("1. `*_original.png` - 原始图片（保持原始尺寸）\n")
            f.write("2. `*_adversarial.png` - 对抗样本（保持原始尺寸，90%区域被攻击）⭐\n")
            f.write("3. `*_comparison.png` - 对比图（原图 vs 对抗样本）\n")
            f.write("4. `*_info.json` - 元数据（分类结果、bbox信息）\n\n")
            
            f.write("## Usage Instructions\n\n")
            f.write("### 手机测试步骤\n")
            f.write("1. 将 `*_adversarial.png` 传输到手机\n")
            f.write("2. 在手机相册中打开图片\n")
            f.write("3. 使用 Google Lens、百度识图等识别\n")
            f.write("4. 对比 `*_original.png` 和 `*_adversarial.png` 的识别结果\n\n")
            
            f.write("### 特点\n")
            f.write("- 保持原始图片尺寸\n")
            f.write("- 90%的图片区域被攻击（尽可能大范围）\n")
            f.write("- 只有边缘10%保持原样\n")
            f.write("- PNG格式，高质量无损\n\n")
            
            f.write("## Tips\n\n")
            f.write("- 确保良好的光照条件\n")
            f.write("- 避免屏幕眩光和反射\n")
            f.write("- 可以打印后用手机拍照测试\n")
        
        print(f"Report generated: {report_path}")
        
    except Exception as e:
        print(f"Failed to generate report: {e}")


def run_physical_world_attack_experiment(input_photos_dir="input_photos", 
                                        detection_method="large_center",
                                        physical_optimization=True,
                                        attack_config=None,
                                        test_name=None):
    """
    运行物理世界攻击实验 - 简化版
    
    Args:
        input_photos_dir: 输入照片目录（用户拍摄的照片）
        detection_method: 裁剪方法（默认large_center，保留90%区域）
        physical_optimization: 是否进行物理世界优化
        attack_config: 攻击配置字典（可选），包含patch_size, patch_k, eps, alpha等参数
        test_name: 测试名称（用于区分不同配置，建议包含配置信息）
    
    输出：
        每张照片生成4个文件（保存在 ResultManager.img_dir）：
        - *_original.png: 原始图片
        - *_adversarial_p{size}x{k}_e{eps}_a{alpha}.png: 对抗样本（相同尺寸）
        - *_comparison_p{size}x{k}_e{eps}_a{alpha}.png: 对比图（原图 vs 对抗样本）
        - *_info_p{size}x{k}_e{eps}_a{alpha}.json: 元数据
    """
    rm = ResultManager.get_instance()
    rm.set_experiment("PhysicalWorldAttack")
    
    # 生成测试名称
    if test_name is None:
        if attack_config is not None:
            # 根据配置生成测试名称
            patch_h, patch_w = attack_config.get("patch_size", (0, 0))
            patch_k = attack_config.get("patch_k", 0)
            eps_val = int(attack_config.get("eps", 0) * 255)
            alpha_val = int(attack_config.get("alpha", 0) * 255)
            test_name = f"config_p{patch_h}x{patch_w}x{patch_k}_e{eps_val}_a{alpha_val}"
        else:
            test_name = "default_config"
    
    rm.set_test(test_name)
    
    rm.log("=== Starting Physical World Attack Experiment ===")
    rm.log(f"Test name: {test_name}")
    rm.log(f"Input directory: {input_photos_dir}")
    rm.log(f"Detection method: {detection_method}")
    rm.log(f"Physical optimization: {physical_optimization}")
    rm.log(f"Output directory: {rm.img_dir}")
    
    if attack_config is not None:
        rm.log(f"Custom attack config: patch_size={attack_config.get('patch_size')}, "
               f"patch_k={attack_config.get('patch_k')}, "
               f"eps={attack_config.get('eps'):.4f}, "
               f"alpha={attack_config.get('alpha'):.4f}")
    else:
        rm.log("Using default attack config")
    
    # 检查输入目录
    if not os.path.exists(input_photos_dir):
        rm.log(f"Input directory not found: {input_photos_dir}")
        rm.log("Creating example directory...")
        os.makedirs(input_photos_dir, exist_ok=True)
        rm.log(f"Please place your photos in: {input_photos_dir}")
        rm.log("Supported formats: .jpg, .jpeg, .png, .bmp")
        return None
    
    # 处理照片批次（output_dir 会被 process_photo_batch 内部用 rm.img_dir 替换）
    results = process_photo_batch(
        input_dir=input_photos_dir,
        output_dir=None,  # 不使用，内部会用 rm.img_dir
        detection_method=detection_method,
        physical_optimization=physical_optimization,
        attack_config=attack_config,
        test_name=test_name
    )
    
    rm.log("\n=== Physical World Attack Experiment Complete ===")
    rm.log(f"Images saved to: {rm.img_dir}")
    rm.log(f"Logs saved to: {rm.log_dir}")
    rm.log(f"Transfer the *_adversarial.png images to your mobile device for testing!")
    
    return results


def batch_test_multiple_configs(input_photos_dir="input_photos"):
    """
    批量测试多个攻击配置
    
    示例：测试不同patch尺寸、数量、扰动强度的组合
    """
    rm = ResultManager.get_instance()
    #rm.set_experiment("PhysicalWorldAttack")
    rm.set_test("batch_test_multiple_configs")
    # 定义多个配置进行测试
    configs = [
        # 配置1: 小patch + 多数量
        {
            "name": "small_patch_many",
            "config": {
                "patch_size": (48, 48),
                "patch_k": 3,
                "patch_type": "square",
                "steps": 1000,
                "eps": 16/255,
                "alpha": 8/255,
                "blend_width": 6,
                "targeted_attack": False,
                "cam_model_name": "resnet50",
                "cam_thresh_ratio": 0.5,
                "mask_thresh_ratio": 0.3,
                "debug": False,
                "save_imgs": False,
                "save_attack_process": False,
            }
        },
 
        {
            "name": "small_patch_many_with_5patches",
            "config": {
                "patch_size": (48, 48),
                "patch_k": 5,
                "patch_type": "square",
                "steps": 1000,
                "eps": 16/255,
                "alpha": 8/255,
                "blend_width": 6,
                "targeted_attack": False,
                "cam_model_name": "resnet50",
                "cam_thresh_ratio": 0.5,
                "mask_thresh_ratio": 0.3,
                "debug": False,
                "save_imgs": False,
                "save_attack_process": False,
            }
        },
        {
            "name": "medium_patch_many_with_3patches",
            "config": {
                "patch_size": (64, 64),
                "patch_k": 3,
                "patch_type": "square",
                "steps": 1000,
                "eps": 16/255,
                "alpha": 8/255,
                "blend_width": 6,
                "targeted_attack": False,
                "cam_model_name": "resnet50",
                "cam_thresh_ratio": 0.5,
                "mask_thresh_ratio": 0.3,
                "debug": False,
                "save_imgs": False,
                "save_attack_process": False,
            }
        },
        {
            "name": "medium_patch_many_with_5patches",
            "config": {
                "patch_size": (64, 64),
                "patch_k": 5,
                "patch_type": "square",
                "steps": 1000,
                "eps": 16/255,
                "alpha": 8/255,
                "blend_width": 6,
                "targeted_attack": False,
                "cam_model_name": "resnet50",
                "cam_thresh_ratio": 0.5,
                "mask_thresh_ratio": 0.3,
                "debug": False,
                "save_imgs": False,
                "save_attack_process": False,
            }
        },
        # 配置2: 大patch + 少数量
        {
            "name": "large_patch_few",
            "config": {
                "patch_size": (120, 120),
                "patch_k": 3,
                "patch_type": "square",
                "steps": 1000,
                "eps": 16/255,
                "alpha": 8/255,
                "blend_width": 6,
                "targeted_attack": False,
                "cam_model_name": "resnet50",
                "cam_thresh_ratio": 0.5,
                "mask_thresh_ratio": 0.3,
                "debug": False,
                "save_imgs": False,
                "save_attack_process": False,
            }
        },
        # 配置3: 强扰动
        {
            "name": "strong_perturbation",
            "config": {
                "patch_size": (120, 120),
                "patch_k": 5,
                "patch_type": "square",
                "steps": 1000,
                "eps": 32/255,  # 更大的扰动
                "alpha": 16/255,
                "blend_width": 6,
                "targeted_attack": False,
                "cam_model_name": "resnet50",
                "cam_thresh_ratio": 0.5,
                "mask_thresh_ratio": 0.3,
                "debug": False,
                "save_imgs": False,
                "save_attack_process": False,
            }
        },
        {
            "name": "small_perturbation",
            "config": {
                "patch_size": (64, 64),
                "patch_k": 3,
                "patch_type": "square",
                "steps": 1000,
                "eps": 6/255,  # 更大的扰动
                "alpha": 16/255,
                "blend_width": 6,
                "targeted_attack": False,
                "cam_model_name": "resnet50",
                "cam_thresh_ratio": 0.5,
                "mask_thresh_ratio": 0.3,
                "debug": False,
                "save_imgs": False,
                "save_attack_process": False,
            }
        },
        {
            "name": "small_perturbation_with_5patches",
            "config": {
                "patch_size": (48, 48),
                "patch_k": 5,
                "patch_type": "square",
                "steps": 1000,
                "eps": 6/255,  # 更大的扰动
                "alpha": 16/255,
                "blend_width": 6,
                "targeted_attack": False,
                "cam_model_name": "resnet50",
                "cam_thresh_ratio": 0.5,
                "mask_thresh_ratio": 0.3,
                "debug": False,
                "save_imgs": False,
                "save_attack_process": False,
            }
        },
       
    ]
    
    all_results = []
    
    for i, cfg in enumerate(configs):
        rm.log(f"\n{'='*60}")
        rm.log(f"Testing Configuration {i+1}/{len(configs)}: {cfg['name']}")
        rm.log(f"{'='*60}")
        
        # 为每个配置使用不同的test_name，避免覆盖
        results = run_physical_world_attack_experiment(
            input_photos_dir=input_photos_dir,
            detection_method="large_center",
            physical_optimization=True,
            attack_config=cfg['config'],
            test_name=cfg['name']  # 使用配置名称作为test_name
        )
        
        all_results.append({
            'config_name': cfg['name'],
            'config': cfg['config'],
            'results': results
        })
    
    rm.log(f"\n{'='*60}")
    rm.log("All configurations tested!")
    rm.log(f"{'='*60}")
    
    # 打印汇总
    for item in all_results:
        if item['results'] is not None:
            rm.log(f"\n{item['config_name']}:")
            rm.log(f"  Attack Success Rate: {item['results'].get('attack_success_rate', 0):.2%}")
    
    return all_results


if __name__ == '__main__':
    """
    简化版物理世界攻击实验
    
    使用方法：
    1. 将照片放入 input_photos/ 目录
    2. 运行此脚本
    3. 查看 ResultManager.img_dir 目录（会自动显示路径）
    
    每张照片生成：
    - *_original.png: 原始图片
    - *_adversarial_p{size}x{k}_e{eps}_a{alpha}.png: 对抗样本⭐
    - *_comparison_p{size}x{k}_e{eps}_a{alpha}.png: 对比图
    - *_info_p{size}x{k}_e{eps}_a{alpha}.json: 元数据
    
    将 *_adversarial.png 传到手机测试！
    """
    
    # 方式1: 使用默认配置
    results = run_physical_world_attack_experiment(
        input_photos_dir="input_photos",
        detection_method="large_center",
        physical_optimization=True
    )
    
    # 方式2: 使用自定义配置
    # custom_config = {
    #     "patch_size": (80, 80),
    #     "patch_k": 7,
    #     "eps": 20/255,
    #     "alpha": 10/255,
    #     ... 其他参数
    # }
    # results = run_physical_world_attack_experiment(
    #     input_photos_dir="input_photos",
    #     attack_config=custom_config
    # )
    
    # 方式3: 批量测试多个配置
    # all_results = batch_test_multiple_configs("input_photos")

