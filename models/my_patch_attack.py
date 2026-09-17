import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
from datetime import datetime

from utils.patch_selector import PatchSelectorBatch
from utils.cam_generator import CAMGenerator
from utils.color_extractor import ColorExtractor
from models.StableFusion.patch_generator import PatchGenerator
from utils.patch_blender import PatchBlender
from models.patch_optimizer import PatchOptimizer
from models.Mask2Former.mask2former import SemanticSegmentor
from utils.result_saver import ResultManager

DEFAULT_CONFIG = {
    "patch_size": (24, 24),
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

class MyPatchAttack_Eva:
    """评估版本的MyPatchAttack，返回对抗样本和每步预测结果"""
    
    def __init__(self, model, device='cuda', config=None, debug=False, save_imgs=False, save_attack_process=False):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.debug = debug
        self.save_imgs = save_imgs
        self.save_attack_process = save_attack_process  # 新增：是否保存攻击过程

        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._init_modules()
        self._init_logger()
        self._init_timestamp()
        self.target = self.config["targeted_attack"]
        if self.target:
            self.rm.log("[Targeted Attack] 启动目标攻击模式")
        else:
            self.rm.log("[Non-targeted Attack] 启动非目标攻击模式")
        
    def _init_modules(self):
        self.segmentor = SemanticSegmentor()
        self.patch_gen = PatchGenerator(device=self.device)
        self.color_extractor = ColorExtractor()
        self.patch_blender = PatchBlender(blend_width=self.config["blend_width"])
        self.optimizer = PatchOptimizer(device=self.device,steps=self.config["steps"],eps=self.config["eps"], alpha=self.config["alpha"])
        self.cam_gen = CAMGenerator(model_name=self.config["cam_model_name"], device=str(self.device))

    def _init_logger(self):
        self.rm = ResultManager.get_instance() 
        if self.rm:
            if self.debug:
                self.rm.log("[Debug Mode] 启动MyPatchAttack调试模式")

    def _init_timestamp(self):
        self._timestamp = datetime.now().strftime("%Y%m%dT%H%M%S") if self.save_imgs else None

    def _log(self, msg):
        if self.debug and self.rm:
            self.rm.log(f"[Debug] {msg}")

    def _save_img(self, vis_func, *args, **kwargs):
        if self.save_imgs:
            vis_func(*args, save_dir=self.rm.img_dir, timestamp_str=self._timestamp, show=False, **kwargs)

    def _save_attack_process(self, step_name, images_dict, sample_idx=0):
        """保存攻击过程的各个步骤图片
        
        Args:
            step_name: 步骤名称，如 'original', 'segmentation', 'cam', 'patches', 'blended', 'adversarial'
            images_dict: 包含图片的字典，键为图片名称，值为图片tensor或PIL图像
            sample_idx: 样本索引
        """
        if not self.save_attack_process:
            return
            
        import os
        from PIL import Image
        import torchvision.utils as vutils
        
        # 创建攻击过程保存目录
        process_dir = os.path.join(self.rm.img_dir, "attack_process")
        os.makedirs(process_dir, exist_ok=True)
        
        # 为每个样本创建子目录
        sample_dir = os.path.join(process_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存每个步骤的图片
        for img_name, img in images_dict.items():
            if img is None:
                continue
                
            # 确保图片是tensor格式
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            elif isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
                
            # 确保是3D tensor [C, H, W]
            if img.dim() == 4:
                img = img[0]  # 取第一个样本
            elif img.dim() == 2:
                img = img.unsqueeze(0).repeat(3, 1, 1)  # 灰度图转RGB
                
            # 确保值在[0,1]范围内
            if img.max() > 1.0:
                img = img / 255.0
                
            # 保存图片
            filename = f"{step_name}_{img_name}.png"
            filepath = os.path.join(sample_dir, filename)
            vutils.save_image(img, filepath)
            
        if self.debug:
            self._log(f"保存攻击过程图片: {step_name} -> {sample_dir}")


        """保存攻击过程的各个步骤图片
        
        Args:
            step_name: 步骤名称，如 'original', 'segmentation', 'cam', 'patches', 'blended', 'adversarial'
            images_dict: 包含图片的字典，键为图片名称，值为图片tensor或PIL图像
            sample_idx: 样本索引
        """
        if not self.save_attack_process:
            return
            
        import os
        from PIL import Image
        import torchvision.utils as vutils
        
        # 创建攻击过程保存目录
        process_dir = os.path.join(self.rm.img_dir, "attack_process")
        os.makedirs(process_dir, exist_ok=True)
        
        # 为每个样本创建子目录
        sample_dir = os.path.join(process_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存每个步骤的图片
        for img_name, img in images_dict.items():
            if img is None:
                continue
                
            # 确保图片是tensor格式
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            elif isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
                
            # 确保是3D tensor [C, H, W]
            if img.dim() == 4:
                img = img[0]  # 取第一个样本
            elif img.dim() == 2:
                img = img.unsqueeze(0).repeat(3, 1, 1)  # 灰度图转RGB
                
            # 确保值在[0,1]范围内
            if img.max() > 1.0:
                img = img / 255.0
                
            # 保存图片
            filename = f"{step_name}_{img_name}.png"
            filepath = os.path.join(sample_dir, filename)
            vutils.save_image(img, filepath)
            
        if self.debug:
            self._log(f"保存攻击过程图片: {step_name} -> {sample_dir}")

    def run(self, x, y, target=None, sample_idx=0):
        """
        评估版本，返回 adv 和 step_preds_list（每步预测标签），与 baseline 评估接口一致
        """
        attack_id = target if self.target and target is not None else y
        
        # 保存原始图片
        if self.save_attack_process:
            self._save_attack_process('original', {'input': x[0]}, sample_idx=sample_idx)
        
        pil_imgs = self._to_pil(x)
        color_names = self._extract_colors(pil_imgs)
        
        # 语义分割步骤
        seg_maps, region_names, region_masks = self._segment_semantics(x)
        if self.save_attack_process:
            # 保存分割结果
            seg_images = {}
            for i, (seg_map, mask) in enumerate(zip(seg_maps, region_masks)):
                if i == 0:  # 只保存第一个样本
                    # 保存region_mask（tensor格式）
                    if hasattr(mask, 'cpu'):
                        seg_images[f'region_mask_{i}'] = mask.cpu()
                    else:
                        seg_images[f'region_mask_{i}'] = torch.tensor(mask)
                    
                    # 生成可视化的语义分割图片
                    try:
                        # 使用Mask2Former的可视化方法
                        vis_img = self.segmentor.visualize_panoptic_and_largest_stuff(
                            x[i:i+1],  # 单个样本
                            [seg_map],  # 单个分割图
                            [mask],     # 单个掩码
                            [region_names[i]]  # 单个区域名称
                        )
                        if vis_img is not None:
                            seg_images[f'visualization_{i}'] = vis_img
                    except Exception as e:
                        if self.debug:
                            self._log(f"语义分割可视化失败: {e}")
                    
                    # 创建简单的掩码可视化
                    try:
                        # 将掩码转换为彩色图片
                        mask_tensor = mask if hasattr(mask, 'cpu') else torch.tensor(mask)
                        if mask_tensor.dim() == 2:
                            # 单通道掩码，转换为3通道彩色图片
                            colored_mask = torch.zeros(3, mask_tensor.shape[0], mask_tensor.shape[1])
                            colored_mask[0] = mask_tensor * 0.8  # 红色通道
                            colored_mask[1] = mask_tensor * 0.2  # 绿色通道
                            colored_mask[2] = mask_tensor * 0.1  # 蓝色通道
                            seg_images[f'colored_mask_{i}'] = colored_mask
                    except Exception as e:
                        if self.debug:
                            self._log(f"掩码可视化失败: {e}")
                    
                    # 尝试从seg_map中提取可视化信息
                    if isinstance(seg_map, dict):
                        # 如果是字典，尝试提取panoptic_seg
                        if 'panoptic_seg' in seg_map:
                            panoptic_seg = seg_map['panoptic_seg']
                            if hasattr(panoptic_seg, 'cpu'):
                                seg_images[f'panoptic_seg_{i}'] = panoptic_seg.cpu()
                            else:
                                seg_images[f'panoptic_seg_{i}'] = torch.tensor(panoptic_seg)
                        
                        # 尝试提取sem_seg
                        if 'sem_seg' in seg_map:
                            sem_seg = seg_map['sem_seg']
                            if hasattr(sem_seg, 'cpu'):
                                seg_images[f'sem_seg_{i}'] = sem_seg.cpu()
                            else:
                                seg_images[f'sem_seg_{i}'] = torch.tensor(sem_seg)
            
            # 只有当有有效图片时才保存
            if seg_images:
                self._save_attack_process('segmentation', seg_images, sample_idx=sample_idx)
        
        # CAM生成步骤
        if self.target:
            cam_list, cam_masks = self._generate_cam_and_masks(x, label=attack_id)
        else:
            cam_list, cam_masks = self._generate_cam_and_masks(x)
        
        if self.save_attack_process:
            # 保存CAM结果
            cam_images = {}
            for i, (cam, mask) in enumerate(zip(cam_list, cam_masks)):
                if i == 0:  # 只保存第一个样本
                    # 确保cam和mask是tensor格式
                    if hasattr(cam, 'cpu'):
                        cam_images[f'cam_{i}'] = cam.cpu()
                    else:
                        cam_images[f'cam_{i}'] = torch.tensor(cam)
                    
                    if hasattr(mask, 'cpu'):
                        cam_images[f'cam_mask_{i}'] = mask.cpu()
                    else:
                        cam_images[f'cam_mask_{i}'] = torch.tensor(mask)
            self._save_attack_process('cam', cam_images, sample_idx=sample_idx)
        
        # 补丁位置选择
        coords_list = self._select_patch_positions(x, region_masks, cam_masks)
        
        # 补丁生成步骤
        if self.target:
            patch_tensor = self._generate_patches(region_names, color_names, attack_id)
        else:
            patch_tensor = self._generate_patches(region_names, color_names)
        
        if self.save_attack_process:
            # 保存生成的补丁
            patch_images = {}
            for i in range(min(3, patch_tensor.size(1))):  # 保存前3个补丁
                patch_images[f'patch_{i}'] = patch_tensor[0, i]
            self._save_attack_process('patches', patch_images, sample_idx=sample_idx)
        
        # 补丁融合步骤
        patched_img = self._blend_patches(x, patch_tensor, coords_list)
        if self.save_attack_process:
            self._save_attack_process('blended', {'patched': patched_img[0]}, sample_idx=sample_idx)
        
        # 优化步骤
        if self.target:
            x_adv, step_preds_list = self._optimize(x, patched_img, coords_list, attack_id)
        else:
            x_adv, step_preds_list = self._optimize(x, patched_img, coords_list, y)
        
        if self.save_attack_process:
            self._save_attack_process('adversarial', {'final': x_adv[0]}, sample_idx=sample_idx)

        # 添加原始图像的预测结果到step_preds_list的开头
        with torch.no_grad():
            preds_orig = torch.argmax(self.model(x), dim=1)
            step_preds_list.insert(0, preds_orig.detach().cpu())

        return x_adv, step_preds_list

    def _to_pil(self, x):
        return [to_pil_image(x[i].cpu()) for i in range(x.shape[0])]

    def _extract_colors(self, pil_imgs):
        color_names = self.color_extractor.get_top_color_names_batch(pil_imgs, topk=3)
        if self.debug:
            self._log(f"Dominant colors: {color_names}")
        return color_names

    def _segment_semantics(self, x):
        seg_maps = self.segmentor.segment_batch_panoptic(x)
        names, masks = self.segmentor.get_largest_stuff_region_batch(seg_maps)
        if self.save_imgs:
            self._save_img(self.segmentor.visualize_panoptic_and_largest_stuff, x, seg_maps, masks, names)
        if self.debug:
            self._log(f"Segmented regions: {names}")
        return seg_maps, names, masks

    def _generate_cam_and_masks(self, x, label=None):
        cams = self.cam_gen.generate_cam_batch(x, target_classes=label)
        cam_masks = self.cam_gen.cam_to_mask_batch(cams, percentile=self.config["cam_percentile"])
        if self.save_imgs:
            self._save_img(self.cam_gen.save_cam_composite_4in1_batch, images_tensor=x, cams=cams, masks_batch=cam_masks)
        if self.debug:
            self._log(f"Generated CAMs for {len(cams)} images with masks.")
        return cams, cam_masks

    def _select_patch_positions(self, x, region_masks, cam_masks):
        selector = PatchSelectorBatch(region_masks, x, self.config["patch_size"], cam_masks,self.config["cam_thresh_ratio"], self.config["mask_thresh_ratio"])
        coords = selector.get_best_patch_positions(k=self.config["patch_k"], stride_ratio=0.5)
        if self.save_imgs:    
            self._save_img(selector.visualize_patch_selection, x, coords)
        return coords

    def _generate_patches(self, region_names, color_names, target_classes=None):
        if self.target:
            patches, prompts = self.patch_gen.generate_patch_by_region_batch(
                region_names=region_names,
                color_names_list=color_names,
                patch_size=self.config["patch_size"],
                per_region_k=self.config["patch_k"],
                target_classes=target_classes
            )
        else:
            patches, prompts = self.patch_gen.generate_patch_by_region_batch(
                region_names=region_names,
                color_names_list=color_names,
                patch_size=self.config["patch_size"],
                per_region_k=self.config["patch_k"]
            )
        if self.debug:
            self._log(f"Prompts: {prompts}")
        if self.save_imgs:
            self._save_img(self.patch_gen.visualize_patch_batch_with_prompts, patch_batch=patches, prompts_batch=prompts)
        to_tensor = transforms.ToTensor()
        return torch.stack([torch.stack([to_tensor(p) for p in plist]) for plist in patches]).to(self.device)

    def _blend_patches(self, x, patch_tensor, coords):
        if self.debug:
            self._log(f"Blending patches into images at coordinates: {coords}")
        blended = self.patch_blender.blend_patch_tensor_batch(
            backgrounds=x,
            patches=patch_tensor,
            positions=coords,
            blend_width=self.config["blend_width"]
        )
        if self.debug:
            self._log(f"Blended patches into images.")
        if self.save_imgs:
            self._save_img(self.patch_blender.visualize_patch_blending_batch, x, patch_tensor, coords, blended)
        
        return blended

    def _optimize(self, x, patched_x, coords, y_or_target):
        targeted = self.config["targeted_attack"]
        if self.debug:
            self._log(f"Optimizing adversarial images with coordinates: {coords} and target: {y_or_target}")
        # 使用return_step_preds=True来获取每步预测结果
        result = self.optimizer.optimize_batch(x, patched_x, coords, y_or_target, targeted=targeted, return_step_preds=True)
        if isinstance(result, tuple):
            x_adv, step_preds_list = result
        else:
            x_adv = result
            step_preds_list = []
            
        if self.debug:
            self._log(f"Optimized adversarial images with targeted={targeted}.")
        if self.save_imgs:
            self._save_img(self.optimizer.visualize_adversarial_comparison, x=x, x_adv=x_adv)
        return x_adv, step_preds_list


class MyPatchAttack:
    """基础版本的MyPatchAttack，只返回对抗样本"""
    
    def __init__(self, model, device='cuda', config=None, debug=False, save_imgs=False, save_attack_process=False):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.debug = debug
        self.save_imgs = save_imgs
        self.save_attack_process = save_attack_process  # 新增：是否保存攻击过程

        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._init_modules()
        self._init_logger()
        self._init_timestamp()
        self.target = self.config["targeted_attack"]
        if self.target:
            self.rm.log("[Targeted Attack] 启动目标攻击模式")
        else:
            self.rm.log("[Non-targeted Attack] 启动非目标攻击模式")
        
    def _init_modules(self):
        self.segmentor = SemanticSegmentor()
        self.patch_gen = PatchGenerator(device=self.device)
        self.color_extractor = ColorExtractor()
        self.patch_blender = PatchBlender(blend_width=self.config["blend_width"])
        self.optimizer = PatchOptimizer(device=self.device, steps=self.config["steps"], eps=self.config["eps"], alpha=self.config["alpha"], model=self.model)
        self.cam_gen = CAMGenerator(model_name=self.config["cam_model_name"], device=str(self.device))

    def _init_logger(self):
        self.rm = ResultManager.get_instance() 
        if self.rm:
            if self.debug:
                self.rm.log("[Debug Mode] 启动MyPatchAttack调试模式")

    def _init_timestamp(self):
        self._timestamp = datetime.now().strftime("%Y%m%dT%H%M%S") if self.save_imgs else None

    def _log(self, msg):
        if self.debug and self.rm:
            self.rm.log(f"[Debug] {msg}")

    def _save_img(self, vis_func, *args, **kwargs):
        if self.save_imgs:
            vis_func(*args, save_dir=self.rm.img_dir, timestamp_str=self._timestamp, show=False, **kwargs)

    def _save_attack_process(self, step_name, images_dict, sample_idx=0):
        """保存攻击过程的各个步骤图片
        
        Args:
            step_name: 步骤名称，如 'original', 'segmentation', 'cam', 'patches', 'blended', 'adversarial'
            images_dict: 包含图片的字典，键为图片名称，值为图片tensor或PIL图像
            sample_idx: 样本索引
        """
        if not self.save_attack_process:
            return
            
        import os
        from PIL import Image
        import torchvision.utils as vutils
        
        # 创建攻击过程保存目录
        process_dir = os.path.join(self.rm.img_dir, "attack_process")
        os.makedirs(process_dir, exist_ok=True)
        
        # 为每个样本创建子目录
        sample_dir = os.path.join(process_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存每个步骤的图片
        for img_name, img in images_dict.items():
            if img is None:
                continue
                
            # 确保图片是tensor格式
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            elif isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
                
            # 确保是3D tensor [C, H, W]
            if img.dim() == 4:
                img = img[0]  # 取第一个样本
            elif img.dim() == 2:
                img = img.unsqueeze(0).repeat(3, 1, 1)  # 灰度图转RGB
            
            # 处理数据类型和值范围
            if img.dtype != torch.float32:
                img = img.float()
                
            # 确保值在[0,1]范围内
            if img.max() > 1.0:
                img = img / 255.0
                
            # 保存图片
            filename = f"{step_name}_{img_name}.png"
            filepath = os.path.join(sample_dir, filename)
            vutils.save_image(img, filepath)
            
        if self.debug:
            self._log(f"保存攻击过程图片: {step_name} -> {sample_dir}")

    def run(self, x, y, target=None, sample_idx=0):
        # 若为目标攻击，使用target；否则使用真实标签y
        attack_id = target if self.target and target is not None else y
        
        # 保存原始图片
        if self.save_attack_process:
            self._save_attack_process('original', {'input': x[0]}, sample_idx=sample_idx)
        
        pil_imgs = self._to_pil(x)
        color_names = self._extract_colors(pil_imgs)
        
        # 语义分割步骤
        seg_maps, region_names, region_masks = self._segment_semantics(x)
        if self.save_attack_process:
            # 保存分割结果
            seg_images = {}
            for i, (seg_map, mask) in enumerate(zip(seg_maps, region_masks)):
                if i == 0:  # 只保存第一个样本
                    # 保存region_mask（tensor格式）
                    if hasattr(mask, 'cpu'):
                        seg_images[f'region_mask_{i}'] = mask.cpu()
                    else:
                        seg_images[f'region_mask_{i}'] = torch.tensor(mask)
                    
                    # 生成可视化的语义分割图片
                    try:
                        # 使用Mask2Former的可视化方法
                        vis_img = self.segmentor.visualize_panoptic_and_largest_stuff(
                            x[i:i+1],  # 单个样本
                            [seg_map],  # 单个分割图
                            [mask],     # 单个掩码
                            [region_names[i]]  # 单个区域名称
                        )
                        if vis_img is not None:
                            seg_images[f'visualization_{i}'] = vis_img
                    except Exception as e:
                        if self.debug:
                            self._log(f"语义分割可视化失败: {e}")
                    
                    # 创建简单的掩码可视化
                    try:
                        # 将掩码转换为彩色图片
                        mask_tensor = mask if hasattr(mask, 'cpu') else torch.tensor(mask)
                        if mask_tensor.dim() == 2:
                            # 单通道掩码，转换为3通道彩色图片
                            colored_mask = torch.zeros(3, mask_tensor.shape[0], mask_tensor.shape[1])
                            colored_mask[0] = mask_tensor * 0.8  # 红色通道
                            colored_mask[1] = mask_tensor * 0.2  # 绿色通道
                            colored_mask[2] = mask_tensor * 0.1  # 蓝色通道
                            seg_images[f'colored_mask_{i}'] = colored_mask
                    except Exception as e:
                        if self.debug:
                            self._log(f"掩码可视化失败: {e}")
                    
                    # 尝试从seg_map中提取可视化信息
                    if isinstance(seg_map, dict):
                        # 如果是字典，尝试提取panoptic_seg
                        if 'panoptic_seg' in seg_map:
                            panoptic_seg = seg_map['panoptic_seg']
                            if hasattr(panoptic_seg, 'cpu'):
                                seg_images[f'panoptic_seg_{i}'] = panoptic_seg.cpu()
                            else:
                                seg_images[f'panoptic_seg_{i}'] = torch.tensor(panoptic_seg)
                        
                        # 尝试提取sem_seg
                        if 'sem_seg' in seg_map:
                            sem_seg = seg_map['sem_seg']
                            if hasattr(sem_seg, 'cpu'):
                                seg_images[f'sem_seg_{i}'] = sem_seg.cpu()
                            else:
                                seg_images[f'sem_seg_{i}'] = torch.tensor(sem_seg)
            
            # 只有当有有效图片时才保存
            if seg_images:
                self._save_attack_process('segmentation', seg_images, sample_idx=sample_idx)
        
        # CAM生成步骤
        if self.target:
            cam_list, cam_masks = self._generate_cam_and_masks(x, label=attack_id)
        else:
            cam_list, cam_masks = self._generate_cam_and_masks(x)
        
        if self.save_attack_process:
            # 保存CAM结果
            cam_images = {}
            for i, (cam, mask) in enumerate(zip(cam_list, cam_masks)):
                if i == 0:  # 只保存第一个样本
                    # 确保cam和mask是tensor格式
                    if hasattr(cam, 'cpu'):
                        cam_images[f'cam_{i}'] = cam.cpu()
                    else:
                        cam_images[f'cam_{i}'] = torch.tensor(cam)
                    
                    if hasattr(mask, 'cpu'):
                        cam_images[f'cam_mask_{i}'] = mask.cpu()
                    else:
                        cam_images[f'cam_mask_{i}'] = torch.tensor(mask)
            self._save_attack_process('cam', cam_images, sample_idx=sample_idx)
        
        # 补丁位置选择
        coords_list = self._select_patch_positions(x, region_masks, cam_masks)
        
        # 补丁生成步骤
        if self.target:
            patch_tensor = self._generate_patches(region_names, color_names, attack_id)
        else:
            patch_tensor = self._generate_patches(region_names, color_names)
        
        if self.save_attack_process:
            # 保存生成的补丁
            patch_images = {}
            for i in range(min(3, patch_tensor.size(1))):  # 保存前3个补丁
                patch_images[f'patch_{i}'] = patch_tensor[0, i]
            self._save_attack_process('patches', patch_images, sample_idx=sample_idx)
        
        # 补丁融合步骤
        patched_img = self._blend_patches(x, patch_tensor, coords_list)
        if self.save_attack_process:
            self._save_attack_process('blended', {'patched': patched_img[0]}, sample_idx=sample_idx)
        
        # 优化步骤
        if self.target:
            x_adv = self._optimize(x, patched_img, coords_list, attack_id)
        else:
            x_adv = self._optimize(x, patched_img, coords_list, y)
        
        if self.save_attack_process:
            self._save_attack_process('adversarial', {'final': x_adv[0]}, sample_idx=sample_idx)
        
        return x_adv

    def _to_pil(self, x):
        return [to_pil_image(x[i].cpu()) for i in range(x.shape[0])]

    def _extract_colors(self, pil_imgs):
        color_names = self.color_extractor.get_top_color_names_batch(pil_imgs, topk=3)
        if self.debug:
            self._log(f"Dominant colors: {color_names}")
        return color_names

    def _segment_semantics(self, x):
        seg_maps = self.segmentor.segment_batch_panoptic(x)
        names, masks = self.segmentor.get_largest_stuff_region_batch(seg_maps)
        if self.save_imgs:
            self._save_img(self.segmentor.visualize_panoptic_and_largest_stuff, x, seg_maps, masks, names)
        if self.debug:
            self._log(f"Segmented regions: {names}")
        return seg_maps, names, masks

    def _generate_cam_and_masks(self, x, label=None):
        cams = self.cam_gen.generate_cam_batch(x, target_classes=label)
        cam_masks = self.cam_gen.cam_to_mask_batch(cams, percentile=self.config["cam_percentile"])
        if self.save_imgs:
            self._save_img(self.cam_gen.save_cam_composite_4in1_batch, images_tensor=x, cams=cams, masks_batch=cam_masks)
        if self.debug:
            self._log(f"Generated CAMs for {len(cams)} images with masks.")
        return cams, cam_masks

    def _select_patch_positions(self, x, region_masks, cam_masks):
        selector = PatchSelectorBatch(region_masks, x, self.config["patch_size"], cam_masks,self.config["cam_thresh_ratio"], self.config["mask_thresh_ratio"])
        coords = selector.get_best_patch_positions(k=self.config["patch_k"], stride_ratio=0.5)
        if self.save_imgs:    
            self._save_img(selector.visualize_patch_selection, x, coords)
        return coords

    def _generate_patches(self, region_names, color_names, target_classes=None):
        if self.target:
            patches, prompts = self.patch_gen.generate_patch_by_region_batch(
                region_names=region_names,
                color_names_list=color_names,
                patch_size=self.config["patch_size"],
                per_region_k=self.config["patch_k"],
                target_classes=target_classes
            )
        else:
            patches, prompts = self.patch_gen.generate_patch_by_region_batch(
                region_names=region_names,
                color_names_list=color_names,
                patch_size=self.config["patch_size"],
                per_region_k=self.config["patch_k"]
            )
        if self.debug:
            self._log(f"Prompts: {prompts}")
        if self.save_imgs:
            self._save_img(self.patch_gen.visualize_patch_batch_with_prompts, patch_batch=patches, prompts_batch=prompts)
        to_tensor = transforms.ToTensor()
        return torch.stack([torch.stack([to_tensor(p) for p in plist]) for plist in patches]).to(self.device)

    def _blend_patches(self, x, patch_tensor, coords):
        if self.debug:
            self._log(f"Blending patches into images at coordinates: {coords}")
        blended = self.patch_blender.blend_patch_tensor_batch(
            backgrounds=x,
            patches=patch_tensor,
            positions=coords,
            blend_width=self.config["blend_width"]
        )
        if self.debug:
            self._log(f"Blended patches into images.")
        if self.save_imgs:
            self._save_img(self.patch_blender.visualize_patch_blending_batch, x, patch_tensor, coords, blended)
        
        return blended

    def _optimize(self, x, patched_x, coords, y_or_target):
        targeted = self.config["targeted_attack"]
        if self.debug:
            self._log(f"Optimizing adversarial images with coordinates: {coords} and target: {y_or_target}")
        x_adv = self.optimizer.optimize_batch(x, patched_x, coords, y_or_target, targeted=targeted)
        if self.debug:
            self._log(f"Optimized adversarial images with targeted={targeted}.")
        if self.save_imgs:
            self._save_img(self.optimizer.visualize_adversarial_comparison, x=x, x_adv=x_adv)
        return x_adv 