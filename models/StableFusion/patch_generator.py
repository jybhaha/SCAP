import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import random
import json
import os
import matplotlib.pyplot as plt
from typing import List
from torchvision.models import ResNet50_Weights


import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class PatchGenerator:
    PROMPT_TEMPLATES = [
        "A {target_class}-shaped {patch_type} on a {region_name}",
        "A pattern resembling a {target_class} on a {region_name}",
        "A silhouette of a {target_class} blended into the {region_name}",
        "A distorted {target_class} shape over the {region_name}",
        "An artistic shape of {target_class} subtly appearing on the {region_name}"
    ]

    def __init__(self, 
                 model_path="/home/jyb/0code/SD-v1-5-model/stable-diffusion-v1-5", 
                 json_path="/home/jyb/0code/SCAR/数据准备/ADE20K_patches.json",
                 device="cuda"):
        self.device = device
        self.idx_to_label = {
            idx: label for idx, label in enumerate(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])
        }

        # 初始化 Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(device)

        # 关闭安全检查器（用于研究测试）
        self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

        # 加载语义类别对应的 patch 类型列表
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到补丁类型定义文件: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            self.region_patch_map = json.load(f)

        self.region2patch = {item["name"]: item["possible_patches"] for item in self.region_patch_map}
        
        
    def get_label_name(self, idx):
        """根据 ImageNet 类别 index 获取对应英文标签"""
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self.idx_to_label.get(idx, "unknown object")
    
    
    def generate_patch(self, prompt, patch_size=(64, 64), num_inference_steps=20, guidance_scale=7.5):
        with torch.autocast(self.device.type):
            image = self.pipe(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        patch_img = image.resize(patch_size, Image.LANCZOS)
        return patch_img

    def generate_patch_by_region_batch(self, region_names, color_names_list=None, patch_size=(24, 24), 
                                   num_inference_steps=20, guidance_scale=7.5, per_region_k=1,
                                   target_classes=None):
        """
        批量生成 patch 图像，并返回对应的 prompt。
        """
        B = len(region_names)
        all_prompts = []

        for idx in range(B):
            region_name = region_names[idx]
            color_names = color_names_list[idx] if color_names_list else []
            target_class = target_classes[idx] if target_classes is not None  else None
       
            # ✅ 若是索引形式，自动转换为标签字符串
            if isinstance(target_class, (int, torch.Tensor)):
        
                target_class = self.get_label_name(target_class)
   
            else:

                target_class = target_class
            
            
            region_prompts = []
            for _ in range(per_region_k):
                patch_type = random.choice(self.region2patch.get(region_name, ["texture"]))
                if color_names:
                    color_desc = " in " + " and ".join(color_names)
                else:
                    color_desc = ""

                if target_class is not None:
                    template = random.choice(self.PROMPT_TEMPLATES)
                    prompt = template.format(
                        target_class=target_class,
                        patch_type=patch_type,
                        region_name=region_name
                    )
                    
                else:
                    prompt = f"A realistic photo of {patch_type} on a {region_name}{color_desc}"

                region_prompts.append(prompt)
            all_prompts.extend(region_prompts)

        print(f"[\U0001f9e0 Batch Prompt Size]: {len(all_prompts)}")
        for i, p in enumerate(all_prompts):
            print(f" - Prompt {i+1}: {p}")

        with torch.autocast(self.device.type):
            images = self.pipe(
                all_prompts,
                height=512,
                width=512,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images

        resized_images = [img.resize(patch_size, Image.LANCZOS) for img in images]
        patch_images_batch = [
            resized_images[i * per_region_k:(i + 1) * per_region_k]
            for i in range(B)
        ]
        prompts_batch = [
            all_prompts[i * per_region_k:(i + 1) * per_region_k]
            for i in range(B)
        ]

        return patch_images_batch, prompts_batch

    def visualize_patch_batch_with_prompts(self,patch_batch: List[List[Image.Image]],
                                       prompts_batch: List[List[str]],
                                       save_dir: str = None,
                                       timestamp_str: str = "default_time",
                                       show: bool = False):
        B = len(patch_batch)
        for i in range(B):
            patches = patch_batch[i]
            prompts = prompts_batch[i]
            k = len(patches)

            fig, axs = plt.subplots(1, k, figsize=(3 * k, 3.2))
            if k == 1:
                axs = [axs]

            for j in range(k):
                axs[j].imshow(patches[j])
                axs[j].axis("off")
                axs[j].set_title(f"Prompt {j+1}:\n{prompts[j]}", fontsize=8, wrap=True)

            fig.suptitle(f"Sample {i}", fontsize=12)

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f"patches_with_prompts_{timestamp_str}_{i}.png"),
                            dpi=300, bbox_inches='tight')
            if show:
                plt.show()
            plt.close(fig)
