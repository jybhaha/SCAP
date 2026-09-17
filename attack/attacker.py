import torch
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
from utils.patch_selector import PatchSelector
from utils.cam_generator import CAMGenerator, cam_to_mask
from utils.color_extractor import ColorExtractor
from models.StableFusion.patch_generator import PatchGenerator
from utils.patch_blender import PatchBlender
from models.patch_optimizer import PatchOptimizer
from utils.success_function import validate_attack_success
from models.Mask2Former.mask2former import SemanticSegmentor
from utils.result_saver import ResultSaver

class MyPatchAttack:
    def __init__(self, model, device='cuda', config=None, debug=False):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.debug = debug

        self.config = config or {
            "patch_size": (24, 24),
            "blend_width": 15,
            "patch_k": 3,
            "cam_model_name": "resnet50",
            "cam_percentile": 90,
            "topk_cam_regions": 1,
            "eval_topk": 1,
            "targeted_attack": False,
        }

        # 初始化模块
        self.segmentor = SemanticSegmentor()
        self.patch_gen = PatchGenerator(device=self.device)
        self.color_extractor = ColorExtractor()
        self.patch_blender = PatchBlender(blend_width=self.config["blend_width"])
        self.optimizer = PatchOptimizer(device=self.device)

        self.resultsaver = ResultSaver("debug_mypatchattack") if debug else None
        if self.resultsaver:
            self.resultsaver.log("[Debug Mode] 启动MyPatchAttack调试模式")

    def run(self, x, y):
        """
        批量版 run 函数
        x: tensor [B, C, H, W]
        y: tensor [B]
        返回：x_adv [B, C, H, W]
        """

        B = x.shape[0]

        # 1. 转成 PIL 图像列表
        pil_imgs = [to_pil_image(x[i].cpu()) for i in range(B)]

        # 2. 批量提取 dominant color names，调用批量接口
        dominant_color_names_batch = self.color_extractor.get_top_color_names_batch(pil_imgs, topk=3)

        # 3. 批量语义分割
        seg_maps = self.segmentor.segment_batch(x)  # 这里假设你有 batch 版本的 segment_batch，返回 list 或 tensor

        x_adv_list = []

        for i in range(B):
            pil_img = pil_imgs[i]
            dominant_color_names = dominant_color_names_batch[i]
            seg_map = seg_maps[i]

            topk_info = self.segmentor.get_topk_regions_info(seg_map, k=self.config["topk_cam_regions"])

            x_i = x[i].to(self.device)
            y_i = y[i].to(self.device)

            x_adv_i = x_i.clone()

            for name, mask, _ in topk_info:
                try:
                    # CAM 这里还是单张处理，因为一般CAM耗时且输入为PIL
                    cam_gen = CAMGenerator(model_name=self.config["cam_model_name"], device=self.device)
                    cam = cam_gen.generate_cam(pil_img)
                    cam_mask = cam_to_mask(cv2.resize(cam, pil_img.size[::-1]), percentile=self.config["cam_percentile"])

                    selector = PatchSelector(mask, pil_img, cam_mask=cam_mask, patch_size=self.config["patch_size"])
                    coords_list = selector.get_best_patch_positions(k=self.config["patch_k"], visualize=self.debug)

                    patch_list = self.patch_gen.generate_patch_by_region(
                        region_name=name,
                        color_names=dominant_color_names,
                        patch_size=self.config["patch_size"],
                        k=self.config["patch_k"]
                    )
                    
                    if len(patch_list) != len(coords_list):
                        continue
                    
                    patched_img = pil_img.copy()
                    patch_mask_np = np.zeros((pil_img.height, pil_img.width), dtype=np.uint8)
                    
                    for patch_img, coords in zip(patch_list, coords_list):
                        if patch_img.mode != "RGBA":
                            patch_img = patch_img.convert("RGBA")
                        position = (coords[0], coords[1])
                        patched_img = self.patch_blender.blend_patch(patched_img, patch_img, position)
                        x1, y1 = position
                        x2, y2 = x1 + self.config["patch_size"][0], y1 + self.config["patch_size"][1]
                        patch_mask_np[y1:y2, x1:x2] = 1

                    if self.debug:
                        self.resultsaver.save_image(
                            imgs=patch_list,
                            filenames=[f"{name}_patch_{j}.png" for j in range(len(patch_list))],
                            titles=[f"{name} patch {j}" for j in range(len(patch_list))],
                            show=False
                        )

                    patch_mask_tensor = torch.tensor(patch_mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                    patched_tensor = transforms.ToTensor()(patched_img).unsqueeze(0).to(self.device)

                    if patch_mask_tensor.sum() == 0:
                        continue

                    x_adv_i = self.optimizer.optimize(
                        image_tensor=x_i,
                        patched_image_tensor=patched_tensor,
                        patch_mask=patch_mask_tensor,
                        label=y_i
                    )
                    
                    if self.debug:
                        self.resultsaver.save_tensor_image(x_adv_i, name=f"x_adv_{name}_{i}")
                        self.resultsaver.log(f"adv_patch区域 {name} 完成，样本索引：{i}，形状：{x_adv_i.shape}")

                    # 一旦对该样本的一个区域成功生成对抗，跳出区域循环
                    break

                except Exception as e:
                    if self.debug:
                        self.resultsaver.log(f"跳过样本{i}区域 {name}，错误：{e}")
                    continue

            x_adv_list.append(x_adv_i)

        return torch.stack(x_adv_list)

