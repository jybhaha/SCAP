import torch
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import os
from utils.result_saver import ResultManager

class PatchOptimizer_ChangeLoss:
    #这个版本把loss改成了
    #对抗损失：交叉熵损失
    #平滑损失：总变差损失
    #颜色一致性损失：颜色一致性损失
    #其中对抗损失、平滑损失、颜色一致性损失的权重可以调整
    #其中平滑损失、颜色一致性损失的权重可以调整
    #其中颜色一致性损失的权重可以调整
    #其中颜色一致性损失的权重可以调整
    def __init__(self, device, steps=25, eps=8/255, alpha=5/255, decay=1.0, model=None,
                 lambda_adv=1.0, lambda_smooth=0.1, lambda_color=0.05):
        self.device = device
        self.steps = steps
        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        
        # 损失函数权重
        self.lambda_adv = lambda_adv
        self.lambda_smooth = lambda_smooth
        self.lambda_color = lambda_color

        # 使用传入的模型，如果没有则使用默认的ResNet50
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = models.resnet50(pretrained=True).to(device)
        self.model.eval()
        self.rm = ResultManager.get_instance()  # 获取结果管理器实例

    def _total_variation_loss(self, patch):
        """计算总变差损失，用于平滑性约束"""
        # patch shape: [B, 3, H, W]
        tv_h = torch.mean(torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1]))
        return tv_h + tv_w

    def _color_consistency_loss(self, patch, target_colors):
        """
        计算颜色一致性损失
        target_colors: 目标颜色调色板 [B, 3, 3] 或 [3, 3]，表示每个batch的3个主要RGB颜色
        """
        B, C, H, W = patch.shape
        
        # 如果target_colors是单一样本，扩展到batch维度
        if len(target_colors.shape) == 2:
            target_colors = target_colors.unsqueeze(0).repeat(B, 1, 1)
        
        # 重塑patch以便计算颜色距离 [B, H*W, 3]
        patch_flat = patch.permute(0, 2, 3, 1).contiguous().view(B, -1, 3)
        
        # 计算每个像素与目标颜色调色板的最小距离
        color_loss = 0
        for i in range(B):
            # 扩展维度以便广播计算 [H*W, 1, 3] 和 [1, 3, 3]
            pixels = patch_flat[i].unsqueeze(1)  # [H*W, 1, 3]
            colors = target_colors[i].unsqueeze(0)  # [1, 3, 3]
            
            # 计算所有像素与所有颜色的距离 [H*W, 3]
            distances = torch.norm(pixels - colors, dim=2)
            
            # 取每个像素与最近颜色的距离
            min_distances = torch.min(distances, dim=1)[0]
            color_loss += torch.mean(min_distances)
        
        return color_loss / B

    def optimize_batch(self, image_tensor, patched_image_tensor, coords_list, label, 
                      target_colors=None, targeted=False, return_step_preds=False):
        """
        批量优化函数 - 使用复合损失函数版本

        参数：
            image_tensor: 原图 (B, 3, H, W)
            patched_image_tensor: 初始带patch图 (B, 3, H, W)
            coords_list: List[List[(x1,y1,x2,y2)]]
            label: (B,) Tensor，若 targeted=True 则为目标类
            target_colors: 目标颜色调色板 [B, 3, 3] 或 [3, 3]
            targeted: 是否为目标攻击
            return_step_preds: 是否返回每步预测结果
        返回：
            x_adv: (B, 3, H, W) 或 (x_adv, step_preds_list)
        """
        B, _, H, W = image_tensor.shape
        device = self.device
        x_adv = patched_image_tensor.clone().detach().to(device)
        ori = image_tensor.clone().detach().to(device)
        label = label.to(device)
        
        # 生成 patch mask
        patch_mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
        patch_regions = []
        for i, coords in enumerate(coords_list):
            patch_regions.append([])
            for (x1, y1, x2, y2) in coords:
                patch_mask[i, 0, y1:y2, x1:x2] = 1.0
                patch_regions[i].append((x1, y1, x2, y2))

        momentum = torch.zeros_like(x_adv).to(device)
        step_preds_list = []
        
        for step in range(self.steps):
            x_adv.requires_grad = True
            outputs = self.model(x_adv)
            
            # 计算复合损失
            adv_loss = F.cross_entropy(outputs, label)
            if targeted:
                adv_loss = -adv_loss
            
            # 提取当前patch区域用于计算辅助损失
            current_patches = []
            for i, regions in enumerate(patch_regions):
                for (x1, y1, x2, y2) in regions:
                    patch = x_adv[i, :, y1:y2, x1:x2]
                    current_patches.append(patch)
            
            if current_patches:
                # 合并所有patch用于损失计算
                combined_patches = torch.cat(current_patches, dim=0)
                
                # 计算平滑损失
                smooth_loss = self._total_variation_loss(combined_patches)
                
                # 计算颜色一致性损失（如果提供了目标颜色）
                color_loss = 0
                if target_colors is not None:
                    color_loss = self._color_consistency_loss(combined_patches, target_colors)
                
                # 总损失
                total_loss = (self.lambda_adv * adv_loss + 
                            self.lambda_smooth * smooth_loss + 
                            self.lambda_color * color_loss)
            else:
                total_loss = adv_loss
            
            # 反向传播和优化
            grad = torch.autograd.grad(total_loss, x_adv, retain_graph=False, create_graph=False)[0]
            grad_norm = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
            momentum = self.decay * momentum + grad_norm
            
            x_adv = x_adv.detach() + self.alpha * momentum.sign() * patch_mask
            x_adv = torch.max(torch.min(x_adv, ori + self.eps), ori - self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # 如果需要返回每步预测，记录当前步的预测结果
            if return_step_preds:
                with torch.no_grad():
                    step_outputs = self.model(x_adv)
                    step_preds = torch.argmax(step_outputs, dim=1)
                    step_preds_list.append(step_preds.detach().cpu())

        if return_step_preds:
            return x_adv, step_preds_list
        else:
            return x_adv

    def visualize_adversarial_comparison(self, x: torch.Tensor, x_adv: torch.Tensor,
                                        save_dir: str = None,
                                        timestamp_str: str = "default_time",
                                        show: bool = False):
        """
        可视化对抗优化前后的图像（原图 vs 对抗图），支持 batch。
        
        参数：
            x: 原始图像 [B, 3, H, W]，取值范围 [0,1]
            x_adv: 优化后图像 [B, 3, H, W]，取值范围 [0,1]
            save_dir: 若指定，则保存每组图像
            timestamp_str: 保存文件用时间戳
            show: 是否显示图像
        """
        B = x.shape[0]
        for i in range(B):
            img_orig = TF.to_pil_image(x[i].cpu())
            img_adv = TF.to_pil_image(x_adv[i].cpu())

            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            axs[0].imshow(img_orig)
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(img_adv)
            axs[1].set_title("Adversarial")
            axs[1].axis("off")

            fig.suptitle(f"Sample {i}", fontsize=12)

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"adv_compare_{timestamp_str}_{i}.png"
                fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')

            if show:
                plt.show()

            plt.close(fig)