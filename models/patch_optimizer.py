import torch
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import os
from utils.result_saver import ResultManager

class PatchOptimizer:
    def __init__(self, device, steps=25, eps=8/255, alpha=5/255, decay=1.0, model=None):
        self.device = device
        self.steps = steps
        self.eps = eps
        self.alpha = alpha
        self.decay = decay

        # 使用传入的模型，如果没有则使用默认的ResNet50
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = models.resnet50(pretrained=True).to(device)
        self.model.eval()
        self.rm = ResultManager.get_instance()  # 获取结果管理器实例

    def optimize_old(self, image_tensor, patched_image_tensor, patch_mask, label):
        """
        image_tensor: 原图 (B, 3, H, W)
        patched_image_tensor: 带初始patch图 (B, 3, H, W)
        patch_mask: patch区域mask (1, 1, H, W) 或 (B, 1, H, W)
        label: (B,) Tensor
        """
        B = image_tensor.size(0)

        x_adv = patched_image_tensor.clone().detach().to(self.device)
        ori = image_tensor.clone().detach().to(self.device)
        label = label.to(self.device)

        # 自动扩展 patch_mask 到 batch 维度
        if patch_mask.shape[0] == 1:
            patch_mask = patch_mask.expand(B, -1, -1, -1).to(self.device)
        else:
            patch_mask = patch_mask.to(self.device)

        momentum = torch.zeros_like(x_adv).to(self.device)

        for _ in range(self.steps):
            x_adv.requires_grad = True
            outputs = self.model(x_adv)
            loss = F.cross_entropy(outputs, label)

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            grad_norm = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
            momentum = self.decay * momentum + grad_norm

            x_adv = x_adv.detach() + self.alpha * momentum.sign() * patch_mask

            x_adv = torch.max(torch.min(x_adv, ori + self.eps), ori - self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv

    def optimize(self, image_tensor, patched_image_tensor, patch_mask, label):
        """
        使用PGD算法进行补丁优化
        image_tensor: 原图 (B, 3, H, W)
        patched_image_tensor: 带初始patch图 (B, 3, H, W)
        patch_mask: patch区域mask (1, 1, H, W) 或 (B, 1, H, W)
        label: (B,) Tensor
        """
        B = image_tensor.size(0)

        x_adv = patched_image_tensor.clone().detach().to(self.device)
        ori = image_tensor.clone().detach().to(self.device)
        label = label.to(self.device)

        # 自动扩展 patch_mask 到 batch 维度
        if patch_mask.shape[0] == 1:
            patch_mask = patch_mask.expand(B, -1, -1, -1).to(self.device)
        else:
            patch_mask = patch_mask.to(self.device)

        # PGD算法：从随机初始化开始，使用更大的扰动
        x_adv = ori + torch.randn_like(ori) * self.eps * 0.3
        x_adv = torch.clamp(x_adv, 0, 1)

        for step in range(self.steps):
            x_adv.requires_grad = True
            outputs = self.model(x_adv)
            loss = F.cross_entropy(outputs, label)

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            
            # PGD更新：使用梯度方向
            x_adv = x_adv.detach() + self.alpha * grad.sign() * patch_mask
            
            # 投影到L∞球内
            delta = x_adv - ori
            delta = torch.clamp(delta, -self.eps, self.eps)
            x_adv = ori + delta
            
            # 确保像素值在[0,1]范围内
            x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv
    
    
    

    def optimize_batch(self, image_tensor, patched_image_tensor, coords_list, label, targeted=False, return_step_preds=False):
        """
        批量优化函数（支持目标攻击）- 原MIFGSM版本

        参数：
            image_tensor: 原图 (B, 3, H, W)
            patched_image_tensor: 初始带patch图 (B, 3, H, W)
            coords_list: List[List[(x1,y1,x2,y2)]]
            label: (B,) Tensor，若 targeted=True 则为目标类
            targeted: 是否为目标攻击
            return_step_preds: 是否返回每步预测结果
        返回：
            x_adv: (B, 3, H, W) 或 (x_adv, step_preds_list)
        """
        #self.rm.log(f"Optimizing batch of {len(coords_list)} images with targeted={targeted}.")
        B, _, H, W = image_tensor.shape
        device = self.device
        x_adv = patched_image_tensor.clone().detach().to(device)
        ori = image_tensor.clone().detach().to(device)
        label = label.to(device)
        #self.rm.log(f"lable: {label.tolist()}")
        #self.rm.log(f"Input images shape: {x_adv.shape}, Original images shape: {ori.shape}, Labels shape: {label.shape}")
        # 生成 patch mask
        patch_mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
        for i, coords in enumerate(coords_list):
            for (x1, y1, x2, y2) in coords:
                patch_mask[i, 0, y1:y2, x1:x2] = 1.0

        momentum = torch.zeros_like(x_adv).to(device)
        #self.rm.log(f"Starting optimization for {B} images with {self.steps} steps.")
        
        step_preds_list = []
        
        for step in range(self.steps):
            x_adv.requires_grad = True
            outputs = self.model(x_adv)
            loss = F.cross_entropy(outputs, label)
            #self.rm.log(f"Step {step+1}/{self.steps}, Loss: {loss.item()}")
            # ⚠️ 根据是否为 targeted 攻击，调整梯度方向
            if targeted:
                loss = -loss

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            grad_norm = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
            momentum = self.decay * momentum + grad_norm
            #self.rm.log(f"Momentum updated for step {step+1}/{self.steps}.")
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

    def optimize_batch_pgd(self, image_tensor, patched_image_tensor, coords_list, label, targeted=False, return_step_preds=False):
        """
        批量优化函数（支持目标攻击）- PGD版本

        参数：
            image_tensor: 原图 (B, 3, H, W)
            patched_image_tensor: 初始带patch图 (B, 3, H, W)
            coords_list: List[List[(x1,y1,x2,y2)]]
            label: (B,) Tensor，若 targeted=True 则为目标类
            targeted: 是否为目标攻击
            return_step_preds: 是否返回每步预测结果
        返回：
            x_adv: (B, 3, H, W) 或 (x_adv, step_preds_list)
        """
        B, _, H, W = image_tensor.shape
        device = self.device
        ori = image_tensor.clone().detach().to(device)
        label = label.to(device)
        
        # 生成 patch mask
        patch_mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
        for i, coords in enumerate(coords_list):
            for (x1, y1, x2, y2) in coords:
                patch_mask[i, 0, y1:y2, x1:x2] = 1.0

        # PGD算法：从随机初始化开始
        x_adv = ori + torch.randn_like(ori) * self.eps * 0.3
        x_adv = torch.clamp(x_adv, 0, 1)
        
        step_preds_list = []
        
        for step in range(self.steps):
            x_adv.requires_grad = True
            outputs = self.model(x_adv)
            loss = F.cross_entropy(outputs, label)
            
            # 根据是否为 targeted 攻击，调整梯度方向
            if targeted:
                loss = -loss

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            
            # PGD更新：使用梯度方向
            x_adv = x_adv.detach() + self.alpha * grad.sign() * patch_mask
            
            # 投影到L∞球内
            delta = x_adv - ori
            delta = torch.clamp(delta, -self.eps, self.eps)
            x_adv = ori + delta
            
            # 确保像素值在[0,1]范围内
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
