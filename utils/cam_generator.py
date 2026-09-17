import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image


class CAMGenerator:
    def __init__(self, model_name='resnet50', device='cuda'):
        self.device = device
        self.model, self.target_layer = self._load_model_and_target_layer(model_name)
        self.model.eval().to(self.device)
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _load_model_and_target_layer(self, model_name):
        model_name = model_name.lower()
        if model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            target_layer = model.layer4[-1]
        elif model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            target_layer = model.layer4[-1]
        elif model_name == 'densenet121':
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            target_layer = model.features[-1]
        elif model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            target_layer = model.features[-1]
        else:
            raise ValueError(f"暂不支持模型: {model_name}")
        return model, target_layer

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def _preprocess(self, input_data):
        """
        支持三种输入类型：PIL.Image / numpy.ndarray / torch.Tensor
        输出 shape: [1, 3, 224, 224]
        """
        if isinstance(input_data, torch.Tensor):
            if input_data.dim() == 3:
                input_data = input_data.unsqueeze(0)  # [1,3,H,W]
            if input_data.shape[2:] != (224, 224):
                input_data = F.interpolate(input_data, size=(224, 224), mode='bilinear', align_corners=False)
            input_data = self._normalize(input_data).to(self.device)
        else:
            if isinstance(input_data, np.ndarray):
                input_data = Image.fromarray(input_data.astype(np.uint8))
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            input_data = transform(input_data).unsqueeze(0).to(self.device)
        return input_data

    def _normalize(self, tensor):
        """对已经是 tensor 的图像做标准化"""
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def generate_cam(self, input_data, target_class=None):
        input_tensor = self._preprocess(input_data)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax().item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        pooled_grad = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted_activations = self.activations * pooled_grad
        cam = torch.sum(weighted_activations, dim=1).squeeze().cpu().numpy()

        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-6)
        return cam

    def overlay_cam_on_image(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        if isinstance(image, torch.Tensor):
            image = image.squeeze().permute(1, 2, 0).cpu().numpy()
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(alpha * heatmap + (1 - alpha) * image)
        return overlay

    def generate_cam_batch(self, input_tensor_batch, target_classes=None):
        """
        批量生成 Grad-CAM，支持目标攻击

        参数：
            input_tensor_batch: torch.Tensor, shape=[B,3,H,W]
            target_classes: None / int / list[int] / torch.Tensor, 指定每张图的目标类

        返回：
            cam_tensor: torch.Tensor, shape = [B, H, W]，归一化至 [0,1]
        """
        self.model.eval()
        B = input_tensor_batch.shape[0]
        input_tensor_batch = self._normalize(input_tensor_batch).to(self.device)
        input_tensor_batch.requires_grad_()

        # 前向传播，保存 activations 和 logits
        output = self.model(input_tensor_batch)  # (B, num_classes)

        # 获取目标类别
        if target_classes is None:
            target_classes = output.argmax(dim=1)  # 取预测类
        elif isinstance(target_classes, int):
            target_classes = torch.full((B,), target_classes, dtype=torch.long, device=self.device)
        elif isinstance(target_classes, (list, tuple)):
            target_classes = torch.tensor(target_classes, dtype=torch.long, device=self.device)
        elif isinstance(target_classes, torch.Tensor):
            target_classes = target_classes.to(self.device)
        else:
            raise ValueError("target_classes 必须是 None, int, list[int], 或 torch.Tensor")

        cam_list = []

        for i in range(B):
            self.model.zero_grad()
            score = output[i, target_classes[i]]
            score.backward(retain_graph=True)

            activations = self.activations[i]   # shape: (C, H, W)
            gradients = self.gradients[i]       # shape: (C, H, W)
            pooled_grad = torch.mean(gradients, dim=(1, 2), keepdim=True)  # (C,1,1)
            weighted_activations = activations * pooled_grad               # (C,H,W)
            cam = torch.sum(weighted_activations, dim=0)                   # (H,W)
            cam = F.relu(cam)  # 去掉负值

            # 归一化至 [0,1]
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-6)

            cam_list.append(cam.detach())

            # 清除梯度
            input_tensor_batch.grad = None
            self.model.zero_grad()

        # 合并为 (B, H, W)
        cam_tensor = torch.stack(cam_list, dim=0)
        return cam_tensor

    def cam_to_mask_batch(self, cams: torch.Tensor, percentile: float = 90): 
        """
        批量将 CAM 转换为二值掩码 (0/1)，支持 GPU 运算

        参数：
            cams: torch.Tensor，形状为 [B, H, W]，值范围应为 [0,1]
            percentile: float，CAM 阈值的百分位 (0-100)，用于生成 mask

        返回：
            List[torch.Tensor]，每个元素形状为 [H, W]，类型为 uint8（0 或 1）
        """
        B, H, W = cams.shape
        flat = cams.view(B, -1)
        k = int((100 - percentile) / 100.0 * H * W)
        thresholds = torch.kthvalue(flat, k + 1, dim=1).values  # kthvalue is 1-indexed

        # [B,1,1] 形状用于广播比较
        thresholds = thresholds.view(B, 1, 1)
        masks = (cams >= thresholds).to(torch.uint8)

        return list(masks)  # 每个 [H, W] 的二值 mask


    def save_cam_composite_4in1_batch(self,images_tensor: torch.Tensor,
                                  cams: torch.Tensor,
                                  masks_batch: list,
                                  save_dir: str,
                                  timestamp_str: str = "default_time",
                                  show: bool = False):
        """
        显示并保存四图合成图：原图、CAM图、CAM定位叠加图、Mask图。

        参数:
            images_tensor: [B, C, H, W]
            cams: [B, H, W]，值归一化 [0,1]
            masks_batch: List[torch.Tensor]，每个元素 [H, W]，值为 0/1
        """
        B = images_tensor.shape[0]
        os.makedirs(save_dir, exist_ok=True)

        images_tensor = images_tensor.detach().cpu()
        cams = cams.detach().cpu()

        for i in range(B):
            # 原图
            img = images_tensor[i]
            if img.max() <= 1.0:
                img = img * 255.0
            img = img.to(torch.uint8)
            img_pil = to_pil_image(img)
            w, h = img_pil.size

            # CAM 热力图（独立）
            cam = cams[i].numpy()
            cam_norm = (cam - cam.min()) / (cam.max() + 1e-6)
            cmap = plt.get_cmap('jet')
            cam_color = cmap(cam_norm)[:, :, :3] * 255
            cam_color = cam_color.astype(np.uint8)
            cam_pil = Image.fromarray(cam_color).resize((w, h), Image.BILINEAR)

            # CAM 定位图（叠加原图）
            overlay = Image.blend(img_pil.convert("RGBA"), cam_pil.convert("RGBA"), alpha=0.5)

            # Mask图（红色高亮）
            mask_np = masks_batch[i].cpu().numpy()
            mask_rgb = np.stack([mask_np * 255, np.zeros_like(mask_np), np.zeros_like(mask_np)], axis=2).astype(np.uint8)
            mask_pil = Image.fromarray(mask_rgb).resize((w, h), Image.NEAREST)

            # 绘图 2x2
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            axs[0, 0].imshow(img_pil)
            axs[0, 0].set_title("原始图像")
            axs[0, 1].imshow(cam_pil)
            axs[0, 1].set_title("CAM 热力图")
            axs[1, 0].imshow(overlay)
            axs[1, 0].set_title("CAM 定位图")
            axs[1, 1].imshow(mask_pil)
            axs[1, 1].set_title("Mask 区域")

            for ax in axs.flatten():
                ax.axis("off")

            fig.tight_layout()
            filename = f"4in1_cam_mask_{timestamp_str}_{i}.png"
            fig.savefig(os.path.join(save_dir, filename), dpi=300)
            plt.close(fig)

            if show:
                plt.figure(figsize=(8, 8))
                plt.imshow(np.asarray(fig.canvas.buffer_rgba()))
                plt.axis("off")
                plt.title(f"Sample {i}")
                plt.show()



    

def cam_to_mask(cam: np.ndarray, percentile: float = 90) -> np.ndarray:
    """
    将 CAM 映射为二值掩码，高于 percentile 的区域标记为 1。
    cam: [H,W]，归一化后的热力图
    """
    threshold = np.percentile(cam, percentile)
    binary_mask = (cam >= threshold).astype(np.uint8)
    return binary_mask

    