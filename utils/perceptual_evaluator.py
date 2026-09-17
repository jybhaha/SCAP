import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
import pyiqa
from typing import List
from utils.result_saver import ResultManager

class PerceptualEvaluator:
    def __init__(self, device='cuda', lpips_net='alex'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lpips_loss = lpips.LPIPS(net=lpips_net).to(self.device)
        # 初始化 NR-IQA 模型
        self.nr_models = {
            'clipiqa': pyiqa.create_metric('clipiqa', device=self.device),
            'musiq': pyiqa.create_metric('musiq', device=self.device),
            'niqe': pyiqa.create_metric('niqe', device=self.device)
        }
        self.rm = ResultManager.get_instance()

    def calc_ssim(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        ssim_list = []
        for i in range(x1.size(0)):
            img1 = x1[i].detach().cpu().numpy().transpose(1, 2, 0)
            img2 = x2[i].detach().cpu().numpy().transpose(1, 2, 0)
            ssim_val = ssim(img1, img2, channel_axis=-1, data_range=1.0)
            ssim_list.append(ssim_val)
        return np.mean(ssim_list)

    def calc_psnr(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        psnr_list = []
        for i in range(x1.size(0)):
            img1 = x1[i].detach().cpu().numpy().transpose(1, 2, 0)
            img2 = x2[i].detach().cpu().numpy().transpose(1, 2, 0)
            psnr_val = psnr(img1, img2, data_range=1.0)
            psnr_list.append(psnr_val)
        return np.mean(psnr_list)

    def calc_l1(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        return F.l1_loss(x1, x2, reduction='mean').item()

    def calc_l2(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        return torch.norm((x1 - x2).view(x1.size(0), -1), p=2, dim=1).mean().item()

    def calc_lpips(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        x1_norm = x1 * 2 - 1
        x2_norm = x2 * 2 - 1
        with torch.no_grad():
            loss = self.lpips_loss(x1_norm.to(self.device), x2_norm.to(self.device))
        return loss.mean().item()

    def calc_lpips_l2(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        x1_norm = x1 * 2 - 1
        x2_norm = x2 * 2 - 1
        with torch.no_grad():
            dist = self.lpips_loss.forward(x1_norm.to(self.device), x2_norm.to(self.device), normalize=True)
        return dist.mean().item()

    def calc_cosine_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        cos = F.cosine_similarity(x1_flat, x2_flat, dim=1)
        return cos.mean().item()

    def calc_fid(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        if x1.size(0) < 10:
            print("[Warning] FID建议至少10张图像，当前为", x1.size(0))
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        fid.update(x1.to(self.device), real=True)
        fid.update(x2.to(self.device), real=False)
        return fid.compute().item()

    

    def calc_no_reference_metrics(self, x: torch.Tensor) -> dict:
        results = {}
        x = x.clamp(0, 1).float().to(self.device)  # 保证范围和类型
        for name, model in self.nr_models.items():
            try:
                with torch.no_grad():
                    val = model(x).mean().item()
                    results[f'{name}'] = val
            except Exception as e:
                results[f'{name}'] = -1
                self.rm.log(f"[Warning] {name} 评估失败: {e}")
                #print(f"[Warning] {name} 评估失败: {e}")
        return results


    def evaluate(self, x1: torch.Tensor, x2: torch.Tensor, include_fid=False, include_no_reference=False) -> dict:
        """
        输入图像范围应为 [0, 1]，shape = (B, 3, H, W)
        """
        results = {
            'SSIM': self.calc_ssim(x1, x2),
            'PSNR': self.calc_psnr(x1, x2),
            'L1': self.calc_l1(x1, x2),
            'L2': self.calc_l2(x1, x2),
            'LPIPS': self.calc_lpips(x1, x2),
            'LPIPS_L2': self.calc_lpips_l2(x1, x2),
            'Cosine': self.calc_cosine_similarity(x1, x2)
        }

        if include_fid:
            results['FID'] = self.calc_fid(x1, x2)

        if include_no_reference:
            nr_results = self.calc_no_reference_metrics(x2)
            results.update(nr_results)

        return results
