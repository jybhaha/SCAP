import torch
import torch.nn as nn
from ..attack import Attack
from IQA.IQA import IQAMetricLoss
import time
from utils import Config, setup_logger, lg

class VNIFGSM(Attack):
    r"""
    VNI-FGSM in the paper 'Enhancing the Transferability of Adversarial Attacks through Variance Tuning
    [https://arxiv.org/abs/2103.15571], Published as a conference paper at CVPR 2021
    Modified from "https://github.com/JHL-HUST/VT"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        N (int): the number of sampled examples in the neighborhood. (Default: 5)
        beta (float): the upper bound of neighborhood. (Default: 3/2)
        targeted (bool): whether the attack is targeted or not. (Default: False)
        target_labels (torch.Tensor): if targeted, provide target labels.
        lable_top2 (torch.Tensor): if untargeted, provide the top-2 wrong class labels.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = VNIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2, targeted=False, lable_top2=top2_labels)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, N=5, beta=3 / 2, targeted=False,iqa_loss_control=1.0,attr = None,k = 1
    ):
        super().__init__("VNIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
        self.targeted = targeted
        self.supported_mode = ["default", "targeted"]
        #self.iqa_loss = IQAMetricLoss()
        #self.iqa_loss_control = iqa_loss_control
        self.attr = attr
        self.k = k


    def forward(self, images, labels,target_labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # 初始化 momentum 和 v
        momentum = torch.zeros_like(images).detach().to(self.device)
        v = torch.zeros_like(images).detach().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        # 初始化监控数据
        monitor_data = {
            "cross_entropy_loss": [],
            "iqa_loss_value": [],
            "attr_value": []
        }
        

        for _ in range(self.steps):
            adv_images.requires_grad = True
            adv_images.grad = None  # 清除当前梯度
            nes_images = adv_images + self.decay * self.alpha * momentum
            outputs = self.get_logits(nes_images)
            #start_time = time.time()
            # 计算 IQA 损失
            iqa_loss_value = self.iqa_loss.evaluate_iqa_method_loss(images, adv_images)
            #end_time = time.time()
            # 计算并打印时间
            #elapsed_time = end_time - start_time
            #print(f"生成loss耗时: {elapsed_time:.4f} 秒")

            # 计算交叉熵损失
            if self.targeted:
                cost = loss_fn(outputs, target_labels) + iqa_loss_value * self.iqa_loss_control
            else:
                cost = loss_fn(outputs, labels)  + iqa_loss_value * self.iqa_loss_control

            # 更新 adversarial images
            adv_grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            grad = (adv_grad + v) / torch.mean(
                torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
            )
            grad = grad + momentum * self.decay
            momentum = grad

            # 计算梯度方差
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + torch.randn_like(
                    images
                ).uniform_(-self.eps * self.beta, self.eps * self.beta)
                neighbor_images.requires_grad = True
                outputs = self.get_logits(neighbor_images)

                # 计算邻域图像的损失
                if self.targeted:
                    cost = loss_fn(outputs, target_labels)
                else:
                    cost = loss_fn(outputs, labels)
                GV_grad += torch.autograd.grad(
                    cost, neighbor_images, retain_graph=False, create_graph=False
                )[0]

            # 获取梯度方差
            v = GV_grad / self.N - adv_grad

            adv_images = adv_images.detach() + self.alpha * grad.sign() * self.k * self.attr * 0.75
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # 监控交叉熵损失和 IQA 损失
            monitor_data["cross_entropy_loss"].append(loss_fn(outputs, target_labels if self.targeted else labels).item())
            monitor_data["iqa_loss_value"].append(iqa_loss_value)
            
        
        # 返回 adversarial images 和监控数据
        return adv_images


class VNIFGSM_vTest2(Attack):
    r"""
    VNI-FGSM in the paper 'Enhancing the Transferability of Adversarial Attacks through Variance Tuning
    [https://arxiv.org/abs/2103.15571], Published as a conference paper at CVPR 2021
    Modified from "https://github.com/JHL-HUST/VT"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        N (int): the number of sampled examples in the neighborhood. (Default: 5)
        beta (float): the upper bound of neighborhood. (Default: 3/2)
        targeted (bool): whether the attack is targeted or not. (Default: False)
        target_labels (torch.Tensor): if targeted, provide target labels.
        lable_top2 (torch.Tensor): if untargeted, provide the top-2 wrong class labels.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = VNIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2, targeted=False, lable_top2=top2_labels)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, N=5, beta=3 / 2, targeted=False,threshold = 1,gradient_scale_up = 1,gradient_scale_down =1
    ):
        super().__init__("VNIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
        self.targeted = targeted
        self.supported_mode = ["default", "targeted"]
        #self.iqa_loss = IQAMetricLoss()
        self.threshold = threshold
        self.gradient_scale_up = gradient_scale_up
        self.gradient_scale_down = gradient_scale_down


    def forward(self, images, labels, attr):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # 初始化 momentum 和 v
        momentum = torch.zeros_like(images).detach().to(self.device)
        v = torch.zeros_like(images).detach().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        
        
        # 创建一个list保存每个step生成的adv_images
        adv_images_list = []
        for _ in range(self.steps):
            adv_images.requires_grad = True
            adv_images.grad = None  # 清除当前梯度
            nes_images = adv_images + self.decay * self.alpha * momentum
            outputs = self.get_logits(nes_images)
            #start_time = time.time()
            # 计算 IQA 损失
            #iqa_loss_value = self.iqa_loss.evaluate_iqa_method_loss(images, adv_images)
            #end_time = time.time()
            # 计算并打印时间
            #elapsed_time = end_time - start_time
            #print(f"生成loss耗时: {elapsed_time:.4f} 秒")

            # 计算交叉熵损失
            if self.targeted:
                cost = -loss_fn(outputs, labels)
                #cost = -loss_fn(outputs, target_labels) + iqa_loss_value * self.iqa_loss_control
            else:
                cost = loss_fn(outputs, labels) 

            # 更新 adversarial images
            adv_grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            grad = (adv_grad + v) / torch.mean(
                torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
            )
            grad = grad + momentum * self.decay
            momentum = grad
            
            
            # 计算梯度方差
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + torch.randn_like(
                    images
                ).uniform_(-self.eps * self.beta, self.eps * self.beta)
                neighbor_images.requires_grad = True
                outputs = self.get_logits(neighbor_images)

                # 计算邻域图像的损失
                if self.targeted:
                    cost = loss_fn(outputs, labels)
                else:
                    cost = loss_fn(outputs, labels)
                GV_grad += torch.autograd.grad(
                    cost, neighbor_images, retain_graph=False, create_graph=False
                )[0]

            # 获取梯度方差
            v = GV_grad / self.N - adv_grad

            #adv_images = adv_images.detach() + self.alpha * grad.sign() * self.k * self.attr * 0.75
            #adv_images = adv_images.detach() + self.alpha * grad.sign()
            #delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            #adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            scaling_factor = torch.where(attr > self.threshold, self.gradient_scale_up, self.gradient_scale_down)  # 高于门限放大，否则缩小
            adjusted_grad = grad.sign() * scaling_factor

            # 更新对抗样本，直接使用调整后的梯度
            adv_images = adv_images.detach() + self.alpha * adjusted_grad
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            
            # 保存每个step生成的adv_images
            adv_images_list.append(adv_images.clone().detach())

        # 返回保存每个step生成的adv_images的list
        return adv_images_list
