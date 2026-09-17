import torch
import torch.nn as nn

from ..attack import Attack

class SINIFGSM_MINE(Attack):
    r"""
    SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Modified from "https://githuba.com/JHL-HUST/SI-NI-FGSM"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        m (int): number of scale copies. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SINIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, m=5,threshold = 1,gradient_scale_up = 1,gradient_scale_down =1):
        super().__init__("SINIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
        self.supported_mode = ["default", "targeted"]
        self.threshold = threshold
        self.gradient_scale_up = gradient_scale_up
        self.gradient_scale_down = gradient_scale_down

    def forward(self, images, labels, attr):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        # 创建一个list保存每个step生成的adv_images
        adv_images_list = []
        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_image = adv_images + self.decay * self.alpha * momentum
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(images).detach().to(self.device)
            for i in torch.arange(self.m):
                nes_images = nes_image / torch.pow(2, i)
                outputs = self.get_logits(nes_images)
                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
                adv_grad += torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
            adv_grad = adv_grad / self.m

            # Update adversarial images
            grad = self.decay * momentum + adv_grad / torch.mean(
                torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True
            )
            momentum = grad
            
            scaling_factor = torch.where(attr > self.threshold, self.gradient_scale_up, self.gradient_scale_down)  # 高于门限放大，否则缩小
            adjusted_grad = grad.sign() * scaling_factor

            # 更新对抗样本，直接使用调整后的梯度
            adv_images = adv_images.detach() + self.alpha * adjusted_grad
            #adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # 保存每个step生成的adv_images
            adv_images_list.append(adv_images.clone().detach())

        # 返回保存每个step生成的adv_images的list
        return adv_images_list
