import torch
import torch.nn as nn

from ..attack import Attack


class FFGSM_MINE(Attack):
    r"""
    New FGSM proposed in 'Fast is better than free: Revisiting adversarial training'
    [https://arxiv.org/abs/2001.03994]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 10/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8 / 255, alpha=10 / 255,threshold = 1,gradient_scale_up = 1,gradient_scale_down =1):
        super().__init__("FFGSM", model)
        self.eps = eps
        self.alpha = alpha
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

        loss = nn.CrossEntropyLoss()

        adv_images = images + torch.randn_like(images).uniform_(
            -self.eps, self.eps
        )  # nopep8
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images.requires_grad = True

        outputs = self.get_logits(adv_images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]

        scaling_factor = torch.where(attr > self.threshold, self.gradient_scale_up, self.gradient_scale_down)  # 高于门限放大，否则缩小
        adjusted_grad = grad.sign() * scaling_factor

        # 更新对抗样本，直接使用调整后的梯度
        adv_images = adv_images.detach() + self.alpha * adjusted_grad
        #adv_images = adv_images + self.alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
