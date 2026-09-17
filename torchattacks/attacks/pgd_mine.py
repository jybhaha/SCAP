import torch
import torch.nn as nn

from ..attack import Attack

from IQA.IQA import IQAMetricLoss

from utils import Config, setup_logger, lg

class PGD_MINE(Attack):
    r"""
    
    这个版本用来返回每个step的adv_images，以便在测试时记录每个step的对抗样本
    
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True,threshold = 1,gradient_scale_up = 1,gradient_scale_down =1):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        #self.iqa_loss = IQAMetricLoss()
        #self.iqa_loss_control = iqa_loss_control
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
        adv_images = images.clone().detach()
        # 计算 IQA 损失
        #iqa_loss_value = self.iqa_loss.evaluate_iqa_method_loss(images, adv_images)
        #lg("iqa_loss_value: {}".format(iqa_loss_value))
        
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
                
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()


        # 创建一个list保存每个step生成的adv_images
        adv_images_list = []
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                #eloss = -loss(outputs, target_labels)
                #iqaloss = -iqa_loss_value * self.iqa_loss_control
                #lg("eloss: {}, iqaloss: {}".format(eloss, iqaloss))
                #cost = eloss + iqaloss
                #cost = -loss(outputs, target_labels)  - iqa_loss_value * self.iqa_loss_control
                cost = -loss(outputs, target_labels)
            else:
                #cost = loss(outputs, labels) + iqa_loss_value * self.iqa_loss_control
                cost = loss(outputs, labels)
            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

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

