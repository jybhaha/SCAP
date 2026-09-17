import torch
import torch.nn as nn

from ..attack import Attack


class VMIFGSM_IQA_LOSS(Attack):
    r"""
    VMIFGSM_IQA in the paper 'Enhancing the Transferability of Adversarial Attacks through Variance Tuning
    [https://arxiv.org/abs/2103.15571], Published as a conference paper at CVPR 2021
    Modified from "https://github.com/JHL-HUST/VT"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of iterations. (Default: 10)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        N (int): the number of sampled examples in the neighborhood. (Default: 5)
        beta (float): the upper bound of neighborhood. (Default: 3/2)
        iqa_loss_weight (float): weight for the IQA loss. (Default: 0.5)
        iqa_metric_loss (IQAMetricLoss): an instance of the IQAMetricLoss class for evaluating IQA loss.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,
        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> iqa_loss = IQAMetricLoss()
        >>> attack = torchattacks.VMIFGSM_IQA(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2, iqa_loss_weight=0.5, iqa_metric_loss=iqa_loss)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, N=5, beta=3 / 2,
        iqa_loss_weight=0.5, iqa_metric_loss=None
    ):
        super().__init__("VMIFGSM_IQA_LOSS", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
        self.iqa_loss_weight = iqa_loss_weight
        self.iqa_metric_loss = iqa_metric_loss
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        v = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate cross-entropy loss
            if self.targeted:
                ce_loss = -loss(outputs, target_labels)
            else:
                ce_loss = loss(outputs, labels)

            # If IQA loss is provided, combine it with cross-entropy loss
            if self.iqa_metric_loss is not None:
                iqa_loss = self.iqa_metric_loss.evaluate_iqa_method_loss(images, adv_images)
                total_loss = ce_loss + self.iqa_loss_weight * iqa_loss
            else:
                total_loss = ce_loss

            # Update adversarial images
            adv_grad = torch.autograd.grad(
                total_loss, adv_images, retain_graph=False, create_graph=False
            )[0]

            grad = (adv_grad + v) / torch.mean(
                torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
            )
            grad = grad + momentum * self.decay
            momentum = grad

            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + torch.randn_like(
                    images
                ).uniform_(-self.eps * self.beta, self.eps * self.beta)
                neighbor_images.requires_grad = True
                outputs = self.get_logits(neighbor_images)

                # Calculate loss for neighbor images
                if self.targeted:
                    neighbor_loss = -loss(outputs, target_labels)
                else:
                    neighbor_loss = loss(outputs, labels)
                GV_grad += torch.autograd.grad(
                    neighbor_loss, neighbor_images, retain_graph=False, create_graph=False
                )[0]
            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class VMIFGSM_IQA_Clamp(Attack):
    def __init__(
        self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, N=5, beta=3 / 2,
        iqa_metric=None, iqa_threshold=0.2
    ):
        super().__init__("VMIFGSM_IQA_Clamp", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.N = N
        self.beta = beta
        self.iqa_metric = iqa_metric
        self.iqa_threshold = iqa_threshold
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        v = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            if self.targeted:
                ce_loss = -loss(outputs, target_labels)
            else:
                ce_loss = loss(outputs, labels)

            # Update adversarial images
            adv_grad = torch.autograd.grad(
                ce_loss, adv_images, retain_graph=False, create_graph=False
            )[0]

            grad = (adv_grad + v) / torch.mean(
                torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
            )
            grad = grad + momentum * self.decay
            momentum = grad

            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + torch.randn_like(
                    images
                ).uniform_(-self.eps * self.beta, self.eps * self.beta)
                neighbor_images.requires_grad = True
                outputs = self.get_logits(neighbor_images)

                if self.targeted:
                    neighbor_loss = -loss(outputs, target_labels)
                else:
                    neighbor_loss = loss(outputs, labels)

                GV_grad += torch.autograd.grad(
                    neighbor_loss, neighbor_images, retain_graph=False, create_graph=False
                )[0]
            v = GV_grad / self.N - adv_grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()

            # Calculate IQA score for the adversarial images
            if self.iqa_metric is not None:
                iqa_score = self.iqa_metric.evaluate_iqa_method_loss(images, adv_images)

                # If IQA score exceeds the threshold, scale the perturbation
                delta = adv_images - images
                if iqa_score > self.iqa_threshold:
                    scaling_factor = self.iqa_threshold / iqa_score
                    delta *= scaling_factor

                # Apply the (scaled) perturbation to the original images
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            else:
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

