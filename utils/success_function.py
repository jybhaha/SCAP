import torch
import torch.nn.functional as F

def validate_attack_success(
    model,
    adv_tensor: torch.Tensor,
    true_label: torch.Tensor,
    target_label: torch.Tensor = None,
    topk: int = 1,
    targeted: bool = False,
    device: str = 'cuda',
    return_probs: bool = False
):
    """
    验证对抗样本是否攻击成功（支持 untargeted / targeted 模式）

    参数:
        model: 分类器模型
        adv_tensor: [1, 3, H, W] 对抗图像张量（需在 device 上）
        true_label: [1] 原始标签
        target_label: [1] 目标标签（仅在 targeted 模式使用）
        topk: int，判断是否在 top-k 中攻击成功
        targeted: bool，是否为 targeted 攻击
        device: str，模型与数据所在设备
        return_probs: 是否返回 softmax 概率分布

    返回:
        success: bool 是否攻击成功
        pred_label: int 模型预测的 top-1 类别
        probs (可选): Tensor 模型 softmax 概率分布
    """
    model.eval()
    with torch.no_grad():
        logits = model(adv_tensor.to(device))  # [1, C]
        probs = F.softmax(logits, dim=1)
        topk_preds = probs.topk(k=topk, dim=1).indices[0]  # shape: [topk]
        pred_label = logits.argmax(dim=1).item()

        if targeted:
            assert target_label is not None, "Targeted 攻击需指定 target_label"
            success = target_label.item() in topk_preds
        else:
            success = true_label.item() not in topk_preds

        if return_probs:
            return success, pred_label, probs.squeeze().cpu()
        return success, pred_label
