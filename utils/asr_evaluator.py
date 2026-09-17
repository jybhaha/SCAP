import torch
from utils.result_saver import ResultManager


class ASREvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model.eval().to(device)
        
        self.device = device
        self.rm = ResultManager.get_instance()
        #self.rm.log(f"模型当前设备: {next(self.model.parameters()).device}")


    def evaluate_dataset(self, dataloader, attack_fn, topk=1):
        """
        原始版本：对完整数据集执行攻击评估。
        """
        total, success = 0, 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            try:
                adv = attack_fn(x, y)
            except Exception as e:
                self.rm.log(f"攻击失败: {e}")
                continue
            batch_total, batch_success = self.evaluate(x, y, adv, topk)
            total += batch_total
            success += batch_success
        return total, success

    def evaluate(self, x: torch.Tensor, y: torch.Tensor, adv: torch.Tensor, topk=1):
        """
        非目标攻击评估：预测不等于原标签即为成功。
        """
        with torch.no_grad():
            output = self.model(adv)
            _, pred = output.topk(topk, dim=1, largest=True, sorted=True)
        y_cpu, pred_cpu = y.cpu(), pred.cpu()
        
        #self.rm.log(f"y_cpu.shape: {y_cpu.shape}, pred_cpu.shape: {pred_cpu.shape}")
        success = sum([y_cpu[i].item() not in pred_cpu[i].tolist() for i in range(len(y))])
        return len(y), success

    def evaluate_targeted(self, x: torch.Tensor, target: torch.Tensor, adv: torch.Tensor, topk=1):
        """
        目标攻击评估：预测中包含目标标签即为成功。
        """
        with torch.no_grad():
            output = self.model(adv)
            _, pred = output.topk(topk, dim=1, largest=True, sorted=True)
        target_cpu, pred_cpu = target.cpu(), pred.cpu()
        #self.rm.log(f"Target labels: {target_cpu.tolist()}")
        #self.rm.log(f"Predicted labels: {pred_cpu.tolist()}")
        success = sum([target_cpu[i].item() in pred_cpu[i].tolist() for i in range(len(target))])
        return len(target), success

    def evaluate_auto(self, x: torch.Tensor, y: torch.Tensor, adv: torch.Tensor, topk=1, target: torch.Tensor = None):
        """
        自动判断是否为目标攻击：
        - 若传入 target → evaluate_targeted()
        - 否则 → evaluate()
        """
        if target is not None:
            return self.evaluate_targeted(x, target, adv, topk)
        else:
            return self.evaluate(x, y, adv, topk)
