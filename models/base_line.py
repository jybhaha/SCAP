import torch

from torchattacks.attacks.pgd import PGD_Eva 
from torchattacks.attacks.fgsm import FGSM_Eva
from torchattacks.attacks.mifgsm import MIFGSM_Eva
from torchattacks.attacks.difgsm import DIFGSM_Eva
from torchattacks.attacks.nifgsm import NIFGSM_Eva
from torchattacks.attacks.vnifgsm import VNIFGSM_Eva
from torch.autograd import Variable
import torch.nn.functional as F
from utils.result_saver import ResultManager

class PixelAttackBaseline:
    def __init__(self, model, eps=8/255, default_steps=10):
        self.model = model
        self.eps = eps
        self.steps = default_steps
        self.result_manager = ResultManager.get_instance()

    def get_attack(self, name, **kwargs):
        name = name.upper()
        #self.result_manager.log(f"Initializing {name} attack with eps={self.eps}")
        if name == "FGSM":
            return FGSM_Eva(self.model, eps=self.eps)
        elif name == "MIFGSM":
            return MIFGSM_Eva(self.model, eps=self.eps, steps=self.steps, decay=1.0, **kwargs)
        elif name == "DIFGSM":
            return DIFGSM_Eva(self.model, eps=self.eps, steps=self.steps, diversity_prob=0.7, **kwargs)
        elif name == "NIFGSM":
            return NIFGSM_Eva(self.model, eps=self.eps, steps=self.steps, **kwargs)
        elif name == "PGD":
            return PGD_Eva(self.model, eps=self.eps, alpha=2/255, steps=self.steps, **kwargs)
        elif name == "VNIFGSM":
            return VNIFGSM_Eva(self.model, eps=self.eps, steps=self.steps, **kwargs)
        else:
            raise ValueError(f"Unsupported attack type: {name}")

    def run(self, attack_name, x, y, target=None, **kwargs):
        attack = self.get_attack(attack_name, **kwargs)

        # 🔑 设置目标攻击模式
        if target is not None:
            if hasattr(attack, "set_mode_targeted_by_label"):
                attack.set_mode_targeted_by_label(quiet=True)
            labels = target
        else:
            labels = y

        # ✅ 调用 attack（兼容 __call__ 或 forward）
        if hasattr(attack, "__call__"):
            x_adv = attack(x, labels)
        else:
            x_adv = attack.forward(x, labels)

        return x_adv




import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# ART 相关
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import DPatch, AdversarialPatch

class PatchAttackBaseline:
    def __init__(self, model, input_size=(224, 224), device='cuda', num_classes=1000):
        self.model = model.to(device).eval()
        self.input_size = input_size
        self.device = device
        self.num_classes = num_classes

        import torch.nn as nn
        import torch.optim as optim
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        self.art_classifier = PyTorchClassifier(
            model=self.model,
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=(3, *input_size),
            nb_classes=num_classes,
            clip_values=(0.0, 1.0),
            device_type='gpu' if 'cuda' in str(device) else 'cpu',
        )

        self.methods = {
            'advpatch': self.advpatch,
            'lavan': self.lavan,
            'lavan_eva': self.lavan_eva,

            'art_advpatch': self.art_advpatch,
            'art_advpatch_eva': self.art_advpatch_eva,
        }
        
        self.result_manager = ResultManager.get_instance()

    def run(self, method, x, y=None, target=None, **kwargs):
        method = method.lower()
        if method not in self.methods:
            raise ValueError(f"不支持的Patch攻击方法: {method}")
        return self.methods[method](x, y, target=target, **kwargs)

    def advpatch(self, x, y=None, target=None, patch_path=None, alpha=0.2, **kwargs):
        if patch_path is None:
            raise ValueError("请提供 patch_path")
        patch = Image.open(patch_path).convert("RGB")
        patch = T.Resize((self.input_size[0]//4, self.input_size[1]//4))(patch)
        patch = T.ToTensor()(patch).unsqueeze(0).to(self.device)
        x_adv = x.clone()
        x_adv[:, :, :patch.shape[2], :patch.shape[3]] = \
            (1 - alpha) * x[:, :, :patch.shape[2], :patch.shape[3]] + alpha * patch
        return x_adv

    def lavan(self, x, y=None, target=None, patch_coords=None, epsilon=0.01, iterations=300, **kwargs):
        device = self.device
        model = self.model
        model.eval()

        b, c, h, w = x.shape
        if patch_coords is None:
            patch_coords = {
                'x_min': w // 2 - 25,
                'x_max': w // 2 + 25,
                'y_min': h // 2 - 25,
                'y_max': h // 2 + 25,
            }

        x_min, x_max = patch_coords['x_min'], patch_coords['x_max']
        y_min, y_max = patch_coords['y_min'], patch_coords['y_max']

        patch = torch.zeros_like(x).to(device)
        mask = torch.zeros_like(x).to(device)
        mask[:, :, y_min:y_max, x_min:x_max] = 1.0
        patch[:, :, y_min:y_max, x_min:x_max] = torch.rand_like(patch[:, :, y_min:y_max, x_min:x_max])

        x_adv = x.clone()

        # 转换标签
        if isinstance(y, int):
            y = torch.tensor([y], dtype=torch.long, device=device)
        elif isinstance(y, torch.Tensor):
            y = y.to(device)

        if target is not None:
            target_tensor = target if isinstance(target, torch.Tensor) else torch.tensor([target], dtype=torch.long, device=device)
        else:
            target_tensor = None

        for _ in range(iterations):
            x_adv = Variable(x_adv.data, requires_grad=True)
            outputs = model(x_adv)

            if target_tensor is not None:
                loss = -F.cross_entropy(outputs, target_tensor)
            else:
                loss = F.cross_entropy(outputs, y)

            loss.backward()

            grad = x_adv.grad.data
            patch_grad = grad * mask

            patch = patch - epsilon * patch_grad
            patch = torch.clamp(patch, 0.0, 1.0)
            x_adv = x * (1 - mask) + patch * mask
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    def lavan_eva(self, x, y=None, target=None, patch_coords=None, epsilon=0.01, iterations=300, **kwargs):
        #self.result_manager.log(f"lavan_eva")
        device = self.device
        model = self.model
        model.eval()
        #self.result_manager.log(f"222")
        b, c, h, w = x.shape
        if patch_coords is None:
            patch_coords = {
                'x_min': w // 2 - 25,
                'x_max': w // 2 + 25,
                'y_min': h // 2 - 25,
                'y_max': h // 2 + 25,
            }
        #self.result_manager.log(f"444")
        x_min, x_max = patch_coords['x_min'], patch_coords['x_max']
        y_min, y_max = patch_coords['y_min'], patch_coords['y_max']
        #self.result_manager.log(f"333")
        patch = torch.zeros_like(x).to(device)
        mask = torch.zeros_like(x).to(device)
        mask[:, :, y_min:y_max, x_min:x_max] = 1.0
        patch[:, :, y_min:y_max, x_min:x_max] = torch.rand_like(patch[:, :, y_min:y_max, x_min:x_max])
        #self.result_manager.log(f"555")
        x_adv = x.clone()
        #self.result_manager.log(f"666")
        # 转换标签
        if isinstance(y, int):
            y = torch.tensor([y], dtype=torch.long, device=device)
        elif isinstance(y, torch.Tensor):
            y = y.to(device)
        #self.result_manager.log(f"777")
        if target is not None:
            target_tensor = target if isinstance(target, torch.Tensor) else torch.tensor([target], dtype=torch.long, device=device)
        else:
            target_tensor = None
        #self.result_manager.log(f"888")
        step_preds_list = []
        #self.result_manager.log(f"999")
        for _ in range(iterations):
            if x_adv is not None:
                x_adv = Variable(x_adv.data, requires_grad=True)
            outputs = model(x_adv)
            #self.result_manager.log(f"1000")
            # 记录每步预测
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                step_preds_list.append(preds.detach().cpu())
            #self.result_manager.log(f"1111")
            loss = None
            if target_tensor is not None:
                loss = -F.cross_entropy(outputs, target_tensor)
            elif y is not None:
                loss = F.cross_entropy(outputs, y)
            #self.result_manager.log(f"1222")
            if loss is not None:
                loss.backward()
                grad = x_adv.grad.data
                patch_grad = grad * mask

                patch = patch - epsilon * patch_grad
                patch = torch.clamp(patch, 0.0, 1.0)
                x_adv = x * (1 - mask) + patch * mask
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv, step_preds_list

    

    def art_advpatch(self, x, y, target=None, patch_shape=None, scale=0.3, max_iter=500, **kwargs):
        x_np = x.detach().cpu().numpy()
        if x_np.max() > 1:
            x_np = x_np / 255.0

        # 如果传入 target，则进行目标攻击
        is_targeted = target is not None
        label_tensor = target if is_targeted else y
        y_np = label_tensor.detach().cpu().numpy() if isinstance(label_tensor, torch.Tensor) else np.array([label_tensor])

        attack = AdversarialPatch(
            classifier=self.art_classifier,
            rotation_max=22.5,
            scale_min=scale / 2,
            scale_max=scale,
            learning_rate=5.0,
            max_iter=max_iter,
            batch_size=x_np.shape[0],
            patch_shape=patch_shape,
            targeted=is_targeted,
            verbose=False,
        )

        patch, _ = attack.generate(x=x_np, y=y_np)
        x_adv_np = attack.apply_patch(x_np, scale=scale, patch_external=patch)
        x_adv = torch.tensor(x_adv_np).to(self.device).float()
        assert x_adv.shape == x.shape, f"输出形状不一致: x_adv {x_adv.shape}, x {x.shape}"
        return x_adv

    def art_advpatch_eva(self, x, y, target=None, patch_shape= (64, 64, 3), scale=0.3, max_iter=500, **kwargs):
        x_np = x.detach().cpu().numpy()
        if x_np.max() > 1:
            x_np = x_np / 255.0

        # 如果传入 target，则进行目标攻击
        is_targeted = target is not None
        label_tensor = target if is_targeted else y
        y_np = label_tensor.detach().cpu().numpy() if isinstance(label_tensor, torch.Tensor) else np.array([label_tensor])
        #self.result_manager.log(f"11111")
        #self.result_manager.log(f"x_np: {x_np.shape}")
        attack = AdversarialPatch(
            classifier=self.art_classifier,
            rotation_max=22.5,
            scale_min=scale / 2,
            scale_max=scale,
            learning_rate=5.0,
            max_iter=max_iter,
            batch_size=len(x_np),
            patch_shape=patch_shape,
            targeted=is_targeted,
            verbose=False,
        )
        #self.result_manager.log(f"22222")
        patch, _ = attack.generate(x=x_np, y=y_np)
        x_adv_np = attack.apply_patch(x_np, scale=scale, patch_external=patch)
        x_adv = torch.tensor(x_adv_np).to(self.device).float()
        assert x_adv.shape == x.shape, f"输出形状不一致: x_adv {x_adv.shape}, x {x.shape}"
        #self.result_manager.log(f"33333")
        # 评估每步预测（这里只能评估原图和最终对抗图）
        step_preds_list = []
        with torch.no_grad():
            preds_orig = torch.argmax(self.model(x), dim=1)
            preds_adv = torch.argmax(self.model(x_adv), dim=1)
            step_preds_list.append(preds_orig.detach().cpu())
            step_preds_list.append(preds_adv.detach().cpu())
        #self.result_manager.log(f"44444")
        return x_adv, step_preds_list

