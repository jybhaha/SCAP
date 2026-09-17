import os
import datetime
import threading
import queue
import json
import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.utils as vutils


class Logger:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.file_handle = open(log_path, "a", encoding="utf-8")

        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.worker_thread.start()

    def info(self, msg: str):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{now_str}] [INFO] {msg}"
        print(log_line)
        self.log_queue.put(log_line)

    def _write_worker(self):
        while not self.stop_event.is_set() or not self.log_queue.empty():
            try:
                line = self.log_queue.get(timeout=0.1)
                self.file_handle.write(line + "\n")
                self.file_handle.flush()
                self.log_queue.task_done()
            except queue.Empty:
                continue

    def close(self):
        self.stop_event.set()
        self.worker_thread.join()
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def __del__(self):
        self.close()


class ResultManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.current_exp = "default_exp"
        self.current_test = "test1"
        self.base_dir = "Results"
        self.logger = None
        self.save_images_enabled = False  # 新增：是否启用保存图片功能

        # 关键修复：固定时间戳，只在初始化时生成一次
        self._date_str = datetime.datetime.now().strftime("%Y%m%d")  # 日期格式：20251010
        self._timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 完整时间戳：20251010_105227
        self._init_dirs()
        self._init_logger()

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls()
        return cls._instance

    def enable_image_saving(self, enabled=True):
        """启用或禁用图片保存功能"""
        self.save_images_enabled = enabled
        if enabled:
            self.log("图片保存功能已启用")
        else:
            self.log("图片保存功能已禁用")

    def _init_dirs(self):
        # 关键修复：使用固定的 self._timestamp，不再重新获取时间
        self.exp_dir = os.path.join(self.base_dir, self._timestamp, self.current_exp)
        self.test_dir = os.path.join(self.exp_dir, self.current_test)
        self.img_dir = os.path.join(self.test_dir, "images")
        self.log_dir = os.path.join(self.test_dir, "logs")
        self.model_dir = os.path.join(self.test_dir, "models")

        for d in [self.img_dir, self.log_dir, self.model_dir]:
            os.makedirs(d, exist_ok=True)

    def _init_logger(self):
        if self.logger is not None:
            self.logger.close()
        log_path = os.path.join(self.log_dir, f"{self.current_test}.log")
        self.logger = Logger(log_path)

    def set_experiment(self, exp_name: str):
        if self.current_exp != exp_name:
            self.current_exp = exp_name
            self._reset_test_dirs_only()
            self._init_logger()  # 确保日志也写到正确 test 路径

    def set_test(self, test_name: str):
        if self.current_test != test_name:
            self.current_test = test_name
            self._reset()  # ✅ 只重新构建 test 目录，不动 exp_dir

    def _reset(self):
        self._init_dirs()
        self._init_logger()

    def log(self, msg: str):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{now_str}] [{self.current_exp}/{self.current_test}] {msg}"
        print(log_line)
        if self.logger:
            self.logger.info(msg)

    def save_image(self, img, filename_prefix="image", ext="png"):
        if isinstance(img, torch.Tensor):
            img = self._tensor_to_pil(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        elif not isinstance(img, Image.Image):
            raise TypeError("Unsupported image type")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.{ext}"
        save_path = os.path.join(self.img_dir, filename)
        img.save(save_path)
        self.log(f"Saved image: {save_path}")
        return save_path

    def save_images_batch(self, imgs, filename_prefix="image", ext="png"):
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        paths = []
        for i, img in enumerate(imgs):
            prefix = f"{filename_prefix}_{i}"
            path = self.save_image(img, filename_prefix=prefix, ext=ext)
            paths.append(path)
        return paths

    def save_attack_images(self, original_img, perturbation, adv_img, batch_idx=0, method_name="attack", ext="png"):
        """保存攻击相关的三张图片：原始图片、扰动、对抗样本
        
        Args:
            original_img: 原始图片 tensor [B, C, H, W] 或 [C, H, W]
            perturbation: 扰动 tensor [B, C, H, W] 或 [C, H, W]
            adv_img: 对抗样本 tensor [B, C, H, W] 或 [C, H, W]
            batch_idx: 批次索引
            method_name: 攻击方法名称
            ext: 图片格式
        """
        if not self.save_images_enabled:
            return None
            
        # 确保输入是4D tensor [B, C, H, W]
        if original_img.dim() == 3:
            original_img = original_img.unsqueeze(0)
        if perturbation.dim() == 3:
            perturbation = perturbation.unsqueeze(0)
        if adv_img.dim() == 3:
            adv_img = adv_img.unsqueeze(0)
            
        # 获取第一个样本（如果有多张图片）
        original = original_img[0] if original_img.size(0) > 0 else original_img
        pert = perturbation[0] if perturbation.size(0) > 0 else perturbation
        adv = adv_img[0] if adv_img.size(0) > 0 else adv_img
        
        # 确保图片在[0,1]范围内
        original = original.clamp(0, 1)
        pert = pert.clamp(-1, 1)  # 扰动可能在[-1,1]范围内
        adv = adv.clamp(0, 1)
        
        # 将扰动归一化到[0,1]用于可视化
        pert_vis = (pert + 1) / 2 if pert.min() < 0 else pert
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始图片
        original_filename = f"{method_name}_original_batch{batch_idx}_{timestamp}.{ext}"
        original_path = os.path.join(self.img_dir, original_filename)
        vutils.save_image(original, original_path)
        
        # 保存扰动
        pert_filename = f"{method_name}_perturbation_batch{batch_idx}_{timestamp}.{ext}"
        pert_path = os.path.join(self.img_dir, pert_filename)
        vutils.save_image(pert_vis, pert_path)
        
        # 保存对抗样本
        adv_filename = f"{method_name}_adversarial_batch{batch_idx}_{timestamp}.{ext}"
        adv_path = os.path.join(self.img_dir, adv_filename)
        vutils.save_image(adv, adv_path)
        
        self.log(f"Saved attack images for {method_name} batch {batch_idx}:")
        self.log(f"  Original: {original_path}")
        self.log(f"  Perturbation: {pert_path}")
        self.log(f"  Adversarial: {adv_path}")
        
        return {
            'original': original_path,
            'perturbation': pert_path,
            'adversarial': adv_path
        }

    def save_model(self, model_state_dict, filename_prefix="model"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pth"
        save_path = os.path.join(self.model_dir, filename)
        torch.save(model_state_dict, save_path)
        self.log(f"Saved model: {save_path}")
        return save_path

    def save_json(self, data, filename_prefix="data"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        save_path = os.path.join(self.test_dir, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        self.log(f"Saved JSON: {save_path}")
        return save_path

    def _tensor_to_pil(self, tensor_img):
        img = tensor_img.detach().cpu().clamp(0, 1)
        if img.ndim == 3:
            return TF.to_pil_image(img)
        elif img.ndim == 4:
            return [TF.to_pil_image(i) for i in img]
        else:
            raise ValueError("Unsupported tensor shape for conversion to image.")
            
    def _reset_test_dirs_only(self):
        # 关键修复：使用固定的 self._timestamp，不再重新获取时间
        self.exp_dir = os.path.join(self.base_dir, self._timestamp, self.current_exp)
        self.test_dir = os.path.join(self.exp_dir, self.current_test)
        self.img_dir = os.path.join(self.test_dir, "images")
        self.log_dir = os.path.join(self.test_dir, "logs")
        self.model_dir = os.path.join(self.test_dir, "models")

        for d in [self.img_dir, self.log_dir, self.model_dir]:
            os.makedirs(d, exist_ok=True)
