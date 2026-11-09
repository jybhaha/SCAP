"""CUDA-aware wall-time and peak-memory profiling with JSON diagnostics."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

import numpy as np
import torch


def synchronize(device=None):
    if torch.cuda.is_available() and (device is None or torch.device(device).type == "cuda"):
        torch.cuda.synchronize(device)


class StageProfiler:
    def __init__(self, device=None):
        self.device = torch.device(device) if device is not None else None
        self.cuda_enabled = torch.cuda.is_available() and self.device is not None and self.device.type == "cuda"
        self.stages = {}
        self.peak_memory_bytes = 0

    def start(self):
        if self.cuda_enabled:
            synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)

    @contextmanager
    def stage(self, name):
        if name in self.stages:
            raise ValueError(f"Stage {name!r} was already recorded; repeated stages need distinct names")
        if self.cuda_enabled:
            synchronize(self.device)
            # Reset immediately before every stage. Otherwise max_memory_allocated()
            # is the cumulative peak since start(), not this stage's own peak.
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.cuda_enabled:
                synchronize(self.device)
            elapsed = time.perf_counter() - start
            stage_peak = int(torch.cuda.max_memory_allocated(self.device)) if self.cuda_enabled else 0
            self.stages[name] = {
                "wall_time_seconds": elapsed,
                "cuda_peak_memory_bytes": stage_peak,
                "memory_measurement": "independent_stage_peak" if self.cuda_enabled else "cpu_no_cuda_memory",
            }
            self.peak_memory_bytes = max(self.peak_memory_bytes, stage_peak)

    def summary(self):
        return {
            "stages": self.stages,
            "total_wall_time_seconds": sum(stage["wall_time_seconds"] for stage in self.stages.values()),
            "cuda_peak_memory_bytes": self.peak_memory_bytes,
            "cuda_synchronized": self.cuda_enabled,
        }


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_diagnostics(diagnostics, path):
    destination = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(destination))
    os.makedirs(parent, exist_ok=True)
    temporary = destination + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(diagnostics), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, destination)
