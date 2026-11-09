"""Backward-compatible name for the unified joint-loss optimiser."""

from models.patch_optimizer import PatchOptimizer


class PatchOptimizer_ChangeLoss(PatchOptimizer):
    """Compatibility subclass; joint TV/CIELab loss now lives in PatchOptimizer."""

    pass
