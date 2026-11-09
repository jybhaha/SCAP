"""Public attack entry point.

The implementation lives in ``models.my_patch_attack`` to avoid divergent pipelines.
"""

from models.my_patch_attack import MyPatchAttack, MyPatchAttack_Eva

__all__ = ["MyPatchAttack", "MyPatchAttack_Eva"]
