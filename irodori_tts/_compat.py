"""Runtime compatibility patches for environments with incomplete torch builds."""

from __future__ import annotations

import enum

import torch.distributed as _dist

if not hasattr(_dist, "ReduceOp"):
    # AMD ROCm Windows builds ship a stripped torch.distributed that omits
    # ReduceOp. audiotools (a dacvae dependency) references dist.ReduceOp at
    # class-definition time, so we inject a stub before dacvae is imported.
    class _ReduceOp(enum.IntEnum):
        SUM = 0
        PRODUCT = 1
        MIN = 2
        MAX = 3
        BAND = 4
        BOR = 5
        BXOR = 6
        AVG = 7
        PREMUL_SUM = 8
        UNUSED = 9

    _dist.ReduceOp = _ReduceOp  # type: ignore[attr-defined]
