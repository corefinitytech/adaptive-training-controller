"""Accelerator resolution.

The single place `torch.device` selection happens. Nothing else in this package calls
`.cuda()` or hardcodes an accelerator string — everything asks `resolve_device()`, so the
same code path that runs on this project's Mac (CPU/MPS) target today also runs on a CUDA
cluster later without a rewrite.
"""

from __future__ import annotations

import torch


def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick the best available accelerator, or honor an explicit choice.

    `preferred` may be any string `torch.device` accepts (`"cpu"`, `"mps"`, `"cuda"`,
    `"cuda:0"`, ...). If given, it is used as-is — the caller is asserting it's available.
    Otherwise: CUDA, then Apple Silicon MPS, then CPU.
    """
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
