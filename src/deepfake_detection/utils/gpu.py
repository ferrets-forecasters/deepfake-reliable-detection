from __future__ import annotations

import torch


def cap_gpu_memory(fraction: float | None) -> None:
    """Keeps this process's CUDA allocations under a share of the card, so a scoring run can sit
    beside trainings without pushing the card into shared system memory."""
    if fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
