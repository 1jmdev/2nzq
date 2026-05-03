from __future__ import annotations

import math


def sigmoid_tau(step: int, total_steps: int, k: float = 10.0) -> float:
    if total_steps <= 0:
        return 1.0
    x = k * (step / total_steps - 0.5)
    return 1.0 / (1.0 + math.exp(-x))


def pqaft_tau(step: int, total_steps: int, k: float = 10.0, warmup_fraction: float = 0.05, hard_fraction: float = 0.9) -> float:
    progress = step / max(total_steps, 1)
    if progress < warmup_fraction:
        return 0.0
    if progress >= hard_fraction:
        return 1.0
    return sigmoid_tau(step, total_steps, k)


def pqaft_lr_factor(step: int, total_steps: int, min_factor: float = 0.01) -> float:
    progress = step / max(total_steps, 1)
    if progress >= 0.9:
        return min_factor
    return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress / 0.9))
