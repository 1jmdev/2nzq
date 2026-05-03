from __future__ import annotations

import math

import torch


LUT_VALUES = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)


def get_lut(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(LUT_VALUES, device=device, dtype=dtype)


def pad_to_group(flat: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    original_numel = flat.numel()
    pad = (-original_numel) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return flat, original_numel


def init_scales(weight: torch.Tensor, group_size: int = 128, eps: float = 1e-8) -> torch.Tensor:
    flat, _ = pad_to_group(weight.detach().float().reshape(-1), group_size)
    groups = flat.reshape(-1, group_size)
    return groups.abs().amax(dim=1).clamp_min(eps)


def quantize_groups(flat: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    flat, original_numel = pad_to_group(flat, group_size)
    groups = flat.reshape(-1, group_size)
    norm = groups / scales.float().reshape(-1, 1).clamp_min(1e-8)
    codes = torch.zeros_like(norm, dtype=torch.uint8)
    codes = torch.where(norm >= -2.0 / 3.0, torch.ones_like(codes), codes)
    codes = torch.where(norm >= 0.0, torch.full_like(codes, 2), codes)
    codes = torch.where(norm >= 2.0 / 3.0, torch.full_like(codes, 3), codes)
    return codes.reshape(-1)[:original_numel]


def dequantize_codes(codes: torch.Tensor, scales: torch.Tensor, shape: torch.Size | tuple[int, ...], group_size: int = 128) -> torch.Tensor:
    codes, original_numel = pad_to_group(codes.reshape(-1), group_size)
    lut = get_lut(codes.device, scales.dtype if scales.is_floating_point() else torch.float32)
    values = lut[codes.long()].reshape(-1, group_size)
    out = values * scales.reshape(-1, 1).to(values.dtype)
    return out.reshape(-1)[:original_numel].reshape(shape)


def quantize_tensor(weight: torch.Tensor, scales: torch.Tensor | None = None, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scales is None:
        scales = init_scales(weight, group_size).to(device=weight.device, dtype=torch.float32)
    codes = quantize_groups(weight.float().reshape(-1), scales.float(), group_size)
    dequantized = dequantize_codes(codes, scales.float(), weight.shape, group_size).to(weight.dtype)
    return codes, scales, dequantized


def ste_quantized_weight(weight: torch.Tensor, scales: torch.Tensor, tau: float, group_size: int = 128) -> torch.Tensor:
    _, _, hard = quantize_tensor(weight, scales, group_size)
    if tau <= 0.0:
        return weight
    hard_ste = hard + weight - weight.detach()
    if tau >= 1.0:
        return hard_ste
    return (1.0 - tau) * weight + tau * hard_ste


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.reshape(-1).to(torch.uint8)
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    groups = flat.reshape(-1, 4)
    return groups[:, 0] | (groups[:, 1] << 2) | (groups[:, 2] << 4) | (groups[:, 3] << 6)


def unpack_codes(packed: torch.Tensor, numel: int) -> torch.Tensor:
    packed = packed.reshape(-1).to(torch.uint8)
    codes = torch.empty(packed.numel() * 4, device=packed.device, dtype=torch.uint8)
    codes[0::4] = packed & 0x03
    codes[1::4] = (packed >> 2) & 0x03
    codes[2::4] = (packed >> 4) & 0x03
    codes[3::4] = (packed >> 6) & 0x03
    return codes[:numel]


def bits_per_weight(num_weights: int, num_groups: int) -> float:
    return ((math.ceil(num_weights / 4) + 2 * num_groups) * 8) / max(num_weights, 1)
