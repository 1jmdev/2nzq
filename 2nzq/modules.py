from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .quantization import dequantize_codes, init_scales, pack_codes, quantize_groups, ste_quantized_weight, unpack_codes


@dataclass(frozen=True)
class QuantizedLinearState:
    name: str
    in_features: int
    out_features: int
    bias: bool
    group_size: int
    weight_numel: int
    weight_shape: tuple[int, int]
    scales: torch.Tensor
    packed_weight: torch.Tensor
    bias_tensor: torch.Tensor | None


class TwoNZQLinear(nn.Module):
    def __init__(self, linear: nn.Linear, group_size: int = 128) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None
        scales = init_scales(self.weight, group_size).to(device=self.weight.device, dtype=torch.float32)
        self.scales = nn.Parameter(scales)
        self.tau = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = ste_quantized_weight(self.weight, self.scales, self.tau if self.training else 1.0, self.group_size)
        return F.linear(x, weight.to(dtype=x.dtype), self.bias.to(dtype=x.dtype) if self.bias is not None else None)

    @torch.no_grad()
    def export_state(self, name: str) -> QuantizedLinearState:
        codes = quantize_groups(self.weight.float().reshape(-1), self.scales.detach().float(), self.group_size)
        return QuantizedLinearState(
            name=name,
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
            group_size=self.group_size,
            weight_numel=self.weight.numel(),
            weight_shape=(self.out_features, self.in_features),
            scales=self.scales.detach().half().cpu(),
            packed_weight=pack_codes(codes.cpu()),
            bias_tensor=self.bias.detach().half().cpu() if self.bias is not None else None,
        )


class TwoNZQInferenceLinear(nn.Module):
    def __init__(self, state: QuantizedLinearState) -> None:
        super().__init__()
        self.in_features = state.in_features
        self.out_features = state.out_features
        self.group_size = state.group_size
        self.weight_numel = state.weight_numel
        self.weight_shape = state.weight_shape
        self.register_buffer("scales", state.scales.contiguous())
        self.register_buffer("packed_weight", state.packed_weight.contiguous())
        if state.bias_tensor is not None:
            self.register_buffer("bias", state.bias_tensor.contiguous())
        else:
            self.bias = None

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        codes = unpack_codes(self.packed_weight, self.weight_numel)
        return dequantize_codes(codes, self.scales.float(), self.weight_shape, self.group_size).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight(x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def should_quantize_linear(name: str, module: nn.Linear, quantize_lm_head: bool = False) -> bool:
    lname = name.lower()
    if not quantize_lm_head and (lname == "lm_head" or lname.endswith(".lm_head")):
        return False
    if "embed" in lname or "embedding" in lname:
        return False
    return module.weight.ndim == 2


def replace_linears_for_qat(model: nn.Module, group_size: int = 128, quantize_lm_head: bool = False) -> list[str]:
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and should_quantize_linear(full_name, child, quantize_lm_head):
                setattr(parent, child_name, TwoNZQLinear(child, group_size))
                replaced.append(full_name)
            else:
                visit(child, full_name)

    visit(model)
    return replaced


def set_tau(model: nn.Module, tau: float) -> None:
    for module in model.modules():
        if isinstance(module, TwoNZQLinear):
            module.tau = float(tau)


def collect_quantized_states(model: nn.Module) -> dict[str, QuantizedLinearState]:
    states: dict[str, QuantizedLinearState] = {}
    for name, module in model.named_modules():
        if isinstance(module, TwoNZQLinear):
            states[name] = module.export_state(name)
    return states


def replace_linears_for_inference(model: nn.Module, states: dict[str, QuantizedLinearState]) -> None:
    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if full_name in states:
                setattr(parent, child_name, TwoNZQInferenceLinear(states[full_name]))
            else:
                visit(child, full_name)

    visit(model)
