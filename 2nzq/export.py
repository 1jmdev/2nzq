from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .config import ensure_dir
from .modules import QuantizedLinearState, collect_quantized_states
from .quantization import bits_per_weight


def _state_to_dict(state: QuantizedLinearState) -> dict[str, object]:
    data = asdict(state)
    return data


def save_2nzq_model(model: nn.Module, output_path: str | Path, base_model: str, group_size: int, quantized_layers: list[str]) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    qstates = collect_quantized_states(model)
    full_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".packed_weight" not in k}
    for name in qstates:
        prefix = f"{name}."
        full_state = {k: v for k, v in full_state.items() if not k.startswith(prefix)}

    qdict = {name: _state_to_dict(state) for name, state in qstates.items()}
    for name, state in qstates.items():
        if not torch.isfinite(state.scales.float()).all():
            raise FloatingPointError(f"Refusing to export non-finite 2NZQ scales in layer {name}")
        if state.bias_tensor is not None and not torch.isfinite(state.bias_tensor.float()).all():
            raise FloatingPointError(f"Refusing to export non-finite bias in layer {name}")
    for name, tensor in full_state.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor.float()).all():
            raise FloatingPointError(f"Refusing to export non-finite FP tensor {name}")
    total_weights = sum(state.weight_numel for state in qstates.values())
    total_groups = sum(state.scales.numel() for state in qstates.values())
    payload = {
        "format": "2NZQ",
        "version": 1,
        "base_model": base_model,
        "group_size": group_size,
        "quantized_layers": quantized_layers,
        "quantized": qdict,
        "fp_state_dict": full_state,
        "stats": {
            "quantized_weights": total_weights,
            "quantized_groups": total_groups,
            "estimated_bits_per_quantized_weight": bits_per_weight(total_weights, total_groups),
        },
    }
    torch.save(payload, output_path)
    return output_path


def load_2nzq_payload(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, object]:
    payload = torch.load(path, map_location=map_location)
    if payload.get("format") != "2NZQ":
        raise ValueError(f"Not a 2NZQ export: {path}")
    return payload


def decode_quantized_states(payload: dict[str, object]) -> dict[str, QuantizedLinearState]:
    states: dict[str, QuantizedLinearState] = {}
    for name, raw in payload["quantized"].items():
        states[name] = QuantizedLinearState(
            name=raw["name"],
            in_features=raw["in_features"],
            out_features=raw["out_features"],
            bias=raw["bias"],
            group_size=raw["group_size"],
            weight_numel=raw["weight_numel"],
            weight_shape=tuple(raw["weight_shape"]),
            scales=raw["scales"],
            packed_weight=raw["packed_weight"],
            bias_tensor=raw["bias_tensor"],
        )
    return states
