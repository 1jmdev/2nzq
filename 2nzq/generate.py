from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .export import decode_quantized_states, load_2nzq_payload
from .modules import replace_linears_for_inference


def _dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _assert_finite_payload(payload: dict[str, object]) -> None:
    for name, raw in payload["quantized"].items():
        if not torch.isfinite(raw["scales"].float()).all():
            raise FloatingPointError(f"Export contains non-finite 2NZQ scales in layer {name}. Re-train/re-export the model.")
        bias = raw.get("bias_tensor")
        if bias is not None and not torch.isfinite(bias.float()).all():
            raise FloatingPointError(f"Export contains non-finite bias in layer {name}. Re-train/re-export the model.")
    for name, tensor in payload["fp_state_dict"].items():
        if tensor.is_floating_point() and not torch.isfinite(tensor.float()).all():
            raise FloatingPointError(f"Export contains non-finite FP tensor {name}. Re-train/re-export the model.")


def load_model(export_path: str | Path, device: str | None = None, dtype_name: str = "float32"):
    payload = load_2nzq_payload(export_path, map_location="cpu")
    _assert_finite_payload(payload)
    base_model = payload["base_model"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_name(dtype_name)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    states = decode_quantized_states(payload)
    replace_linears_for_inference(model, states)
    missing, unexpected = model.load_state_dict(payload["fp_state_dict"], strict=False)
    unexpected = [x for x in unexpected if "packed_weight" not in x]
    if unexpected:
        raise RuntimeError(f"Unexpected state dict keys: {unexpected[:10]}")
    model.to(device)
    model.eval()
    return model, base_model, missing


@torch.inference_mode()
def generate(
    export_path: str | Path,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: str | None = None,
    dtype_name: str = "float32",
) -> str:
    model, base_model, _ = load_model(export_path, device, dtype_name)
    tokenizer_dir = Path(export_path).parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
        remove_invalid_values=True,
        renormalize_logits=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with an exported 2NZQ model.")
    parser.add_argument("--model", default="runs/smol135m-2nzq/model.2nzq.pt")
    parser.add_argument("--prompt", default="The future of efficient language models is")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    args = parser.parse_args()
    print(generate(args.model, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.device, args.dtype))


if __name__ == "__main__":
    main()
