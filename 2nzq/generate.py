from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .export import decode_quantized_states, load_2nzq_payload
from .modules import replace_linears_for_inference


def load_model(export_path: str | Path, device: str | None = None):
    payload = load_2nzq_payload(export_path, map_location="cpu")
    base_model = payload["base_model"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
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
def generate(export_path: str | Path, prompt: str, max_new_tokens: int = 80, temperature: float = 0.8, top_p: float = 0.95, device: str | None = None) -> str:
    model, base_model, _ = load_model(export_path, device)
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
    args = parser.parse_args()
    print(generate(args.model, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.device))


if __name__ == "__main__":
    main()
