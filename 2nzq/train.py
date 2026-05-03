from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ensure_dir, load_config
from .data import TokenBlockDataset, tokenize_fineweb_sample
from .export import save_2nzq_model
from .modules import clamp_scales, replace_linears_for_qat, set_tau
from .schedule import pqaft_lr_factor, pqaft_tau


def prepare_data(cfg: dict) -> Path:
    tokenized_path = Path(cfg["data"]["tokenized_path"])
    if tokenized_path.exists() and not cfg["data"].get("retokenize", False):
        return tokenized_path
    return tokenize_fineweb_sample(
        model_name=cfg["model"]["name"],
        dataset_name=cfg["data"].get("dataset_name", "HuggingFaceFW/fineweb"),
        dataset_config=cfg["data"].get("dataset_config", "sample-10BT"),
        split=cfg["data"].get("split", "train"),
        output_path=tokenized_path,
        max_bytes=int(cfg["data"].get("max_bytes", 100_000_000)),
        sequence_length=int(cfg["training"].get("sequence_length", 1024)),
    )


def train(cfg: dict) -> Path:
    output_dir = ensure_dir(cfg["output_dir"])
    tokenized_path = prepare_data(cfg)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype_name = cfg["training"].get("dtype", "float32")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], dtype=dtype)
    model.config.use_cache = False
    quantized_layers = replace_linears_for_qat(
        model,
        group_size=int(cfg["quantization"].get("group_size", 128)),
        quantize_lm_head=bool(cfg["quantization"].get("quantize_lm_head", False)),
    )
    model.to(device)
    model.train()

    dataset = TokenBlockDataset(tokenized_path)
    loader = DataLoader(dataset, batch_size=int(cfg["training"].get("batch_size", 1)), shuffle=True, drop_last=True)
    total_steps = int(cfg["training"].get("total_steps", 1000))
    grad_accum = int(cfg["training"].get("gradient_accumulation_steps", 1))
    base_lr = float(cfg["training"].get("learning_rate", 2e-5))
    scale_lr = float(cfg["training"].get("scale_learning_rate", base_lr * 10))

    scale_params = [p for n, p in model.named_parameters() if n.endswith(".scales")]
    other_params = [p for n, p in model.named_parameters() if not n.endswith(".scales")]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": base_lr, "weight_decay": float(cfg["training"].get("weight_decay", 0.1))},
            {"params": scale_params, "lr": scale_lr, "weight_decay": 0.0},
        ],
        betas=(float(cfg["training"].get("beta1", 0.9)), float(cfg["training"].get("beta2", 0.95))),
    )

    step = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps, desc="PQAFT")
    while step < total_steps:
        for batch in loader:
            tau = pqaft_tau(step, total_steps, k=float(cfg["training"].get("tau_k", 10.0)))
            set_tau(model, tau)
            factor = pqaft_lr_factor(step, total_steps)
            optimizer.param_groups[0]["lr"] = base_lr * factor
            optimizer.param_groups[1]["lr"] = scale_lr * factor

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at step {step}. Use training.dtype=float32 or lower the learning rates."
                )
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("max_grad_norm", 1.0)))
                optimizer.step()
                clamp_scales(model)
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{loss.item() * grad_accum:.4f}", tau=f"{tau:.3f}")
            progress.update(1)
            step += 1
            if step >= total_steps:
                break
    progress.close()

    set_tau(model, 1.0)
    export_path = output_dir / cfg.get("export_name", "model.2nzq.pt")
    save_2nzq_model(
        model=model,
        output_path=export_path,
        base_model=cfg["model"]["name"],
        group_size=int(cfg["quantization"].get("group_size", 128)),
        quantized_layers=quantized_layers,
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    return export_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and export a 2NZQ model with PQAFT.")
    parser.add_argument("--config", default="config/smol135m_fineweb100m.json")
    args = parser.parse_args()
    path = train(load_config(args.config))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
