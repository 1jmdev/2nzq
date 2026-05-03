from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import ensure_dir


def _text_stream(dataset_name: str, dataset_config: str | None, split: str) -> Iterable[str]:
    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True) if dataset_config else load_dataset(dataset_name, split=split, streaming=True)
    for row in dataset:
        text = row.get("text") if isinstance(row, dict) else None
        if text:
            yield text


def tokenize_fineweb_sample(
    model_name: str,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    output_path: str | Path,
    max_bytes: int = 100_000_000,
    sequence_length: int = 1024,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_chunks: list[torch.Tensor] = []
    buffer: list[int] = []
    seen_bytes = 0
    progress = tqdm(total=max_bytes, unit="B", unit_scale=True, desc="FineWeb sample")
    for text in _text_stream(dataset_name, dataset_config, split):
        encoded = tokenizer(text + tokenizer.eos_token, add_special_tokens=False).input_ids
        buffer.extend(encoded)
        seen = len(text.encode("utf-8"))
        seen_bytes += seen
        progress.update(min(seen, max(0, max_bytes - progress.n)))
        while len(buffer) >= sequence_length + 1:
            token_chunks.append(torch.tensor(buffer[: sequence_length + 1], dtype=torch.long))
            del buffer[: sequence_length + 1]
        if seen_bytes >= max_bytes:
            break
    progress.close()
    if not token_chunks:
        raise RuntimeError("No token chunks were produced; check dataset access and byte budget.")
    data = torch.stack(token_chunks)
    torch.save({"input_ids": data, "sequence_length": sequence_length, "model_name": model_name}, output_path)
    return output_path


class TokenBlockDataset(Dataset):
    def __init__(self, tokenized_path: str | Path) -> None:
        payload = torch.load(tokenized_path, map_location="cpu")
        self.input_ids = payload["input_ids"].long()

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.input_ids[idx]
        ids = ids[:-1]
        # HF causal LM heads shift labels internally; passing pre-shifted labels
        # trains the model to predict the wrong token offset.
        return {"input_ids": ids, "labels": ids.clone()}
