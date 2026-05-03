#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib

from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    config_mod = importlib.import_module("2nzq.config")
    parser = argparse.ArgumentParser(description="Download the configured Hugging Face model and tokenizer.")
    parser.add_argument("--config", default="config/smol135m_fineweb100m.json")
    args = parser.parse_args()
    cfg = config_mod.load_config(args.config)
    model_name = cfg["model"]["name"]
    AutoTokenizer.from_pretrained(model_name, use_fast=True)
    AutoModelForCausalLM.from_pretrained(model_name)
    print(f"downloaded {model_name}")


if __name__ == "__main__":
    main()
