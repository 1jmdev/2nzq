from __future__ import annotations

import argparse

from .config import load_config
from .data import tokenize_fineweb_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and tokenize a byte-limited FineWeb sample.")
    parser.add_argument("--config", default="config/smol135m_fineweb100m.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = tokenize_fineweb_sample(
        model_name=cfg["model"]["name"],
        dataset_name=cfg["data"].get("dataset_name", "HuggingFaceFW/fineweb"),
        dataset_config=cfg["data"].get("dataset_config", "sample-10BT"),
        split=cfg["data"].get("split", "train"),
        output_path=cfg["data"]["tokenized_path"],
        max_bytes=int(cfg["data"].get("max_bytes", 100_000_000)),
        sequence_length=int(cfg["training"].get("sequence_length", 1024)),
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
