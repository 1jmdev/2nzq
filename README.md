# 2NZQ

Python implementation of 2NZQ two-bit non-zero quantization with Progressive Quantization-Aware Fine-Tuning for Hugging Face causal language models.

The default config uses `HuggingFaceTB/SmolLM-135M` and streams only `100000000` bytes from FineWeb via `datasets`, then tokenizes that sample locally.

## Layout

- `2nzq`: core package only.
- `config`: runnable JSON configs.
- `scripts`: command-line entrypoints.
- No `pyproject.toml` is used.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Download model:

```bash
python scripts/download_model.py --config config/smol135m_fineweb100m.json
```

Download and tokenize 100MB FineWeb sample:

```bash
python scripts/tokenize_dataset.py --config config/smol135m_fineweb100m.json
```

Fine-tune with PQAFT and export packed 2NZQ weights:

```bash
python scripts/train_pqaft.py --config config/smol135m_fineweb100m.json
```

The default training dtype is `float32` because AdamW on raw FP16 weights can produce NaNs during QAT. If your GPU supports BF16 well, `"dtype": "bfloat16"` in the config is a reasonable faster alternative.

Generate from the exported model:

```bash
python scripts/generate.py --model runs/smol135m-2nzq/model.2nzq.pt --prompt "The future of efficient language models is"
```

The export is a `torch.save` payload containing FP weights for excluded parameters plus packed 2-bit weights and FP16 group scales for quantized linear layers.
