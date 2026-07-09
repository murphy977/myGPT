# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

All code lives in `nanoGPT-master/`. Run every command below from that directory. This is Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — a minimal (~300-line model, ~300-line training loop) GPT-2 reproduction/finetuning codebase. Not a git repo.

## Install

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```
`transformers` is only needed to load OpenAI GPT-2 checkpoints; `datasets` only to build OpenWebText; `wandb` is optional logging (off by default).

## Core workflow: prepare → train → sample

Every dataset must be tokenized into `train.bin`/`val.bin` before training. Each `data/<dataset>/prepare.py` writes those files (raw `uint16` token streams) into its own directory; char-level datasets also write `meta.pkl` (holding `vocab_size`, `stoi`, `itos`).

```sh
# 1. Prepare data (pick one)
python data/shakespeare_char/prepare.py    # char-level, 1MB, ~instant — best for local/debug
python data/shakespeare/prepare.py         # GPT-2 BPE tokenized tiny shakespeare (for finetuning)
python data/openwebtext/prepare.py         # full OWT, downloads + tokenizes (slow, large)

# 2. Train (config file first, then --key=value overrides)
python train.py config/train_shakespeare_char.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False  # macbook/CPU

# 3. Sample from a trained checkpoint or a pretrained GPT-2
python sample.py --out_dir=out-shakespeare-char
python sample.py --init_from=gpt2-xl --start="Hello" --num_samples=5 --max_new_tokens=100
python sample.py --start=FILE:prompt.txt   # prompt from a file
```

`bench.py` reproduces the inner training step for profiling only (no data prep, logging, or checkpointing).

## The configurator pattern (critical, non-obvious)

There is no argparse. `train.py`, `sample.py`, and `bench.py` define all hyperparameters as **module-level globals**, then run `exec(open('configurator.py').read())`. `configurator.py` walks `sys.argv` and mutates `globals()` directly:
- A bare arg (no `=`) is treated as a Python config file and `exec`'d, overriding globals (e.g. `config/train_gpt2.py`).
- A `--key=value` arg overrides one global. Values go through `ast.literal_eval`, and **the type must match the existing global's type** (asserted), so `--flag=False` works but the key must already exist or it raises.

Consequences when editing:
- To add a new tunable, declare it as a global in the `# -----` config block of `train.py` *before* the `exec` line — otherwise `--newkey=...` fails with "Unknown config key".
- Config files under `config/` are plain Python snippets (not modules); they just assign globals. See `config/train_gpt2.py` (multi-GPU 124M run), `config/train_shakespeare_char.py` (baby model), `config/finetune_shakespeare.py`, and `config/eval_gpt2*.py` (set `eval_only=True`).

## model.py architecture

`GPTConfig` (dataclass) + `GPT(nn.Module)`. Standard decoder-only transformer: token + learned positional embeddings → `n_layer` × `Block` (pre-LN: LayerNorm → CausalSelfAttention → residual, LayerNorm → MLP → residual) → final LayerNorm → `lm_head`. Notable details:
- **Weight tying**: `transformer.wte.weight` and `lm_head.weight` are the same tensor.
- **Flash attention**: `CausalSelfAttention` uses `F.scaled_dot_product_attention` when available (PyTorch ≥2.0), else a manual masked-softmax fallback.
- `bias=False` by default (unlike real GPT-2) — slightly faster/better.
- `vocab_size` defaults to 50304 (50257 padded to a multiple of 64 for efficiency).
- Key methods: `from_pretrained(model_type)` loads HF GPT-2 weights (transposes Conv1D weights); `crop_block_size()` does model surgery to shrink context; `configure_optimizers()` splits params into weight-decay (2D+) vs no-decay (1D) groups and uses fused AdamW on CUDA; `generate()` autoregressive sampling with temperature/top-k; `estimate_mfu()` reports model FLOPs utilization.

## train.py execution model

Single loop in `train.py` handles both single-GPU and multi-GPU. DDP is auto-detected from the `RANK` env var (set by `torchrun`), so launch multi-GPU as:
```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```
- **Gradient accumulation** (`gradient_accumulation_steps`) simulates large batches; under DDP it is divided by world size, and grad sync is disabled on all but the last micro-step.
- **Mixed precision** via `torch.autocast`; `dtype='float16'` auto-enables a `GradScaler`, `bfloat16` does not.
- `init_from` selects `'scratch'` | `'resume'` (loads `out_dir/ckpt.pt`) | `'gpt2*'` (OpenAI weights). Resuming strips a `_orig_mod.` prefix that `torch.compile` adds to state-dict keys.
- Checkpoints (`ckpt.pt`) are written to `out_dir` at each `eval_interval`; they bundle model + optimizer state + `model_args` + `config`.
- `torch.compile` is on by default (`--compile=False` to disable, required on CPU/Windows/unsupported platforms).
- Data loading is a deliberately simple `get_batch()` using `np.memmap` re-opened every batch (avoids a memmap memory leak).

## Conventions

- Deterministic seed is `1337 + seed_offset` (offset per DDP rank).
- `out_dir` defaults to `out`; example configs use `out-shakespeare-char`, `out-shakespeare`.
- Notebooks `scaling_laws.ipynb` and `transformer_sizing.ipynb` are analysis/exploration, not part of the train/sample path.
