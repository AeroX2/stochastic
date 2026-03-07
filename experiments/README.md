# Experiments: Spiking and Stochastic GPT Variants

Three model variants for testing the hypothesis that **spiking + stochastic computing** can speed up LLM training and inference:

| Variant | Flag | What changes |
|--------|------|----------------|
| **Spiking only** | `--variant=spiking` | MLP uses a spiking activation (binary/ternary + surrogate gradient) instead of ReLU². |
| **Stochastic only** | `--variant=stochastic` | MLP uses `StochasticLinear` (input mapped to [0,1] unipolar) instead of standard Linear. |
| **Both** | `--variant=both` | MLP uses `StochasticLinear` + spiking activation. |
| **Baseline** | `--variant=baseline` | Unchanged nanochat GPT. |

Attention, embeddings, and the rest of the architecture are the same as nanochat.

## Layout

- **`layers/`** – Shared building blocks:
  - `spiking.py`: `SpikingActivation` (LIF-style threshold + surrogate gradient).
  - `stochastic.py`: `StochasticLinear` (sigmoid input, differentiable).
- **`models/`** – Model definitions:
  - `gpt_spiking.py`: `GPTSpiking`, `SpikingMLP`, `SpikingBlock`.
  - `gpt_stochastic.py`: `GPTStochastic`, `StochasticMLP`, `StochasticBlock`.
  - `gpt_spiking_stochastic.py`: `GPTSpikingStochastic`, `SpikingStochasticMLP`, `SpikingStochasticBlock`.
- **`train.py`** – Entrypoint: selects the variant, patches `nanochat.gpt.GPT`, then runs nanochat’s `base_train`.

## How to run

From the **stochastic** repo root (parent of `nanochat` and `experiments`):

```bash
# Spiking-only (small run for sanity check)
python -m experiments.train --variant=spiking --depth=12 --run=exp-spiking --num-iterations=10 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1

# Stochastic-only
python -m experiments.train --variant=stochastic --depth=12 --run=exp-stochastic ...

# Both
python -m experiments.train --variant=both --depth=12 --run=exp-both ...

# Baseline (same as nanochat)
python -m experiments.train --variant=baseline --depth=12 --run=exp-baseline ...
```

All other arguments are passed through to nanochat’s `base_train` (e.g. `--depth`, `--device-batch-size`, `--total-batch-size`).  
For real training you’ll need a GPU (and likely an AWS/cloud node); reduce `--device-batch-size` if you hit OOM.

## Dependencies

- **nanochat** and its environment (see `nanochat/README.md`).  
- No extra deps for the experiments package; it uses nanochat’s `GPT`, tokenizer, dataloader, and training loop.
- Use Python 3.10–3.12 for training (nanochat uses `torch.compile`, which is not supported on Python 3.14+).

## Python version (pyenv)

nanochat uses `torch.compile`, which **does not support Python 3.14+**. Use **Python 3.12** for training and validation.

A `.python-version` file at the repo root is set to `3.12.10`. With pyenv:

1. **Use 3.12 in this repo** (optional; makes `python` here be 3.12):
   ```bash
   cd stochastic
   pyenv local 3.12.10   # or leave the existing .python-version
   ```

2. **Install deps for 3.12** (torch, nanochat, etc.). From the **nanochat** directory with that Python active:
   ```bash
   cd nanochat
   pyenv local 3.12.10
   uv venv && uv sync --extra gpu   # or: pip install torch ...
   source .venv/bin/activate       # or on Windows: .venv\Scripts\activate
   ```

3. **Run validation from repo root** with the 3.12 env active:
   ```bash
   cd stochastic
   python -m experiments.run_validation
   ```
   (If you use a venv inside nanochat, activate it first so `python` is 3.12 with torch.)

## Validation (small-scale, before renting a GPU)

From repo root, with Python 3.12 active (e.g. via `.python-version` or `pyenv run 3.12.10`):

```bash
python -m experiments.run_validation
```

This runs **all four variants** (baseline = normal nanochat GPT, spiking, stochastic, both) with a tiny model (depth=4, 4×64 tokens per step) for 10 steps each on **synthetic random data**. No tokenizer or dataset needed. It reports loss and peak GPU memory so you can confirm everything works and compare memory use before running full training.

## Smoke test (one forward/backward, no optimizer)

```bash
python -m experiments.run_smoke_test
```

Builds all four models with depth=2 and runs one forward/backward. Use Python 3.12 (same as above).
