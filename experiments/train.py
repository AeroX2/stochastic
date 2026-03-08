"""
Train spiking / stochastic / both variants by patching nanochat.gpt.GPT then running nanochat's base_train.

Usage (from repo root: stochastic/):
  python -m experiments.train --variant=spiking --depth=12 ...
  python -m experiments.train --variant=stochastic --depth=12 ...
  python -m experiments.train --variant=both --depth=12 ...
  python -m experiments.train --variant=baseline --depth=12 ...

All other args are passed to nanochat's base_train (e.g. --depth, --run, --device-batch-size, ...).
Requires nanochat and its deps; run from the stochastic repo root so both nanochat and experiments are on the path.

We run base_train via runpy as __main__ so the process looks like speedrun (torchrun -m scripts.base_train).
That avoids torch.compile + main-module differences that can cause a hang after step 0 (0% CPU/GPU).
"""

import sys
import argparse
import runpy
from pathlib import Path

# Repo root (stochastic/) and nanochat package
ROOT = Path(__file__).resolve().parents[1]
NANOCHAT_DIR = ROOT / "nanochat"
SCRIPTS_DIR = NANOCHAT_DIR / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(NANOCHAT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_DIR))


def main():
    # Parse our args and strip them so base_train doesn't see them
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "spiking", "stochastic", "both"])
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile (e.g. when MSVC not installed on Windows)")
    args, rest = parser.parse_known_args()
    variant = args.variant
    if args.no_compile:
        import os
        os.environ["STOCHASTIC_NO_COMPILE"] = "1"
    # So base_train's parser sees only its args (no --variant / --no-compile)
    sys.argv = [sys.argv[0]] + rest

    # Optionally disable torch.compile (avoids MSVC requirement on Windows)
    import os
    if os.environ.get("STOCHASTIC_NO_COMPILE") == "1":
        import torch
        def _no_compile(fn=None, *args, **kwargs):
            if fn is None:
                return lambda f: f
            return fn
        torch.compile = _no_compile
        print("torch.compile disabled (STOCHASTIC_NO_COMPILE=1)")

    # Patch GPT before base_train imports it
    if variant != "baseline":
        if variant == "spiking":
            from experiments.models.gpt_spiking import GPTSpiking
            ModelClass = GPTSpiking
            print("Using model: GPTSpiking (spiking-only)")
        elif variant == "stochastic":
            from experiments.models.gpt_stochastic import GPTStochastic
            ModelClass = GPTStochastic
            print("Using model: GPTStochastic (stochastic-only)")
        elif variant == "both":
            from experiments.models.gpt_spiking_stochastic import GPTSpikingStochastic
            ModelClass = GPTSpikingStochastic
            print("Using model: GPTSpikingStochastic (spiking + stochastic)")
        import nanochat.gpt as nanochat_gpt
        nanochat_gpt.GPT = ModelClass

    # Run base_train as __main__ (same as speedrun: torchrun -m scripts.base_train).
    # Preserving argv and using run_path keeps base_train's parser seeing our rest args.
    sys.path.insert(0, str(SCRIPTS_DIR))
    base_train_path = str(SCRIPTS_DIR / "base_train.py")
    sys.argv = [base_train_path] + rest
    runpy.run_path(base_train_path, run_name="__main__")
    return None


if __name__ == "__main__":
    main()
