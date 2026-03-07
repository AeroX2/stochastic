"""
Evaluate a trained model (same variant as training) by patching nanochat.gpt.GPT then running base_eval.

Usage (from repo root):
  python -m experiments.eval --variant=baseline --model-tag d12 --device-batch-size=16
  torchrun --nproc_per_node=8 -m experiments.eval --variant=baseline --model-tag d24 --device-batch-size=16

All args other than --variant are passed to nanochat's base_eval (--model-tag, --device-batch-size, etc.).
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NANOCHAT_DIR = ROOT / "nanochat"
SCRIPTS_DIR = NANOCHAT_DIR / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(NANOCHAT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_DIR))

parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "spiking", "stochastic", "both"])
args, rest = parser.parse_known_args()
variant = args.variant
sys.argv = [sys.argv[0]] + rest

# Patch GPT so load_model builds the same variant as was trained
if variant != "baseline":
    if variant == "spiking":
        from experiments.models.gpt_spiking import GPTSpiking
        ModelClass = GPTSpiking
    elif variant == "stochastic":
        from experiments.models.gpt_stochastic import GPTStochastic
        ModelClass = GPTStochastic
    elif variant == "both":
        from experiments.models.gpt_spiking_stochastic import GPTSpikingStochastic
        ModelClass = GPTSpikingStochastic
    import nanochat.gpt as nanochat_gpt
    nanochat_gpt.GPT = ModelClass
    print(f"Using model variant: {variant}")

sys.path.insert(0, str(SCRIPTS_DIR))
import base_eval
base_eval.main()
