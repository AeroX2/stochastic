"""
Minimal smoke test: build each variant (depth=2, tiny size), run one forward + backward.
Run from repo root: python -m experiments.run_smoke_test
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
NANOCHAT_DIR = ROOT / "nanochat"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NANOCHAT_DIR))

import torch

def main():
    from nanochat.gpt import GPT, GPTConfig
    from experiments.models.gpt_spiking import GPTSpiking
    from experiments.models.gpt_stochastic import GPTStochastic
    from experiments.models.gpt_spiking_stochastic import GPTSpikingStochastic

    config = GPTConfig(sequence_len=128, vocab_size=1000, n_layer=2, n_head=2, n_kv_head=2, n_embd=64, window_pattern="L")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T = 2, 16

    for name, ModelClass in [
        ("baseline", GPT),
        ("spiking", GPTSpiking),
        ("stochastic", GPTStochastic),
        ("both", GPTSpikingStochastic),
    ]:
        try:
            with torch.device("meta"):
                model = ModelClass(config)
            model.to_empty(device=device)
            model.init_weights()
            model.train()
            idx = torch.randint(0, config.vocab_size, (B, T), device=device)
            targets = torch.randint(0, config.vocab_size, (B, T), device=device)
            loss = model(idx, targets=targets)
            loss.backward()
            print(f"  {name}: ok (loss={loss.item():.4f})")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            raise

    print("Smoke test passed.")

if __name__ == "__main__":
    main()
