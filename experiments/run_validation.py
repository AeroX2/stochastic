"""
Small-scale validation for all four variants (baseline, spiking, stochastic, both).

Run with Python 3.10–3.12 (nanochat uses torch.compile, not supported on 3.14+).
From repo root:
  pyenv local 3.12.10
  python -m experiments.run_validation

Or one-off:
  pyenv run 3.12.10 python -m experiments.run_validation

Uses synthetic random data (no tokenizer/dataset). Runs a few steps per variant and reports
loss + peak GPU memory so you can confirm everything works before renting a GPU.
"""

import sys
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NANOCHAT_DIR = ROOT / "nanochat"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NANOCHAT_DIR))

def main():
    if sys.version_info >= (3, 14):
        print("WARNING: Python 3.14+ is not supported (torch.compile). Use: pyenv local 3.12.10")
        sys.exit(1)

    import torch
    from nanochat.gpt import GPT, GPTConfig
    from experiments.models.gpt_spiking import GPTSpiking
    from experiments.models.gpt_stochastic import GPTStochastic
    from experiments.models.gpt_spiking_stochastic import GPTSpikingStochastic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Tiny config: fast, fits in 12GB
    depth = 4
    head_dim = 64
    base_dim = depth * 64
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    config = GPTConfig(
        sequence_len=256,
        vocab_size=1000,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern="L",
    )
    B, T = 4, 64
    n_steps = 10

    variants = [
        ("baseline", GPT),
        ("spiking", GPTSpiking),
        ("stochastic", GPTStochastic),
        ("both", GPTSpikingStochastic),
    ]

    results = []
    for name, ModelClass in variants:
        print(f"\n--- {name} ---")
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            with torch.device("meta"):
                model = ModelClass(config)
            model.to_empty(device=device)
            model.init_weights()
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            losses = []
            for step in range(n_steps):
                idx = torch.randint(0, config.vocab_size, (B, T), device=device)
                targets = torch.randint(0, config.vocab_size, (B, T), device=device)
                loss = model(idx, targets=targets)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses.append(loss.item())
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
            results.append((name, losses[0], losses[-1], peak_mb, None))
            print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}  peak GPU: {peak_mb:.0f} MB")
        except Exception as e:
            results.append((name, None, None, None, str(e)))
            print(f"  FAILED: {e}")
        model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n--- Summary ---")
    for name, l0, lN, peak, err in results:
        if err:
            print(f"  {name}: FAILED ({err})")
        else:
            print(f"  {name}: loss {l0:.4f} -> {lN:.4f}, peak GPU {peak:.0f} MB")
    failed = [r[0] for r in results if r[4] is not None]
    if failed:
        sys.exit(1)
    print("\nAll variants ran successfully. You can run full training with:")
    print("  python -m experiments.train --variant=baseline --depth=12 ...")
    print("  python -m experiments.train --variant=spiking ...")
    print("  (etc.)")

if __name__ == "__main__":
    main()
