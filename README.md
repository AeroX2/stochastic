# Stochastic Spiking LLMs

Exploring the hypothesis that combining **spiking neural networks** with **stochastic computing** can lead to substantially faster LLM training and inference.

## Thesis

Traditional LLMs rely on dense floating-point matrix multiplications. By replacing these with:
1. **Spiking activations** (binary event-driven signals instead of continuous values)
2. **Stochastic computing** (encoding values as bitstreams, where multiplication reduces to AND gates)

...we may achieve dramatic speedups, especially on hardware designed for sparse/binary operations.

## Included Repos

### SpikeGPT (`SpikeGPT/`)
- **Paper**: [arXiv:2302.13939](https://arxiv.org/abs/2302.13939) (2023)
- **Architecture**: RWKV-based recurrent model with LIF (Leaky Integrate-and-Fire) spiking neurons
- **Spiking**: True spiking via SpikingJelly's `MultiStepLIFNode` with surrogate gradients (ATan)
- **Key insight**: Each transformer block passes activations through LIF neurons, producing binary spikes
- **Strengths**: Genuine end-to-end spiking training, CUDA-optimized WKV kernel, pre-trained 216M model available
- **Limitations**: Based on older RWKV-v2, no modern attention, requires CuPy for spiking neuron backend, small scale (216M params max)

### SpikingBrain-7B (`SpikingBrain-7B/`)
- **Paper**: [arXiv:2509.05276](https://arxiv.org/abs/2509.05276) (2025)
- **Architecture**: Hybrid GLA (Gated Linear Attention) + sliding window attention + MoE
- **Spiking**: **Pseudo-spiking only** - activations approximated as spike-like signals at the tensor level, not true event-driven spiking
- **Key insight**: W8ASpike quantization converts integer activations into spike sequences (binary/ternary/bitwise LIF encoding) for efficient matmul
- **Strengths**: 7B scale, vLLM integration, Int2Spike encoding library, hybrid attention architecture
- **Limitations**: Not truly spiking during training, pseudo-spiking is inference-only optimization, requires MetaX/NVIDIA GPU cluster

### nanochat (`nanochat/`)
- **Paper**: [Karpathy, 2025](https://github.com/karpathy/nanochat)
- **Architecture**: GPT-2 style transformer with modern improvements (RoPE, GQA, Flash Attention 3, ReLU^2 MLP, Muon optimizer)
- **Spiking**: None - standard dense transformer
- **Key insight**: Single-dial complexity via `--depth`, trains GPT-2 capability in ~2hrs on 8xH100 for ~$48
- **Strengths**: Clean/hackable codebase, full pipeline (pretrain -> SFT -> RL -> eval -> chat UI), well-tested, active development
- **Limitations**: No spiking components, requires 8xH100 for full speedrun (can run on single GPU with grad accumulation)

## Related Work & Resources

### Most Relevant Papers
| Paper | Year | Key Contribution |
|-------|------|-----------------|
| **Stochastic Spiking Attention** (Song et al.) | 2024 | **Directly combines stochastic computing + spiking attention.** 6.3x energy reduction, 48x lower latency on FPGA vs GPU |
| **SpikeLLM** (Xing et al., ICLR 2025) | 2025 | First spiking LLM at 7-70B scale via GIF neurons + Optimal Brain Spiking |
| **SpikingLLM** (ICLR 2026 submission) | 2025 | Fully binary spike-driven LLM from scratch, 4-6% of ANN compute cost |
| **SpikeLM** (Xing et al., ICML 2024) | 2024 | Elastic bi-spiking for language modeling, bridges SNN-ANN gap |
| **Xpikeformer** | 2024 | Hybrid analog-digital arch combining in-memory computing + stochastic spiking attention, 13x energy reduction |

### Cloned for Study (Phase 1)
| Repo | Path | Study notes |
|------|------|-------------|
| **SpikeLM** | `SpikeLM/` | ICML 2024 elastic bi-spiking; see **STUDY_SpikeLM_and_SC-PyTorch.md** |
| **SC-PyTorch** | `SC-PyTorch/` | Stochastic Conv/Linear via CUDA (inference-only); see STUDY doc |

### Additional Repos to Consider
| Repo | URL | Why |
|------|-----|-----|
| **SpikeLLM** | https://github.com/Xingrun-Xing2/SpikeLLM | ICLR 2025, scales to 7-70B params |
| **Spike-Driven-Transformer-V3** | https://github.com/BICLab/Spike-Driven-Transformer-V3 | IEEE T-PAMI 2025, efficient spike firing approximation |
| **SpikingBERT** | https://github.com/NeuroCompLab-psu/SpikingBERT | Knowledge distillation approach for spiking language models |

## Ready for training (environment)

A **Python 3.12** venv with PyTorch and nanochat deps is in `nanochat/.venv312` so `torch.compile` works.

- **Validation (all four variants, synthetic data):**  
  From repo root:  
  `.\run_validation.ps1`  
  Or:  
  `$env:PYTHONPATH=".;nanochat"; nanochat\.venv312\Scripts\python -m experiments.run_validation`

- **First-time setup** (if `.venv312` is missing):  
  1. `cd nanochat` then `pyenv local 3.12.10` (so `python` is 3.12).  
  2. Run `.\setup_training_env.ps1` from repo root (creates venv, installs torch + deps).

- **GPU training:** Right now the venv has **PyTorch CPU**. For GPU, install [CUDA 12.x](https://developer.nvidia.com/cuda-downloads), then in the same venv:  
  `nanochat\.venv312\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126`  
  Then run training as below (or on an AWS/cloud GPU node with this env recreated there).

- **One-time before first training:** Prepare data and tokenizer (from `nanochat` with `.venv312` active):  
  `python -m nanochat.dataset -n 8`  
  then  
  `python -m scripts.tok_train --max-chars=2000000000`.  
  See nanochat README for details.
- **Run training:**  
  `$env:PYTHONPATH=".;nanochat"; nanochat\.venv312\Scripts\python -m experiments.train --variant=baseline --depth=12 ...`  
  Same for `--variant=spiking`, `--variant=stochastic`, `--variant=both`. See `experiments/README.md`.

### Single script (Linux / WSL / AWS GPU)

On a Linux box (WSL or an AWS GPU instance), from the **stochastic** repo root you can do minimal setup and training in one go:

```bash
chmod +x setup_and_train.sh
./setup_and_train.sh --variant=baseline
```

If your WSL default distro is **Docker** (not Ubuntu), run the script inside Ubuntu:

```bash
wsl -d Ubuntu -- bash -c "cd /mnt/c/path/to/stochastic && ./setup_and_train.sh --variant=baseline"
```
(Replace the path with your repo location; Windows `C:\` is `/mnt/c/` in WSL.)

The script will: create a venv (if needed), install PyTorch (CUDA if `nvidia-smi` works) and deps, download data and train the tokenizer (if not already present), then run training. Options:

- `--variant=baseline|spiking|stochastic|both` (required)
- `--depth=N` (default 12)
- `--run=NAME` (default `dummy`; see **wandb** below)
- `--skip-setup` — reuse existing venv
- `--skip-data` — skip dataset/tokenizer (use existing)
- `--num-iterations=N` — e.g. `100` for a short test
- `--device-batch-size=N` (default 32)
- `--hf-repo=USERNAME/REPO` — after training, upload the checkpoint to Hugging Face (set `HF_TOKEN` or run `huggingface-cli login` first)
- `--nproc-per-node=N` — use `torchrun` with N GPUs (e.g. `8` for 8× H100)
- `--save-every=N` — save a checkpoint every N steps (recommended on **interruptible** instances)
- `--hf-upload-interval=N` — when using `--hf-repo`, upload to HF every N seconds in the background (default 600 = 10 min)

**Example: 8× H100 on Vast.ai (interruptible), then upload and shut down**

```bash
# Create a model repo at huggingface.co (e.g. myuser/baseline-d12) and set your token once:
export HF_TOKEN=your_token   # or: huggingface-cli login

./setup_and_train.sh --variant=baseline --nproc-per-node=8 \
  --hf-repo=myuser/baseline-d12 --save-every=500
```

When training **finishes**, the script uploads the checkpoint dir to the HF repo. If you use `--hf-repo`, the script also **uploads every 10 minutes in the background** (override with `--hf-upload-interval=N`). So if the instance is **preempted**, the last periodic upload (up to ~10 min ago) is already on Hugging Face. Use `--save-every=N` (e.g. 500) so nanochat writes checkpoints often; then the background upload pushes whatever is on disk to HF. When training ends, the background upload is stopped and a final upload runs. Then stop the instance from the Vast dashboard to stop billing.

Example short test on GPU:  
`./setup_and_train.sh --variant=spiking --depth=6 --num-iterations=100`

### wandb (logging)

nanochat can log metrics (loss, etc.) to **Weights & Biases** (wandb). If you use a real run name, e.g. `--run=my-exp`, the training script will call `wandb.init()` and try to upload; that requires an API key.

- **No wandb:** use `--run=dummy`. No login, no upload; metrics are only printed locally.
- **With wandb:** create an account at [wandb.com](https://wandb.com), then once on the machine run `wandb login` and paste your API key. After that, use any `--run=...` name and runs will appear in your wandb project.

## Project Plan

- **Phase 1 (done):** Clone SpikeLM and SC-PyTorch; study implementations. Notes: **STUDY_SpikeLM_and_SC-PyTorch.md**.
- **Phase 2 (done):** Build three model variants + baseline in `experiments/`, validation script, env setup.
- **Phase 3 (needs GPU):** Train and compare; use an AWS/cloud GPU node for real training.
