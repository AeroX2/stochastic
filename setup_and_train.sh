#!/usr/bin/env bash
#
# Single script: minimal setup + training on Linux / WSL / AWS GPU.
# Run from the stochastic repo root (parent of nanochat/ and experiments/).
#
# Usage:
#   ./setup_and_train.sh [OPTIONS] --variant=baseline|spiking|stochastic|both
#
# Options:
#   --variant=...     (required) baseline | spiking | stochastic | both
#   --depth=N         model depth (default: 12)
#   --run=NAME        run name for logging (default: dummy = no wandb)
#   --skip-setup      skip venv/create and pip install (env already ready)
#   --skip-data       skip dataset download and tokenizer training
#   --num-iterations=N  training steps (default: from nanochat scaling; use e.g. 100 for a short test)
#   --device-batch-size=N  per-device batch size (default: 32)
#   --hf-repo=USERNAME/REPO  upload checkpoint to Hugging Face when done (set HF_TOKEN or run huggingface-cli login)
#   --nproc-per-node=N      use torchrun with N GPUs (e.g. 8 for 8x H100); default 1 = single process
#   --save-every=N          save checkpoint every N steps (for interruptible instances; passed to nanochat)
#   --hf-upload-interval=N  when using --hf-repo, also upload every N seconds in background (default 600 = 10 min)
#
# Examples:
#   ./setup_and_train.sh --variant=baseline
#   ./setup_and_train.sh --variant=baseline --hf-repo=myuser/baseline-d12 --save-every=500
#   ./setup_and_train.sh --variant=baseline --nproc-per-node=8 --hf-repo=myuser/baseline-8h100
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOCHAT_DIR="$REPO_ROOT/nanochat"
VENV_DIR="${VENV_DIR:-$NANOCHAT_DIR/.venv}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

# Defaults
VARIANT=""
DEPTH=12
RUN="dummy"
SKIP_SETUP=false
SKIP_DATA=false
NUM_ITERATIONS=""
DEVICE_BATCH_SIZE=32
HF_REPO=""
NPROC_PER_NODE=1
SAVE_EVERY=""
HF_UPLOAD_INTERVAL=600
EXTRA_ARGS=()

# Parse script-specific args; pass the rest to training
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant=*)
      VARIANT="${1#--variant=}"
      shift
      ;;
    --depth=*)
      DEPTH="${1#--depth=}"
      shift
      ;;
    --run=*)
      RUN="${1#--run=}"
      shift
      ;;
    --skip-setup)
      SKIP_SETUP=true
      shift
      ;;
    --skip-data)
      SKIP_DATA=true
      shift
      ;;
    --num-iterations=*)
      NUM_ITERATIONS="${1#--num-iterations=}"
      shift
      ;;
    --device-batch-size=*)
      DEVICE_BATCH_SIZE="${1#--device-batch-size=}"
      shift
      ;;
    --hf-repo=*)
      HF_REPO="${1#--hf-repo=}"
      shift
      ;;
    --nproc-per-node=*)
      NPROC_PER_NODE="${1#--nproc-per-node=}"
      shift
      ;;
    --save-every=*)
      SAVE_EVERY="${1#--save-every=}"
      shift
      ;;
    --hf-upload-interval=*)
      HF_UPLOAD_INTERVAL="${1#--hf-upload-interval=}"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$VARIANT" ]]; then
  echo "Usage: $0 --variant=baseline|spiking|stochastic|both [OPTIONS]" >&2
  echo "  --depth=N  --run=NAME  --skip-setup  --skip-data  --hf-repo=USER/REPO  --nproc-per-node=N  --save-every=N" >&2
  exit 1
fi

if [[ "$VARIANT" != "baseline" && "$VARIANT" != "spiking" && "$VARIANT" != "stochastic" && "$VARIANT" != "both" ]]; then
  echo "Error: --variant must be one of: baseline, spiking, stochastic, both" >&2
  exit 1
fi

mkdir -p "$NANOCHAT_BASE_DIR"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d${DEPTH}"

# -----------------------------------------------------------------------------
# Setup: Python 3.12 venv + PyTorch (CUDA if available) + deps
# -----------------------------------------------------------------------------
if [[ "$SKIP_SETUP" != true ]]; then
  if ! command -v python3.12 &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "Error: Python 3 not found. Install Python 3.12 (e.g. apt install python3.12-venv python3.12-dev, or pyenv)." >&2
    exit 1
  fi
  PYTHON=""
  for p in python3.12 python3; do
    if command -v "$p" &>/dev/null; then
      if "$p" -c 'import sys; exit(0 if sys.version_info >= (3,10) and sys.version_info < (3,14) else 1)' 2>/dev/null; then
        PYTHON="$p"
        break
      fi
    fi
  done
  if [[ -z "$PYTHON" ]]; then
    echo "Error: Need Python 3.10–3.12 (nanochat uses torch.compile). Found: $(python3 --version 2>/dev/null || true)" >&2
    exit 1
  fi
  echo "Using $PYTHON: $($PYTHON --version)"

  mkdir -p "$(dirname "$VENV_DIR")"
  if [[ ! -d "$VENV_DIR" ]] || [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
  fi
  source "$VENV_DIR/bin/activate"

  # Prefer CUDA build if nvidia-smi works
  if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "GPU detected; installing PyTorch with CUDA 12.x ..."
    python -m pip install -q --upgrade pip
    python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
  else
    echo "No GPU detected; installing PyTorch CPU ..."
    python -m pip install -q --upgrade pip
    python -m pip install -q torch
  fi

  # nanochat deps (from pyproject.toml, minus torch already installed)
  python -m pip install -q datasets transformers tiktoken wandb accelerate regex tokenizers zstandard scipy setuptools
  python -m pip install -q rustbpe
  echo "Venv ready at $VENV_DIR"
else
  source "$VENV_DIR/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}:${NANOCHAT_DIR}"

# -----------------------------------------------------------------------------
# Data + tokenizer (one-time)
# -----------------------------------------------------------------------------
if [[ "$SKIP_DATA" != true ]]; then
  if [[ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]]; then
    echo "Downloading dataset and training tokenizer ..."
    (cd "$NANOCHAT_DIR" && python -m nanochat.dataset -n 8)
    (cd "$NANOCHAT_DIR" && PYTHONPATH="${REPO_ROOT}:${NANOCHAT_DIR}" python -m scripts.tok_train --max-chars=2000000000)
    echo "Tokenizer saved to $NANOCHAT_BASE_DIR/tokenizer/"
  else
    echo "Tokenizer already present at $NANOCHAT_BASE_DIR/tokenizer/"
  fi
else
  echo "Skipping data/tokenizer (--skip-data)"
fi

# -----------------------------------------------------------------------------
# Train (no --no-compile so torch.compile is used on Linux)
# -----------------------------------------------------------------------------
TRAIN_ARGS=(
  --variant="$VARIANT"
  --depth="$DEPTH"
  --run="$RUN"
  --device-batch-size="$DEVICE_BATCH_SIZE"
)
[[ -n "$NUM_ITERATIONS" ]] && TRAIN_ARGS+=(--num-iterations="$NUM_ITERATIONS")
[[ -n "$SAVE_EVERY" ]] && TRAIN_ARGS+=(--save-every="$SAVE_EVERY")
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

# -----------------------------------------------------------------------------
# Optional: install HF hub and start background periodic upload (for interruptible instances)
# -----------------------------------------------------------------------------
HF_UPLOAD_PID=""
if [[ -n "$HF_REPO" ]] && [[ "${HF_UPLOAD_INTERVAL:-0}" -gt 0 ]]; then
  python -m pip install -q huggingface_hub
  export CHECKPOINT_DIR HF_REPO
  (
    while true; do
      sleep "$HF_UPLOAD_INTERVAL"
      if [[ -d "$CHECKPOINT_DIR" ]] && compgen -G "${CHECKPOINT_DIR}/model_*.pt" >/dev/null 2>&1; then
        echo "[HF background] Uploading checkpoint to $HF_REPO ..."
        python -c "
import os
from huggingface_hub import HfApi
api = HfApi()
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
api.upload_folder(folder_path=os.environ['CHECKPOINT_DIR'], repo_id=os.environ['HF_REPO'], repo_type='model', token=token)
print('[HF background] Upload done.')
" || true
      fi
    done
  ) &
  HF_UPLOAD_PID=$!
  echo "Started background HF upload every ${HF_UPLOAD_INTERVAL}s (PID $HF_UPLOAD_PID)."
fi

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  echo "Running: torchrun --nproc_per_node=$NPROC_PER_NODE -m experiments.train ${TRAIN_ARGS[*]}"
  torchrun --nproc_per_node="$NPROC_PER_NODE" -m experiments.train "${TRAIN_ARGS[@]}"
else
  echo "Running: python -m experiments.train ${TRAIN_ARGS[*]}"
  python -m experiments.train "${TRAIN_ARGS[@]}"
fi
TRAIN_EXIT=$?

# Stop background upload loop
if [[ -n "$HF_UPLOAD_PID" ]]; then
  kill "$HF_UPLOAD_PID" 2>/dev/null || true
  echo "Stopped background HF upload."
fi

# -----------------------------------------------------------------------------
# Final upload checkpoint to Hugging Face (if requested)
# -----------------------------------------------------------------------------
if [[ -n "$HF_REPO" ]]; then
  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Warning: Checkpoint dir not found ($CHECKPOINT_DIR), skipping HF upload." >&2
  else
    echo "Uploading checkpoint to Hugging Face: $HF_REPO"
    python -m pip install -q huggingface_hub 2>/dev/null || true
    HF_REPO="$HF_REPO" CHECKPOINT_DIR="$CHECKPOINT_DIR" python -c "
import os
from huggingface_hub import HfApi
api = HfApi()
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
folder = os.environ['CHECKPOINT_DIR']
repo = os.environ['HF_REPO']
api.upload_folder(folder_path=folder, repo_id=repo, repo_type='model', token=token)
print('Done. Checkpoint uploaded to https://huggingface.co/' + repo)
"
    echo "HF upload finished."
  fi
fi

exit "$TRAIN_EXIT"
