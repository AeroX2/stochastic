"""
Stochastic-only variant: MLP uses StochasticLinear (input mapped to [0,1] unipolar) instead of standard Linear.
Attention and rest unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import (
    GPT,
    GPTConfig,
    Linear,
    norm,
    has_ve,
    apply_rotary_emb,
    CausalSelfAttention,
    Block as BaseBlock,
)
from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn

from experiments.layers.stochastic import StochasticLinear


class StochasticMLP(nn.Module):
    """MLP with stochastic-computing-style linear layers (sigmoid input) and ReLU^2 activation."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = StochasticLinear(config.n_embd, 6 * config.n_embd, bias=False)
        self.c_proj = Linear(6 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class StochasticBlock(nn.Module):
    """Same as nanochat Block but MLP is StochasticMLP."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = StochasticMLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPTStochastic(GPT):
    """GPT with stochastic MLP only (attention unchanged)."""

    def __init__(self, config, pad_vocab_size_to=64):
        nn.Module.__init__(self)
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([
                StochasticBlock(config, i) if (i % 2 == 0) else BaseBlock(config, i)
                for i in range(config.n_layer)
            ]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
