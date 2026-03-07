"""
Spiking-only variant: same as nanochat GPT but MLP uses LIF-style spiking activation (binary/ternary + surrogate gradient).
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
)
from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn

from experiments.layers.spiking import SpikingActivation


class SpikingMLP(nn.Module):
    """MLP with spiking activation instead of ReLU^2."""

    def __init__(self, config, ternary=False):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.spike_act = SpikingActivation(threshold=0.0, alpha=2.0, ternary=ternary, learnable_scale=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.spike_act(x)
        x = self.c_proj(x)
        return x


class SpikingBlock(nn.Module):
    """Same as nanochat Block but MLP is SpikingMLP."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = SpikingMLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPTSpiking(GPT):
    """GPT with spiking MLP only (attention unchanged)."""

    def __init__(self, config, pad_vocab_size_to=64):
        nn.Module.__init__(self)
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([SpikingBlock(config, i) for i in range(config.n_layer)]),
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

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        for block in self.transformer.h:
            if hasattr(block.mlp, "spike_act") and hasattr(block.mlp.spike_act, "scale"):
                if isinstance(block.mlp.spike_act.scale, nn.Parameter):
                    block.mlp.spike_act.scale.fill_(1.0)
