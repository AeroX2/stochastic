# Study Notes: SpikeLM & SC-PyTorch (Cloned Repos)

We cloned and studied these repos before building our three variants (spiking-only, stochastic-only, both). Here’s what we learned.

---

## SpikeLM (Xingrun-Xing/SpikeLM)

**Location:** `SpikeLM/` (root: `spikeLM-BERT/`)

**Paper:** SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking Mechanisms (ICML 2024)

### Core idea: Elastic Bi-Spiking

Spikes are not just {0,1}. They use **direction** (sign), **amplitude** (learnable scale α), and **temporal unrolling** over T steps so that spike trains can carry more information while still being additive.

### Key implementation files

| File | Role |
|------|------|
| `spiking.py` | `ElasticBiSpiking`, `SpikeLinear`, `QuantizeEmbedding` |
| `spike_bert.py` | BERT with SpikeLinear / ElasticBiSpiking throughout |
| `run_pretrain.py` | Pretraining script (Accelerate, MLM); uses `spikingjelly.clock_driven.functional` for reset |

### 1. ElasticBiSpiking (`spiking.py`)

- **Custom `torch.autograd.Function`** (inspired by Learned Step-size Quantization).
- **Forward:** Quantize input to binary (num_bits=1: `sign`) or ternary (num_bits=2: round and clamp to {-1, 0, 1}), then scale by learnable **α** (per-layer/per-step): `w_q = q_w * alpha`.
- **Backward:** Straight-through style:
  - `grad_input = indicate_middle * grad_output` (only pass gradient where not clamped).
  - `grad_alpha` from a scaled sum involving `grad_output` and the quantized values.
- **AlphaInit:** Parameter wrapper that lazily initializes α from the first input (e.g. `2 * tensor.abs().mean() / sqrt(Qp)`).

So “elastic” = learnable scale α; “bi” = binary or ternary; differentiable via STE.

### 2. SpikeLinear (`spiking.py`)

- **Input shape:** `(T, B, D)` — T timesteps, batch B, dim D.
- **Recurrent membrane over time:**  
  - Step 0: `mem = input[0]`.  
  - Step t>0: `mem = mem_old * 0.25 * (act_clip_val[t-1] - output[t-1]) + input[t]` (LIF-like leak and reset).
- **Output per step:** `output[t] = ElasticBiSpiking(mem, act_clip_val[t], input_bits, True)`.
- Then **standard linear:** `out = F.linear(output, weight) + bias` (output still (T,B,D_out)).
- So: temporal unrolling + binary/ternary activation with learnable α, then dense linear. Weights are full precision in this layer; quantization is only in the config (weight_bits) for other parts.

### 3. Spike BERT usage (`spike_bert.py`)

- **BertSelfAttention:** `query` / `key` / `value` are `SpikeLinear`; key/value are then passed through `ElasticBiSpiking` per timestep (clip_key, clip_value). Attention scores use **softmax** (no spiking in the softmax).
- **BertSelfOutput, BertIntermediate, BertOutput:** Use `SpikeLinear` for the dense layers.
- **Embeddings:** Can use `QuantizeEmbedding` (1-bit or 2-bit weights with straight-through).
- **Config:** `T` (timesteps), `weight_bits`, `input_bits`, `quantize_act`, `clip_val`, etc.

Takeaway: SpikeLM keeps the same BERT topology but replaces linear + activation with SpikeLinear (temporal LIF + ElasticBiSpiking) and optional quantized embeddings. Good reference for “spiking-only” in our decoder-only GPT: we can use a similar LIF + elastic binary/ternary activation in the MLP and optionally in embeddings.

---

## SC-PyTorch (JavierFo/SC-PyTorch)

**Location:** `SC-PyTorch/`

**README:** “Stochastic Computing extension to PyTorch Neural Network's 2D convolution and fully connected layer.”

### Core idea: Stochastic computing

Values in [0,1] are represented as **bitstreams**. Multiplication of two values in [0,1] becomes **AND** of two random bitstreams whose expectations are those values; the count of 1s in the product bitstream (normalized) estimates the product. So matmul can be done with very low-precision / low-energy ops (AND + counters).

### Key implementation

- **C++/CUDA extension** (`ScCudaTorch2.cpp`, `ScCudaTorch_UniPinConst_2.cu` or similar) loaded via `torch.utils.cpp_extension.load`.
- **No Python autograd through SC:** The SC layers are used in **inference only**. A pre-trained standard NN (e.g. LeNet on MNIST, VGG9 on CIFAR) is loaded; weights and activations are normalized to [0,1]; then `ScCudaConv2d` and `ScCudaFcLayer` are called with:
  - Flattened weights and inputs (as lists),
  - **Pre-generated random matrices** (one per layer) for converting float → bitstream,
  - A **bitstream length** (e.g. 1280).
- **Output:** `(count / (num_accumulations * bitstream_Length)) * num_accumulations` plus bias — i.e. stochastic estimate of the deterministic linear/conv output.
- **Training:** Not done with SC in this repo; they train a normal NN, then replace forward with SC for inference. So we **cannot** directly use their CUDA SC ops for training.

### Implications for our “stochastic-only” and “both” variants

- For **training** we need a **differentiable** stochastic linear in PyTorch. Options:
  1. **Expectation-based:** In forward, treat stochastic multiply as expectation (e.g. `E[x*y] = x*y` when x,y in [0,1]) and do normal matmul; optionally add noise for regularization. Backward is standard.
  2. **Sample-based + STE:** Sample bitstreams (e.g. Bernoulli( x )), do AND/count in forward, backward pass gradient as if the operation were the expectation (straight-through).
  3. **Hybrid:** Use expectation in backward and optionally sampled SC in forward to mimic hardware.
- We will implement a **pure-PyTorch** stochastic linear (no C++/CUDA dependency) so it runs on any device and is easy to plug into nanochat. We can later swap in a CUDA kernel if needed.

---

## Summary Table

| Aspect | SpikeLM | SC-PyTorch |
|--------|--------|------------|
| Spiking | ✅ Elastic bi-spiking (binary/ternary + α), LIF-like membrane over T steps | ❌ |
| Stochastic computing | ❌ | ✅ Bitstream conv/linear via CUDA (inference only) |
| Differentiable | ✅ Custom autograd (STE-style) for ElasticBiSpiking and SpikeLinear | ❌ (inference-only SC) |
| Architecture | BERT (encoder), SpikeLinear everywhere | LeNet / VGG9 (small CNNs) |
| Our use | Ideas for LIF + binary/ternary activation in MLP (and optionally embed) | Ideas for stochastic matmul; we add our own differentiable PyTorch SC layer for training |

---

## Next steps (for our experiments)

1. **Spiking-only variant:** Reuse ideas from SpikeLM: LIF-like recurrence + ElasticBiSpiking (or a simpler LIF with surrogate gradient) in the **MLP** of nanochat’s GPT; keep attention and rest as in nanochat. No temporal T in the sequence dimension — we use one “step” per token or a fixed T per block, similar to SpikeGPT’s use of MultiStepLIF.
2. **Stochastic-only variant:** Implement a **differentiable** `StochasticLinear` in PyTorch (expectation or sample+STE, no C++), and use it in the MLP (and optionally elsewhere).
3. **Both variant:** Use the same spiking MLP as in (1) but with StochasticLinear instead of standard Linear for the spiking MLP (or for all linears in the block).

We will place all new code under `experiments/` and keep nanochat unchanged.
