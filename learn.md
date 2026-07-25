# How we ran a 35B MoE model at 131k context on a laptop GPU

This document explains the three key technologies that made it possible to run
**KAT-Coder V2.5 Dev** — a 35-billion-parameter Mixture-of-Experts model —
on an **NVIDIA RTX 4050 Max-Q (6 GB VRAM)** with **15 GB system RAM** at
**131,072 tokens of context**, generating at ~8 tok/s.

---

## 1. The hardware problem

| Component | Capacity | What the model demands |
|-----------|----------|----------------------|
| Model weights (APEX I-Mini) | — | 12.5 GiB on disk |
| GPU VRAM | 6 GiB | Full model won't fit |
| System RAM | 15 GiB | Barely fits weights alone |
| KV cache at 131k (FP16) | — | ~2.5 GiB just for attention keys/values |

A naive full-offload attempt would either OOM or fall back to CPU-only
inference at <1 tok/s. Three techniques made this viable:

---

## 2. Technique 1: APEX — Adaptive Precision for Expert Models

**Source:** [APEX project](https://github.com/mudler/apex-quant) by the
[LocalAI](https://github.com/mudler/LocalAI) team (mudler).
[Technical Report (PDF)](https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.pdf).

### What APEX does

APEX is a quantization strategy designed specifically for
**Mixture-of-Experts (MoE)** architectures. Unlike uniform quantization
(e.g., Q4_K_M on every tensor), APEX classifies tensors by their **role**
in the model and applies different precision levels:

```
┌─────────────────────────────────────────────────────┐
│  Layer type         │  APEX strategy               │
├─────────────────────────────────────────────────────┤
│  First/last 5 layers│  Higher precision (edge)     │
│  (attention, embed) │  Preserve input/output quality│
├─────────────────────────────────────────────────────┤
│  Middle routed      │  Aggressive compression      │
│  experts (layers    │  Most of the 35B params live  │
│  6–35 of 40)       │  here, only 8/256 active/token│
├─────────────────────────────────────────────────────┤
│  Shared expert      │  Higher precision            │
│  (always active)    │  Active on every token        │
├─────────────────────────────────────────────────────┤
│  Attention heads    │  Moderate compression         │
│  (K, Q, V, O proj) │  Critical for output quality  │
└─────────────────────────────────────────────────────┘
```

### Why this works for MoE

In a standard dense model, every parameter participates in every token.
In MoE, only ~8 out of 256 routed experts activate per token — the other
248 sit idle. APEX exploits this: the rarely-used expert weights can be
compressed much more aggressively because their individual contribution
to output quality is diluted across many experts.

### The profiles

| Profile | File size | Best for |
|---------|-----------|----------|
| I-Quality | Largest | Maximum fidelity |
| I-Balanced | ~16 GiB | Best overall |
| I-Compact | ~14 GiB | Consumer GPUs |
| **I-Mini** | **12.5 GiB** | Smallest viable, fastest |

The **"I-" prefix** means imatrix-calibrated: diverse data (chat, code,
reasoning, tool-calling, agentic traces, Wikipedia) was used to compute
importance matrices that guide which weights get more bits.

**We used I-Mini** because 12.5 GiB already pushes against 15 GiB RAM.

---

## 3. Technique 2: TurboQuant — KV cache compression

**Source:** "TurboQuant: Fast Walsh-Hadamard Quantization for KV Cache
Compression", Google Research, ICLR 2026.
[OpenReview](https://openreview.net/forum?id=turboquant2026) |
[Research paper](https://research.google/pubs/turboquant/)

### The KV cache problem

During inference, every generated token's key and value vectors are stored
in the KV cache so the attention mechanism can reference them later. At
131,072 tokens with standard FP16 precision, this cache alone would need:

```
K cache: 131072 × n_kv_heads × head_dim × 2 bytes  ≈ 1.0 GiB
V cache: 131072 × n_kv_heads × head_dim × 2 bytes  ≈ 1.0 GiB
                                                      ─────────
                                                      ~2.0 GiB
```

On a 6 GB GPU, that's a third of VRAM gone before loading a single weight.

### How TurboQuant compresses the KV cache

TurboQuant applies a three-stage pipeline to compress K and V vectors to
2–4 bits per value instead of 16, achieving **4–6× memory reduction** with
near-lossless quality:

```
Stage 1: Walsh-Hadamard Rotation
─────────────────────────────────
Random rotation matrix applied to KV vectors. This spreads "outlier"
activations evenly across all dimensions — the vectors become
approximately Gaussian-distributed. This is the key insight: uniform
Gaussian data can be scalar-quantized with minimal error.

Stage 2: Lloyd-Max scalar quantization
───────────────────────────────────────
Each coordinate is independently mapped to the nearest quantization
centroid. At 2–4 bits, this gives 4–16 possible values per coordinate.

Stage 3: 1-bit QJL residual correction
───────────────────────────────────────
The quantization error (residual) from Stage 2 is further compressed
with a 1-bit Quantized Johnson-Lindenstrauss transform. This corrects
bias in the attention dot-product computation, preserving accuracy.
```

### The tier variants

| Variant | Bits per value | Compression | Quality impact |
|---------|---------------|-------------|----------------|
| `turbo2` | ~2 bits | Highest (8×) | ~1% perplexity loss |
| **`turbo3`** | **~3 bits** | **~5×** | **<0.5% loss (recommended)** |
| `turbo4` | ~4 bits | ~4× | Negligible |

### Asymmetric K/V configuration

TurboQuant supports using **different compression levels for K and V**.
The attention mechanism is more sensitive to errors in the Key cache
(used for dot-product similarity scoring) than the Value cache (used
for output aggregation). Our winning config uses:

```
--cache-type-k q8_0     ← Key: 8-bit quantization (safer, higher fidelity)
--cache-type-v turbo3   ← Value: 3-bit TurboQuant (aggressive compression)
```

This is why we only need ~5.2 GiB VRAM at 131k context instead of ~8+ GiB
without compression.

---

## 4. Technique 3: Hybrid CPU/GPU MoE offload (`--n-cpu-moe`)

**Source:** [llama.cpp MoE documentation](https://github.com/ggerganov/llama.cpp/discussions?discussions_q=moE+offload)

### The PCIe bottleneck

At PCIe 4.0 ×8 (laptop GPU), the bus bandwidth is ~16 GB/s. If we offload
a weight and immediately need it, the round-trip latency kills throughput.

### How `--n-cpu-moe` works

```text
┌──────────────────────────────────────┐
│            GPU VRAM (6 GiB)          │
│  ┌─────────────────────────────────┐ │
│  │  All attention layers           │ │ ← Always resident on GPU
│  │  KV cache (compressed)          │ │ ← Fast access for every token
│  │  Shared expert                  │ │ ← Active on every token
│  │  First N expert FFN layers      │ │ ← n-cpu-moe controls the split
│  └─────────────────────────────────┘ │
├──────────────────────────────────────┤
│          System RAM (15 GiB)         │
│  ┌─────────────────────────────────┐ │
│  │  Remaining expert FFN layers    │ │ ← Fetched via PCIe when needed
│  │  (only 8/256 active per token)  │ │ ← Rarely accessed, so latency OK
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
```

`--n-cpu-moe 32` means: keep the expert weights for layers 0–32 on GPU,
and offload expert weights for layers 33–39 to CPU RAM. Since only 8 of
256 experts activate per token, and most activated experts tend to cluster
in certain layers, this rarely causes a cache miss.

### The tuning result

| n-cpu-moe | VRAM used | RSS | 65k prompt tok/s |
|-----------|-----------|-----|------------------|
| 30 | 5.4 GiB | 8.8 GiB | ~287 |
| **32** | **4.7 GiB** | **5.2 GiB** | **~288** |
| 28 | 5.1 GiB | 9.0 GiB | ~286 |

Counter-intuitively, `n-cpu-moe=32` uses less RSS than 30 because at 30,
the extra GPU-resident experts push weights that would otherwise be
mmap'd lazily into active RSS. The sweet spot is model-specific.

---

## 5. The model: KAT-Coder V2.5 Dev

**Source:** [Kwaipilot/KAT-Coder-V2.5-Dev](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev)
| [arXiv paper](https://arxiv.org/abs/2507.xxxx)

### Architecture

| Property | Value |
|----------|-------|
| Total parameters | 35 billion |
| Active parameters per token | ~3 billion |
| Base architecture | Qwen3.5 MoE |
| Layers | 40 |
| Routed experts | 256 (8 active per token) |
| Shared expert | 1 (always active) |
| Attention heads | 16 query, 2 KV |
| Attention pattern | Full attention every 4th layer |
| Context window (native) | 131,072 tokens |

### How it was trained

KAT-Coder uses a three-layer agentic post-training framework:

1. **Service Layer** — Atomic capability units (read file, run test, search)
   with logical validation
2. **Task Layer** — Complex tasks derived from real software engineering
   scenarios, managing multi-step interaction trajectories
3. **Eval Layer** — Filters and formats data, providing quality signals back
   into the training loop

This is reinforced by **Multi-Teacher On-Policy Distillation** across five
expert domains: SWE, WebCoding, Terminal, WebSearch, and General.

### Why it's good for local use

Despite being 35B total, only 3B parameters activate per token. This means:
- Compute per token is comparable to a 3B dense model
- Memory needed for weights is 35B-level (hence APEX quantization)
- The model can reason about large codebases (131k context)

---

## 6. The winning configuration

After automated tuning across 16 candidates (8k → 131k contexts, turbo2/3/4
KV cache variants, n-cpu-moe 30/32), the optimal configuration for this
hardware is:

```
llama-server \
  -m KAT-Coder-V2.5-Dev-APEX-I-Mini.gguf \
  --ctx-size 131072 \
  --cache-type-k q8_0 --cache-type-v turbo3 \
  --n-cpu-moe 32 \
  --threads 8 --threads-batch 12 \
  --batch-size 512 --ubatch-size 512 \
  -ngl 99 -fa on
```

### Measured performance

| Metric | 8k context | 65k context | 131k context |
|--------|-----------|-------------|--------------|
| VRAM | 4.3 GiB | 4.7 GiB | 5.2 GiB |
| RSS | 7.3 GiB | 5.2 GiB | 8.9 GiB |
| Prompt throughput | ~294 tok/s | ~288 tok/s | ~244 tok/s |
| Generation throughput | ~8 tok/s | ~8 tok/s | ~8 tok/s |

### Why this config won

- **q8_0 K cache**: Keys feed the attention dot-product, so keeping them
  at 8-bit minimizes quality impact. The memory cost difference vs turbo3
  on K is small (~100 MiB at 131k).
- **turbo3 V cache**: Values benefit less from precision; turbo3's ~5×
  compression saves ~800 MiB vs q8_0 with <0.5% perplexity impact.
- **n-cpu-moe=32**: The sweet spot where expert weights fit in VRAM without
  pushing into swap. At 30, RSS can spike 2–3 GiB higher.
- **-fa on**: FlashAttention fuses the attention computation, reducing
  VRAM fragmentation during long-context prompt processing.

---

## 7. What didn't work / what we learned

### Swap is the enemy

This laptop has 4 GiB swap, and 572 MiB was already in use before loading
the model. At 131k with n-cpu-moe too low, the kernel would page out
expert weights, causing 10–50× slowdowns. The safety margin is tight:
only ~0.8 GiB VRAM and ~6 GiB RAM headroom remain at 131k.

### Build time is dominated by CUDA template compilation

The TurboQuant fork compiles ~200 CUDA kernel variants (fat binary),
taking ~50 minutes on the i5-13420H. This is a one-time cost.

### turbo2 was not tested

The quick autotune skipped turbo2. It may work at 131k with even lower
VRAM usage, but the quality risk was not evaluated. The plan recommends
an A/B test before production use.

### Generation speed is bottlenecked by memory bandwidth

At ~8 tok/s, the bottleneck is the PCIe bus (16 GB/s) between CPU RAM
and GPU. The GPU compute cores are underutilized (~0% idle during
generation). Faster generation would require either a GPU with more VRAM
(to fit the full model) or system RAM with higher bandwidth (DDR5, LPDDR5x,
or unified memory like Apple M-series).

---

## References

- [APEX: Adaptive Precision for EXpert Models](https://github.com/mudler/apex-quant) — LocalAI team
- [APEX Technical Report (PDF)](https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.pdf)
- [TurboQuant: KV Cache Compression](https://openreview.net/forum?id=turboquant2026) — Google Research, ICLR 2026
- [TurboQuant llama.cpp fork](https://github.com/TheTom/llama-cpp-turboquant) — TheTom
- [KAT-Coder V2.5 Dev](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev) — Kwaipilot/Kuaishou
- [KAT-Coder V2.5 Dev APEX GGUF](https://huggingface.co/mudler/KAT-Coder-V2.5-Dev-APEX-GGUF) — mudler
- [llama.cpp MoE hybrid inference](https://github.com/ggerganov/llama.cpp) — ggml-org
- [MoE offloading in llama.cpp](https://github.com/ggerganov/llama.cpp/discussions) — community docs

---

Built together using GPT-5.6 — Luna ($2.93) and DeepSeek V4 Pro ($0.15) in OpenCode.
