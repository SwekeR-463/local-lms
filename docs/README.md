# Docs

The [README](../README.md) is the entry point: quick start, configuration order, and a headline-results table. This directory holds everything else.

- [workflow.md](workflow.md) — included profiles, adding a model, vision projector, Pi and Codex integration, network exposure.
- [benchmarks.md](benchmarks.md) — what the benchmark suites measure and how sampling modes work.
- [models/](models/) — one page per measured model family.
- [studies/](studies/) — cross-model investigations.

## Model pages

| Page | Profile(s) covered |
|---|---|
| [Qwen3.8 27B](models/qwen3.8-27b.md) | `qwen38-27b-iq3`, `qwen38-27b-ud-q3`, `qwen38-27b-ad-iq3` |
| [Ornith 1.5 35B](models/ornith-1.5-35b.md) | `ornith15-35b-apex-compact`, `ornith-35b-i-mini`, `ornith15-9b-q8` |
| [KAT-Coder](models/kat-coder.md) | `kat-coder`, `kat-coder-mtp` |
| [Qwen3.6 27B Q2](models/qwen3.6-27b-q2.md) | `qwen36-27b-q2`, `qwen36-27b-q2-mtp`, `qwen36-27b-q2-dflash` |
| [Muse Glimmer](models/muse-glimmer.md) | `muse-glimmer`, `muse-glimmer-dflash` |
| [BTL-4 Compact](models/btl-4-compact.md) | `btl-4-compact` |

## Studies

| Page | Coverage |
|---|---|
| [131k context decode](studies/131k-decode.md) | Ternary Bonsai PQ2_0 vs GSQ-RCO IQ3_S, ngram speculation ceiling |
| [CPU MoE offload](studies/cpu-moe-offload.md) | Mac CPU-offload penalty at two context sizes |
