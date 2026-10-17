# Local GGUF Workbench

A reproducible GGUF launcher, TurboQuant KV-cache autotuner, and benchmark runner for trying local models on Apple Silicon and Linux/NVIDIA. It builds [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), a llama.cpp fork.

## Quick start

Requirements: Bash, Git, CMake, curl, jq, and either Xcode command-line tools on macOS or CUDA on Linux.

```bash
scripts/preflight.sh qwen3.6-35b-a3b --allow-missing-model --allow-missing-runtime --diagnose
scripts/build.sh
scripts/download-model.sh qwen3.6-35b-a3b --resolve-only
scripts/download-model.sh qwen3.6-35b-a3b
scripts/autotune.sh qwen3.6-35b-a3b --quick
scripts/run.sh qwen3.6-35b-a3b
scripts/benchmark.sh qwen3.6-35b-a3b
scripts/stop.sh
```

Omit the model ID to use `DEFAULT_MODEL` from `config/default.env`.

## Documentation

Everything beyond quick start and configuration lives in [docs/](docs/README.md):

| Doc | Contents |
|---|---|
| [docs/workflow.md](docs/workflow.md) | included profiles, adding a model, vision projector, Pi and Codex integration, network exposure |
| [docs/benchmarks.md](docs/benchmarks.md) | what `benchmark.sh` measures, sampling modes, the short-python suite, the agent-51 subset |
| [docs/models/](docs/models/) | one page per measured model family |
| [docs/studies/](docs/studies/) | cross-model decode and memory studies |

## Configuration order

Later files override earlier files:

1. `config/default.env`: portable runtime defaults
2. `config/models/<id>.env`: model metadata and context targets
3. `config/runtime.env`: generated llama.cpp TurboQuant build path
4. `config/local/<id>.env`: generated machine-specific tuning winner

Models are stored under ignored `models/`. Generated runs are stored under `results/` and include the model ID.

## Headline results

Best measured generation throughput per profile, on the Apple Silicon host unless noted:

| Profile | Context | Generation | Note |
|---|---:|---:|---|
| `btl-4-compact` | 32k smoke | 71.41 tok/s | 9.3 GiB IQ2_XXS MoE, short-prompt smoke |
| `kat-coder-mtp` | 131k | 64.24 tok/s | MTP + ngram-mod, +15% over original |
| `ornith15-35b-apex-compact` | 65k | 49.07 tok/s | fastest large model measured |
| `ternary-bonsai-2-27b-pq2` | 131k | 22.96 tok/s | idle decode, 1.44x the GSQ-RCO IQ3_S |
| `qwen38-27b-iq3` | 65k | 15.37 tok/s | dense thinking baseline |
| `muse-glimmer-dflash` | 65k | 13.07 tok/s | DFlash, +14.5% over baseline |
| `qwen36-27b-q2-mtp` | 65k | 10.93 tok/s | MTP gained only 4.8% |

These are smoke-benchmark numbers from one machine; sustained load cuts decode by 10-20%. Each [docs page](docs/README.md) carries the full protocol, caveats, plots, and the machine-readable record under `results/` with the model ID.

## Network access

The server binds to `127.0.0.1` by default; see [network exposure rules](docs/workflow.md#network-access).
