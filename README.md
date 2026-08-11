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

## Included profiles

```text
kat-coder
kat-coder-mtp
muse-glimmer
muse-glimmer-dflash
qwen36-27b-q2
qwen36-27b-q2-mtp
qwen3.6-35b-a3b
qwen3.5-27b
```

The Qwen3.6 27B profiles require recent upstream llama.cpp; the pinned TurboQuant fork produced slower MTP results in the earlier Q3_K_M experiment.

Add a model by copying a profile:

```bash
cp config/models/qwen3.5-27b.env config/models/my-model.env
```

At minimum set:

```bash
MODEL_NAME="My model"
MODEL_REPO="owner/gguf-repository"
MODEL_FILE="exact-file.gguf"
```

Then every command accepts `my-model` as its first argument. `MODEL_PATH` can point at an existing local GGUF instead.

## Configuration order

Later files override earlier files:

1. `config/default.env` — portable runtime defaults
2. `config/models/<id>.env` — model metadata and context targets
3. `config/runtime.env` — generated llama.cpp TurboQuant build path
4. `config/local/<id>.env` — generated machine-specific tuning winner

Models are stored under ignored `models/`. Generated runs are stored under `results/` and include the model ID.

## Benchmark output

`benchmark.sh` sends the fixed prompt in `config/prompts/benchmark.txt` to a running server and writes JSON containing the model/profile, quant file, context, hardware, memory, prompt throughput, generation throughput, speculative acceptance, and response. This is a throughput smoke benchmark, not a quality leaderboard.

## Mac results

On an Apple M5 Pro, the measured winner kept every MoE layer on Metal and used 10 performance-core threads, `q8_0/turbo3` KV cache, and `1024/1024` batch/ubatch.

| Configured context | Actual prompt | Prompt processing | Generation | Process RSS |
|---:|---:|---:|---:|---:|
| 65,536 | 60,504 | 372.14 tok/s | **30.91 tok/s** | 13.63 GiB |
| 98,304 | 90,751 | 229.16 tok/s | **22.80 tok/s** | 13.97 GiB |

CPU MoE offload did not help: even 8 offloaded layers reduced generation by about 16% at both contexts, while RSS stayed effectively unchanged.

![CPU MoE generation throughput](results/plots/cpu_moe_generation.png)

![CPU MoE prompt throughput](results/plots/cpu_moe_prompt.png)

## Linux/NVIDIA reference graphs

These plots show the earlier KAT-Coder I-Mini measurements on an RTX 4050 Max-Q. The hardware and fixed configuration are included in each chart.

![Memory scaling and throughput across context sizes](results/plots/full_comparison.png)

![Context vs VRAM and RSS](results/plots/context_vs_memory.png)

![Context vs prompt and generation throughput](results/plots/context_vs_throughput.png)

![KV cache comparison at 65k context](results/plots/kv_cache_comparison.png)

## KAT-Coder MTP benchmark

The `kat-coder-mtp` profile uses the embedded MTP head with the publisher's recommended `draft-mtp,ngram-mod` configuration. Two deterministic 512-token runs were measured at each configured context with `q8_0/turbo3`, full Metal offload, and a 38-token coding prompt.

| Context | Original | MTP + ngram | Gain | Draft acceptance |
|---:|---:|---:|---:|---:|
| 65,536 | 55.00 tok/s | **63.37 tok/s** | **15.2%** | 96.7% |
| 98,304 | 55.48 tok/s | **63.62 tok/s** | **14.7%** | 96.7% |
| 131,072 | 55.41 tok/s | **64.24 tok/s** | **15.9%** | 96.7% |

The MTP-file-only matrix produced these means:

| Context | None | ngram | MTP | MTP + ngram | Winner |
|---:|---:|---:|---:|---:|---|
| 65,536 | 56.26 | 55.84 | 61.64 | **63.37** | MTP + ngram |
| 98,304 | 64.14 | 64.40 | **69.45** | 63.62 | MTP |
| 131,072 | 64.15 | 63.38 | **64.54** | 64.24 | MTP (near tie) |

All three context sizes loaded and completed. These are short-prompt generation tests, not near-window context validation; the two-run 131k MTP result was noisy. Raw metrics and aggregates are in `results/mtp-benchmark-results.json`.

A longer 8,192-token video-preextractor prompt exposed quality failures in both models. The original hit the output limit during its tests; MTP stopped after 2,391 tokens in the middle of the implementation. Both hallucinated the PyTorchVideo API (`EncodedVideo(path)` and unsupported `get_clip` arguments instead of the documented `EncodedVideo.from_path(path)` flow). MTP was 12.7% faster, but neither response was runnable. See `results/mtp-video-preextractor-results.json`.

Sequential Pi coding-agent runs reached the same conclusion. MTP completed its tool loop 5.13x faster (5.06 versus 25.95 minutes), but both agents wrote tests around invented APIs and then reported success. Independent review found neither implementation runnable with PyTorchVideo. Workspaces and review data are under `results/pi-subagents/`.

## Qwen3.6 27B Q2 MTP benchmark

Qwen3.6 27B `UD-Q2_K_XL` baseline and MTP GGUFs were tested at 65,536 context using upstream llama.cpp commit `dd1ea524333b1e697489067d7a4c39c60d32beee`. One warm-up and two measured 512-token runs used full Metal offload, Flash Attention, one slot, and the publisher's `draft-mtp` maximum of two draft tokens.

| Mode | Generation | RSS | Draft acceptance |
|---|---:|---:|---:|
| Baseline | 10.43 tok/s | 15.95 GiB | — |
| MTP | **10.93 tok/s** | 15.61 GiB | 93.9% |

![Qwen3.6 Q2 baseline and MTP throughput](results/plots/qwen36_q2_mtp.png)

MTP improved generation by only **4.8%**, despite high acceptance, and all six deterministic responses were byte-identical. This is better than the earlier TurboQuant Q3_K_M result, where MTP was 40.7% slower, but too small to justify a coding-agent comparison yet. Raw aggregates are in `results/qwen36-27b-q2-mtp-results.json`.

A targeted 65k autotune compared three KV combinations and `1024/2048` batch and ubatch sizes. `f16/f16` with `1024/1024` won at 98.53 prompt tok/s and 9.49 generation tok/s on a 60,504-token prompt. The best `q8_0/q8_0` candidate used about 1.9 GiB less RSS but was 9.5% slower for prompt processing and 17.2% slower for generation; mixed `q8_0/f16` candidates exceeded the 900-second request timeout. See `results/qwen36-27b-q2-autotune-results.json`.

## Muse Glimmer DFlash benchmark

Muse Glimmer 30B `UD-Q2_K_XL` was tested on the M5 Pro at 65,536 configured context using upstream llama.cpp commit `dd1ea524333b1e697489067d7a4c39c60d32beee`. The target GGUF is 11.59 GiB; Meta's DFlash drafter adds 1.52 GiB. One warm-up and two measured 512-token runs used full Metal offload, Flash Attention, one slot, and `f16/f16` KV cache.

| Mode | Generation | RSS | Draft acceptance |
|---|---:|---:|---:|
| Baseline | 11.42 tok/s | 12.69 GiB | — |
| DFlash | **13.07 tok/s** | 14.52 GiB | 87.5% |

![Muse Glimmer DFlash throughput and memory](results/plots/muse_glimmer_dflash.png)

DFlash improved generation by **14.5%** and produced byte-identical deterministic responses. This is useful but much smaller than Meta's M5 Max ExecuTorch result; backend and hardware differences matter. Raw aggregates are in `results/muse-glimmer-dflash-results.json`.

An isolated Pi coding agent completed the video-preextractor task in 11.16 minutes with 13 tool calls and no tool errors. It used the documented PyTorchVideo API correctly, unlike the earlier KAT agents, but independent review still found missing worker support, filename collisions, stale overwrite manifests, and vacuous resume/corruption tests. The workspace and review are under `results/pi-subagents/muse-glimmer-dflash/`.

Muse Glimmer requires a recent upstream llama.cpp with the `muse-glimmer` architecture; the pinned TurboQuant commit does not load it. `scripts/download-model.sh muse-glimmer-dflash` resolves both the target and companion draft GGUF. The DFlash profile uses `--model-draft`, `--spec-type draft-dflash`, and `--spec-draft-n-max 15`.

## Use a model with Pi

[Pi](https://pi.dev) can use the running server through its OpenAI-compatible API.

1. Start the selected model:

   ```bash
   scripts/run.sh qwen3.6-35b-a3b
   ```

2. Merge `config/pi-models.example.json` into `~/.pi/agent/models.json`. Keep any providers already present in that file. Change the model ID, name, and context window when running another profile.

3. Start Pi and select the local model:

   ```bash
   pi --provider local-workbench --model qwen3.6-35b-a3b
   ```

   Alternatively, choose it interactively with `/model`.

The dummy `apiKey` only makes Pi expose the keyless local provider; llama-server does not require it. `qwen-chat-template` lets Pi control Qwen thinking through llama.cpp's chat-template arguments.

## Network access

The server binds to `127.0.0.1` by default. Set `HOST=0.0.0.0` and `EXPOSE_NETWORK=1` in a local config only when remote access is intentional.
