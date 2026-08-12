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
btl-4-compact
kat-coder
kat-coder-mtp
ling3-tiny-q8
muse-glimmer
muse-glimmer-dflash
ornith-35b-i-mini
qwen36-27b-q2
qwen36-27b-q2-mtp
qwen36-27b-q2-dflash
qwen3.6-35b-a3b
qwen3.5-27b
```

BTL-4 Compact and the Qwen3.6 27B profiles require recent upstream llama.cpp. The pinned TurboQuant fork produced slower MTP results in the earlier Qwen Q3_K_M experiment.

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

The same target was also tested with Alittlehammmer's recommended Q8_0 DFlash drafter and `--spec-draft-n-max 6`. Because sequential exploratory runs varied with system state, the reported comparison restarted each server and spaced its runs by 30 seconds.

| Mode | Generation | RSS | Draft acceptance |
|---|---:|---:|---:|
| Paired baseline | 9.54 tok/s | 15.95 GiB | — |
| DFlash Q8_0 | **10.71 tok/s** | 18.96 GiB | 91.9% |

![Qwen3.6 Q2 baseline and DFlash throughput](results/plots/qwen36_q2_dflash.png)

DFlash gained **12.2%** in the controlled pair and preserved byte-identical output, but added about 3.0 GiB RSS. Earlier exploratory DFlash runs ranged from 10.53 to 18.36 tok/s, so the isolated peak is not treated as sustained performance. Results are in `results/qwen36-27b-q2-dflash-results.json`.

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

## BTL-4 Compact

[BTL-4 Compact](https://huggingface.co/badtheorylabs/BTL-4-Compact) packages the 35.1B-parameter, roughly 2.1B-active MoE as a 9.3 GiB `IQ2_XXS` GGUF. It requires recent upstream llama.cpp with `qwen35moe` support. The included profile runs at 65,536 context with q8_0 K/V cache and the model card's required Jinja and DeepSeek reasoning settings:

```bash
scripts/download-model.sh btl-4-compact
scripts/run.sh btl-4-compact
pi --provider local-workbench --model btl-4-compact
```

On the M5 Pro, a 32k smoke test generated at **71.41 tok/s**, used about **9.82 GiB RSS** after loading, returned the requested exact response, and emitted a correct OpenAI-compatible tool call. At 65k, idle RSS was about **10.16 GiB** and peak observed RSS during the coding run was about **10.54 GiB**.

The 65k direct-prompt agent completed a TypeScript project in 7.94 minutes without context compaction. Five supervised correction rounds made its mocked tests and TypeScript checks pass, but independent review still rejected the production path: face detection and ffmpeg trimming remained placeholders, command execution used a shell, and destination key construction was wrong. BTL-4 is promising as a fast supervised pair programmer, not a production-reliable autonomous agent; subjective assessment is **4/10 autonomous, 6/10 with five-round supervision**.

![BTL-4 Compact runtime and coding-agent assessment](results/plots/btl4_compact_summary.png)

The chart intentionally leaves 64k generation blank because no controlled 64k throughput smoke benchmark was run. Machine-readable measurements are in `results/btl-4-compact-results.json`.

`--jinja` and `--reasoning-format deepseek` are required for this model's tool-call template and reasoning separation. `scripts/run.sh` reads the latter from `REASONING_FORMAT` in the profile.

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
