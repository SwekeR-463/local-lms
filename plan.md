# KAT-Coder on a 6 GB GPU: implementation plan

## Objective

Run the KAT-Coder V2.5 Dev APEX GGUF model locally with:

- the `mudler/KAT-Coder-V2.5-Dev-APEX-GGUF` repository;
- its Mini profile, preferably the imatrix-calibrated `I-Mini` file;
- NVIDIA GPU offload using the laptop's 6 GB RTX 4050 Max-Q;
- CPU execution for the remaining MoE weights;
- a preferred 131,072-token context, with 65,536 tokens accepted as the minimum successful target;
- an automatic tuner that finds the fastest configuration that does not OOM;
- a reproducible OpenAI-compatible local server launcher.

This is a best-effort hardware-tuning project. A successful build is not the same as a successful 131k run: the model is approximately 35B total parameters, and this laptop has 15 GB system RAM. The Mini file should be materially smaller than the APEX Compact file, but the exact size must be read from the repository rather than hard-coded. The tuner must report infeasibility instead of silently relying on swap.

## Hardware baseline to record

The current machine was detected as:

- Ubuntu 24.04, x86_64;
- Intel Core i5-13420H, 8 physical cores / 12 threads;
- NVIDIA RTX 4050 Max-Q, 6 GB VRAM;
- 15 GB system RAM;
- one NUMA node.

Do not hard-code these values. `scripts/preflight.sh` must detect and print the actual hardware every time.

## Required repository layout

Create the following files:

```text
config/default.env             # user-editable model and server settings
scripts/preflight.sh           # dependency, driver, RAM, disk, and flag checks
scripts/build.sh               # build or install the TurboQuant server
scripts/download-model.sh      # verify and download the selected GGUF
scripts/autotune.sh            # benchmark candidate configurations
scripts/run.sh                 # launch the selected configuration
scripts/stop.sh                # stop only this project's server process
scripts/lib/common.sh           # shared logging, validation, and process helpers
results/.gitkeep               # benchmark output; ignore generated results
README.md                      # installation, tuning, launch, and troubleshooting
.gitignore                     # model/cache/log/result exclusions
```

Keep generated binaries, model files, logs, and benchmark results outside Git. Never commit model weights.

## Configuration contract

Add `config/default.env` with these defaults, allowing every value to be overridden by environment variables:

```bash
MODEL_REPO="mudler/KAT-Coder-V2.5-Dev-APEX-GGUF"
MODEL_PROFILE="I-Mini"
MODEL_FILE="KAT-Coder-V2.5-Dev-APEX-I-Mini.gguf"
MODEL_PATH=""
SERVER_BIN=""
BUILD_DIR=".cache/llama-cpp-turboquant/build"
RESULTS_DIR="results"
HOST="127.0.0.1"
PORT="8000"
PREFERRED_CONTEXT="131072"
MIN_ACCEPTED_CONTEXT="65536"
PARALLEL="1"
SEED="42"
TUNE_TIMEOUT_SECONDS="180"
MIN_RAM_HEADROOM_MB="1024"
MIN_VRAM_HEADROOM_MB="256"
ALLOW_VULKAN_FALLBACK="0"
ALLOW_SWAP="0"
```

The target repository is valid and the live repository API currently lists the exact default file as `KAT-Coder-V2.5-Dev-APEX-I-Mini.gguf`. The resolver still verifies the file on every run because Hugging Face's web file listing is not stable enough to use as the only source of truth. `download-model.sh` must:

1. Query `https://huggingface.co/api/models/$MODEL_REPO` and list `.siblings[].rfilename`.
2. If `MODEL_FILE` is set, require an exact match.
3. Otherwise, when `MODEL_PROFILE=I-Mini`, prefer a filename ending in `APEX-I-Mini.gguf`, then one ending in `I-Mini.gguf`.
4. If no imatrix Mini file exists, try `APEX-Mini.gguf`/`-Mini.gguf` and print that the non-imatrix fallback was selected.
5. Never silently fall back to `I-Compact`, Balanced, a normal Q4 GGUF, or an MTP file. Those require an explicit profile.
6. Record the resolved filename, exact byte size, SHA256, repository revision, and profile in `results/model.json`.

The implementation should support these explicit profiles: `I-Mini`, `Mini`, `I-Compact`, `Compact`, and `MTP-I-Compact`. The default is `I-Mini` because the target machine has limited RAM/VRAM. The MTP file is a separate experiment and must not be selected just because it has a similar name.

Context policy is flexible: `PREFERRED_CONTEXT=131072` is the stretch goal, while `MIN_ACCEPTED_CONTEXT=65536` is a valid success. Both values must be configurable. A failed 131k search must not invalidate a stable 64k result.

## Phase 1: preflight

Implement `scripts/preflight.sh` first. It must:

1. Use `set -Eeuo pipefail` and produce readable diagnostics.
2. Check `cmake`, a C/C++ compiler, `git`, `curl`, `awk`, `sed`, `timeout`, and `jq`.
3. Detect the GPU with `nvidia-smi`.
   - If the PCI device exists but `nvidia-smi` fails, stop and explain that the NVIDIA driver/runtime is unavailable.
   - Do not claim CUDA support merely because an NVIDIA PCI device is present.
4. Print GPU name, VRAM, driver, CPU threads/cores, RAM, swap, and free disk.
5. Refuse to proceed when swap is active unless `ALLOW_SWAP=1`. Swap is not an acceptable solution for a 131k production run.
6. Check the model file size and estimate available memory before launching.
7. Verify that the selected server supports all requested flags by parsing `SERVER_BIN --help`. Unsupported optional flags must be removed or reported; do not pass unknown flags and hope they work.
8. Check that the port is unused and that a prior PID belongs to this project before stopping it.

The preflight output must be saved as `results/preflight-<timestamp>.txt`.

## Phase 2: obtain/build the runtime

Use the TurboQuant fork at:

```text
https://github.com/TheTom/llama-cpp-turboquant
```

Prefer a tagged/prebuilt release when it supports the current GPU. Otherwise:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON
cmake --build build --config Release -j"$(nproc)"
```

Locate `llama-server` after the build and store its absolute path in the generated tuning metadata. If CUDA cannot be initialized and `ALLOW_VULKAN_FALLBACK=1`, build/test Vulkan separately; otherwise stop rather than silently falling back to CPU-only execution.

The implementation must verify the actual command name. The original draft uses `lm-server-tq`, while the referenced project documents `llama-server`; treat `lm-server-tq` as a local wrapper only if it exists and its `--help` output proves compatibility.

## Phase 3: download and inspect the model

`scripts/download-model.sh` must:

1. Verify the repository and resolve/verify the filename according to `MODEL_PROFILE` before downloading.
2. Download to a user-selected model directory with resume support.
3. Verify the final file exists and has non-zero size.
4. Run the project's GGUF/model inspection utility if available.
5. Print architecture, context advertised by the model, total experts, active experts, and quantization metadata.
6. Record SHA256, file size, repository revision, and download date in `results/model.json`.

The model is an MoE model with about 35B total and about 3B active parameters per token. Active parameters reduce compute, not the amount of expert weights that must be stored or mapped. The Mini profile reduces weight storage, but does not make a 131k KV cache free.

## Phase 4: establish a known-good baseline

Before attempting 131k, launch a short-context baseline at 8,192 tokens with GPU offload and no speculative decoding. Use the least risky supported KV configuration:

```text
--cache-type-k q8_0
--cache-type-v turbo4
```

Use `-fa on` if supported. Start with `-ngl 99` (or the equivalent full-offload request) and let `--n-cpu-moe` control CPU MoE placement when supported. Start `--n-cpu-moe` at 32, then tune it; do not assume 36 is optimal.

Baseline acceptance criteria:

- server starts and stays alive for at least 30 seconds;
- `/health` reports ready;
- one chat/completion request returns valid text;
- no CUDA error, assertion, NaN, repeated-token loop, or process crash;
- GPU utilization and memory are visible in `nvidia-smi`;
- prompt and generation tokens/second are recorded.

Save the complete command, environment, server stderr, RSS, peak VRAM, and response to `results/baseline-<timestamp>/`.

## Phase 5: automatic tuner

Implement `scripts/autotune.sh` as a bounded, resumable search. Every candidate must be launched in a fresh process and killed through the recorded PID only.

### Candidate dimensions

Search in this order, stopping early when the preferred target is met:

1. Context: `8192`, `16384`, `32768`, `65536`, `98304`, `131072`.
2. KV precision, from safest to most aggressive:
   - `q8_0 / turbo4`;
   - `q8_0 / turbo3`;
   - `q8_0 / turbo2`;
   - `f16 / turbo4` only if quality validation needs it.
3. CPU MoE count: model-specific starting point `32`, then `28`, `36`, `24` if supported.
4. Batch sizes: `512`, `1024`, `2048`.
5. Ubatch sizes: `256`, `512`, `1024`.
6. Threads: physical core count, then physical core count minus one. Batch threads may use all logical threads.

Do not search symmetric `turbo3/turbo3` or `turbo2/turbo2` configurations initially. The TurboQuant documentation specifically recommends keeping K at higher precision than V. Add symmetric modes only behind an explicit `ALLOW_RISKY_KV=1` override.

### Candidate execution

For each candidate:

1. Generate a unique run directory and command manifest.
2. Start the server with `timeout` and redirect stdout/stderr there.
3. Wait for `/health`, with a bounded startup timeout.
4. Send a short deterministic smoke prompt.
5. If the context is greater than 8k, call `/tokenize` when available and submit a synthetic repeated prompt near the requested token count. Otherwise mark context validation as unavailable, not successful.
6. Sample `nvidia-smi` and `/proc/<pid>/status` during startup and inference.
7. Record startup time, prompt throughput, generation throughput, peak VRAM, peak RSS, exit code, and failure classification.
8. Kill the candidate and verify the process is gone before trying the next one.

Failure classifications must include: unsupported flag, model load failure, CUDA initialization failure, VRAM OOM, RAM OOM, timeout, HTTP failure, invalid output, and unknown crash.

### Selection policy

A candidate is eligible only when:

- the server reaches ready state;
- the requested context is actually accepted;
- the smoke request returns valid output;
- peak VRAM leaves at least `MIN_VRAM_HEADROOM_MB`;
- peak RSS plus the configured safety margin does not exceed physical RAM;
- no swap-in activity is observed;
- no CUDA/runtime errors occur.

Selection must happen in two stages:

1. Prefer the highest eligible context, with `PREFERRED_CONTEXT` winning when available.
2. Within that context, select the highest generation throughput. If throughput is within 5%, select the lower-memory candidate.

If no candidate reaches `PREFERRED_CONTEXT`, select the fastest eligible candidate at the highest context that is at least `MIN_ACCEPTED_CONTEXT`. Mark the result as `preferred-context-unavailable` but `success-minimum-context`. If no candidate reaches `MIN_ACCEPTED_CONTEXT`, mark the run failed and preserve the highest stable context for diagnosis. Write the winner to `config/selected.env` and `results/autotune-summary.json`.

The tuner must support `--resume`, `--max-context`, `--quick`, and `--dry-run`.

## Phase 6: production launcher

Implement `scripts/run.sh` to:

1. Load `config/default.env` and `config/selected.env`.
2. Run preflight unless `SKIP_PREFLIGHT=1`.
3. Require an explicit model path.
4. Use the selected candidate rather than reusing the original rough command blindly.
5. Bind to `127.0.0.1` by default. Require `EXPOSE_NETWORK=1` before using `0.0.0.0`.
6. Write a PID file and log path under `results/`.
7. Support `--foreground` and print the final OpenAI-compatible endpoint.
8. Refuse to start if another process owns the configured port.

Use these flags only when `SERVER_BIN --help` confirms them:

```text
-m MODEL_PATH
--host HOST --port PORT
-fa on
--ctx-size CONTEXT
--threads THREADS --threads-batch THREADS_BATCH
--parallel 1
--cache-type-k K_TYPE --cache-type-v V_TYPE
--ubatch-size UBATCH --batch-size BATCH
--n-cpu-moe N_CPU_MOE
--metrics
```

Treat `--no-mmap`, `--fit off`, `--cache-ram`, `--ctx-checkpoints`, `--spec-type ngram-mod`, CPU affinity, and NUMA flags as optional experiments. Record them only when supported and when a benchmark demonstrates a benefit. On this laptop, avoid `--no-mmap` unless required by the runtime because it can increase resident memory pressure.

Speculative decoding is a second-stage optimization. First establish a stable non-speculative winner. Then test `ngram-mod` with a separate A/B benchmark; it must not be enabled if the selected binary does not expose the flag or if acceptance rate/throughput gets worse.

## Phase 7: validation

Add `scripts/validate.sh` with three test levels:

1. `--short`: 8k context, deterministic smoke prompt, server API check.
2. `--accepted`: test the selected context; this must be at least `MIN_ACCEPTED_CONTEXT` for success.
3. `--preferred`: test near-`PREFERRED_CONTEXT`; report unavailable rather than failing the whole run when the selected configuration is 64k.
4. `--quality`: same prompts across the baseline and selected KV configurations; detect empty output, malformed tool/chat delimiters, repetition, and obvious corruption.

Validation must report “not tested” when it cannot create the requested token count. Do not report a 131k success based only on `--ctx-size 131072` appearing in the command line.

## Stop conditions and expected outcome

Stop and report the blocker if any of these occur:

- NVIDIA driver cannot initialize;
- exact target GGUF cannot be found;
- model load requires more RAM than available without swap;
- all candidates at or above `MIN_ACCEPTED_CONTEXT` OOM or time out;
- the server accepts the flag but fails during a real long-context request;
- no candidate produces valid output.

If 131k fails but 64k passes, report 64k as a successful fallback and preserve the 131k failure logs. Do not weaken memory checks or enable swap merely to label the task successful. Only report the overall run as failed when no candidate reaches `MIN_ACCEPTED_CONTEXT`.

## Acceptance criteria for the implementation

Another agent can consider this plan implemented only when:

- all scripts are executable and pass `shellcheck` where available;
- `preflight.sh` gives actionable diagnostics;
- build and model download are reproducible;
- baseline inference succeeds;
- autotuning produces a machine-readable result and selected configuration;
- `run.sh` starts the selected configuration safely;
- `validate.sh` distinguishes configured context from actually tested context and treats 64k as an accepted fallback;
- README documents the exact commands and known hardware limitations;
- no model, binary, credential, or generated log is committed.

## Original command: reference only

The original command is retained here only as a record of the intended settings. It must not be copied into the launcher without the capability checks and tuning above. In particular, its Q/K settings, model alias, server binary name, speculative-decoding flags, `--no-mmap`, NUMA settings, and CPU affinities all require validation on the target machine.
