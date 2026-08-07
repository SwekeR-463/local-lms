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
qwen3.6-35b-a3b
qwen3.5-27b
```

Qwen3.6 currently has a 35B-A3B release; the included dense 27B profile is Qwen3.5.

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

`benchmark.sh` sends the fixed prompt in `config/prompts/benchmark.txt` to a running server and writes JSON containing the model/profile, quant file, context, hardware, memory, prompt throughput, generation throughput, and response. This is a throughput smoke benchmark, not a quality leaderboard.

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
