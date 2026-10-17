# Workflow

Operating manual for the workbench: profiles, configuration precedence, adding models, and client integration.

## Included profiles

```text
btl-4-compact
kat-coder
kat-coder-mtp
ling3-tiny-q8
muse-glimmer
muse-glimmer-dflash
ornith-35b-i-mini
ornith15-35b-apex-compact
ornith15-9b-q8
qwen3.5-27b
qwen3.6-35b-a3b
qwen36-27b-q2
qwen36-27b-q2-dflash
qwen36-27b-q2-mtp
qwen38-27b-ad-iq3
qwen38-27b-ad-iq3-dflash2
qwen38-27b-gsq-rco-iq3
qwen38-27b-iq3
qwen38-27b-ud-q3
qwen38-27b-ud-q3-dflash2
qwen38-27b-ud-q3-v3
ternary-bonsai-2-27b-pq2
```

BTL-4 Compact and the Qwen3.6 27B profiles require recent upstream llama.cpp. The pinned TurboQuant fork produced slower MTP results in the earlier Qwen Q3_K_M experiment.

Measured detail for the profiles above lives in [models/](models/) and [studies/](studies/).

## Adding a model

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

## Vision projector

Set `MODEL_MMPROJ_FILE` to also fetch and serve a vision projector:

```bash
MODEL_MMPROJ_FILE="mmproj-my-model.gguf"
```

`scripts/download-model.sh` stores it next to the main GGUF, `scripts/preflight.sh` reports it, and `scripts/run.sh` passes `--mmproj` whenever the server build supports the flag.

## Configuration order

Later files override earlier files:

1. `config/default.env`: portable runtime defaults
2. `config/models/<id>.env`: model metadata and context targets
3. `config/runtime.env`: generated llama.cpp TurboQuant build path
4. `config/local/<id>.env`: generated machine-specific tuning winner

Models are stored under ignored `models/`. Generated runs are stored under `results/` and include the model ID.

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

## Codex against a local model

Run Codex against a local model with automatic server cleanup:

```bash
scripts/codex-local.sh ornith15-35b-apex-compact
```

The wrapper uses port 8001 by default; set `CODEX_LOCAL_PORT` to change it.

## Network access

The server binds to `127.0.0.1` by default. Set `HOST=0.0.0.0` and `EXPOSE_NETWORK=1` in a local config only when remote access is intentional.
