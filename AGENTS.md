# Agent instructions

This repository is a reproducible GGUF workbench built on the llama.cpp TurboQuant fork for Apple Silicon and Linux/NVIDIA.

## Rules

- Keep scripts and documentation model-agnostic. Model-specific values belong only in `config/models/<id>.env`.
- Keep machine-generated settings under ignored `config/local/`; never commit local paths or tuned winners.
- Reuse `scripts/lib/common.sh`; do not introduce another CLI or configuration layer.
- Prefer portable Bash plus existing tools (`curl`, `jq`, CMake). Do not add dependencies without a measured need.
- Preserve TurboQuant KV-cache tuning, OpenAI-compatible llama-server behavior, and localhost binding by default.
- Results must include the model ID and remain machine-readable JSON.

## Checks

```bash
bash -n scripts/*.sh scripts/lib/*.sh
scripts/run.sh --help
scripts/autotune.sh kat-coder --dry-run --quick
```

Do not download models or run expensive benchmarks unless explicitly requested.
