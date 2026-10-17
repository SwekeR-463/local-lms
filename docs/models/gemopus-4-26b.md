# Gemopus 4 26B A4B

Profiles: `gemopus-4-26b`. Measurements are from the RTX 4050 Laptop (6 GB) Linux host.

## Overview

[Gemopus-4-26B-A4B-it](https://huggingface.co/Jackrong/Gemopus-4-26B-A4B-it-Preview) is a supervised fine-tune of Gemma 4 26B instruction, evaluated here through mudler's [APEX I-Mini quant](https://huggingface.co/mudler/Gemopus-4-26B-A4B-it-Preview-APEX-GGUF). The 12,984,460,192-byte GGUF is a 30-layer MoE with 128 routed experts, 8 active per token, and a shared expert; attention uses a 5:1 sliding-window pattern (1,024-token window on 24 layers, full attention with 2 KV heads on 6 layers). The sampling parameters follow the model card: temperature 1.0, top-p 0.95, top-k 64.

```bash
scripts/download-model.sh gemopus-4-26b
scripts/run.sh gemopus-4-26b
scripts/benchmark.sh gemopus-4-26b
```

Mudler's own native-eval table puts I-Mini within sight of the larger APEX profiles (HellaSwag 52.0, MMLU 34.4), which is why this workbench uses the same I-Mini tier the kat-coder profile used on this 6 GB GPU.

## Tuning on the RTX 4050

TODO: fill in after the cpu-moe sweep and 131,072-token probe.

## Benchmarks

TODO: fill in after the throughput smoke benchmark and short-python suite.
