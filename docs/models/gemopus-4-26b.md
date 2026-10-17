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

A `--cpu-moe-sweep` walked `n-cpu-moe` 26/28/30 at 32k and 65k with `q8_0/turbo4`, then a 131,072-token probe compared the same three values against `q8_0/turbo3` and `q8_0/turbo4`. Batch and ubatch stayed at 512 throughout; the I-Mini expert tensors are small enough (~250 MB per layer) that every candidate fit the 6 GB GPU even at 131k, so the sweep maximized throughput instead of triaging out-of-memory failures.

| context | n-cpu-moe | pp tok/s | tg tok/s |
| --- | --- | --- | --- |
| 32,768 | 26 | **355** | **18** |
| 32,768 | 28 | 336 | 17 |
| 32,768 | 30 | 322 | 16 |
| 65,536 | 26 | **339** | **15** |
| 65,536 | 28 | 266 | 7 |
| 65,536 | 30 | 240 | 7 |
| 131,072 | 26 | 276 | 10 |
| 131,072 | 28 | 265 | 10 |
| 131,072 | 30 | 264 | 9 |

All values above are `q8_0/turbo4`. The 131,072-token probe then repeated the matrix with `q8_0/turbo3`:

| n-cpu-moe | pp tok/s | tg tok/s |
| --- | --- | --- |
| 26 | **307** | **11** |
| 28 | 291 | 11 |
| 30 | 285 | 10 |

`turbo3` beat `turbo4` at the full window on both prompt and generation throughput, and 26 CPU-expert layers won at every context. The winner written to `config/local/gemopus-4-26b.env` is **131,072 context, `q8_0/turbo3`, `n-cpu-moe=26`, batch/ubatch 512, 12 threads**. Threads 8 vs 12 were a tie on the smoke benchmark (24.64 vs 24.30 tok/s), so the box's usual 12 was kept.

## Benchmarks

The smoke benchmark (37-token prompt, 512 generated tokens) sustained **24.6 tok/s** generation at ~4.9 GiB VRAM and 8.6 GiB RSS, with tern's 508 MiB and the desktop resident alongside.

The short-python suite (six tasks, greedy, seed 42, 1,024-token cap) scored **5/6**: merge-intervals, flatten-dict, dependency-order, parse-env, and batch-iterable passed. `binary-metrics` failed the evaluator's strict dictionary equality — the model's metrics were close but not exact, a recurring trip-up for every model tested so far.

### Paired comparison with KAT-Coder V2.5 Dev

KAT-Coder's July 4050 tuning only produced throughput numbers; it had never run the code suite. Both models were therefore served with the identical recipe (131,072 context, `q8_0/turbo3`, batch/ubatch 512, 12 threads) and measured back to back on this box. KAT-Coder needed `n-cpu-moe=36` instead of July's 32 because the 519 MiB compute buffer no longer fit alongside tern's resident 508 MiB; gemopus used its tuned `n-cpu-moe=26`.

| measurement | Gemopus 4 26B | KAT-Coder V2.5 Dev |
| --- | --- | --- |
| 131k deep probe pp / tg (this session) | 307 / **11 tok/s** | **414** / **11 tok/s** |
| smoke benchmark tg (512 tokens) | **24.6 tok/s** | 20.1 tok/s |
| short-python suite | **5/6** | 4/6 |
| failure detail | binary-metrics: values off strict equality | parse-env: spent all 1,024 tokens inside `think`, emitted no code; binary-metrics: zeroed metrics |

KAT-Coder prompt-processes the full 131,072-token window ~35% faster (414 vs 307 tok/s), while generation throughput is a dead tie at this depth. At short context Gemopus generates faster and it scores one task higher on the suite. KAT-Coder's `parse-env` failure is a reasoning-budget artifact: its `think` block consumed the entire token cap.
