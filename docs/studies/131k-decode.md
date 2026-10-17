# 131k context decode

Cross-model study: Ternary Bonsai PQ2_0 versus GSQ-RCO IQ3_S at 131,072-token context, plus the ngram speculation ceiling. Profiles: `ternary-bonsai-2-27b-pq2`, `qwen38-27b-gsq-rco-iq3`. All measurements are from the Apple Silicon host.

## PQ2_0 versus IQ3_S

Two 27B-class GGUFs were measured at a configured 131,072-token context with the `f16/f16` KV cache, full Metal offload, `1024/1024` batch/ubatch, temperature 0, and seed 42. `scripts/benchmark.sh` supplied the fixed prompt from `config/prompts/benchmark.txt` (80 prompt tokens) and 512 generated tokens, with one warmup plus two measured runs after ten minutes without GPU load.

| Model | Quant | File | Decode, idle | Decode, after sustained load | Decode at 116.6k depth | Prefill at 116.6k depth | Idle RSS |
|---|---|---|---:|---:|---:|---:|---:|
| ISTA-DASLab `Qwen3.8-27B-GSQ-RCO-IQ3_S` | 3.50 bpw mixed | 11.77 GB | **15.92 tok/s** | 10.02 tok/s | 5.43 tok/s | 64.9 tok/s | 19.29 GiB |
| prism-ml `Ternary-Bonsai-2-27B-PQ2_0` | 2.13 bpw ternary | 7.21 GB | **22.96 tok/s** | 13.86 tok/s | 6.86 tok/s | 64.5 tok/s | 15.02 GiB |

The GSQ-RCO file is standard GGUF and loads unmodified in the pinned upstream build, so the `qwen38-27b-gsq-rco-iq3` profile needs no local override; `PQ2_0` only runs on the PrismML fork, which `config/local/ternary-bonsai-2-27b-pq2.env` selects.

Idle decode at 131,072 context is 1.44x faster on Bonsai PQ2_0 than on GSQ-RCO IQ3_S, and 1.38x faster after sustained load. The gap narrows to 1.26x at 116,621 tokens of real prompt depth, where attention over the KV cache dominates the per-token cost and both models fall to 30-34% of their shallow-context speed. Prompt processing is effectively identical at that depth: 64.9 versus 64.5 tok/s. A replicate idle session for GSQ-RCO IQ3_S measured 16.09 tok/s (16.57 and 15.62), so its idle result reproduces within about 1%.

Decode throughput on this laptop depends strongly on how long the GPU has been loaded. Within one 11-minute Bonsai session holding the same binary, model, context, and KV cache, `benchmark.sh` measured 19.37 tok/s at a warm start, 15.99 and 16.21 tok/s immediately after a 3.7-minute 32,928-token prefill, and 23.70 tok/s after five idle minutes. The sustained-load columns above were sampled right after a 30-minute prefill, so they are lower bounds for a cool machine rather than steady-state ratings. The earlier 2026-09-18 session recorded 14.91 tok/s for PQ2_0 at 131,072 context with the same fork commit, which falls inside today's observed 10-24 tok/s range.

The publisher lists 28.1 tok/s TG128 for PQ2_0 on an M5 Pro laptop, measured with `llama-bench` at depth 0; this workbench's server benchmark holds a 131,072-token KV allocation and a full chat template, so the two figures are not directly comparable. The machine-readable record, including each file's SHA-256, the pinned runtime commits, every source file, and the thermal series, is `results/131k-decode-comparison.json`.

Both profiles now carry their vision projector (`MODEL_MMPROJ_FILE`): `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (600 MiB) and `mmproj-Qwen3.8-27B-BF16.gguf` (888 MiB). With the projector loaded the servers idle at 15.72 GiB and 20.28 GiB RSS, and each one named all four quadrants of a red/blue/green/yellow test image correctly (`results/vision-test-*.json`).

Prescribed sampling does not change throughput. Repeating the idle protocol with `scripts/benchmark.sh MODEL --sampling profile` (temperature 1.0, top-p 0.95, top-k 20, min-p 0.05 for Bonsai and 0.0 for the GSQ-RCO file, xhigh reasoning) measured 15.92 tok/s for GSQ-RCO IQ3_S and 25.76 tok/s for Bonsai PQ2_0, matching the greedy table above within the session-to-session spread. Runs issued at the end of a session are 10-20% slower than runs at the start regardless of sampler, which is the load sensitivity described above; the Bonsai profile also gained the `min_p=0.05` the card prescribes, which its earlier `min_p=0.0` was overriding.

## Decode ceiling and ngram speculation

A pthread STREAM probe measured this M5 Pro's read path at 313 GB/s (15 threads; Apple rates the chip at 307 GB/s), so the dense 11.77 GB GSQ-RCO file is capped at ~26 tok/s for novel text no matter the settings; 30-40 tok/s would need 353-471 GB/s. Measured on 2026-10-12: 16.6-16.7 tok/s idle at 131k without speculation (reproducing the 2026-10-10 result) and 14.6 tok/s on first-occurrence text with `--spec-type ngram-mod`.

ngram-mod pays off when output repeats content already in context: draft acceptance rises to 95-100% and one weight read emits several tokens. A fresh repetitive prompt measured 34.61 tok/s versus 16.72 without speculation (byte-identical output), and identical repeated requests reached 44-68 tok/s. Enable with `SPEC_TYPE=ngram-mod scripts/run.sh qwen38-27b-gsq-rco-iq3`; expect roughly a 12% penalty on novel text.

The six-task `short-python-v1` suite ran at 131k on the ngram server: 3/6 with the suite's greedy/low defaults and a 1,024-token cap (three cap truncations), and 4/6 with the prescribed sampling at a 4,096-token cap (two cap truncations). Decode settled at 9-11 tok/s under sustained load after 17 tok/s on the first task, so 30-40 tok/s is not sustained for coding workloads on this machine. The machine-readable record is `results/131k-ngram-speculation-qwen38-27b-gsq-rco-iq3.json`.
