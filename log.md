# Implementation log

This file is the handoff point for continuing agents. Append a dated entry after every meaningful action. Do not delete earlier entries.

## 2026-07-25 — project setup

Phase 0—1: scaffold, model download, preflight checks.

- `2026-07-25 07:58:17 IST` — preflight completed; report saved at results/preflight-20260725-075816.txt

- `2026-07-25` — shell scaffold passed `bash -n` and help checks for all scripts.
- `2026-07-25` — diagnostic preflight passed; confirmed 15 GiB RAM, one NUMA node, active swap, missing usable NVIDIA driver, no runtime binary, and no model downloaded.
- `2026-07-25` — live Hugging Face API query confirmed `KAT-Coder-V2.5-Dev-APEX-I-Mini.gguf` in `mudler/KAT-Coder-V2.5-Dev-APEX-GGUF`; API reported no byte size.
- `2026-07-25` — remote headers confirmed the Mini GGUF is 13,467,211,136 bytes (about 12.5 GiB); download has not started yet.
- `2026-07-25` — added `--resolve-only` to the downloader so future agents can verify the remote artifact without starting a large download.
- `2026-07-25` — downloader `--resolve-only` passed end to end and confirmed `KAT-Coder-V2.5-Dev-APEX-I-Mini.gguf`, 13,467,211,136 bytes (~12.5 GiB).
- `2026-07-25` — started the approved 12.5 GiB model download to `models/KAT-Coder-V2.5-Dev-APEX-I-Mini.gguf.part`; transfer is in progress and will be verified before rename.
- `2026-07-25` — improved preflight to load generated runtime/model configuration and improved long-context validation to submit a context-sized deterministic prompt rather than only mentioning the target length.
- `2026-07-25` — TurboQuant source clone completed at `.cache/llama-cpp-turboquant/src`; build stopped as expected because `nvcc` is not installed. No runtime binary exists yet.
- `2026-07-25` — added build-script help/argument validation after the initial probe attempted a clone for `--help`.
- `2026-07-25` — inspected the cloned source: `turbo2`, `turbo3`, `turbo4`, `--n-cpu-moe`, `--cache-ram`, `--ctx-checkpoints`, and `ngram-mod` are implemented in this branch; generated README tables omit the Turbo cache names, so runtime `--help` checks remain mandatory.
- `2026-07-25` — source checkout is branch `feature/turboquant-kv-cache` at commit `c26cbdf` and occupies about 193 MiB.
- `2026-07-25` — model download passed 4 GiB and is still running; partial file remains `.part` until checksum/size verification succeeds.
- `2026-07-25` — partial download begins with a valid GGUF v3 header (`GGUF`), providing an early integrity sanity check.
- `2026-07-25` — user confirmed to continue with the I-Mini profile rather than switch to I-Compact; use case is local data-pipeline writing and monitoring, prioritizing memory headroom and stable 64k context.
- `2026-07-25` — after the user's confirmation, the I-Mini transfer continued past 6 GiB; no model variant switch was made.
- `2026-07-25 08:27` — I-Mini download completed. Final size is 13,467,211,136 bytes; SHA256 verified; no `.part` file remains.
- `2026-07-25` — post-download inspection found NVIDIA driver and CUDA 12.8 installed on the host; sandboxed `nvidia-smi`/`nvcc` checks were misleading because device/tool paths are isolated. Build script now discovers `/usr/local/cuda*/bin/nvcc` and sets `CUDACXX`.

## 2026-07-25 — build, baseline, autotune

### Build

- `2026-07-25` — confirmed NVIDIA GPU working: RTX 4050 Max-Q, 6141 MiB VRAM, driver 580.x. nvcc at `/usr/local/cuda-12.8/bin/nvcc`.
- `2026-07-25 10:07:24 IST` — TurboQuant build completed with CUDA 12.8. Binary at `.cache/llama-cpp-turboquant/build/bin/llama-server` (commit `c26cbdf`, branch `feature/turboquant-kv-cache`). Build took ~50 min (CUDA template compilation dominates). All key flags confirmed: `--cache-type-k/v` supports `turbo2`/`turbo3`/`turbo4`, `--n-cpu-moe`, `-ngl`, `-fa`, `--cache-ram`, `--ctx-checkpoints`, `--metrics`, `--jinja`, `ngram-mod` speculative decoding.

### Baseline (8k warm-up)

- `2026-07-25` — baseline at 8k context, q8_0/turbo4, n-cpu-moe=32, gpu-layers=99:
  - Server startup: 38s to `/health`
  - RSS: 7.3 GiB, VRAM: 4.3 GiB (of 6 GiB), GPU util: 0% idle after load
  - Prompt throughput: ~5.4 tok/s, Generation throughput: ~8.0 tok/s
  - Status: healthy, valid code generation output
  - Results saved to `results/baseline-20260725-100751/`

### Autotune — first attempts (failed)

- `2026-07-25 10:15:20 IST` — quick autotune (12 candidates). All 8k and 32k candidates passed, but all 4x 65k candidates returned `http-failure`. Root cause: `make_prompt()` generates `context * 4` chars (262k chars for 65k), and passing this via `jq --arg` exceeds Linux `ARG_MAX` (~2 MiB). The prompt string itself fits, but the jq argument encoding makes the command line too long.
- Fix: switched `jq --arg prompt "${prompt}"` → `jq --rawfile prompt tempfile.txt`. Same fix applied to `scripts/validate.sh`.

- `2026-07-25 10:42:00 IST` — re-ran after rawfile fix. All 8k and 32k candidates still pass. 65k candidates now start correctly but all 4 fail with `http-failure`. Root cause: `TUNE_TIMEOUT_SECONDS=180` curl timeout expires before the server finishes processing the 60k+ token prompt. The server logs show it was processing at ~285 tok/s and needed ~230s for 65k tokens, but curl killed the request after 180s.
- Fix: increased `TUNE_TIMEOUT_SECONDS` from 180 → 480 in `config/default.env`.

### Autotune — successful quick run (12 candidates)

- `2026-07-25 11:14:45 IST` — quick autotune succeeded with all 12 candidates completing. Candidate matrix:

  | # | ctx | K | V | n-cpu-moe | status | RSS | VRAM |
  |---|-----|---|---|-----------|--------|-----|------|
  | 1 | 8k | q8_0 | turbo4 | 30 | ready | 5.6 GiB | 4.9 GiB |
  | 2 | 8k | q8_0 | turbo4 | 32 | ready | 7.3 GiB | 4.3 GiB |
  | 3 | 8k | q8_0 | turbo3 | 30 | ready | 9.0 GiB | 4.9 GiB |
  | 4 | 8k | q8_0 | turbo3 | 32 | ready | 8.9 GiB | 4.2 GiB |
  | 5 | 32k | q8_0 | turbo4 | 30 | ready | 9.0 GiB | 5.1 GiB |
  | 6 | 32k | q8_0 | turbo4 | 32 | ready | 8.8 GiB | 4.5 GiB |
  | 7 | 32k | q8_0 | turbo3 | 30 | ready | 6.3 GiB | 5.1 GiB |
  | 8 | 32k | q8_0 | turbo3 | 32 | ready | 9.1 GiB | 4.4 GiB |
  | 9 | 65k | q8_0 | turbo4 | 30 | ready | 8.8 GiB | 5.4 GiB |
  | 10 | 65k | q8_0 | turbo4 | 32 | ready | 7.4 GiB | 4.7 GiB |
  | 11 | 65k | q8_0 | turbo3 | 30 | ready | 4.8 GiB | 5.4 GiB |
  | 12 | 65k | q8_0 | turbo3 | 32 | ready | 5.2 GiB | 4.7 GiB |

  Quick search winner: **65,536 context, q8_0/turbo3, n-cpu-moe=32, batch/ubatch=512**. RSS 5.2 GiB, VRAM 4.7 GiB. Status: `success-minimum-context` (131k preferred not yet tested in quick mode). Both turbo3 and turbo4 work at all tested contexts; turbo3 tends to use slightly less VRAM.

### Autotune — 131k manual validation

- `2026-07-25` — focused manual test at 98k and 131k (4 candidates: turbo3/turbo4 × 98k/131k, all n-cpu-moe=32):

  | ctx | K | V | n-cpu-moe | startup | prompt tok | inference | RSS | VRAM |
  |-----|---|---|-----------|---------|------------|-----------|-----|------|
  | 98k | q8_0 | turbo3 | 32 | 27s | 78,658 | 288s (~273 tok/s) | 6.6 GiB | 5.0 GiB |
  | 98k | q8_0 | turbo4 | 32 | 22s | 78,658 | 288s (~273 tok/s) | 6.3 GiB | 5.0 GiB |
  | 131k | q8_0 | turbo3 | 32 | 16s | 104,874 | 429s (~244 tok/s) | 8.9 GiB | 5.2 GiB |
  | 131k | q8_0 | turbo4 | 32 | 18s | 104,874 | 432s (~243 tok/s) | 9.1 GiB | 5.3 GiB |

  All four passed. Memory is within limits at all tested contexts (6 GiB VRAM, 15 GiB RAM). The VRAM headroom at 131k is narrow (~0.8 GiB free) but stable. The KV cache growth from 8k→131k adds only ~1 GiB VRAM (4.3→5.2) because the turbo3/turbo4 compression is efficient.

  **Final winner: 131,072 context, q8_0/turbo3, n-cpu-moe=32, batch/ubatch=512.** turbo3 selected over turbo4 for: slightly faster inference (429s vs 432s), slightly less VRAM (5.2 vs 5.3 GiB), and slightly less RSS (8.9 vs 9.1 GiB). Status: `success-preferred-context`. Written to `config/selected.env` and `results/autotune-summary.json`.

### Key observations from autotuning

- **n-cpu-moe does not vary much at this scale.** At 8k, n-cpu-moe=30 uses 2 GiB less RSS than 32 (5.6 vs 7.3) with q8_0/turbo4, but this gap closes at higher contexts. At 65k, the RSS difference between 30 and 32 is negligible.
- **turbo3 vs turbo4: marginal difference.** turbo3 consistently matches or edges out turbo4 in speed and memory, so it was selected. turbo2 was not tested (quick search skipped it).
- **Prompt throughput drops predictably with context:** ~294 tok/s at 8k, ~288 tok/s at 65k, ~244 tok/s at 131k. This is expected due to MoE expert loading overhead and larger KV cache lookups.
- **Swap is the major concern for production.** 572 MiB swap is already used (pre-existing), and the 131k configuration leaves only ~6 GiB free RAM. Long runs should be monitored for swap growth.

Current status: **131k context achieved** on RTX 4050 Max-Q with the I-Mini GGUF. All scripts are functional and syntax-checked. Next steps for another agent:

1. Run `scripts/run.sh` to launch the production server with the 131k winner.
2. Run `scripts/validate.sh --accepted` to verify the running server handles 131k.
3. Optionally run full `scripts/autotune.sh` — but 800+ candidates would take 12+ hours.
4. Consider A/B testing `ngram-mod` speculative decoding and `--cache-ram` as second-stage optimizations.
5. Monitor swap usage during prolonged 131k runs.

- `2026-08-04 22:27:46 IST` — preflight completed; report saved at results/preflight-20260804-222746.txt

- `2026-08-04 22:27:47 IST` — server started with PID 20349; log results/server-20260804-222746.log

- `2026-08-04 22:30:45 IST` — stopped project server PID 20349

- `2026-08-05 00:30:23 IST` — autotune selected 98304-token configuration; summary saved at results/autotune-summary.json

## Apple Silicon tuning — 2026-08-05

- Ported the runner to macOS while retaining Linux/CUDA support. Built TurboQuant commit `0967f499714dd6018494b480b710b849ca45b156` with Metal and Accelerate on an M5 Pro (16 GPU cores, 48 GiB unified memory).
- Downloaded and verified the 12.5 GiB I-Mini model. Short API validation passed.
- Full 36-candidate Metal search completed: 28 accepted. Best completed configuration was 98,304 context, `q8_0/turbo3`, `n-cpu-moe=0`, batch/ubatch `1024/1024`, 10 threads, and full Metal offload. The long-prompt request took 386 seconds.
- All 131,072-context candidates hit the old 480-second HTTP timeout; this is not evidence of an out-of-memory or Metal failure. Timeout was raised to 900 seconds and an exact-context retry was added, then the retry was stopped at user request.
- `n-cpu-moe=0` is only the all-Metal baseline. llama.cpp supports CPU expert offload on Metal; future tuning should compare `0/8/16/24/32/40`. Unified memory avoids a PCIe copy, but CPU execution and CPU/GPU synchronization can still reduce throughput, so offload should be selected by measurement rather than assumed beneficial.
- Current selected configuration remains the completed 98,304-context winner. Autotuning is stopped and port 8000 is free.

### CPU MoE sweep results

Two complete sweeps tested `n-cpu-moe=0/8/16/24/32/40` with fixed `q8_0/turbo3`, batch/ubatch `1024/1024`, 10 threads, deterministic prompts, and 128 output tokens.

| Context | CPU MoE | Prompt tok/s | Generation tok/s | RSS GiB |
|---:|---:|---:|---:|---:|
| 65,536 | **0** | **372.14** | **30.91** | 13.63 |
| 65,536 | 8 | 343.40 | 26.08 | 13.63 |
| 65,536 | 16 | 323.51 | 23.48 | 13.63 |
| 65,536 | 24 | 307.84 | 20.93 | 13.63 |
| 65,536 | 32 | 278.58 | 18.15 | 13.63 |
| 65,536 | 40 | 260.69 | 14.79 | 13.63 |
| 98,304 | **0** | **229.16** | **22.80** | 13.97 |
| 98,304 | 8 | 219.04 | 19.18 | 13.98 |
| 98,304 | 16 | 215.24 | 16.94 | 13.98 |
| 98,304 | 24 | 213.54 | 16.99 | 13.98 |
| 98,304 | 32 | 211.14 | 16.12 | 13.98 |
| 98,304 | 40 | 206.62 | 15.38 | 13.97 |

- Full Metal (`n-cpu-moe=0`) won at both contexts. Offloading eight layers reduced generation throughput by ~16%; additional offload generally worsened it. RSS differences were noise-level because CPU and GPU share unified memory.
- I-Mini remains the selected model per user scope. Current configuration: 98,304 context, `q8_0/turbo3`, `n-cpu-moe=0`, `1024/1024`, 10 threads.
- Raw runs: `results/cpu-moe-run1.json`, `results/cpu-moe-run2.json`. Aggregate: `results/cpu-moe-results.json`. Charts: `results/plots/cpu_moe_generation.png` and `results/plots/cpu_moe_prompt.png`.
- `README.md` now contains the measured M5 Pro results and charts.

## General-purpose workbench refactor — 2026-08-08

- Refactored the KAT-specific runner into a clean, model-agnostic GGUF workbench while preserving the llama.cpp TurboQuant runtime and KV-cache tuning.
- Added reusable model profiles, per-model local tuning state, namespaced JSON results, a comparable benchmark command, Pi integration guidance, and contributor instructions in `AGENTS.md`.
- Retained KAT-Coder as the default profile and added Qwen3.6 35B-A3B and Qwen3.5 27B profiles. Shell syntax, dry-run autotuning, Pi connectivity, and JSON validation passed.

## KAT-Coder MTP benchmark — 2026-08-08

- Downloaded and verified `Kwaipilot_KAT-Coder-V2.5-Dev-MTP-APEX-I-Mini.gguf` (14,366,220,928 bytes / 13.4 GiB) from `gbuzhf/KAT-Coder-V2.5-Dev-MTP-GGUF`.
- Added the `kat-coder-mtp` profile and optional speculative-decoding flags to `scripts/run.sh`.
- Fixed `scripts/benchmark.sh` to retain reasoning-only responses and report latency, draft tokens, accepted draft tokens, and acceptance rate.
- Held `q8_0/turbo3`, `n-cpu-moe=0`, batch/ubatch `1024/1024`, 10 threads, seed 42, a 38-token coding prompt, and 512 generated tokens constant. Each case ran twice.
- Practical replacement A/B (`kat-coder` without speculation vs MTP with `draft-mtp,ngram-mod`):

  | Context | Original tok/s | MTP tok/s | Gain | Acceptance |
  |---:|---:|---:|---:|---:|
  | 65,536 | 55.00 | 63.37 | +15.2% | 96.7% |
  | 98,304 | 55.48 | 63.62 | +14.7% | 96.7% |
  | 131,072 | 55.41 | 64.24 | +15.9% | 96.7% |

- MTP-file matrix winners: `draft-mtp,ngram-mod` at 65,536 (63.37 tok/s), `draft-mtp` at 98,304 (69.45 tok/s), and `draft-mtp` at 131,072 (64.54 tok/s, effectively tied with the other modes given two-run noise).
- `ngram-mod` alone did not help this novel coding prompt. Combined MTP + ngram beat ngram-only by 13.5% at 65k, lost 1.2% at 98k, and gained 1.3% at 131k.
- All configured contexts loaded and completed. This benchmark used a short prompt and therefore does not prove near-window 131k prompt throughput. Detailed runs and aggregates are in `results/mtp-benchmark-results.json`.
- Port 8000 is free. The existing `kat-coder` local 98,304-token winner remains unchanged.

### Real video-preextractor prompt

- Compared the original model and `kat-coder-mtp` at 65,536 configured context using a 297-token production-style PyTorchVideo pre-extractor request and an 8,192-token output allowance.
- Original: 8,192 completion tokens, 58.21 tok/s, 141.0 seconds, and `finish_reason=length`. Its main implementation block parsed, but the response was truncated during tests.
- MTP + ngram: 2,391 completion tokens, 65.61 tok/s, 36.8 seconds, 95.3% draft acceptance, and `finish_reason=stop`. It was 12.7% faster, but stopped mid-function with malformed tool markup and omitted the CLI/tests.
- Neither response was runnable. Both used invalid documented PyTorchVideo calls: `EncodedVideo(path)` rather than `EncodedVideo.from_path(path)`, plus unsupported `get_clip` keyword arguments. The original also assumed unavailable metadata attributes.
- Saved the complete prompt, responses, timings, and assessment in `results/mtp-video-preextractor-results.json`.
- Skipped the tentative 98,304-context repeat: the request used only 297 prompt tokens, so a larger configured KV window would not fix API hallucinations or truncation. Repeat only if context-dependent throughput is specifically needed.

## Sequential Pi coding subagents — 2026-08-08

- Added `kat-coder` and `kat-coder-mtp` to `~/.pi/agent/models.json` and the checked-in `config/pi-models.example.json`. Both use the localhost OpenAI-compatible provider; only one server was loaded at a time.
- Ran independent `pi --mode json` coding agents at 65,536 context in isolated workspaces with the same video-preextractor prompt and read/write/edit/bash tools.
- Original agent: 25.95 minutes, 24 assistant turns, 49 tool calls, 14 tool errors, and 15 self-authored tests passing.
- MTP agent: 5.06 minutes, 8 assistant turns, 17 tool calls, 2 tool errors, and 22 self-authored tests passing. The full agent loop was 5.13x faster than the original run.
- Independent review rejected both implementations despite their green tests. The tests mocked each model's invented video API rather than the documented contract.
- Original used nonexistent `EncodedVideo.from_file`, `get_video_info`, and `decode_video` methods, used an invalid interpolation mode for 5D tensors, and ignored `--workers`.
- MTP used `EncodedVideo(path)` and unsupported frame-based `get_clip` arguments, emitted `T x C x H x W` while claiming `C x T x H x W`, passed an integer descriptor to `torch.save`, ignored `--workers`, and contained vacuous resume tests.
- The documented interface is `EncodedVideo.from_path(path)` followed by `get_clip(start_sec, end_sec)`. Passing mocked tests therefore did not establish real PyTorchVideo compatibility.
- Preserved each workspace, final response, run metadata, and the independent review under `results/pi-subagents/`; aggregate: `results/pi-subagents/summary.json`. Port 8000 is free.
