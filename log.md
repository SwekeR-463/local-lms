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

- Ported the runner to macOS while retaining Linux/CUDA support. Built TurboQuant commit `0967f499714dd6018494b480b710b849ca45b156` with Metal and Accelerate on a Mac (16 GPU cores, 48 GiB unified memory).
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
- `README.md` now contains the measured Mac results and charts.

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

## Failed Qwen3.6 27B Q3_K_M MTP experiment — 2026-08-08

- Tested separate baseline and MTP Q3_K_M GGUFs from `unsloth/Qwen3.6-27B-GGUF` and `unsloth/Qwen3.6-27B-MTP-GGUF` on the Mac.
- Held 65,536 context, full Metal offload, Flash Attention, one server slot, `f16/f16` KV cache, a 38-token prompt, and 512 generated tokens constant. MTP used `--spec-type draft-mtp --spec-draft-n-max 2`.
- Baseline generation runs measured 15.24 and 13.22 tok/s, averaging **14.23 tok/s**.
- MTP generation runs measured 8.63 and 8.25 tok/s, averaging **8.44 tok/s**, despite **95.2% mean draft acceptance**. MTP was therefore **40.7% slower** than baseline with this TurboQuant llama.cpp build on Metal.
- The baseline Pi coding agent completed the same isolated video-preextractor task in 26.68 minutes with 25 tool calls and 21 self-authored tests passing. It correctly used `EncodedVideo.from_path(path)` and `get_clip(start_sec, end_sec)`, but explicitly left `--workers` unused; the mocked suite was not treated as real integration proof.
- The MTP Pi coding-agent run was stopped before completion at user request to avoid further sustained load on a work laptop. No coding-quality comparison was claimed.
- The experiment branch, Qwen profiles, generated Qwen results, and Pi entries were discarded. All downloaded GGUF files, including the prior KAT-Coder files, were deleted; `models/` is empty and port 8000 is free.

- `2026-08-10 22:01:09 IST` — muse-glimmer: model resolver selected Muse-Glimmer-30B-UD-Q2_K_XL.gguf (12444212256 bytes)

- `2026-08-10 22:01:51 IST` — preflight completed; report saved at results/preflight-20260810-220151.txt

- `2026-08-10 22:01:52 IST` — muse-glimmer: server started with PID 69555; log results/server-20260810-220151.log

- `2026-08-10 22:04:31 IST` — stopped project server PID 69555

- `2026-08-10 22:04:34 IST` — preflight completed; report saved at results/preflight-20260810-220433.txt

- `2026-08-10 22:04:35 IST` — muse-glimmer: server started with PID 70325; log results/server-20260810-220434.log

- `2026-08-10 22:07:17 IST` — stopped project server PID 70325

- `2026-08-10 22:07:20 IST` — preflight completed; report saved at results/preflight-20260810-220719.txt

- `2026-08-10 22:07:22 IST` — muse-glimmer-dflash: server started with PID 71130; log results/server-20260810-220721.log

- `2026-08-10 22:25:51 IST` — stopped project server PID 71130

## Muse Glimmer 30B DFlash experiment — 2026-08-10

- Built latest upstream llama.cpp commit `dd1ea524333b1e697489067d7a4c39c60d32beee` with Metal because the pinned TurboQuant fork does not yet contain the `muse-glimmer` architecture. TurboQuant remains untouched.
- Downloaded and verified `Muse-Glimmer-30B-UD-Q2_K_XL.gguf` (12,444,212,256 bytes / 11.59 GiB) and Meta's `dflash-kquant.gguf` (1,631,205,312 bytes / 1.52 GiB).
- Added baseline and DFlash profiles plus optional external draft-model and model-specific sampling support in `scripts/run.sh`. Meta's recommended `temperature=1.0`, `top_p=0.95`, and `top_k=64` are stored in the profiles.
- Both modes loaded successfully at 65,536 configured context with full Metal, Flash Attention, one slot, and `f16/f16` KV cache.
- After one warm-up, baseline measured 11.42 and 11.41 tok/s (**11.42 tok/s mean**). DFlash measured 12.86 and 13.28 tok/s (**13.07 tok/s mean**), a **14.5% gain** with **87.5% draft acceptance** and 4.54 accepted-token mean draft length.
- All six benchmark responses had the same SHA-256, confirming deterministic output parity. DFlash increased RSS from 12.69 GiB to 14.52 GiB.
- The isolated 65k Pi coding agent completed in 11.16 minutes with 14 assistant turns, 13 tool calls, and no tool errors. It used the documented `EncodedVideo.from_path(path)` and `get_clip(start_sec, end_sec)` API and did not falsely report passing tests.
- Independent review did not accept the generated implementation as production-ready: `--workers` was missing, fractional starts and duplicate source basenames can collide, overwrite leaves stale manifest entries, and resume/corruption tests do not exercise those behaviors. Pytest collection also could not run because torch is not installed locally.
- Machine-readable benchmark: `results/muse-glimmer-dflash-results.json`; chart: `results/plots/muse_glimmer_dflash.png`. Agent workspace and review: `results/pi-subagents/muse-glimmer-dflash/`. Server stopped; port 8000 is free.

- `2026-08-10 22:31:11 IST` — muse-glimmer-dflash: model resolver selected Muse-Glimmer-30B-UD-Q2_K_XL.gguf (12444212256 bytes)

- `2026-08-10 22:31:12 IST` — muse-glimmer-dflash: draft model ready: dflash-kquant.gguf (1631205312 bytes)

- `2026-08-11 00:01:14 IST` — qwen36-27b-q2: model resolver selected Qwen3.6-27B-UD-Q2_K_XL.gguf (11849779424 bytes)

- `2026-08-11 08:19:59 IST` — qwen36-27b-q2-mtp: model resolver selected Qwen3.6-27B-UD-Q2_K_XL.gguf (12040512640 bytes)

- `2026-08-11 10:21:00 IST` — autotune selected 65536-token configuration; summary saved at results/qwen36-27b-q2-autotune-results.json

- `2026-08-11 10:21:19 IST` — preflight completed; report saved at results/preflight-20260811-102118.txt

- `2026-08-11 10:21:20 IST` — qwen36-27b-q2: server started with PID 83518; log results/server-20260811-102119.log

- `2026-08-11 10:24:11 IST` — stopped project server PID 83518

- `2026-08-11 10:24:11 IST` — preflight completed; report saved at results/preflight-20260811-102411.txt

- `2026-08-11 10:24:12 IST` — qwen36-27b-q2-mtp: server started with PID 84741; log results/server-20260811-102411.log

- `2026-08-11 10:27:15 IST` — stopped project server PID 84741

## Qwen3.6 27B UD-Q2_K_XL MTP experiment — 2026-08-11

- Downloaded and verified the separate Unsloth baseline and MTP GGUFs: 11,849,779,424 bytes with SHA-256 `3db422cf36c7efacb027396a11df287c0fc469829bd7daf1867a3505a9e44af6`, and 12,040,512,640 bytes with SHA-256 `16fb3f81a522faaecfed0402890c3471e970e732c0e3e1914f1c0d9d9253be00`.
- Used upstream llama.cpp commit `dd1ea524333b1e697489067d7a4c39c60d32beee`, full Metal offload, Flash Attention, one slot, and 65,536 configured context. MTP used the publisher's `--spec-type draft-mtp --spec-draft-n-max 2` settings.
- Extended `scripts/autotune.sh` so profiles can provide candidate KV/batch matrices and speculative profiles are actually tuned with their draft flags.
- Ran a nine-candidate baseline autotune over `f16/f16`, `q8_0/f16`, and `q8_0/q8_0` KV cache with `1024/2048` batch and ubatch combinations. The `f16/f16`, `1024/1024` winner processed the 60,504-token prompt at 98.53 tok/s and generated at 9.49 tok/s with 15.90 GiB RSS.
- The best `q8_0/q8_0` candidate saved about 1.9 GiB RSS but was 9.5% slower for prompt processing and 17.2% slower for generation. All three mixed `q8_0/f16` requests exceeded the 900-second timeout.
- After one warm-up, baseline measured 10.72 and 10.14 tok/s (**10.43 tok/s mean**). MTP measured 11.02 and 10.84 tok/s (**10.93 tok/s mean**), a modest **4.8% gain** with **93.9% draft acceptance**.
- All six benchmark responses had SHA-256 `c94d52619f9a3e43f7ac0a70a2ba2b0b2caf0d1aa0550687df2fc5471088e706`. A coding-agent comparison was skipped because the 4.8% gain does not justify another sustained run yet.
- Machine-readable data: `results/qwen36-27b-q2-autotune-results.json` and `results/qwen36-27b-q2-mtp-results.json`; chart: `results/plots/qwen36_q2_mtp.png`. Server stopped; port 8000 is free.

- `2026-08-11 10:55:33 IST` — qwen36-27b-q2-dflash: model resolver selected Qwen3.6-27B-UD-Q2_K_XL.gguf (11849779424 bytes)

- `2026-08-11 10:58:04 IST` — qwen36-27b-q2-dflash: draft model ready: Qwen3.6-27B-DFlash-Q8_0.gguf (1849481440 bytes)

- `2026-08-11 10:58:54 IST` — preflight completed; report saved at results/preflight-20260811-105854.txt

- `2026-08-11 10:58:56 IST` — qwen36-27b-q2-dflash: server started with PID 99460; log results/server-20260811-105855.log

- `2026-08-11 11:05:09 IST` — stopped project server PID 99460

- `2026-08-11 11:05:10 IST` — preflight completed; report saved at results/preflight-20260811-110509.txt

- `2026-08-11 11:05:11 IST` — qwen36-27b-q2-dflash: server started with PID 4649; log results/server-20260811-110510.log

- `2026-08-11 11:09:29 IST` — stopped project server PID 4649

- `2026-08-11 11:09:30 IST` — preflight completed; report saved at results/preflight-20260811-110929.txt

- `2026-08-11 11:09:32 IST` — qwen36-27b-q2: server started with PID 7052; log results/server-20260811-110931.log

- `2026-08-11 11:14:01 IST` — stopped project server PID 7052

## Qwen3.6 27B DFlash experiment — 2026-08-11

- Downloaded Alittlehammmer's recommended `Qwen3.6-27B-DFlash-Q8_0.gguf`: 1,849,481,440 bytes with SHA-256 `23b6c8ebcc51b3b4107709342fd2960167e88397af36e394923b8d5895ddf7ea`.
- Used the existing plain `UD-Q2_K_XL` target and tuned `f16/f16`, `1024/1024` setup at 65,536 context. DFlash used the publisher's `--spec-type draft-dflash --spec-draft-n-max 6` settings.
- The drafter initialized successfully. llama.cpp emitted its documented normal warning while probing draft-model memory, then loaded the DFlash context with block size 16 and five extracted tokens.
- Initial exploratory DFlash runs varied from 18.36 down to 10.53 tok/s despite identical 91.9% acceptance and byte-identical output. These peaks are retained in the JSON but not presented as sustained throughput.
- Repeated the comparison with a server restart per mode and 30 seconds between requests. After one warm-up, baseline measured 9.38 and 9.70 tok/s (**9.54 tok/s mean**); DFlash measured 10.71 and 10.70 tok/s (**10.71 tok/s mean**).
- The controlled DFlash gain was **12.2%** with **91.9% acceptance** and 4.17 mean accepted draft length. RSS increased from 15.95 GiB to 18.96 GiB.
- Machine-readable data: `results/qwen36-27b-q2-dflash-results.json`; chart: `results/plots/qwen36_q2_dflash.png`. No coding-agent run was started. Server stopped; port 8000 is free.

- `2026-08-11 12:44:15 IST` — preflight completed; report saved at results/preflight-20260811-124414.txt

- `2026-08-11 12:44:16 IST` — muse-glimmer-dflash: server started with PID 42281; log results/server-20260811-124415.log

- `2026-08-11 13:39:18 IST` — stopped project server PID 42281

- `2026-08-11 14:27:26 IST` — kat-coder: model resolver selected KAT-Coder-V2.5-Dev-APEX-I-Mini.gguf (13467211136 bytes)

- `2026-08-11 14:27:36 IST` — preflight completed; report saved at results/preflight-20260811-142735.txt

- `2026-08-11 14:36:23 IST` — preflight completed; report saved at results/preflight-20260811-143622.txt

- `2026-08-11 14:36:25 IST` — kat-coder: server started with PID 83203; log results/server-20260811-143624.log

- `2026-08-11 16:49:50 IST` — stopped project server PID 83203

- `2026-08-11 17:39:26 IST` — ornith-35b-i-mini: model resolver selected Ornith-1.0-35B-APEX-I-Mini.gguf (13467210752 bytes)

- `2026-08-11 17:39:54 IST` — preflight completed; report saved at results/preflight-20260811-173954.txt

- `2026-08-11 17:39:55 IST` — ornith-35b-i-mini: server started with PID 54112; log results/server-20260811-173954.log

- `2026-08-11 19:42:02 IST` — stopped project server PID 54112

- `2026-08-12 00:01:46 IST` — ling3-tiny-q8: model resolver selected Ling-3.0-tiny-UD-Q8_K_XL.gguf (11188839264 bytes)

- `2026-08-12 00:02:31 IST` — preflight completed; report saved at results/preflight-20260812-000231.txt

- `2026-08-12 00:02:35 IST` — preflight completed; report saved at results/preflight-20260812-000235.txt

- `2026-08-12 00:02:36 IST` — ling3-tiny-q8: server started with PID 3492; log results/server-20260812-000235.log

- `2026-08-12 01:14:04 IST` — stopped project server PID 3492

- `2026-08-12 08:59:14 IST` — preflight completed; report saved at results/preflight-20260812-085912.txt

- `2026-08-12 08:59:16 IST` — ling3-tiny-q8: server started with PID 64879; log results/server-20260812-085915.log

- `2026-08-12 10:17:45 IST` — stopped project server PID 64879

## Ling 3.0 Tiny Q8 experiment — 2026-08-12

- Stock llama.cpp and TurboQuant cannot load Ling's `bailingmoe3` Q-LoRA architecture. Built `aetherbird/llama.cpp` branch `bailingmoe3-support` separately at commit `d8d8625`, preserving both existing runtimes.
- Downloaded and verified `Ling-3.0-tiny-UD-Q8_K_XL.gguf`: 11,188,839,264 bytes with SHA-256 `4fefbf341330722c97d10f7ce90a9b494663269911ba0d1fc3dbe50f43693e25`.
- Metal loaded the full **131,072-token context** with `f16/f16` KV, batch/ubatch 1024, all layers offloaded, and one request slot. Idle RSS was about 11.4 GiB.
- A complete deterministic smoke response generated at **87.99 tok/s** and returned `LING_READY`; a 245-token tool-call prompt processed at **852.21 tok/s**, generated at **88.90 tok/s**, and emitted the correct function and arguments. Pi integration also returned `PI_LING_READY`.
- The model spends heavily on hidden reasoning. With Pi's original 16,384-token output limit, all three coding tasks exhausted the limit: one wrote no files, one wrote seven incomplete files, and none produced a final answer. Raising the limit to 65,536 still led to long, unproductive loops, so further agent evaluation was stopped.
- Lesson: high raw generation speed and valid tool-call syntax do not imply efficient autonomous coding. Small reasoning models need an output budget above 16k, but a larger budget can amplify looping rather than improve completion.
- The local GGUF was deleted after testing. The reusable model profile remains; generated agent artifacts are intentionally ignored under `gen-outputs/`.

- `2026-08-12 16:16:02 IST` — btl-4-compact: model resolver selected BTL-4-IQ2_XXS.gguf (9967966240 bytes)

## BTL-4 Compact IQ2_XXS experiment — 2026-08-12

- Downloaded and verified `BTL-4-IQ2_XXS.gguf`: 9,967,966,240 bytes with SHA-256 `6b7c298cf909fc04428ecf360a29dcc578188b1c90aa6ed435159f5a0d351496`.
- Loaded at 32,768 context on Metal with q8_0/q8_0 KV, `--jinja`, and `--reasoning-format deepseek`. Idle RSS was about 9.82 GiB; `/v1/models` reported the native 262,144-token training context and 34.66B parameters.
- Chat smoke test returned exactly `BTL_READY` at 71.41 generation tok/s. Tool-call smoke test correctly called `lookup_status` with `{"job_id":"abc123"}` at 71.28 generation tok/s and separated reasoning from content.
- The in-flight processor Pi task ran for 900.78 seconds with five tool calls and two tool errors. It hit the output limit four times, compacted context twice, wrote only two duplicate/incomplete Python modules, produced no tests or documentation, and gave no final response.
- Independent review rejected the implementation: missing runtime imports, an ffmpeg output path without a usable extension/format, ignored `--dry-run`, incorrect S3 missing-object handling, and per-key validation errors that abort the whole run.
- Lesson: BTL-4 is fast and its advertised chat/tool template works, but this 32k autonomous coding run repeated the same failure mode seen in other reasoning models: token-heavy planning and rewriting displaced completion and verification.

- `2026-08-12 16:16:09 IST` — preflight completed; report saved at results/preflight-20260812-161608.txt

- `2026-08-12 16:16:10 IST` — btl-4-compact: server started with PID 43836; log results/server-20260812-161609.log

- `2026-08-12 16:33:09 IST` — stopped project server PID 43836

- `2026-08-12 17:13:56 IST` — preflight completed; report saved at results/preflight-20260812-171356.txt

- `2026-08-12 17:13:58 IST` — btl-4-compact: server started with PID 64338; log results/server-20260812-171357.log

- `2026-08-12 17:23:13 IST` — stopped project server PID 64338

### BTL-4 64k direct-prompt coding run

- Reloaded BTL-4 at 65,536 context with q8_0/q8_0 KV. Idle RSS was about 10.16 GiB and peak observed RSS after the agent run was about 10.54 GiB.
- Used only this two-line prompt: create an in-flight S3 processor that trims clips based on human-face presence and uploads them to another bucket; implement and test without AWS or real video processing.
- Pi completed in 476.21 seconds with 52 tool calls, 12 tool errors, no context compaction, and a final response. Unlike the detailed 32k attempt, it produced a complete-looking TypeScript project and stayed within context.
- Independent validation rejected it. `npm test` fails with `ERR_MODULE_NOT_FOUND`; both real face detection and real trimming only throw; the real S3 client ignores configured buckets and mis-parses keys; the no-face test always reports a face and has no assertions.
- Lesson: 64k context plus a direct prompt improved task completion and artifact coverage, but not correctness. The model recovered repeatedly from build errors and then claimed success despite leaving the documented test command broken and core production paths unimplemented.

### BTL-4 supervised repair — five rounds

- Ran five short correction rounds against the same Pi session: test runner/assertions, a repeated module-resolution correction, an exact NodeNext compile/run instruction, S3 bucket/key handling, and finally real face-detection/ffmpeg command paths.
- Rounds 1 and 2 both exhausted their output allowance and left the same `ERR_MODULE_NOT_FOUND` failure. Round 3 followed the explicit compile-then-run direction and made `npm test` pass with face/no-face assertions. Round 4 normalized configured S3 buckets and used them in AWS commands.
- Across the rounds the model made 55 tool calls with 14 tool errors. Final independent checks passed `npm test` and `npx tsc --noEmit`.
- The final result remains rejected. Despite an explicit fifth-round instruction, `RealFaceDetector` and `RealVideoTrimmer` still unconditionally throw, `RealCommandExecutor` uses shell-interpolated `exec`, the destination bucket is duplicated into the uploaded object key, and the new tests do not inspect actual AWS command inputs or production command construction.
- Lesson: focused back-and-forth can repair mechanical build/test failures, but five supervised turns did not overcome BTL-4's tendency to substitute mocks for required production behavior and declare success after partial compliance.

- `2026-08-12 17:28:15 IST` — preflight completed; report saved at results/preflight-20260812-172814.txt

- `2026-08-12 17:28:16 IST` — btl-4-compact: server started with PID 70117; log results/server-20260812-172815.log

- `2026-08-12 18:05:08 IST` — stopped project server PID 70117

- `2026-08-12` — added the flattened `results/plots/btl4_compact_summary.png` two-panel chart covering measured runtime and autonomous versus supervised coding assessment; source measurements are preserved in `results/btl-4-compact-results.json`.

### BTL-4 CSV classification metrics task

- Ran a fresh 65,536-context one-shot task asking for a Python CSV metrics CLI and PNG graphs for accuracy, precision, recall, F1, and false positives, with binary/multiclass support and synthetic tests.
- Stopped after the 30-minute harness timeout. The model made 242 tool calls with seven tool errors, compacted context three times, and never produced a final response.
- Independent validation rejected the output. `tests.py` has a syntax error; binary string labels are passed to sklearn without a `pos_label`; false positives are always `None`; multiclass plot values are assigned to the wrong bars; and no PNG was successfully generated.
- Lesson: even a smaller, familiar data-analysis task triggered prolonged edit loops at 64k. BTL-4's strong raw speed and tool syntax still do not translate into reliable autonomous completion.

- `2026-08-12 18:42:24 IST` — preflight completed; report saved at results/preflight-20260812-184223.txt

- `2026-08-12 18:42:25 IST` — btl-4-compact: server started with PID 99512; log results/server-20260812-184224.log

- `2026-08-12 19:13:59 IST` — stopped project server PID 99512

- `2026-08-14 22:32:07 IST` — qwen38-27b-iq3: model resolver selected Qwen3.8-27B-UD-IQ3_XXS.gguf (11913559104 bytes)

- `2026-08-14 22:32:40 IST` — preflight completed; report saved at results/preflight-20260814-223240.txt

- `2026-08-14 22:32:41 IST` — qwen38-27b-iq3: server started with PID 27254; log results/server-20260814-223240.log

- `2026-08-14 22:34:13 IST` — stopped project server PID 27254

- `2026-08-14 22:34:14 IST` — preflight completed; report saved at results/preflight-20260814-223413.txt

- `2026-08-14 22:34:15 IST` — qwen38-27b-iq3: server started with PID 28467; log results/server-20260814-223414.log

## Qwen3.8 27B UD-IQ3_XXS initial test — 2026-08-14

- Downloaded and verified `Qwen3.8-27B-UD-IQ3_XXS.gguf` from `unsloth/Qwen3.8-27B-GGUF`: 11,913,559,104 bytes, SHA-256 `0a6129dcbbbe72f423dc67e0e3bbfbbdf3e923981a3637687ebb96a46c59d6be`.
- Loaded at 65,536 context with full Metal offload and `f16/f16` KV cache. Loaded RSS was about 15.2 GiB; a 512-token benchmark measured 90.46 prompt tok/s and 15.37 generation tok/s at about 15.97 GiB RSS.
- Thinking mode uses the recommended `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`, zero presence penalty, xhigh reasoning, and preserved thinking. Added profile-driven sampler and reasoning flags to `scripts/run.sh`.
- Exact chat and Pi smoke tests passed. A required `lookup_status` tool call succeeded with the exact `{"job_id":"abc123"}` argument after switching from deprecated template kwargs to llama.cpp's native reasoning flags.
- Xhigh reasoning exhausted 2,048 output tokens without reaching a final answer on a small interval-merging task. Per-request low reasoning completed the same task correctly in 1,309 tokens; reasoning effort materially affects usability and should be evaluated before coding-agent runs.

- `2026-08-14 23:14:09 IST` — stopped project server PID 28467

- `2026-08-14 23:14:10 IST` — preflight completed; report saved at results/preflight-20260814-231409.txt

- `2026-08-14 23:14:11 IST` — qwen38-27b-iq3: server started with PID 47668; log results/server-20260814-231410.log
- Started the full classic in-flight S3 processor prompt at xhigh reasoning, but stopped it before completion after it remained in an extended generation phase without making a tool call. Preserved the interrupted transcript under `gen-outputs/qwen38-27b-iq3/s3-inflight-processor/`.
- Retried with a focused single-object S3 processor prompt at low reasoning. It completed in 1,120.947 seconds with six tool calls, no tool errors, and no compaction; all three mocked unittests passed independently.
- Independent verdict: accepted with limitations. The implementation correctly used fixed temporary filenames, argument-array subprocess execution, injected S3/runner boundaries, upload-after-success ordering, and automatic cleanup. It did not validate configured prefixes, did not explicitly verify ffmpeg created an output file, and did not test cleanup after download/upload exceptions.

- `2026-08-14 23:35:11 IST` — stopped project server PID 47668

- `2026-08-14 23:38:02 IST` — preflight completed; report saved at results/preflight-20260814-233802.txt

- `2026-08-14 23:38:04 IST` — qwen38-27b-iq3: server started with PID 63098; log results/server-20260814-233803.log
- Ran a medium-reasoning binary CSV evaluation task with a fixed 20-row fixture (`TP=6`, `TN=8`, `FP=2`, `FN=4`). It completed in 1,411.872 seconds with 21 tool calls, no tool errors, and no compaction.
- The generated CLI correctly reported accuracy 0.70, precision 0.75, recall 0.60, F1 0.6667, and false-positive rate 0.20. It generated four valid, labeled PNG charts, and all 11 generated tests passed.
- Independent validation rejected the result: `compute_metrics(tp=0, tn=5, fp=2, fn=3)` raises `ZeroDivisionError` because precision and recall are both zero. The generated tests omitted this case despite claiming complete zero-denominator coverage. FPR itself was correctly defined as `FP / (FP + TN)` and reported as undefined when no actual negatives exist.

- `2026-08-15 00:03:19 IST` — stopped project server PID 63098

- `2026-08-15 20:03:29 IST` — qwen38-27b-ud-q3: model resolver selected Qwen3.8-27B-UD-Q3_K_XL.gguf (13441059904 bytes)

- `2026-08-15 20:28:56 IST` — qwen38-27b-ad-iq3: model resolver selected Qwen3.8-27B-AD-IQ3_S.gguf (13838267872 bytes)

- `2026-08-15 20:29:12 IST` — preflight completed; report saved at results/preflight-20260815-202911.txt

- `2026-08-15 20:29:13 IST` — qwen38-27b-ud-q3: server started with PID 91057; log results/server-20260815-202912.log

- `2026-08-15 20:30:05 IST` — stopped project server PID 91057

- `2026-08-15 20:30:05 IST` — preflight completed; report saved at results/preflight-20260815-203005.txt

- `2026-08-15 20:30:06 IST` — qwen38-27b-ad-iq3: server started with PID 91485; log results/server-20260815-203005.log

- `2026-08-15 20:32:35 IST` — stopped project server PID 91485

- `2026-08-15 20:32:35 IST` — preflight completed; report saved at results/preflight-20260815-203235.txt

- `2026-08-15 20:32:37 IST` — qwen38-27b-ud-q3: server started with PID 92356; log results/server-20260815-203236.log

- `2026-08-15 20:36:00 IST` — stopped project server PID 92356

- `2026-08-15 20:50:00 IST` — preflight completed; report saved at results/preflight-20260815-205000.txt

- `2026-08-15 20:50:02 IST` — qwen38-27b-ud-q3: server started with PID 3134; log results/server-20260815-205001.log

- `2026-08-15 20:59:20 IST` — stopped project server PID 3134

- `2026-08-15 20:59:20 IST` — preflight completed; report saved at results/preflight-20260815-205920.txt

- `2026-08-15 20:59:21 IST` — qwen38-27b-ad-iq3: server started with PID 11949; log results/server-20260815-205920.log

- `2026-08-15 21:09:49 IST` — stopped project server PID 11949

## Qwen3.8 27B Q3 comparison — 2026-08-15

- Downloaded and checksum-verified Unsloth `UD-Q3_K_XL` (13,441,059,904 bytes; `00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2`) and AtomicChat `AD-IQ3_S` (13,838,267,872 bytes; `3e30f93acafc11705a8e4891a0b2aa3c138ffcaf2ca832b1c6a11ea4b5b7b620`).
- Initial cold 512-token runs were effectively tied at 15.57 and 15.22 generation tok/s. Repeated back-to-back runs thermally throttled and are not treated as a stable throughput comparison.
- AtomicChat's matched-corpus measurements favor AD-IQ3_S: mean KL divergence 0.03247 versus 0.03972 and same-top-token agreement 92.411% versus 91.869%.
- Added `benchmark-data/short-python/`, a six-task coding suite with held-out executable evaluators, and `scripts/benchmark-code.sh`. Both quants scored 4/6 under the fixed 1,024-token low-reasoning budget; both truncated before complete answers on binary metrics and dependency ordering.
- Added `benchmark-data/agent-51/manifest.json`, pinning the custom 20 SWE-bench Verified, 25 Terminal-Bench 2.1, and six MLE-bench task subset. It is explicitly labeled as a custom subset rather than an official full-suite score.
- Repeated the short suite at medium reasoning with a 2,048-token cap. Both quants improved to 5/6 by completing binary metrics correctly; both still exhausted the limit before returning complete dependency-order source. Results are in `results/qwen38-27b-short-python-medium-comparison.json`.
- Tested AD-IQ3_S with upstream `f16/f16`, `q8_0/q8_0`, and `q4_0/q4_0`; TurboQuant `q8_0/turbo3` and `q8_0/turbo4`; and upstream embedded MTP. The cold 512-token runs measured 15.92, 15.67, 15.52, 15.51, 9.40, and 12.63 tok/s respectively. MTP was slower despite 94.2% acceptance. No configuration reached 20 tok/s; upstream `f16/f16` remained fastest and measured 16.08 tok/s on UD-Q3_K_XL. Results are in `results/qwen38-27b-ad-speed-matrix.json` and `results/qwen38-27b-ud-f16-confirmation.json`.
- Ran one Terminal-Bench 2.1 `cancel-async-tasks` smoke task. The first UD verifier was invalid because `uvx` crashed under QEMU. After enabling Rosetta temporarily, the valid UD trial scored 0.0 because the submitted API was synchronous instead of async. The AD trial was interrupted at user request. This is not a paired benchmark; see `results/qwen38-27b-terminal-bench-smoke.json`. Raw Harbor jobs remain locally ignored.

- `2026-08-15 21:13:24 IST` — preflight completed; report saved at results/preflight-20260815-211324.txt

- `2026-08-15 21:13:26 IST` — qwen38-27b-ud-q3: server started with PID 21831; log results/server-20260815-211325.log

- `2026-08-15 21:26:20 IST` — stopped project server PID 21831

- `2026-08-15 21:26:20 IST` — preflight completed; report saved at results/preflight-20260815-212620.txt

- `2026-08-15 21:26:21 IST` — qwen38-27b-ad-iq3: server started with PID 31342; log results/server-20260815-212620.log

- `2026-08-15 21:38:29 IST` — stopped project server PID 31342

- `2026-08-15 22:26:40 IST` — preflight completed; report saved at results/preflight-20260815-222640.txt

- `2026-08-15 22:26:41 IST` — qwen38-27b-ud-q3: server started with PID 70333; log results/server-20260815-222640.log

- `2026-08-15 23:23:30 IST` — stopped project server PID 70333

- `2026-08-15 23:23:30 IST` — preflight completed; report saved at results/preflight-20260815-232330.txt

- `2026-08-15 23:23:32 IST` — qwen38-27b-ad-iq3: server started with PID 3570; log results/server-20260815-232331.log

- `2026-08-17 15:39:07 IST` — preflight completed; report saved at results/preflight-20260817-153907.txt

- `2026-08-17 15:39:08 IST` — qwen38-27b-ad-iq3: server started with PID 14294; log results/server-20260817-153907.log

- `2026-08-17 15:39:45 IST` — stopped project server PID 14294

- `2026-08-17 15:40:16 IST` — preflight completed; report saved at results/preflight-20260817-154015.txt

- `2026-08-17 15:40:17 IST` — qwen38-27b-ad-iq3: server started with PID 14632; log results/server-20260817-154016.log

- `2026-08-17 15:40:52 IST` — stopped project server PID 14632

- `2026-08-17 15:41:23 IST` — preflight completed; report saved at results/preflight-20260817-154122.txt

- `2026-08-17 15:41:24 IST` — qwen38-27b-ad-iq3: server started with PID 14961; log results/server-20260817-154123.log

- `2026-08-17 15:42:00 IST` — stopped project server PID 14961

- `2026-08-17 15:42:30 IST` — preflight completed; report saved at results/preflight-20260817-154230.txt

- `2026-08-17 15:42:31 IST` — qwen38-27b-ad-iq3: server started with PID 21822; log results/server-20260817-154230.log

- `2026-08-17 15:43:07 IST` — stopped project server PID 21822

- `2026-08-17 15:43:38 IST` — preflight completed; report saved at results/preflight-20260817-154337.txt

- `2026-08-17 15:43:39 IST` — qwen38-27b-ad-iq3: server started with PID 22158; log results/server-20260817-154338.log

- `2026-08-17 15:44:36 IST` — stopped project server PID 22158

- `2026-08-17 15:45:07 IST` — preflight completed; report saved at results/preflight-20260817-154506.txt

- `2026-08-17 15:45:08 IST` — qwen38-27b-ad-iq3: server started with PID 22531; log results/server-20260817-154507.log

- `2026-08-17 15:45:51 IST` — stopped project server PID 22531

- `2026-08-17 15:46:46 IST` — preflight completed; report saved at results/preflight-20260817-154646.txt

- `2026-08-17 15:46:48 IST` — qwen38-27b-ud-q3: server started with PID 22897; log results/server-20260817-154647.log

- `2026-08-17 15:47:23 IST` — stopped project server PID 22897

- `2026-08-19 16:45:15 IST` — qwen38-27b-ad-iq3-dflash2: model resolver selected Qwen3.8-27B-AD-IQ3_S.gguf (13838267872 bytes)

- `2026-08-19 16:48:09 IST` — qwen38-27b-ad-iq3-dflash2: draft model ready: Qwen3.8-27B-DFlash2-Q4_K_M.gguf (1143006752 bytes)

- `2026-08-19 16:48:52 IST` — preflight completed; report saved at results/preflight-20260819-164851.txt

- `2026-08-19 16:48:53 IST` — qwen38-27b-ad-iq3: server started with PID 86106; log results/server-20260819-164852.log

- `2026-08-19 16:49:30 IST` — stopped project server PID 86106

- `2026-08-19 16:50:00 IST` — preflight completed; report saved at results/preflight-20260819-165000.txt

- `2026-08-19 16:50:02 IST` — qwen38-27b-ad-iq3-dflash2: server started with PID 88992; log results/server-20260819-165000.log

- `2026-08-19 16:51:02 IST` — stopped project server PID 88992

- `2026-08-19 16:51:58 IST` — preflight completed; report saved at results/preflight-20260819-165158.txt

- `2026-08-19 16:52:00 IST` — qwen38-27b-ad-iq3-dflash2: server started with PID 89391; log results/server-20260819-165159.log

- `2026-08-19 16:53:15 IST` — stopped project server PID 89391

- `2026-08-19 16:54:17 IST` — preflight completed; report saved at results/preflight-20260819-165417.txt

- `2026-08-19 16:54:18 IST` — qwen38-27b-ud-q3: server started with PID 89767; log results/server-20260819-165417.log

- `2026-08-19 16:55:00 IST` — stopped project server PID 89767

- `2026-08-19 16:55:30 IST` — preflight completed; report saved at results/preflight-20260819-165530.txt

- `2026-08-19 16:55:31 IST` — qwen38-27b-ud-q3-dflash2: server started with PID 90082; log results/server-20260819-165530.log

- `2026-08-19 16:56:43 IST` — stopped project server PID 90082

- `2026-08-19 16:57:45 IST` — preflight completed; report saved at results/preflight-20260819-165745.txt

- `2026-08-19 16:57:46 IST` — qwen38-27b-ad-iq3-dflash2: server started with PID 90452; log results/server-20260819-165745.log

- `2026-08-19 16:58:35 IST` — stopped project server PID 90452

- `2026-08-19 16:59:06 IST` — preflight completed; report saved at results/preflight-20260819-165905.txt

- `2026-08-19 16:59:07 IST` — qwen38-27b-ad-iq3-dflash2: server started with PID 90777; log results/server-20260819-165906.log

- `2026-08-19 16:59:58 IST` — stopped project server PID 90777

- `2026-08-19 17:01:51 IST` — preflight completed; report saved at results/preflight-20260819-170151.txt

- `2026-08-19 17:01:52 IST` — qwen38-27b-ad-iq3-dflash2: server started with PID 91176; log results/server-20260819-170151.log

- `2026-08-19 17:02:31 IST` — stopped project server PID 91176

- `2026-08-19 17:03:17 IST` — preflight completed; report saved at results/preflight-20260819-170316.txt

- `2026-08-19 17:03:18 IST` — qwen38-27b-ad-iq3: server started with PID 91484; log results/server-20260819-170317.log

- `2026-08-19 17:03:53 IST` — stopped project server PID 91484

- `2026-08-19 17:08:03 IST` — preflight completed; report saved at results/preflight-20260819-170803.txt

- `2026-08-19 17:08:05 IST` — qwen38-27b-ad-iq3: server started with PID 98708; log results/server-20260819-170804.log

## Qwen3.8 27B DFlash2 smoke tests — 2026-08-19

- Built llama.cpp PR #27342 at commit `5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4` with Metal in a separate ignored worktree, leaving the pinned upstream build unchanged.
- Downloaded and checksum-verified `incoai/Qwen3.8-27B-DFlash2-GGUF` revision `6cb5872e2cee6b4e780a8414922350be8e42d65c`, file `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1,143,006,752 bytes; SHA-256 `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594`).
- Used 65,536 context, `f16/f16` KV cache, xhigh reasoning, temperature 0, seed 42, one request at a time, and the deterministic 512-token benchmark prompt.
- AD-IQ3_S autoregressive decoding measured 15.19 and 15.91 tok/s, for a 15.55 tok/s median. DFlash2 `n-max=3` measured 11.69 and 14.33 tok/s, for a 13.01 tok/s median: 16.3% slower with 68.8% draft-token acceptance. Exploratory `n-max=5` and `n-max=7` runs measured 10.57 and 7.10 tok/s.
- UD-Q3_K_XL measured 13.77 tok/s without speculation and 7.53 tok/s with DFlash2 `n-max=7`, 45.3% slower with 37.4% draft-token acceptance.
- Every speculative response was byte-identical to its matching deterministic baseline. The Q4_K_M draft added roughly 1.7–2.5 GiB process RSS.
- DFlash2 did not accelerate either Q3 target in these short smoke tests. The faster IQ3/Q3 target decoding and lower acceptance did not offset draft and verification overhead. These are not sustained-performance estimates.
- Machine-readable results are in `results/qwen38-27b-dflash2-smoke-results.json`; individual raw runs remain locally under ignored `results/qwen38-dflash2-raw/`.

- `2026-08-19 17:40:23 IST` — stopped project server PID 98708

- `2026-08-19 17:40:23 IST` — preflight completed; report saved at results/preflight-20260819-174023.txt

- `2026-08-19 17:40:24 IST` — qwen38-27b-ad-iq3: server started with PID 15649; log results/server-20260819-174023.log

- `2026-08-19 19:43:21 IST` — stopped project server PID 15649

## Qwen3.8 27B UD-Q3_K_XL Dynamic V3.0 — 2026-08-19

- Unsloth replaced the original UD-Q3_K_XL with a Dynamic V3.0 quant at repository commit `313447f257f7ebde0b968e4778feef774546ed81`.
- Downloaded it alongside the original under `models/qwen38-27b-ud-q3-v3/`, rather than overwriting the prior benchmark model.
- Verified size: 13,146,393,504 bytes; SHA-256: `8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e`.
- The original file is 13,441,059,904 bytes with SHA-256 `00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2`.
- Both files contain 866 tensors, but their tensor-type allocations differ substantially, confirming that this is a new quantization recipe rather than a metadata-only update.
- The prior matched-corpus figure recorded in this workbench was 91.869% top-token agreement. A later independent WikiText-2 comparison measured the original file at 92.396% top-1 agreement and 0.031355 mean KLD; those figures use a different evaluation corpus.
- Unsloth's Dynamic V3.0 graph places the new UD-Q3_K_XL at approximately 93% top-1 agreement and approximately 0.022 mean KLD. The graph does not publish exact tabular values, so these are plot readings and must not be presented as exact or directly mixed with the earlier corpus results.
- Unsloth describes the August 19 Dynamic V3.0 update as roughly 10% more accuracy at the same size. Independent coding and runtime comparisons are still pending.

- `2026-08-19 22:44:12 IST` — preflight completed; report saved at results/preflight-20260819-224411.txt

- `2026-08-19 22:51:18 IST` — preflight completed; report saved at results/preflight-20260819-225117.txt

- `2026-08-19 22:51:19 IST` — qwen38-27b-ad-iq3: server started with PID 72965; log results/server-20260819-225118.log

- `2026-08-19 23:01:31 IST` — stopped project server PID 72965

- `2026-08-19 23:02:32 IST` — preflight completed; report saved at results/preflight-20260819-230231.txt

- `2026-08-19 23:02:33 IST` — qwen38-27b-ud-q3-v3: server started with PID 73958; log results/server-20260819-230232.log

- `2026-08-19 23:15:00 IST` — stopped project server PID 73958

- `2026-08-19 23:16:13 IST` — preflight completed; report saved at results/preflight-20260819-231612.txt

- `2026-08-19 23:16:14 IST` — qwen38-27b-ud-q3-v3: server started with PID 81631; log results/server-20260819-231613.log

- `2026-08-19 23:16:51 IST` — stopped project server PID 81631

- `2026-08-19 23:17:51 IST` — preflight completed; report saved at results/preflight-20260819-231751.txt

- `2026-08-19 23:17:52 IST` — qwen38-27b-ad-iq3: server started with PID 81971; log results/server-20260819-231751.log

- `2026-08-19 23:18:31 IST` — stopped project server PID 81971

## AD-IQ3_S versus UD-Q3_K_XL Dynamic V3.0 — 2026-08-19

- Compared both models with the same llama.cpp PR #27342 runtime, 65,536 context, `f16/f16` KV cache, temperature 0, seed 42, and one active request.
- Medium-reasoning short-Python result: 5/6 for each model. Both passed the same five tasks and exhausted the 2,048-token limit on `dependency-order`; the failure is therefore a constrained-completion tie, not a demonstrated algorithmic regression.
- AD-IQ3_S generation runs: 13.81 and 15.20 tok/s; median 14.50 tok/s.
- UD-Q3_K_XL V3 generation runs: 11.91 and 15.21 tok/s; median 13.56 tok/s.
- AD's two-run median was 7.0% faster, but the reverse-order cool runs were effectively tied at 15.20 tok/s, so the small sample shows substantial thermal/order sensitivity.
- UD V3 is 691,874,368 bytes (5.0%) smaller. It saves storage and some RSS, but did not improve this coding gate over AD-IQ3_S.
- Machine-readable comparison: `results/qwen38-27b-ad-vs-ud-v3-comparison.json`. Raw benchmark runs remain ignored.

- `2026-08-19 23:33:52 IST` — preflight completed; report saved at results/preflight-20260819-233351.txt

- `2026-08-19 23:33:53 IST` — qwen38-27b-ad-iq3: server started with PID 89945; log results/server-20260819-233352.log

- `2026-08-19 23:35:09 IST` — stopped project server PID 89945

- `2026-08-19 23:35:23 IST` — preflight completed; report saved at results/preflight-20260819-233523.txt

- `2026-08-19 23:35:24 IST` — qwen38-27b-ad-iq3: server started with PID 90470; log results/server-20260819-233523.log

- `2026-08-19 23:59:18 IST` — preflight completed; report saved at results/preflight-20260819-235918.txt

- `2026-08-19 23:59:19 IST` — ornith15-9b-q8: server started with PID 5403; log results/server-20260819-235918.log

- `2026-08-20 00:06:23 IST` — stopped project server PID 5403

## Ornith 1.5 9B Q8_0 smoke test — 2026-08-19

- Downloaded and checksum-verified `ornith-ai/Ornith-1.5-9B-GGUF` file `Ornith-1.5-9B-Q8_0.gguf` (9,527,501,248 bytes; SHA-256 `6874eeb25c71081dc8f0bbe88f3ebb786312447132745371cd980bce95d259b9`).
- At 65,536 context with `f16/f16` KV cache, the 512-token speed smoke measured 28.02 generation tok/s, 258.13 prompt tok/s, and 11,658,800 KiB RSS.
- The fixed medium-reasoning, 2,048-token short-Python gate scored 3/6. One completed answer used the wrong false-positive-rate denominator; `flatten-dict` and `dependency-order` exhausted the token limit without emitting code.
- The model was nearly twice as fast as the tested Qwen3.8 27B Q3 models, but was not as reliable under the constrained coding budget. Machine-readable results: `results/ornith15-9b-q8-smoke-results.json`.

- `2026-08-20 00:07:16 IST` — preflight completed; report saved at results/preflight-20260820-000716.txt

- `2026-08-20 00:07:18 IST` — ornith15-9b-q8: server started with PID 12829; log results/server-20260820-000717.log

- `2026-08-25 23:45:34 IST` — preflight completed; report saved at results/preflight-20260825-234528.txt

- `2026-08-25 23:45:36 IST` — qwen38-27b-ud-q3-v3: server started with PID 4837; log results/server-20260825-234535.log

- `2026-08-26 00:36:28 IST` — stopped project server PID 4837

- `2026-08-26 14:56:11 IST` — preflight completed; report saved at results/preflight-20260826-145610.txt

- `2026-08-26 14:56:12 IST` — qwen38-27b-ud-q3-v3: server started with PID 64443; log results/server-20260826-145611.log

- `2026-08-26 14:56:27 IST` — stopped project server PID 64443

## Upstream llama.cpp refresh — 2026-08-26

- Fast-forwarded the isolated upstream checkout from `dd1ea524333b1e697489067d7a4c39c60d32beee` to current `origin/master` commit `11cd98842874cc1b87ac274bd2d5cceb38102bb2` (`0.3.0-dev`, build 278).
- Rebuilt `llama-server` in Release mode with Metal and Accelerate; the pinned TurboQuant checkout and separate DFlash2 worktree were not changed.
- Updated the ignored local runtime pointer and removed the UD V3 Terminal-Bench override so `scripts/run.sh qwen38-27b-ud-q3-v3` now uses the refreshed upstream build and its tracked xhigh profile.
- Verified the Dynamic V3.0 GGUF loads at 65,536 context, reports healthy, exposes the expected model path through `/v1/models`, and returns output through `/v1/chat/completions`.

- `2026-08-27 00:00:12 IST` — preflight completed; report saved at results/preflight-20260827-000011.txt

- `2026-08-27 00:37:56 IST` — ornith15-35b-apex-compact: model resolver selected Ornith-1.5-35B-A3B-APEX-Compact.gguf (16538851328 bytes)

- `2026-08-27 00:38:09 IST` — preflight completed; report saved at results/preflight-20260827-003809.txt

- `2026-08-27 00:40:09 IST` — preflight completed; report saved at results/preflight-20260827-004008.txt

- `2026-08-27 00:40:11 IST` — ornith15-35b-apex-compact: server started with PID 76829; log results/server-20260827-004010.log

- `2026-08-27 00:43:26 IST` — stopped project server PID 76829

- `2026-08-27 00:44:17 IST` — preflight completed; report saved at results/preflight-20260827-004416.txt

- `2026-08-27 00:44:18 IST` — qwen38-27b-ad-iq3: server started with PID 77478; log results/server-20260827-004417.log

- `2026-08-27 00:55:35 IST` — stopped project server PID 77478

- `2026-08-27 00:56:05 IST` — preflight completed; report saved at results/preflight-20260827-005605.txt

- `2026-08-27 00:56:06 IST` — qwen38-27b-ud-q3-v3: server started with PID 78761; log results/server-20260827-005605.log

- `2026-08-28 20:40:07 IST` — preflight completed; report saved at results/preflight-20260828-204006.txt

- `2026-08-28 20:40:08 IST` — ornith15-35b-apex-compact: server started with PID 58771; log results/server-20260828-204007.log

- `2026-08-28 20:50:30 IST` — ornith15-35b-apex-compact: preferred validation passed at context 131072

- `2026-08-28 20:50:31 IST` — stopped project server PID 58771

- `2026-08-28 20:52:35 IST` — preflight completed; report saved at results/preflight-20260828-205235.txt

- `2026-08-28 20:52:37 IST` — ornith15-35b-apex-compact: server started with PID 60357; log results/server-20260828-205236.log

- `2026-08-28 20:57:15 IST` — stopped project server PID 60357

## Ornith 1.5 35B A3B APEX Compact — 2026-08-27 to 2026-08-28

- Downloaded and checksum-verified `mudler/Ornith-1.5-35B-A3B-APEX-GGUF` file `Ornith-1.5-35B-A3B-APEX-Compact.gguf` (16,538,851,328 bytes; SHA-256 `846eb4121c1b28df0e2dff06c3f3d174084231a7400c649ba02023843ea41021`).
- With `f16/f16` KV cache, the 65,536-context speed smoke measured 49.07 generation tok/s and 17,795,376 KiB RSS. At 131,072 configured context it measured 32.24 generation tok/s and 18,877,280 KiB RSS. RSS includes mmap-backed pages and is not entirely private physical memory.
- A long-context request processed 97,108 actual prompt tokens at 162.53 tok/s, generated 79 tokens at 17.08 tok/s, and stopped normally. The validation harness's character estimate undershot the configured 131,072-token ceiling.
- The fixed 2,048-token coding gate scored 4/6; both failures exhausted the output limit. Repeating at 131,072 context with a 16,384-token allowance scored 6/6. The formerly truncated tasks completed in 3,689 and 2,182 tokens, confirming constrained-completion failures rather than wrong submitted code.
- The second run requested `xhigh`, but Ornith reasons natively and its template may ignore named reasoning-effort levels. Machine-readable results: `results/ornith15-35b-apex-compact-results.json`.
- Reduced the tracked Pi example and the local Pi configuration to Ornith Compact and Qwen3.8 UD-Q3_K_XL Dynamic V3.0, each with a 16,384-token output limit.

- `2026-08-28 21:32:40 IST` — preflight completed; report saved at results/preflight-20260828-213240.txt
- `2026-09-08 14:47:54 IST` — ornith15-35b-apex-compact: server started with PID 56229; log results/server-20260908-144753.log
- `2026-09-08 14:48:05 IST` — ornith15-35b-apex-compact: server started with PID 56409; log results/server-20260908-144804.log
- `2026-09-08 14:48:19 IST` — ornith15-35b-apex-compact: server started with PID 56573; log results/server-20260908-144818.log

- `2026-09-18 10:40:02 IST` — ternary-bonsai-2-27b: server started with PID 113898; log results/server-20260918-104001.log

- `2026-09-18 10:42:53 IST` — ternary-bonsai-2-27b: ran prism-ml/Ternary-Bonsai-2-27B-gguf via PrismML llama.cpp fork (prism @ 5d80cff, CUDA sm_89); stock llama.cpp cannot load PTQ1_0/PQ2_0. Winner PQ2_0 -ngl 47 -c 8192 -ub 128: 0.88 tok/s decode, 4.17 tok/s prompt, 5.3-5.7 GiB VRAM, verified via scripts/run.sh. PTQ1_0 cannot fully offload (weights 5.67 GiB vs 5666 MiB free VRAM; auto-fit cut to 39/65) and its CPU matmul is scalar-only (arch-fallback.h). Measurements in results/ternary-bonsai-2-27b-offload.json; profile config/models/ternary-bonsai-2-27b.env, machine winner in config/local/.

- `2026-09-18 11:02:17 IST` — ternary-bonsai-2-27b: server started with PID 119854; log results/server-20260918-110216.log

- `2026-09-18 11:03:14 IST` — ternary-bonsai-2-27b CPU-only measured: PQ2_0 0.26 tok/s decode / 0.96 tok/s prompt (-ngl 0 -t 12), PTQ1_0 0.11 tok/s decode; CPU-only is ~3.4x slower than the tuned 47-layer GPU offload (0.88/4.17). Winner config restored via scripts/run.sh.

- `2026-09-18 12:35:26 IST` — ternary-bonsai-2-27b: server started with PID 148898; log results/server-20260918-123524.log

- `2026-09-18 12:59:52 IST` — ternary-bonsai-2-27b: server started with PID 155832; log results/server-20260918-125951.log

- `2026-09-18 13:01:39 IST` — ternary-bonsai-2-27b: wrote AVX2/AVX-VNNI PQ2_0 CPU vec_dot kernel patch (patches/pq2_0-avx2-cpu-kernel.patch, 30+/14- in arch/x86/quants.c): standalone 4.48x vs scalar generic, unit-test PASS 512/512 incl. all-code-3 stress. In-server: 47-layer split decode 0.88 -> 1.27 tok/s (t8), prompt 4.17 -> 5.25 tok/s, CPU-only 0.26 -> 0.33 tok/s. 3-5 tok/s target not reached: residual ~600 ms/token serial floor (unfused GDN recurrence + per-layer small-op scheduling; fused GDN auto-disabled by device-mismatch probe, no CLI override; ngl 48 core dumps, 47 is the VRAM ceiling). Final config t8 ngl47 restored via run.sh, verified (7*8=56, 1.24 tok/s).

- `2026-09-18 13:19:48 IST` — ternary-bonsai-2-27b: server started with PID 162574; log results/server-20260918-131947.log

- `2026-09-18 13:22:10 IST` — ternary-bonsai-2-27b: forced fused GDN cross-device via env-gated patch (patches/force-fgdn-crossdevice.patch, GGML_FORCE_FGDN=1): split decode 1.27 -> 1.42 tok/s (t8, verified Tokyo answer); Amdahl re-fit shows serial floor only ~605 -> ~530 ms/token (t4 1.13 / t8 1.42), remaining cost is the per-layer CPU<->GPU pipeline, not GDN/matmul. ngl 48 core dumps, 47 stays the ceiling. 27B-only focus confirmed by user: deleted the downloaded Ternary-Bonsai-8B file (unwanted; was verifier-directed fallback), best config restored via scripts/run.sh and recorded in config/local.

- `2026-09-18 14:27:17 IST` — ternary-bonsai-2-27b: recorded screen video of the model (PQ2_0, 47/65 layers, GGML_FORCE_FGDN) running a ZCode-style session analyzing /home/sweker/work: live find/du (43 projects, 96G), then streamed reasoning + full 8-line filesystem analysis via OpenAI API on Xvfb+kitty captured with ffmpeg x11grab. Deliverable: results/ternary-bonsai-27b-fs-analysis.mp4 (384 s, 1280x800@10fps, h264, 2.4 MB); verified frames: intro, real du output, streaming analysis, complete final answer.

- `2026-09-18 15:36:57 IST` — ternary-bonsai-2-27b: recorded screen video of the model running inside the ZCode app (llama-server WebUI in the in-app browser pane): typed the work/ analysis request (company dir renamed to acme-corp at user request), live prompt-processing stats and streamed reasoning + final 8-line analysis (715 tok, 1.20 t/s). Captured via in-app browser tab recording (6x90s webm segments chained, stitched to mp4, 8.4 min). Old terminal take showing that name deleted; intermediate webms cleaned. Deliverable: results/ternary-bonsai-27b-zcode-chat.mp4.
- `2026-09-18 17:01:26 IST` — ternary-bonsai-2-27b: model resolver selected Ternary-Bonsai-2-27B-PQ2_0.gguf (7206168928 bytes)
- `2026-09-18 17:01:34 IST` — preflight completed; report saved at results/preflight-20260918-170133.txt
- `2026-09-18 17:01:35 IST` — ternary-bonsai-2-27b: server started with PID 12287; log results/server-20260918-170134.log
- `2026-09-18 17:02:18 IST` — preflight completed; report saved at results/preflight-20260918-170217.txt
- `2026-09-18 17:04:53 IST` — preflight completed; report saved at results/preflight-20260918-170452.txt
- `2026-09-18 17:07:27 IST` — preflight completed; report saved at results/preflight-20260918-170726.txt
- `2026-09-18 17:09:02 IST` — preflight completed; report saved at results/preflight-20260918-170902.txt
- `2026-09-18 17:09:45 IST` — preflight completed; report saved at results/preflight-20260918-170945.txt

- `2026-09-19 18:30:43 IST` — models cleanup at user request: deleted Ternary-Bonsai-2-27B-PQ2_0.gguf and PTQ1_0.gguf (~13.1 GB freed, disk 225G->213G used); llama-server stopped first (was holding PQ2_0 open). KAT-Coder-V2.5-Dev-APEX-I-Mini.gguf (13 GB, pre-existing, default kat-coder profile) left in place pending user decision. Re-download via scripts/download-model.sh <model-id>.

- `2026-09-19 18:36:59 IST` — deleted KAT-Coder-V2.5-Dev-APEX-I-Mini.gguf (13 GB) at user request; models/ directory now empty. All models re-downloadable via scripts/download-model.sh <model-id>.

- `2026-09-26 22:56:47 IST` — ternary-bonsai-2-27b: server started with PID 404846; log results/server-20260926-225646.log

- `2026-09-26 22:59:23 IST` — ternary-bonsai-2-27b: replicated the X post recipe (PTQ1_0 + -ngl 99 + -ot output.weight=CPU + new AVX2/VNNI SIMD PTQ1_0 head kernel): decode 1.76 -> 5.38 tok/s (scalar head baseline -> SIMD), pp 14.2 -> 33.5 tok/s, all 65 transformer layers on GPU, fused GDN auto-enabled, 5.5 GiB VRAM, -np 1 + q4_0 KV needed to fit. Unit test PASS (512/512 vs scalar generic, 4.24x standalone). Patch: patches/prism-fork-avx2-kernels.patch (all fork changes combined); run.sh gained a model-agnostic EXTRA_SERVER_ARGS passthrough; profiles updated to PTQ1_0. Server verified via scripts/run.sh (6*7=42, 5.22 tok/s). vs the tweet: 11.8 recorded — residual gap likely their CPU/RAM or leaner kernel; ours is 3.8x our previous best.

- `2026-09-26 23:19:26 IST` — ternary-bonsai-2-27b: server started with PID 410405; log results/server-20260926-231925.log

- `2026-09-26 23:20:14 IST` — ROOT CAUSE of the speed gap found: laptop was in power-saver platform profile (EPP=power, CPU pinned at 1.0 GHz of 4.6 GHz max, GPU mostly idle waiting). powerprofilesctl set performance -> CPU 4.5 GHz under load. PTQ1_0 + -ngl 99 + -ot output.weight=CPU + SIMD head kernel now: 12.7-13.2 tok/s decode, 65.7 tok/s pp (matches the X post 11.8; 9.1x the session start 1.42). Final config: PTQ1_0, t12, q4_0 KV, -np 1, -ub 128, 8k ctx; verified via scripts/run.sh.
