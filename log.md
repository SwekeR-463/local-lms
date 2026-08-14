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
