# Benchmarks

## Throughput smoke benchmark

`benchmark.sh` sends the fixed prompt in `config/prompts/benchmark.txt` to a running server and writes JSON containing the model/profile, quant file, context, hardware, memory, sampling, prompt throughput, generation throughput, speculative acceptance, and response. This is a throughput smoke benchmark, not a quality leaderboard.

The request is greedy (temperature 0) by default so results stay comparable across sessions. `scripts/benchmark.sh MODEL --sampling profile` instead sends the sampling parameters declared by the profile: temperature, top-p, top-k, min-p, presence and repetition penalties, plus chat-template kwargs such as reasoning effort. Every saved JSON records `sampling_mode` and the exact `sampling` block it used.

## Short Python suite

`benchmark-data/short-python/` contains six deterministic coding tasks with held-out evaluators. With one model server running, use `scripts/benchmark-code.sh MODEL`; raw answers, extracted candidates, evaluator logs, and a model-tagged JSON summary are saved under `results/`. `--sampling profile` switches the suite from its greedy/low defaults to the sampling parameters the profile declares.

## Agent-51 evaluation subset

`benchmark-data/agent-51/manifest.json` pins the separate custom 20-task SWE-bench Verified, 25-task Terminal-Bench 2.1, and six-task MLE-bench evaluation subset.
