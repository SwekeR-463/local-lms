# Benchmark data

This directory pins the workbench's public comparison suites without copying upstream datasets.

- `short-python/`: six runnable, deterministic Python coding tasks. The model receives only each `prompt.txt`; evaluators are held out during generation.
- `agent-51/`: the fixed 51-task agent-suite manifest. Run those tasks with their official harnesses and disclose this as a custom subset, not an official full-suite score.

Results belong under `results/` and must include the model ID. Run one loaded model and one request at a time.
