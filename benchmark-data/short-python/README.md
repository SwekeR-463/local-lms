# Short Python comparison

Start one model with `scripts/run.sh MODEL`, then run:

```bash
scripts/benchmark-code.sh MODEL
```

The runner sends all six prompts sequentially with temperature 0 and seed 42. It defaults to low reasoning and 1,024 tokens; override those limits without changing the suite:

```bash
BENCHMARK_REASONING_EFFORT=medium BENCHMARK_MAX_TOKENS=2048 scripts/benchmark-code.sh MODEL
```

It preserves each raw response and candidate, runs the task evaluator, and writes a JSON summary under `results/short-python-MODEL-TIMESTAMP/`.

Evaluators are public for reproducibility but are not included in model prompts. They execute generated Python locally: inspect candidates or use an isolated machine for untrusted models.
