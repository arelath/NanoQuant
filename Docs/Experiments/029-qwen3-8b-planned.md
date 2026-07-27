# Experiment 029: Planned Qwen3 8B baseline

## Status

**Not run.**

- Model: `Qwen/Qwen3-8B`
- Launcher: `experiments/029-compress-and-benchmark-qwen3-8b.py`
- Retained results: none

## Question

Could the Qwen3 0.6B workflow scale to 8B while avoiding the multi-sequence CUDA numerical instability seen during
protocol development?

## Intended method

The launcher selected serial llama.cpp quality evaluation and conservative large-model execution so that the
deployment runtime, rather than the unstable batched CUDA path, would be authoritative.

## Results

The experiment was not run. No completion, memory, speed, quality, or thinking-mode claim can be made.

## What we learned

The planned protocol captured a known execution risk and a sensible evaluation fallback. That is design history, not
evidence that the fallback or 8B workflow succeeded.

## Disposition

Planned work only; superseded conceptually by the behavior-aware confirmation design in Experiment 031.
