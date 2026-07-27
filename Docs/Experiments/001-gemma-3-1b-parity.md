# Experiment 001: Gemma 3 1B rewrite parity

## Status

**Completed campaign.** This experiment evolved into the principal legacy-versus-rewrite parity effort rather than
remaining a single immutable run.

- Model: `google/gemma-3-1b-it`
- Launcher: `experiments/001-compress-gemma-3-1b-it.py`
- Durable evidence: [`evidence/m4`](../../evidence/m4/) and
  [the parity audit](../23-gemma-1b-parity-completion-audit.md)

## Question

Could the rewrite reproduce the legacy NanoQuant algorithm, artifacts, memory behavior, resumability, quality, and
runtime on the pinned Gemma workload?

## What we did

We built the resident, resumable 26-block/182-layer compression path; matched legacy-style factorization and tuning;
exported a GGUF; audited artifact integrity; and compared the result with the contemporary legacy implementation and
the BF16 source model.

## Results

- Effective BPW: **0.996318**
- Factor rank sum: **105,856**
- WikiText-2 perplexity: rewrite **453.571**, legacy **444.333** (**+2.079%**)
- Model KD: **8 epochs / 2,048 steps**
- Peak CUDA / host memory: **7.63 GB / 12.79 GB**
- GGUF size: **699,863,936 bytes**
- Runtime: **160.74 tok/s**, or **87.12%** of the compared llama.cpp rate

## What we learned

The rewrite reached measured behavioral and quality parity with the contemporary legacy path. That result also made
the larger problem unambiguous: both implementations' shared recipe was far behind the BF16 source model. Rewrite
parity was therefore a foundation, not the final quality goal. Durable journals, rolling activation retention, and
fresh artifact validation were necessary parts of real parity rather than operational conveniences.

## Disposition

Accepted as the rewrite parity baseline. Subsequent experiments target source-model quality, allocation, transfer,
and deployment behavior.
