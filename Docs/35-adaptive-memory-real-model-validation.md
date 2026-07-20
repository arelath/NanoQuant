# Adaptive Memory Real-Model Validation

Status: bounded four-model stage validation and complete base-recipe 270M/1B execution complete; protocol-matched
promotion gates pending

Date: 2026-07-20

## Purpose

This note records the designated-host evidence used to calibrate and validate the adaptive memory controller in
`Docs/34-adaptive-memory-planning-and-execution.md`. It distinguishes safe bounded probes from complete compression
and quality evidence. A passing probe proves that the selected placement and stage batch fit this envelope; it does
not prove final BPW, quality, resume, or end-to-end wall-time parity.

## Host and protocol

- GPU: NVIDIA RTX 4000 Ada Generation, 12,282 MiB, WDDM, driver 596.36.
- Balanced policy: 1 GiB CUDA reserve, 25% estimator allowance, minimum 1.25 GiB uncertainty.
- All CUDA probes ran sequentially under `device_lease.py`; the device returned to about 0.7 GiB graphics usage and
  0% utilization afterward.
- The benchmark uses full configured sequence length, 64 samples for no-gradient block forward, and one logical
  tuning batch for forward/backward. Each throughput candidate is measured five times after full-workload warm-up
  and compared
  by median wall time. A new value must beat the configured baseline by at least 5%.
- The model-load and largest-block probes use the exact local snapshot named below. Retained historical runs provide
  longer block-depth evidence but are not silently treated as completed when interrupted.

Reproduce the matrix with:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_adaptive_memory.py `
  --output evidence\perf\2026-07-19-adaptive-memory-matrix\matrix.json `
  --probe-block-forward
```

The adjacent generated Markdown is a compact rendering; the JSON contains full envelopes, plans, candidate
timings, observed counters, and retained-run measurements.

## Models and retained evidence

| Requested model | Revision | Retained run | Completed blocks | Retained peak reserved |
| --- | --- | --- | ---: | ---: |
| `google/gemma-3-270m-it` | `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3` | Experiment 016, architecture-equivalent Unsloth mirror | 18 | 4.32 GiB |
| `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | Experiment 017, exact source | 26 | 8.52 GiB |
| `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f574b556790deb65791e0c5aa438b6` | Experiment 019, exact source, interrupted | 4 | 4.79 GiB |
| `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` | Experiment 018, exact source, interrupted | 5 | 11.09 GiB |

The interrupted runs ended by `KeyboardInterrupt`, not a caught OOM. They remain useful bounded memory evidence.
Gemma 4B reached zero driver-reported CUDA free memory under resident execution, so the new planner rejects that
placement on this 12 GiB host and selects CPU offload.

## Complete adaptive execution evidence

`tools/run_adaptive_memory_canary.py` composes the normal resident workflow rather than calling the resident engine
below the planning boundary. It requires exact local snapshots, can materialize a validated deterministic
calibration input when network access is unavailable, and keeps all calibration, factorization, tuning, commit, and
quality semantics visible in the canonical config.

Two exact-source base-recipe canaries now complete and pass a fresh `validate_resident_run.py --require-complete`
transitive hash audit:

| Model | Blocks / layers | Selected forward / tuning / refit | Effective BPW | Reference / compressed NLL | Peak CUDA | Peak host | Summed block wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 270M | 18 / 126 | 5 / 2 / 2 | 1.00020 | 20.4904 / 12.0303 | 2.41 GiB | 6.75 GiB | 5,541.9 s |
| Gemma 1B | 26 / 182 | 2 / 2 / 2 | 1.01697 | 9.9726 / 11.1676 | 4.29 GiB | 10.67 GiB | 9,513.0 s |

The 270M run deliberately interrupted after block 0. Resume reused the persisted measured plan and all eight
block-0 journal records; the final audit found 144 contiguous records, one semantic identity, and 670 valid
transitive artifacts.

The 1B run experienced external/transient disk pressure while writing block 18's activation generation. Blocks
0-17 and all seven block-18 layer commits remained valid. The 698-artifact partial graph was copied to a high-space
`C:` workspace, audited byte-for-byte, and resumed without changing semantic identity. The completed graph contains
208 contiguous records and 966 validated artifacts. This failure added a live pre-commit disk admission check to
the implementation: immediately before each multi-gigabyte activation write, current free space is resampled and
the configured reserve plus the next generation's largest write must fit. A pressure change now produces a clear
`RES001` before serialization instead of a partial safetensors write.

These complete runs validate adaptive execution, bounded memory, durable revision reuse, and recovery. They are not
the final fixed-versus-adaptive quality/performance comparison. Retained Experiment 017 uses reconstruction-aware
ranks and stacked QKV (five physical tuning units per block), while the complete adaptive 1B canary intentionally
uses the current independent-layer base recipe (seven units). The fixed graph therefore has 130 layer/group journal
records versus 182 independent layer records, a different plan hash, and a different semantic config hash. Its
8.52 GiB versus 4.29 GiB CUDA peak is useful resource context, but its 7,733.4-second block wall and 9.6331 compressed
NLL must not be presented as an apples-to-apples adaptive speed or quality result.

## Final balanced plans and observed safety

The plan first computes a conservative safe upper bound. Runtime autotuning then chooses the fastest measured
candidate within that bound and persists it as a memory-plan revision before commit identity or tuning checkpoint
discovery.

| Model | Executor | Forward fixed -> selected (safe max) | Max forward reserved / capacity | Tuning fixed -> selected (safe max) | Max tuning reserved / capacity |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 270M | resident | 8 -> 5 (40) | 9.28 / 9.80 GiB | 8 -> 2 (8) | 4.02 / 9.80 GiB |
| Gemma 1B | resident | 8 -> 2 (19) | 6.34 / 9.80 GiB | 8 -> 2 (8) | 7.00 / 9.80 GiB |
| Llama 1B | resident | 4 -> 2 (32) | 8.35 / 9.80 GiB | 1 -> 1 (4) | 3.93 / 9.80 GiB |
| Gemma 4B | CPU offload | 4 -> 19 (19) | 9.16 / 9.80 GiB | 1 -> 1 (4) | 5.15 / 9.80 GiB |

All model-load, largest-block forward, and largest-admitted tuning probes passed both allocator and policy bounds.
The result also disproves the original shortcut that the largest fitting batch is necessarily fastest. On this
host, smaller batches won several eager-attention and transfer-sensitive workloads; Llama and Gemma 4B tuning kept
microbatch 1 because no larger candidate cleared the 5% threshold.

The selected bounded-stage speedups over the former fixed values were:

| Model | Block-forward canary | Tuning forward/backward canary |
| --- | ---: | ---: |
| Gemma 270M | 1.06x | 1.14x |
| Gemma 1B | 1.09x | 1.08x |
| Llama 1B | 1.05x | 1.00x |
| Gemma 4B | 1.05x | 1.00x |

These sub-second canaries select among safe candidates; they are not end-to-end performance claims.

## Calibration history

The first shape-only estimator materially underpredicted real peaks and admitted resident Gemma 4B. The implemented
calibration added:

- the complete model as the resident fixed baseline rather than only the active block;
- retained BF16 factor state when completed blocks stay resident for inline quality;
- eager-attention score/softmax workspace using head count, sequence length, and GQA query expansion;
- selected-batch pinned-host staging rather than charging the configured maximum to every candidate;
- a measured 25% / 1.25 GiB balanced uncertainty allowance; and
- executor admission across every stage, so a downstream resident rejection can select CPU offload.

Before the eager-attention term was calibrated, Gemma 270M batch 102 attempted a 6.38 GiB allocation and failed.
Intermediate safe maxima also exceeded allocator capacity for Gemma 270M and 4B. Those failed development probes
were released cleanly and directly informed the conservative final formula; the final four-model matrix passes.

## Host and disk constraints

Memory safety is not only VRAM. The measured research artifact multiplier projects about 83 GiB of durable/scratch
space for Gemma 4B. At initial validation time the repository's `D:` volume had about 58.4 GiB free, so a full 4B run rooted
there is correctly refused even though CPU-offloaded CUDA stages fit. The system temporary volume on `C:` had about
556 GiB free and was used for metadata planning/probes.

The complete 1B canary also proved why preflight free space is not sufficient by itself. Its static plan required
25.2 GiB and its retained run was only 10.5 GiB at the failure boundary, but live available space collapsed during
the high-private-memory process and recovered to 45.2 GiB after process exit. The new boundary guard closes this
online-pressure gap. A complete 4B canary still needs a writable output/artifact workspace with at least the admitted
capacity plus stable OS/pagefile headroom.

## Remaining promotion gates

- Run a protocol-matched adaptive Gemma 1B compression using Experiment 017's reconstruction-aware stacked-QKV
  recipe, then compare exact quality/BPW and end-to-end wall time with the fixed run.
- Run the corresponding complete Gemma 4B canary on a volume with sufficient durable space. The retained partial run
  is not a substitute.
- Compare an uninterrupted protocol-matched run with an interrupted/resumed run within the approved numerical
  parity contract. Persisted-plan and no-repeated-work behavior is proven; cross-run numerical equality remains.
- Extend measured planning to model-level KD, evaluation, prefetch depth, and learned cross-run estimator profiles.
- Do not enable adaptive mode in the canonical Gemma recipe until those complete-run gates pass.
