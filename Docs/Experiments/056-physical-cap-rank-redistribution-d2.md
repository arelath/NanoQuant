# Experiment 056: physical-cap D2 rank redistribution

## Question

Experiment 055 allowed binary rank above the algebraic dimension and regressed
retained WikiText-2 perplexity from 203.84 to 285.50. Was the useful part of
that experiment the wider measured allocation range, while the harmful part
was spending rank on already saturated owners?

## Controlled change

Experiment 056 is a one-policy ablation of Experiment 055:

```text
allocation.bounds.ceiling_fraction_of_uniform:       1.5 (unchanged)
allocation.bounds.overcomplete_rank_ceiling_fraction: 1.5 -> 1.0
```

Every owner is capped at its aligned physical dimension. The reconstruction
allocator is global: after a high-utility owner reaches that cap, it removes
the owner from the marginal-rank candidate set and continues buying the best
available rank increment in any other unsaturated unit or block.

The outlier policy does not grow at saturation. It remains exactly the same as
Experiments 054 and 055: residual selection, 0.1% of input columns, BF16
storage, and exclusion from the 1.0-BPW rank budget. Experiments 054 and 055
both stored 17,097,106 outlier bits including indices, confirming that the
over-complete experiment did not increase this payload.

All other settings are inherited from Experiment 055:

- pinned Gemma model and tokenizer revision;
- the retained exact-unit KL profile from Experiment 054;
- measured rank-response probes with 100 by five ADMM iterations;
- 1.5x response/allocation ceiling and 1.0 target BPW;
- 2x V member objective weight;
- final 64-pass control-then-tabu binary-factor search;
- functional binary learning rate `3e-5`;
- block tuning, post-block refit, global distillation, export, and retained
  quality protocol.

Fresh response probes are required. The Experiment 055 profile contains
over-complete sample points and its probe-plan identity is incompatible with a
physical hard cap; adopting it would mix invalid evidence into the new plan.

Experiment 056 uses the ordinary single-process resumable workflow. It does
not retain Experiment 055's worker-environment switch or forced one-block
process interruptions. If non-finite state escapes transactional rollback,
that execution bug must be fixed at its source rather than hidden by process
restarts.

## First-run failure and root fix

The first production attempt failed after block 0's post-block refit produced a
non-finite optimizer step. Transactional rollback restored finite parameters
and the committed teacher/compressed activation tensors were independently
audited as entirely finite, but the same process's block-1 dense teacher forward
was non-finite. A fresh process replay of that exact 256 x 2048 x 1152 teacher
boundary through the pinned block-1 weights was finite, matching the previously
documented process-local CUDA contamination in Experiment 054.

The first root fix removed the known failed-step allocations: a non-finite tuning
rollback closes and drops staging/optimizer scratch, synchronizes CUDA, and
unconditionally releases the caching allocator. Ordinary successful tuning
retains the existing pressure-gated release. The hash-valid block-0 activation
boundary remained the resume point because its two safetensors contained zero
non-finite values; only process-local scratch was contaminated.

A later single-process resume committed finite blocks 1 and 2, then exposed the
same process-local failure at block 3 even though block 2 had no reported
non-finite rollback. The committed block-2 teacher and compressed streams were
audited as entirely finite, block-3 calibration importance was finite, and an
exact fresh-process replay of all 32 pinned block-3 teacher microbatches was
finite with outputs from -1020 through 5952. Version 59 therefore treats dense
teacher propagation itself as the isolation boundary: it synchronizes and
releases dead CUDA scratch before the forward, validates every output batch,
and performs at most two quarantined identical-input retries. A persistent
failure remains fatal and identifies the batch; a transient recovery is a
visible `block_teacher_forward.nonfinite_retry` event. Existing version-58
commits remain retained evidence but cannot be adopted by the new semantic
identity.

## Promotion gate

Status: **not run**.

Promotion requires:

1. no final owner exceeds its aligned physical rank;
2. at least one unsaturated owner receives rank that it did not receive in
   Experiment 055, demonstrating actual cross-unit redistribution;
3. the outlier count and bit payload remain identical to Experiment 055;
4. effective BPW does not exceed Experiment 055 except for unavoidable aligned
   scale-bit residue;
5. the complete run and export pass fresh artifact validation; and
6. the retained WikiText-2 protocol materially recovers quality versus 055 and
   is compared directly with the stronger 054 physical-cap baseline.

A changed rank plan or improved reconstruction objective alone is diagnostic;
the complete held-out quality result remains binding.
