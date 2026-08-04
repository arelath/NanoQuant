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
