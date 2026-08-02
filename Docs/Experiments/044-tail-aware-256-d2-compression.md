# Experiment 044: Fresh Tail-Aware 256-Step D2 Compression

## Status

Predeclared. No Experiment 044 model-dependent evidence has been opened.

## Question

Experiment 043 established on one frozen Experiment 042 factorization that:

- conditional KD at 256 steps decisively beats the production-default
  2,048-step conditional trajectory on held-out NLL and full KL;
- at the matched 256-step horizon, top-64 plus aggregated-tail KD decisively
  beats conditional-only KD on fresh WikiText and C4;
- the six-task 1,000-example paired guardrail found no established regression.

Experiment 044 asks whether that two-part policy transfers to a fresh complete
compression run and survives the packed/GGUF lifecycle.

## Frozen treatment

The factorization and allocation policy matches Experiment 042: pinned Gemma
3 1B IT, exact D2 KL allocation profile, unchanged effective-BPW target, and
the same block/layer tuning recipe. The sole intended post-factorization
treatment is:

- primary global KD objective: top-64 plus aggregated tail;
- tail mass weight: 0.5;
- epochs: 8;
- maximum batches per epoch: 32;
- required observed optimizer steps: 256.

The following Experiment 042 stages are explicitly disabled:

- one-sided mass-floor correction;
- fixed 1.015 final-norm fold;
- temperature fitting;
- block-25 refit or factor overlay.

These values come directly from the matched Experiment 043 comparison. No
coefficient, horizon, checkpoint, or calibration scalar may be changed after
launch.

## Gates

1. The complete resident run must contain one coherent identity, all 26 blocks
   and 130 owners, exactly 256 primary KD steps, no correction/fold artifact,
   and pass `tools/validate_resident_run.py --require-complete` with fresh hash
   validation.
2. Effective BPW and packed/GGUF sizes must remain within the Experiment 042
   storage contract; the objective may change quality and compute, not the
   factor budget.
3. The complete workflow must assemble and load the frozen factorized model,
   serialize and reload the packed model, export GGUF, and evaluate both packed
   and llama.cpp paths.
4. The canonical quality result must improve over Experiment 042's completed
   candidate on protocol-matched WikiText perplexity or remain within a
   predeclared 2% relative non-inferiority bound, with no established task
   regression on the full 1,000-example six-task follow-up.
5. Fresh transfer gates must use newly reserved, non-overlapping WikiText and
   C4 intervals. Previously opened offsets 104, 152, 200, 248, 300, and 348
   cannot select or accept this run.

The ordinary workflow's fixed WikiText/task suite remains a regression
benchmark for implementation comparability; it cannot be used to tune the KD
policy. Any additional transfer gate is reserved before evaluation under
[`../evaluation-slice-registry.json`](../evaluation-slice-registry.json).

## Interpretation

Passing Experiment 043 was necessary but not sufficient: it reused one frozen
factorization. Experiment 044 is the first production-shaped transfer test of
the policy. Byte-exact resume, artifact validity, or a successful export do not
substitute for fresh-model quality. Conversely, selected top-k mass is reported
with uncertainty as a calibration diagnostic and is not a capability gate.

The base recipe is not changed globally by this launcher. Promotion requires a
completed Experiment 044 result satisfying every gate above.
