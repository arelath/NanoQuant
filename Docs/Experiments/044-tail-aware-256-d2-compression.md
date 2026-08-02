# Experiment 044: Fresh Tail-Aware 256-Step D2 Compression

## Status

Completed and rejected for promotion. The fresh-factorization WikiText and C4
transfer gates passed, the task guardrail did not establish a regression, and
the complete artifact/export lifecycle passed. The binding deployment-quality
gate failed: protocol-matched packed perplexity was 4.94% worse than Experiment
042, outside the predeclared 2% non-inferiority bound. No production default was
changed.

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

## Completed run

The complete run is
`evidence/044/044-tail-aware-256-d2-compress-and-benchmark-gemma-3-1b-it`.
It was produced from launcher revision `e700a7a` with resident identity:

- config `sha256:08d2e59056ddc6c4878d847b0a8802fd2b3b194bbc7b6567052a83cfd96def0b`;
- model `sha256:32d5b5d041e98027bc7415107bc79b580f9cce407535b4e30134e8f8aed3b130`;
- plan `sha256-21c3d0b066e0e377384e3eda8856d2b2492efd0b1ee5ce9a899f7675bc56545e`.

The immutable epoch-8 distillation checkpoint is
`sha256-958f801f4c09f2ef9fb032ecd5f5b68f598c8a3b4cef863e9eebac30fcfc2567`.
Its receipt records exactly 8 completed epochs and 256 optimizer steps. The
committed global-tuning result is
`sha256-eb4a8f736d51b94b8b8b561d3dc2a0a48970be9f359b508e8c0d4140c46233cf`
and independently records 256 steps. The resolved run config records
`top_k_tail`, tail weight 0.5, 32 maximum batches per epoch, and explicitly
disabled mass-floor correction and final-norm calibration. There is no
correction, fold, temperature-fit, or block-25 overlay artifact.

Fresh strict validation checked 713 transitive artifacts, all 26 blocks and
130 committed owners, one journal identity, and a contiguous 0-25 block
prefix. The result is complete with effective BPW 1.0244176654. The packed
artifact contains 26 blocks, 130 layers, and 89,473,944 weight bytes; the GGUF
is 417,333,824 bytes with SHA-256
`18d7b3043a8e76bf97669d13ff65458889b18456bf2927c5caf05aed4aa0bb29`.

## Fresh-factorization transfer gates

Both gates compare the immutable 044 pre-KD state with its 256-step post-KD
state. This holds factorization, allocation, calibration data, and evaluation
sequences fixed and isolates the transferred KD policy.

The reserved WikiText validation slice at offset 396, 48x512, token hash
`sha256:8121a0e488a85602dd808ea2be915272ad5025f87f4d1621e1e2e5b5a39ddbb0`
passed:

- NLL: 4.90159381 -> 4.44005115, delta -0.46154266, 95% paired CI
  [-0.48596835, -0.43704672];
- full KL: 1.59333768 -> 1.37006794, delta -0.22326974, CI
  [-0.24038826, -0.20650365];
- top-k-plus-tail KL: 1.49985629 -> 1.29281014, delta -0.20704615, CI
  [-0.22325594, -0.19105339].

Selected top-64 mass fell from 0.82444531 to 0.70365714 and its absolute
teacher-mass error rose from 0.11921246 to 0.22873189. This is a calibration
diagnostic, not a capability rejection. It shows that tail weight 0.5 reduces
NLL/KL decisively but does not eliminate confidence drift.

The reserved C4 validation slice at offset 200, 48x512, token hash
`sha256:d8b3fb059626683523f1ab05676f4c14dd809cb322c22476cf075c45bf2d5615`
also passed:

- NLL: 5.06214916 -> 4.71862530, delta -0.34352386, 95% paired CI
  [-0.36077745, -0.32676730];
- KL: 1.28430321 -> 1.13175852, delta -0.15254469, CI
  [-0.16699155, -0.13829220].

Both slices were reserved before evaluation and are now permanently retired in
`Docs/evaluation-slice-registry.json` with their evidence paths.

## Deployment-quality decision

The ordinary complete workflow evaluated both packed and llama.cpp/GGUF
paths. Packed perplexity was 180.36413287 versus Experiment 042's
171.87089667, a 4.9416% regression. The predeclared non-inferiority ceiling was
175.30831460, so this gate failed. GGUF perplexity independently agreed:
179.50214933 versus 170.18196275, a 5.4766% regression.

The generated complete-run summary's top-level `passed: true` records workflow
health and its configured generic finite-result checks; it is not a promotion
decision. The predeclared baseline-relative gate above is the binding 044
decision and rejects the candidate.

The 1,000-example six-task comparison against Experiment 042's completed
candidate produced means 0.44933333 versus 0.45850000. The paired
task-stratified delta was -0.00916667 with 95% CI [-0.02000000, 0.00166667].
It did not establish either improvement or regression. Individual 044 scores
were PIQA 0.588, ARC Easy 0.379, ARC Challenge 0.210, HellaSwag 0.394,
Winogrande 0.505, and BoolQ 0.620.

Therefore the policy transfers in the narrow causal sense--it decisively
improves the fresh factorization over its own pre-KD state on WikiText and
C4--but it does not meet the deployment comparison against 042. The result is
not promoted, the 2% bound is not widened after observing the failure, and no
additional correction or fold is fitted on these gate slices.

## Evidence

- `evidence/044/044-tail-aware-256-d2-compress-and-benchmark-gemma-3-1b-it/strict-validation.json`
- `evidence/044/experiment044-wikitext-validation396-48x512-prekd-vs-postkd.json`
- `evidence/044/experiment044-c4-validation200-48x512-prekd-vs-postkd.json`
- `evidence/044/experiment044-tail256-tasklimit1000-quality.json`
- `evidence/044/experiment044-vs-experiment042-candidate-tasklimit1000-paired.json`
- `Results/044/044-tail-aware-256-d2-compress-and-benchmark-gemma-3-1b-it-summary.json`
- `Results/044/gemma-3-1b-it-nanoquant.export-summary.json`
