# Experiment 035: Foldable-MLP D2 Compression

## Status

Prepared for execution on 2026-07-31. The numbered launcher is
`experiments/035-foldable-mlp-d2-compress-and-benchmark-gemma-3-1b-it.py`.

## Purpose

Experiment 035 turns the Phase C method from
[71-composed-context-mlp-scale-refit-plan.md](../71-composed-context-mlp-scale-refit-plan.md)
into a configured, resumable stage of the complete compression workflow. It
runs after global top-k distillation and before logical export. Logical export,
packed conversion, GGUF conversion, and retained quality therefore consume the
same hash-bound folded component state.

This experiment deliberately does not import Experiment 022's accepted
six-block component overlay. That overlay is bound to Experiment 022's commit
and global-tuning identities. Experiment 035 instead starts all multiplier
families at identity on its freshly compressed and distilled state. This is the
clean test of whether the global method generalizes as a pipeline stage rather
than merely improving a hand-selected initialization.

## Matched baseline and recipe

The baseline is Experiment 022. Experiment 035 preserves its pinned Gemma 3 1B
model revision, calibration recipe, architecture-protected D2 allocation,
same-campaign exact-unit KL measurement, same-run rank-response measurement,
factorization and block tuning, eight-epoch global top-k distillation, packed
format, and complete retained quality protocol.

The only intended numerical delta after global distillation is:

- train FP32 log multipliers for gate output, up output, down input, and down
  output on every compatible MLP block;
- use the retained global-distillation teacher targets for 64 steps;
- use learning rate `1e-4` with cosine decay, family-balanced identity penalty
  `100`, gradient clipping `1.0`, and a multiplier safety limit of `4.0`;
- disable gradient checkpointing, matching the deployment-faithful Phase C
  confirmation path;
- checkpoint every 16 steps so an interruption restarts at most one bounded
  interval;
- fold the selected values through scales, floating outliers, and both sides
  of correction patches with no new tensors in the deployed representation.

## Pipeline contract

The post-KD stage writes a run-local training checkpoint, report, active
receipt, and factor-component overlay. The overlay is bound to all of:

- committed model, configuration, and allocation-plan hashes;
- the active global-tuning artifact;
- calibration-token hash;
- complete foldable-multiplier protocol hash.

The complete compression exporter refuses to proceed when the stage is enabled
but no matching active component state exists. Fresh logical validation compares
every exported term against the overlay, and packed validation then proves exact
logical-to-packed conversion. The folded forward must exactly replay the
deployment-faithful optimization forward before the overlay can become active.

## Acceptance gates

Experiment 035 is accepted only if all of the following hold:

1. 64 steps complete with finite losses and complete finite gradient coverage.
2. Folded-versus-unfolded replay has zero measured error on the replay input.
3. Component replacement changes no shapes, dtypes, or payload bytes.
4. Logical and packed export validate exactly.
5. Packed weight bytes and effective BPW do not exceed Experiment 022.
6. The complete retained protocol runs: WikiText 64x128 and 200 examples each
   of PIQA, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, and BoolQ.
7. WikiText perplexity improves over Experiment 022 without a broad downstream
   task-quality regression.

The accepted Phase C probe initialized from the six-block composed-context
state and reached perplexity `169.481866`; that result is a useful upper
comparison, not the primary matched baseline. If identity initialization does
not recover the gain, the next experiment should productionize the composed-
context coordinate initializer instead of importing an identity-bound overlay.

## Evidence locations

- Run and stage state:
  `evidence/035/035-foldable-mlp-d2-compress-and-benchmark-gemma-3-1b-it/`
- Logical, packed, summary, and quality artifacts: `outputs/035/`
- GGUF and published reports: `Results/035/`

This document must be updated with measured identities, bytes, BPW, timing,
resource use, and full quality results after the workflow completes.
