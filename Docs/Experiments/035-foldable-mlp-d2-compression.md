# Experiment 035: Foldable-MLP D2 Compression

## Status

Completed on 2026-07-31 and **rejected as the replacement candidate**. The
complete pipeline integration worked, but identity-initialized multiplier
tuning did not pass the plan's untouched held-out NLL gate. The numbered
launcher is
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

## Measured execution

The complete workflow finished without reuse of candidate block commits. A
fresh `tools/validate_resident_run.py --require-complete` audit validated 708
transitive artifacts, all 26 contiguous block commits, 130 represented layers,
and the rolling-retention exception. The committed identity is:

```text
model  sha256:32d5b5d041e98027bc7415107bc79b580f9cce407535b4e30134e8f8aed3b130
config sha256:abb5bd5e9dfc2cb4ada9ba5a932ee979bb940f3c60d854a979b9c869d81aceee
plan   sha256-69716c4d153c13fe60b5fdd0adfb5ae333860d18156d1ed96abfbc9a3dbcbdd2
```

Resource and timing measurements:

| Measurement | Result |
| --- | ---: |
| Resident quantization | 9,443.57 s |
| Global distillation | 1,563.04 s |
| Foldable MLP continuation | 51.31 s |
| Complete compression/export | 11,153.24 s |
| Retained quality | 263.41 s |
| End-to-end wall time | 11,416.88 s |
| Peak device bytes | 9,118,416,896 |
| Peak host bytes | 11,544,555,520 |

## Foldable-stage evidence

All 64 configured steps completed. The first and final cached top-k losses were
`1.631042` and `1.467640`. Every tensor in all four families had a finite,
nonzero gradient at the checked boundaries. The 569,088 multipliers remained
conservative:

| Family | Minimum | Median | Maximum | Bound hits |
| --- | ---: | ---: | ---: | ---: |
| gate output | 0.997934 | 0.999997 | 1.002105 | 0 |
| up output | 0.997751 | 0.999995 | 1.002004 | 0 |
| down input | 0.997751 | 0.999995 | 1.002004 | 0 |
| down output | 0.998011 | 1.000008 | 1.001933 | 0 |

Folding replaced 234 component tensors and `3,115,008` bytes with exactly
`3,115,008` bytes. The folded replay maximum absolute error was zero. The
component SHA-256 is
`ac501e8326e38760e3a656359cedf46b8d6795fa42fbce8dabacf7f2a4bd13c9`,
and the protocol hash is
`sha256:df61a6fd2940a67259ac22e485b81ce42479ffe065c305a96e10e18a915656e6`.

Logical and packed export validated exactly:

- 26 blocks, 130 layers, and 910 tensors;
- logical weight bytes: `2,739,492,504`;
- packed weight bytes: `89,480,664`;
- effective BPW: `1.0244947118`;
- packed descriptor SHA-256:
  `3e87ee6854609e2a143eff1b06ef70d4b04353eae7225a5f38f420ff437d677a`;
- GGUF bytes: `417,340,544`.

The packed payload is eight bytes larger than Experiment 022's `89,480,656`
bytes, even though logical bit cost, effective BPW, and GGUF bytes are equal.
The foldable stage did not cause this: it changes no tensor shape, dtype, or
payload byte count. The difference belongs to the freshly measured upstream D2
campaign. It nevertheless means the literal no-greater-packed-bytes comparison
against Experiment 022 does not pass.

## Same-run stage ablation

The complete retained protocol was rerun on the same Experiment 035 global-KD
state without the component overlay. This isolates the new stage from the
fresh D2 allocation:

| Benchmark | Same run, post-KD | Foldable stage | Change |
| --- | ---: | ---: | ---: |
| WikiText perplexity | 221.035336 | **220.879050** | -0.071% |
| PIQA `acc_norm` | 0.615 | 0.615 | 0.000 |
| ARC-Easy `acc_norm` | 0.370 | **0.380** | +0.010 |
| ARC-Challenge `acc_norm` | **0.235** | 0.230 | -0.005 |
| HellaSwag `acc_norm` | **0.430** | 0.425 | -0.005 |
| WinoGrande `acc` | 0.540 | **0.555** | +0.015 |
| BoolQ `acc` | **0.645** | 0.640 | -0.005 |
| Mean primary task score | 0.472500 | **0.474167** | +0.001667 |

The retained benchmark point estimates are slightly favorable, but the
predeclared untouched held-out confirmation does not reproduce the NLL gain.
On validation sequences 104-151 at 48x512, folded minus post-KD is:

| Metric | Point delta | Paired 95% interval |
| --- | ---: | --- |
| NLL | +0.000173 | [-0.000564, +0.000881] |
| Full-vocabulary teacher KL | +0.0000368 | [-0.000550, +0.000605] |

Neither regression is statistically supported, but the NLL point estimate is
not an improvement and its upper interval is not below zero. Identity
initialization therefore fails the Phase C advancement gate.

## Comparison with Experiment 022

The retained inputs are protocol-identical to Experiment 022: WikiText
fingerprint and token hash, tokenizer hash, task semantic keys, prompt hashes,
sample counts, and task ordering all match.

| Benchmark | Experiment 022 | Experiment 035 | Change |
| --- | ---: | ---: | ---: |
| WikiText perplexity | 228.550618 | **220.879050** | -3.357% |
| PIQA `acc_norm` | 0.605 | **0.615** | +0.010 |
| ARC-Easy `acc_norm` | 0.380 | 0.380 | 0.000 |
| ARC-Challenge `acc_norm` | 0.215 | **0.230** | +0.015 |
| HellaSwag `acc_norm` | **0.460** | 0.425 | -0.035 |
| WinoGrande `acc` | 0.520 | **0.555** | +0.035 |
| BoolQ `acc` | 0.635 | **0.640** | +0.005 |
| Mean primary task score | 0.469167 | **0.474167** | +0.005000 |

This comparison is favorable overall, but the same-run ablation shows that
most of it is due to the fresh D2 campaign rather than the foldable stage.

## Decision

Experiment 035 proves that the stage is correctly integrated, resumable,
foldable, exportable, and benchmarked by the mandatory full workflow. It does
not prove that identity initialization is the right numerical policy. The
candidate is rejected because:

1. untouched 48x512 confirmation does not improve paired NLL;
2. the fresh packed payload is eight bytes above Experiment 022; and
3. its `220.879050` perplexity remains far behind the six-block-initialized
   Phase C result of `169.481866` at effectively the same storage.

The next production experiment should integrate the confirmed composed-context
coordinate initializer before the conservative global continuation. The
identity-only setting should remain available as a control, not become the
default compression policy.

Additional evidence:

- `evidence/035/035-resident-validation.json`;
- `evidence/035/experiment035-postkd-no-foldable-quality.json`;
- `evidence/035/experiment035-postkd-foldable-confirm104-48x512-kl.json`.
