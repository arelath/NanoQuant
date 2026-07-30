# Experiment 034 — Gemma 3 1B post-refit QKV covariance

**Status:** Completed; rejected by pre-KD and post-KD perplexity gates

## Question

Does the same-rank post-refit covariance placement selected in Document 50
survive the complete Experiment 022 compression, global-distillation, export,
and retained-quality lifecycle?

## Method

Experiment 034 changes Experiment 022 only by enabling post-refit covariance
refinement for the fused-QKV owner in blocks 5, 11, 24, and 25. It retains
the diagonal calibration objective, D2 rank allocation, ranks, outliers,
factor format, global KD, export, and quality protocols.

The refinement captures 8,192 input rows for each selected owner and runs
after factorized tuning and post-block refit. No new representation field or
bit allocation is introduced.

## Gate

The experiment must complete strict validation and the mandatory packed,
checkpoint, GGUF, and retained-quality lifecycle. Promotion requires:

- no effective-BPW increase versus Experiment 022;
- exact pre-KD quality consistent with the bounded −6.79% perplexity result;
- final post-KD quality better than Experiment 022;
- valid resume, artifacts, and exported runtime format.

## Result

The experiment completed after resuming from durable block 7. The resumed
worker reused 48 valid commits, completed all 26 blocks, ran eight epochs of
global top-k distillation, and produced the mandatory packed artifact,
checkpoint, GGUF, quality report, and `Results/034` publication.

Fresh strict validation reported:

- 156 active journal records with one identity and no inactive records;
- 26 contiguous block records and 130 committed layer/group records;
- 712 validated transitive artifacts, including four persisted
  `covariance-binary-refinement` artifacts;
- effective BPW 1.024496179;
- a valid 417,340,672-byte GGUF.

All four selected refinements improved the production covariance objective,
by 6.45% in block 5, 7.50% in block 11, 5.09% in block 24, and 4.02% in
block 25. They did not improve retained language modeling:

| Metric | Experiment 022 | Experiment 034 | Relative change |
| --- | ---: | ---: | ---: |
| Pre-KD WikiText perplexity | 273.516089 | 286.081276 | +4.59% |
| Post-KD WikiText perplexity | 228.550618 | 241.121781 | +5.50% |

The retained protocol and token hash are identical. Global KD reduced
Experiment 034 perplexity by 15.71%, but the regression introduced before KD
survived.

The small task suite moved in the opposite direction: PIQA, ARC Easy, ARC
Challenge, HellaSwag, and Winogrande improved versus Experiment 022, while
BoolQ tied. The unweighted six-task mean rose from 0.4692 to 0.4842. Because
each task used only 200 rows and the pre-registered exact WikiText gate
failed, this is retained as a follow-up signal rather than an acceptance.

Experiment 034's effective BPW was only 0.00000147 above Experiment 022 and
its GGUF was 6,016 bytes larger. The refinement itself adds no representation
field; a fresh D2 profile allocated 64 additional total rank units. This is
operationally negligible, but it also means the literal no-increase gate did
not pass.

## Decision

Reject the selected post-refit QKV covariance refinement as a default. The
experiment proves that the implementation, resume path, immutable artifact
contract, global KD interaction, and export lifecycle work. It also proves
that the offline WikiText placement gain did not transfer to covariance
captured from the production calibration stream.

Future sparse factor edits must be selected with a held-out functional metric
representative of the target distribution and tested compositionally. Local
covariance improvement, even at hand-selected blocks and unchanged format
capacity, is not sufficient.
