# Block-25 Anomaly and Top-K Tail-Mass Audit

## Status

Completed on 2026-07-31 after the Experiment 035 review probes reported that
a fresh post-KD block-25 MLP scale refit improved much more than all other
screened blocks. The audit treated the result as a possible measurement,
overlay, or checkpoint error before attempting an architectural explanation.

The recommended objective ablation is now complete. See
[75-topk-tail-kd-objective-ablation.md](75-topk-tail-kd-objective-ablation.md).
At 256 matched steps, the tail-aware arm improves held-out NLL and full KL
while retaining 0.826 student mass on the teacher's top 64; the conditional
control collapses that mass to 0.580 and worsens full KL despite improving its
own conditional loss. The original fresh block-25 refit was then repeated
from the tail-aware checkpoint and reversed from a large gain to a
statistically clear NLL/KL regression. That closes the causal explanation:
block 25 was compensating for the conditional objective's distribution error.

## Verdict

The measured gain is real, but the original block-local interpretation was
wrong. Block 25 is not uniquely reconstructed badly, and its refit does not
move its weights closer to the BF16 teacher. It is the highest-leverage place
to compensate a model-wide probability-mass error created by conditional
top-64 distillation.

The fundamental mistake is in the inherited objective. Both the legacy and
rewrite implementations apply softmax independently to the teacher and
student logits at the teacher's 64 selected vocabulary entries. They do not
represent the probability mass outside those entries. Consequently, the loss
can improve while the student moves most of its probability mass into the
unobserved vocabulary tail.

Experiment 036 should remain paused. The next controlled experiment should
repair the KD objective on the retained Experiment 035 pre-KD state before a
fresh block-25 refit is promoted as a production stage.

## Mistake audit

The following checks did not find a block-selection or evaluation defect:

- all pre-KD, post-KD, dense-overlay, and factor-overlay arms have the expected
  frozen identity and global-tuning reference;
- fitting, local validation, confirmation, and quality windows are disjoint;
- the factor-compatible overlay contains exactly the nine block-25 component
  tensors and is rejected by the loader if its identity, shapes, dtypes, or
  replaced-byte contract differ;
- dense and factor-compatible block-25 forwards agree within the existing
  smoke gate, with zero represented-byte delta;
- the effect confirms on untouched 48x512 validation windows, the independent
  WikiText test protocol, and the complete six-task quality inventory;
- Experiment 022 and Experiment 035 independently reproduce the same final
  block pathology.

The paired metric arithmetic was also recomputed directly from the retained
per-sequence values. No baseline inversion, token-count weighting error, or
bootstrap pairing mismatch was found.

## The defect appears during KD in two campaigns

The global KD stage already records pre/post hidden-state MSE against a fixed
teacher reference. Its last six blocks show the same discontinuity in both
campaigns:

| Campaign | block 23 relative MSE change | block 24 | block 25 |
| --- | ---: | ---: | ---: |
| Experiment 022 | +0.216 | +0.440 | **+6.818** |
| Experiment 035 | +0.226 | +0.480 | **+6.746** |

This is not caused by a disproportionately large block-25 optimizer update.
From the first to final Experiment 035 KD checkpoint, its scale and outlier
families move about 2.0%, and relative to the pre-KD values they move roughly
2.7-2.9%. Earlier blocks generally receive larger aggregate parameter
movement. Small final-block parameter movement is therefore being amplified
by the function and its location.

On validation sequences 104-107, the RMS of the final MLP residual
contribution is:

| State | block 24 | block 25 |
| --- | ---: | ---: |
| BF16 teacher | 74.56 | 95.93 |
| compressed pre-KD | 75.30 | 87.12 |
| compressed post-KD | 98.35 | **171.57** |

The post-KD block-25 contribution nearly doubles even though its dense weights
move only about 4.6-5.2% from pre-KD. This is consistent with multiplicative
gate/up/down scaling in the gated MLP and with the final RMS normalization
making raw hidden magnitude weakly constrained.

## The refit is a compensator, not a weight repair

The corrected block-25 dense weights move farther from the teacher:

| Projection | pre-KD weight NRMSE | post-KD | corrected |
| --- | ---: | ---: | ---: |
| gate | 0.6238 | 0.6259 | **0.7209** |
| up | 0.6234 | 0.6247 | **0.9155** |
| down | 0.5738 | 0.5756 | **0.6815** |

The correction direction has little cosine alignment with reversal of the KD
update. It is not restoring original weights.

An isolated-versus-full negative control makes the role clear. Each refitted
block was evaluated in two contexts on validation sequences 104-111 at
128 tokens:

1. only that MLP is spliced into the otherwise BF16 teacher;
2. that MLP is replaced inside the complete post-KD compressed model.

| Block | isolated NLL delta | full-model NLL delta | context amplification | isolated KL delta | full-model KL delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | -0.0025 | -0.0065 | -0.0039 | -0.0110 | -0.0189 |
| 17 | +0.0326 | -0.0173 | -0.0498 | +0.0134 | -0.0217 |
| 21 | +0.0359 | -0.0211 | -0.0570 | +0.0059 | -0.0297 |
| 23 | +0.0011 | -0.0377 | -0.0388 | +0.0014 | -0.0344 |
| 24 | +0.0585 | -0.0691 | -0.1276 | -0.0006 | -0.0746 |
| 25 | **+0.2369** | **-0.3176** | **-0.5544** | -0.0452 | **-0.4773** |

The block-25 refit is harmful to isolated teacher-context NLL but strongly
helpful after all upstream compression errors are present. Blocks 24 and 25
show a rapidly increasing depth effect. The final MLP has no later transformer
block to retransform its correction, so it can directly rotate the state seen
by the final RMS norm and tied vocabulary head. It is effectively a
high-capacity nonlinear output calibrator for cumulative upstream error.

This also explains why independently positive earlier-block refits do not add
to block 25. Once block 25 compensates the incumbent upstream error field,
changing an earlier block changes the field it was compensating. Marginal
effects are not additive.

## Exact top-64 probability-mass blind spot

The repeatable audit is implemented by
`tools/probe_topk_tail_mass.py` and retained at
`evidence/035/experiment035-block25-topk-tail-mass-validation104-8x128.json`.
It measures full KL, the current conditional top-64 KL, and a 65-category KL
containing the 64 selected tokens plus one aggregated tail bucket.

| State | NLL | conditional top-64 KL | top-64 + tail KL | full KL | student mass on teacher top 64 | absolute tail-mass error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-KD | 3.7520 | 1.1545 | 1.1811 | 1.2238 | 0.8977 | 0.0736 |
| post-KD | 3.7384 | **0.7566** | 1.4568 | 1.5029 | **0.5133** | **0.4539** |
| corrected block 25 | **3.4286** | 0.8793 | **0.9927** | **1.0307** | 0.8357 | 0.1322 |

The teacher places 0.9671 probability mass on its top 64 entries. Global KD
improves exactly what it observes: conditional top-64 KL falls by 0.3979.
At the same time, student mass on those entries collapses from 0.8977 to
0.5133, the tail-mass error grows by 0.3803, and full KL worsens by 0.2791.

The block-25 correction moves against the current training objective:
conditional top-64 KL worsens by 0.1228 relative to post-KD. Nevertheless, it
restores 0.3224 selected-token mass and improves full KL by 0.4722. The
top-64-plus-tail metric improves by 0.4641, accounting for 98.3% of the full-KL
gain. Only about 0.0081 nats of the gain requires modeling distinctions within
the remaining 262,080-token tail.

## Root cause in the objective

The current loss computes:

```text
teacher = softmax(teacher_logits[top64])
student = log_softmax(student_logits[top64])
loss = cross_entropy(teacher, student)
```

This is a conditional distribution given that the token is in the teacher's
top 64. It is invariant to moving all 64 selected student logits together
relative to the rest of the vocabulary. The cached target has neither the
teacher full-vocabulary log-normalizer nor an aggregated tail probability, so
the missing mass is unobservable.

The legacy implementation in
`NanoQuant-OfficalCode/src/nanoquant/core/compress_model.py` uses the same
conditional softmax. The rewrite is behaviorally compatible here; parity has
carried forward an objective defect rather than introducing a new one.

Block 25 is the natural sink for this freedom because it is the final
trainable nonlinear transformation before the final normalization and head.
Its gated MLP multiplies two learned branches, scale changes compound, and no
subsequent transformer block constrains its hidden-state representation.

## Recommended next experiment

Do not resume Experiment 036 or immediately productionize the block-25 patch.
Run a retained-state KD objective ablation from Experiment 035 pre-KD:

1. preserve the current conditional top-64 arm as the exact control;
2. cache the teacher's top-64 entries plus its full log-normalizer or top-64
   mass per selected token;
3. compute the student's vocabulary log-sum-exp in bounded chunks and add one
   aggregated tail bucket, avoiding a resident full-vocabulary tensor;
4. checkpoint every epoch and measure top-64 mass, tail-bucket KL, full KL,
   NLL, and block-output MSE, especially blocks 24-25;
5. select the checkpoint on held-out NLL/full or tail-bucket KL rather than the
   conditional objective alone;
6. only then refit block 25. Its remaining marginal tells us how much is a
   genuine compression-placement opportunity versus recovery from the broken
   objective;
7. if the objective arm passes, complete the retained quality benchmark before
   starting a fresh compression campaign.

The 65-category metric captures nearly the entire observed full-KL correction,
so it is the highest-value first attempt. Full-vocabulary KL remains the
reference arm, but is not required as the first production implementation.
