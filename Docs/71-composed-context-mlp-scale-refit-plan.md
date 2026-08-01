# Composed-Context MLP Scale Refit Plan

## Purpose

Experiment 022's accepted five-block MLP scale overlay improves the complete
packed quality result at unchanged bytes and effective BPW. The next work asks
whether the remaining refit capacity can be used more effectively by fitting
and optimizing it in the context in which the compressed model actually runs.

This plan responds to the independent review in
[70-tuned-mlp-scale-placement-analysis.md](70-tuned-mlp-scale-placement-analysis.md).
The review identifies a real mismatch: the current refits are fitted using
dense-teacher MLP inputs but deployed on hidden states produced by the composed
student. It does not, by itself, prove that this mismatch causes the observed
placement behavior. The first experiment therefore distinguishes that
hypothesis from local-objective mismatch, aggressive refits, and downstream
interaction before committing to a larger optimizer change.

The incumbent remains the packed Experiment 022 model plus the accepted policy:

```text
0:output, 17:joint, 18:joint, 23:joint, 24:joint
```

Its protocol-matched WikiText perplexity is `216.241794472612`, its packed
weight payload is `89,480,656` bytes, and its effective BPW is
`1.0244947117998688`. No new candidate replaces it without fresh functional
confirmation, exact folding/export validation, and the complete retained
quality benchmark.

## Questions

1. Does fitting on composed-student inputs improve the transfer of a local MLP
   refit into composed-model NLL and KL?
2. Is it better to recover the dense MLP function at the student's actual
   input, or to correct the student's accumulated hidden-state error at the
   block output?
3. Can the accepted five placements improve further when each is re-fitted
   with the other four already installed?
4. Can all refit-eligible MLP multipliers be optimized under the composed
   language objective without placement search or additional representation
   bytes?
5. Do any gains survive BF16 folding, logical and packed export, and the full
   retained quality benchmark?

## Hypotheses

### H1: input-context mismatch is causal

For middle and late blocks, a refit fitted on the student's actual MLP input
will transfer better than the current teacher-input refit. This hypothesis is
supported only if the student-context arm improves fresh composed NLL relative
to both the frozen baseline and the corresponding teacher-context arm.

### H2: state recovery is more useful than function recovery

The compressed block receives a residual stream that already differs from the
teacher. Matching only the dense MLP function need not correct that difference.
A state-recovery target may transfer better because it asks the student MLP to
produce

```text
teacher block output - student post-attention residual
```

so that the student's residual addition approaches the teacher block output.
This target is diagnostic rather than assumed correct: forcing an intermediate
teacher state can disrupt downstream co-adaptation and must pass the composed
functional gate.

### H3: coordinate re-fitting improves the incumbent

The five accepted refits were fitted independently. Re-fitting them in forward
order while the other accepted changes are installed should remove part of
that inconsistency without adding bits.

### H4: globally optimized foldable multipliers improve on greedy placement

The four positive multiplier families expose 21,888 variables per MLP block:

- 6,912 gate output multipliers;
- 6,912 up output multipliers;
- 6,912 down input multipliers;
- 1,152 down output multipliers.

Across 26 blocks this is 569,088 scalar variables. A composed-objective
optimizer may keep unhelpful blocks near identity and coordinate useful
changes across blocks.

This is not ordinary raw scale-only tuning. Experiment 022's global KD already
trained stored pre/mid/post scales jointly with outliers, patches, biases, and
norms. The new parameterization must instead train explicit positive
row/column multipliers and fold each multiplier covariantly through the
existing factor, floating-outlier, and correction-patch terms. That preserves
the exact separable transformation already validated by
`rescale_factorized_terms` and keeps every stored shape, dtype, and byte count
unchanged.

## Phase A: context ablation

### Fixed candidates

The diagnostic set intentionally includes successful, inconclusive, and
harmful prior placements:

| Block | Existing policy | Prior role |
| ---: | --- | --- |
| 0 | output | accepted; minimal upstream drift |
| 18 | joint | accepted; later block |
| 23 | joint | accepted marginal addition |
| 4 | joint | clearly harmful marginal candidate |
| 19 | joint | negative but inconclusive marginal estimate |
| 21 | joint | near-zero/inconclusive marginal estimate |
| 25 | joint | negative but inconclusive marginal estimate |

Before Phase B, blocks 17 and 24 receive the same context comparison on these
already-fixed windows because they are members of the incumbent coordinate
sweep but were not needed to discriminate the original representative-block
hypothesis. Their results may choose only their Phase B fitting context; they
cannot retroactively change the Phase A candidate inventory or gates.

The policy is held fixed within a block so the experiment tests context rather
than conducting another policy search.

### Context arms

Each block has three predeclared arms:

1. **Teacher function:** reproduce the existing method by capturing the dense
   teacher's MLP input and fitting toward its dense gate/up/MLP outputs.
2. **Student-input function recovery:** capture the current composed student's
   actual MLP input and evaluate both the dense MLP weights and compressed MLP
   weights on that same input. This isolates operator error at the deployed
   operating point without attempting to repair upstream hidden-state drift.
3. **Student-input state recovery:** use the same student MLP input, retain the
   function-recovery gate/up targets, and fit the final MLP output toward the
   teacher block output minus the student's post-attention residual. This asks
   the residual block output to recover the teacher state. On architectures
   such as Gemma 3 that normalize the MLP branch after `down_proj`, project
   this desired residual contribution back through the student's
   post-feed-forward RMS-normalization direction before fitting, and evaluate
   the resulting candidate only after reapplying that normalization. A raw MLP
   output must never be added directly to the residual on those blocks.

Teacher and student values must be paired by exact token position. Hooked
values are captured in BF16 to match deployment, while scale solves and metric
accumulation use FP32. Padding positions are excluded consistently from every
arm.

### Fresh data

- Fit: WikiText sequences 380-383, length 512.
- Local validation: sequences 384-387, length 512.
- Composed functional screen: sequences 388-411, 24x512.
- Fresh confirmation: sequences 412-459, 48x512.

The exact retained 64x128 benchmark and all six retained task inventories are
forbidden as selection data.

### Recorded diagnostics

For every block and arm record:

- local fit and validation normalized RMSE;
- composed NLL and teacher KL per sequence;
- paired arm-minus-baseline and arm-minus-teacher-context intervals;
- multiplier minimum, maximum, median, and quantiles;
- exact counts and fractions at each bound;
- hidden-state RMSE at MLP input, post-attention residual, MLP output, and block
  output;
- whether improvement comes from function recovery or cancellation of upstream
  state error.

Local RMSE is diagnostic only. It cannot nominate or accept a deployed
candidate.

### Gate

Within each block, local validation chooses at most one student-context arm for
the functional screen. A context method advances only when:

1. screen NLL is lower than the frozen comparison base;
2. its paired 95% upper interval is below zero;
3. KL does not show a statistically supported regression; and
4. the same direction passes the untouched 48-sequence confirmation.

Inconclusive results remain inconclusive; they are not labeled harmful or
helpful from their point estimate alone. The analysis will report NLL and KL
even when they disagree.

## Phase B: composed-context coordinate sweep

Starting from the incumbent five-block overlay, re-fit blocks in forward order:

```text
0, 17, 18, 23, 24
```

Use the winning Phase A context target. Keep all other accepted refits installed
while fitting each coordinate. Run at most two complete sweeps; retain a
coordinate update only when it improves the sweep validation inventory. A
second sweep is allowed only if the first sweep improves validation NLL.

Fresh inventories:

- sweep fit: sequences 460-467, 8x512;
- sweep validation: sequences 468-475, 8x512;
- functional screen: sequences 476-499, 24x512;
- confirmation: sequences 500-547, 48x512.

The candidate must improve confirmation NLL with a paired 95% upper interval
below zero relative to the incumbent five-block overlay. KL, multiplier-bound
fractions, and hidden-state diagnostics remain required secondary checks.

### Confirmed-addition composition

Phase A additions that independently pass both its 24-sequence screen and
48-sequence confirmation are composed with the accepted coordinate-sweep base
before global multiplier tuning. The prefix is fixed before examining these
new windows:

1. coordinate base plus block 25 student-function refit;
2. previous prefix plus block 4 student-function refit;
3. previous prefix plus block 21 teacher-function refit.

The WikiText test stream is exhausted at this context length, so this gate uses
the separately pinned WikiText validation split: screen sequences 0-23
(24x512) and confirm on sequences 24-71 (48x512). Advance only the longest
prefix for which every marginal addition improves paired NLL with its 95% upper
interval below zero and does not have a statistically supported KL regression.
These validation-split windows were not used to fit, rank, or confirm either
the Phase A additions or coordinate base. The final retained benchmark remains
on the original test-split protocol and is not used for selection.

## Phase C: globally tuned foldable multipliers

### Parameterization

- Freeze binary factors, stored scales, outliers, patches, biases, norms, and
  every non-MLP parameter.
- Introduce FP32 log-multipliers for the four eligible multiplier families.
- Materialize positive multipliers in the forward pass.
- Apply them to all represented terms exactly as
  `rescale_factorized_terms` does, including outlier rows/columns and the two
  sides of a correction patch.
- Fold the selected multiplier state into the original BF16 components before
  evaluation. Unfolded FP32 multipliers are never part of the representation.

This avoids the scale/outlier mismatch that raw `scale_pre`/`scale_post`
training would create and avoids the redundant pre/mid/post factor-scale
degrees of freedom in the existing global KD parameterization.

### Initializations

Two predeclared initializations are compared:

1. identity multipliers on the Experiment 022 post-KD state;
2. the incumbent five-block multipliers with identity elsewhere.

If Phase B passes, its coordinate-refitted state replaces initialization 2.

### Objective and regularization

- Primary training loss: the existing global top-k teacher objective on the
  existing KD training inventory, initially with its retained top-k setting.
- Identity penalty: mean squared log-multiplier, measured per family so the
  largest vectors do not dominate solely by parameter count.
- Held-out monitoring: both causal NLL and teacher KL on data excluded from
  optimization.
- Optimization state: FP32 moments/master multipliers, deployment-faithful
  factorized forward, gradient clipping, and finite-value checks.
- Early stopping: select the earliest checkpoint within one standard error of
  the best held-out NLL, provided held-out KL has not significantly regressed.

The first run is a short learning-rate and regularization stability probe. A
full run is launched only after confirming finite gradients, non-degenerate
multiplier distributions, and exact BF16 folding replay.

### Required comparisons

- Experiment 022 post-KD baseline;
- incumbent five-block overlay;
- best Phase B candidate, if different;
- global multiplier candidate before folding;
- saved BF16 component replay after folding.

The global candidate advances only if its folded replay improves fresh paired
NLL over the incumbent and its folded-versus-unfolded difference is negligible
on the same inventory.

## Regularization follow-ups

If context-correct candidates remain aggressive, test these in order without
changing the selection inventories:

1. log-space ridge toward identity;
2. per-family damping selected on sweep validation;
3. channel cross-fit acceptance using disjoint halves of the fit and local
   validation inventories;
4. a downstream-sensitivity-weighted local objective using existing diagonal
   Fisher machinery.

Hard clamp limits remain safety bounds, not the primary regularizer. Future
evidence must report bound-hit fractions rather than inferring saturation from
only minima and maxima.

## Export and completion gates

Any candidate that passes a fresh functional confirmation must complete all of
the following:

1. fold multipliers into existing components with zero changed shapes or
   dtypes;
2. prove identical replacement and source payload bytes;
3. save a hash-bound component overlay;
4. replay the saved components and compare them with the in-memory candidate;
5. export and freshly validate the complete logical model;
6. export and exactly validate the packed model;
7. prove packed weight bytes and effective BPW do not increase;
8. run the complete retained quality protocol: WikiText 64x128 plus 200
   examples each of PIQA, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, and
   BoolQ;
9. compare task input identities, prompt hashes, token hash, BF16 baseline, and
   packed descriptor identity with the retained protocol.

The work is not complete from a local RMSE gain, a short functional probe, an
unfolded training result, or WikiText alone.

## Decision interpretation

- If student-input function recovery wins, deployed input distribution was a
  material limitation and becomes the default local fitting context.
- If state recovery wins, upstream drift is exploitable but intermediate-state
  correction must still be treated as a composed-model intervention.
- If neither student-context arm beats teacher context, the review's root-cause
  hypothesis is rejected for these candidates and effort moves to global-loss
  alignment and regularization.
- If coordinate sweeps improve the incumbent, placement interaction is real
  and independent one-shot fitting is retired.
- If global foldable-multiplier tuning wins, it becomes a general post-KD stage
  for compatible dense decoder MLP adapters, subject to per-model held-out
  selection and full quality validation.
- If no phase improves the incumbent, retain the five-block artifact and record
  the negative result rather than weakening the gate.

## Execution results through Phase B

The Phase A and Phase B work was executed on 2026-07-31 using the inventories
declared above. The main findings are:

- Student context is materially useful but not universally better. It wins
  composed NLL for blocks 17, 23, 25, and 4; teacher context remains better for
  blocks 21 and 24. Block 18's state arm improves KL more, but its NLL advantage
  over teacher context does not confirm.
- Block 0 refitted on the new inventory regresses, so its incumbent refit is
  retained rather than replaced.
- The independently fitted block-25 candidate is the dominant new result. On
  the 48x512 Phase A confirmation it improves the post-KD baseline by
  `-0.438717` NLL and `-0.633946` KL.

The incumbent coordinate sweep accepted block 18 in both sweeps and block 23
in the first. It reduced its private validation NLL from `4.520325` to
`4.513504`. Against the original five-block incumbent it then passed:

| Inventory | NLL delta | Paired 95% interval | KL delta | Paired 95% interval |
| --- | ---: | --- | ---: | --- |
| Test sequences 476-499, 24x512 | -0.007382 | [-0.010611, -0.003870] | not measured | — |
| Test sequences 500-547, 48x512 | -0.008891 | [-0.011612, -0.006052] | -0.010181 | [-0.013464, -0.006859] |

On the separately pinned validation split, adding block 25 to that coordinate
base passed both inventories:

| Inventory | NLL delta | Paired 95% interval | KL delta | Paired 95% interval |
| --- | ---: | --- | ---: | --- |
| Validation sequences 0-23, 24x512 | -0.194598 | [-0.226037, -0.164420] | -0.386060 | [-0.421513, -0.351211] |
| Validation sequences 24-71, 48x512 | -0.216572 | [-0.237944, -0.195221] | -0.411620 | [-0.436082, -0.387885] |

Adding block 4 after block 25 regressed screen NLL by `+0.009191`, despite a
small KL improvement, so the fixed prefix stopped at block 25. Adding block 21
also regressed both metrics.

### Representation and complete benchmark

The accepted six-block state is:

```text
0:output, 17:joint, 18:joint, 23:joint, 24:joint, 25:joint
```

Blocks 18 and 23 contain the accepted composed-context coordinate updates;
block 25 is the confirmed student-function addition. Folding this state into
the existing factor terms replaces 54 tensors and `718,848` bytes with exactly
`718,848` bytes. Its component SHA-256 is
`22340b12d8db4d77bb4b574bfe2c82d2dd50e208d06cd71d79a3d5b03c4aa2cb`.
The saved factor replay differs from the dense candidate by only `+0.000041`
NLL, with interval `[-0.001417, +0.001453]`.

Logical and packed export are exact:

- 26 blocks, 130 layers, and 910 logical tensors;
- logical weight bytes: `2,739,492,456`;
- packed weight bytes: `89,480,656`, unchanged from Experiment 022 and the
  prior incumbent;
- exact logical-to-packed conversion;
- zero maximum reference error across 459,264 output elements;
- packed descriptor SHA-256
  `dbd38bdf61b4f8c445d65fdf1a4db567d451a18545a2525c5f2f214cb237f7d7`.

The complete packed benchmark reports:

| Benchmark | Experiment 022 | Prior five-block | New six-block | Change vs prior |
| --- | ---: | ---: | ---: | ---: |
| WikiText perplexity | 228.550618 | 216.241794 | **172.377804** | **-20.285%** |
| PIQA `acc_norm` | 0.605 | **0.630** | 0.605 | -0.025 |
| ARC-Easy `acc_norm` | 0.380 | 0.370 | **0.395** | +0.025 |
| ARC-Challenge `acc_norm` | 0.215 | 0.230 | **0.250** | +0.020 |
| HellaSwag `acc_norm` | 0.460 | **0.485** | 0.440 | -0.045 |
| WinoGrande `acc` | 0.520 | 0.525 | **0.545** | +0.020 |
| BoolQ `acc` | 0.635 | **0.640** | 0.630 | -0.010 |
| Mean primary task score | 0.469167 | **0.480000** | 0.477500 | -0.002500 |

The new model is a large perplexity improvement at identical bytes, and its
mean task score remains `+0.008333` above Experiment 022. It is not an
unqualified task improvement over the five-block incumbent: three tasks
improve, three regress, and the net primary-metric count is three lower out of
1,200 examples. Phase C should therefore monitor task-relevant held-out
behavior in addition to NLL/KL and must not assume that further perplexity gains
imply uniformly better downstream accuracy.

Evidence:

- `experiment022-postkd-context-ablation-fit380-val384-screen388-24x512-v2.json`
- `experiment022-postkd-context-ablation-fit380-val384-confirm412-48x512.json`
- `experiment022-postkd-incumbent-composed-context-coordinate-sweep.json`
- `phaseB-coordinate-confirm500-48x512-kl.json`
- `confirmed-additions-validation-screen0-24x512-kl.json`
- `confirmed-additions-validation-confirm24-48x512-kl.json`
- `experiment022-postkd-coordinate-plus25-factor-compatible-direct64x128.json`
- `experiment022-postkd-coordinate-plus25-logical-validation.json`
- `experiment022-postkd-coordinate-plus25-packed-validation.json`
- `experiment022-postkd-coordinate-plus25-packed-quality.json`

## Phase C execution results

Phase C was executed against Experiment 022's original retained eight-epoch,
411,041,792-byte top-k teacher cache. The six-block component overlay above was
the initialization, and every raw factor, stored scale, outlier, patch, bias,
norm, attention parameter, and non-MLP parameter remained frozen. The only
optimizer variables were the 569,088 FP32 log-multipliers declared in this
plan.

### Stability correction and gradient gate

The first smoke implementation multiplied BF16 activations by near-identity
multipliers. That exposed a BF16 dead zone: the FP32 master values changed, but
the cast activation multipliers still rounded to one, while subsequently
folding the same values into BF16 component scales changed the model. This was
not an acceptable deployment-faithful training path.

The corrected forward dynamically materializes the rescaled BF16 factorized
terms, including floating outliers and both correction-patch sides, before the
factorized linear. It therefore executes the exact representation that folding
will save. The repeated smoke then established:

- all 26 parameter tensors in each of the four multiplier families received
  finite, nonzero gradients at both the first and final checked steps;
- no multiplier approached its safety bound;
- eight steps improved held-out NLL by `-0.001962`, interval
  `[-0.003570, -0.000200]`;
- the saved folded replay and unfolded optimization forward had exactly zero
  NLL difference.

### Objective mismatch and checkpoint selection

A full 2,048-step run used all eight retained teacher-cache epochs and monitored
fresh validation sequences 72-79 every 64 steps. Teacher top-k KL continued to
improve throughout training, but causal NLL was best after 64 steps and then
steadily regressed. This is direct evidence that the retained top-k objective
is not a sufficient checkpoint-selection metric for these multipliers.

The full-run step-64 state improved NLL on validation sequences 80-103 by
`-0.039404`, but full-vocabulary teacher KL regressed by `+0.033411`, interval
`[+0.023689, +0.042220]`, so it was rejected. A shorter cosine-decayed
64-step sweep retained the NLL gain with less movement. The `3e-4` candidate
still showed a confirmed full-KL regression on validation sequences 104-151
and was also rejected.

The accepted setting is the conservative `1e-4` learning-rate, family-balanced
identity penalty `100`, 64-step cosine-decayed candidate. Relative to the
six-block incumbent it produced:

| Inventory | NLL delta | Paired 95% interval | KL delta | Paired 95% interval |
| --- | ---: | --- | ---: | --- |
| Validation sequences 80-103, 24x512 | -0.013079 | [-0.015221, -0.011021] | -0.001261 | [-0.002838, +0.000253] |
| Validation sequences 104-151, 48x512 | -0.011699 | [-0.012882, -0.010540] | +0.000883 | [-0.000123, +0.001898] |

The NLL improvement confirms cleanly. The confirmation KL point estimate is a
small regression, but its interval crosses zero, so it is not a statistically
supported regression under the predeclared gate. The multiplier range remains
very conservative: `0.996706` to `1.003299`, with every family median within
`0.00003` of one and zero bound hits.

### Folded representation and complete benchmark

The accepted state folds into all 78 MLP projections. Its component overlay
replaces 234 existing tensors and `3,115,008` bytes with exactly `3,115,008`
bytes. Shapes, dtypes, packed bytes, and effective BPW are unchanged. The
component tensor SHA-256 is
`95902baa448c1ca153fd11db8f367bfced13e6e6d6e190fa39186cf8195e739e`.

Fresh logical and packed validation reports:

- 26 blocks, 130 layers, and 910 logical tensors;
- logical weight bytes: `2,739,492,456`;
- packed weight bytes: `89,480,656`;
- effective BPW: `1.0244947118`, unchanged;
- exact logical-to-packed conversion;
- zero maximum reference error across 459,264 outputs;
- packed descriptor SHA-256
  `b1ec7016d305296c84e95e88041094ce2967b27f43fc5753fae81e2a504439a4`.

The complete retained packed benchmark passed and reproduced the BF16 baseline,
WikiText token hash, task inputs, and prompts:

| Benchmark | Experiment 022 | Six-block incumbent | Phase C | Change vs six-block |
| --- | ---: | ---: | ---: | ---: |
| WikiText perplexity | 228.550618 | 172.377804 | **169.481866** | **-1.680%** |
| PIQA `acc_norm` | 0.605 | 0.605 | 0.600 | -0.005 |
| ARC-Easy `acc_norm` | 0.380 | 0.395 | 0.395 | 0.000 |
| ARC-Challenge `acc_norm` | 0.215 | 0.250 | 0.240 | -0.010 |
| HellaSwag `acc_norm` | 0.460 | 0.440 | 0.445 | +0.005 |
| WinoGrande `acc` | 0.520 | 0.545 | **0.570** | +0.025 |
| BoolQ `acc` | 0.635 | 0.630 | 0.625 | -0.005 |
| Mean primary task score | 0.469167 | 0.477500 | **0.479167** | +0.001667 |

Phase C lowers perplexity by 25.845% relative to Experiment 022 and by 1.680%
relative to the already much stronger six-block model at identical packed
bytes. Its primary task count is a net two correct answers higher than the
six-block incumbent across 1,200 examples. Individual task movements remain
too small to interpret strongly, but the complete benchmark rules out a broad
task-quality tradeoff and accepts this conservative Phase C state as the new
candidate.

Phase C evidence:

- `phaseC-global-multiplier-smoke-v2.json`
- `phaseC-sweep-lr1e4-reg100.json`
- `phaseC-global-multiplier-full.json`
- `phaseC-global-multiplier-screen80-candidates-24x512-kl.json`
- `phaseC-global-multiplier-confirm104-48x512-kl.json`
- `phaseC-global-multiplier-lr1e4-confirm104-48x512-kl.json`
- `experiment022-postkd-coordinate-plus25-phaseC-lr1e4-logical-validation.json`
- `experiment022-postkd-coordinate-plus25-phaseC-lr1e4-packed-validation.json`
- `experiment022-postkd-coordinate-plus25-phaseC-lr1e4-packed-quality.json`

## Experiment 035 production-integration result

Experiment 035 moved the foldable-multiplier method into the canonical complete
compression workflow. The stage now runs after global top-k distillation and
before logical export, checkpoints every 16 steps, writes a hash-bound active
component state, and is mandatory input to fresh logical validation whenever
enabled. The complete numbered workflow ran a fresh D2 campaign, all 26 block
commits, global KD, the 64-step continuation, exact logical and packed export,
GGUF conversion, and the full retained quality protocol.

The integration contract passed:

- fresh resident validation audited 708 transitive artifacts and all 26 blocks;
- 64/64 multiplier steps completed with complete finite gradient coverage;
- folded replay maximum absolute error was zero;
- 234 tensors and `3,115,008` bytes replaced the same byte count;
- logical and packed exports were exact;
- effective BPW remained `1.0244947118`;
- the full 64x128 plus six-task benchmark completed and passed its runtime
  validity checks.

Identity initialization did **not** pass the numerical advancement gate. On the
untouched validation 104-151 confirmation, folded minus same-run post-KD NLL was
`+0.000173`, interval `[-0.000564,+0.000881]`, and full-vocabulary KL was
`+0.0000368`, interval `[-0.000550,+0.000605]`. The retained 128-token benchmark
did move slightly (`221.035336` to `220.879050` perplexity), and mean task score
moved `0.472500` to `0.474167`, but these small post-hoc gains do not override
the predeclared held-out NLL gate.

Relative to Experiment 022, Experiment 035 improves perplexity by 3.357% and
mean task score by 0.005000. The same-run ablation shows that most of this gain
comes from the fresh D2 campaign, not the multiplier continuation. Its packed
payload is also eight bytes above Experiment 022 (`89,480,664` versus
`89,480,656`) despite identical effective BPW and GGUF size. The foldable stage
itself remains byte-neutral; the eight-byte difference is upstream packing
variation from the fresh campaign.

Decision: the production integration is accepted, but identity-only numerical
policy is rejected. The confirmed six-block composed-context initialization
remains essential: its Phase C perplexity is `169.481866`, far ahead of the
identity-initialized `220.879050`. The next production experiment should
integrate that composed-context coordinate initializer before running the
conservative global continuation.

Experiment 035 evidence is documented in
`Docs/Experiments/035-foldable-mlp-d2-compression.md` and retained under
`evidence/035`, `outputs/035`, and `Results/035`.
