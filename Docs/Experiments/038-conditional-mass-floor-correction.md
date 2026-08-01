# Experiment 038: Conditional KD with a One-Sided Mass Floor

## Status

Complete; the exact candidate is rejected at the broad mass gate. This
experiment reused the validated Experiment 037 frozen state and conditional
checkpoint. It did not rerun factorization and did not resume Experiment 036.

## Motivation

Experiment 037 separated two failures that had previously been coupled:

- conditional top-k KD retained the strongest downstream task score, but its
  teacher-top-64 mass collapsed;
- top-k-plus-tail KD improved broad NLL and KL, but lost about `0.0068` task
  mean on the larger 1,000-example inventory;
- the 1.06 final-norm fold barely changed the task mean, but gave back enough
  WikiText and C4 NLL to fail the matched C4 gate.

The existing tail-aware loss is not merely conditional top-k KD plus a
constraint. Its conditional cross-entropy is weighted by the teacher's
selected probability mass, and its binary mass cross-entropy continues to
push the student toward the teacher's exact mass even after the deployment
floor is satisfied. Both effects can move a task-friendly conditional
solution unnecessarily.

## Candidate objective

Start from the completed Experiment 037 conditional checkpoint rather than
from the common pre-KD state. Preserve the original normalized conditional
top-k cross-entropy exactly. For each training batch, compute the teacher and
student probability mass assigned to the teacher's selected top-64 entries.
The target is

```text
target_mass = 0.80 * mean(teacher_top64_mass)
```

and the additional loss is

```text
mass_deficit = relu(logit(target_mass) - logit(mean(student_top64_mass)))
loss = conditional_topk_cross_entropy + weight * mass_deficit
```

Token weights, when present, apply to both the conditional loss and the two
batch means. The mass term has exactly zero value and gradient once the batch
floor is met. It therefore differs from both full binary tail cross-entropy
and a global logit scale.

The ratio is relative to the teacher instead of hard-coding a model-specific
absolute probability. On the retained broad gate the teacher's top-64 mass is
about 0.936, so a ratio of 0.80 corresponds to approximately 0.749. The
deployment acceptance floor remains the separately measured absolute value
0.75.

## Training protocol

The first probe is deliberately a correction stage:

1. load the ordinary serialized Experiment 037 conditional global-tuning
   result;
2. retain its absolute trainable factor, scale, outlier, bias, patch, and norm
   values as the correction initializer;
3. reuse the retained 256 calibration samples and deterministic teacher top-k
   selections;
4. compute the missing full-vocabulary teacher normalizers with the production
   chunked algorithm;
5. train at most four 32-batch epochs with a fresh optimizer, checkpointing
   every epoch;
6. use a disjoint WikiText-validation 16x512 fit monitor to select the first
   checkpoint that satisfies its mass floor without reversing NLL or KL;
7. leave Experiment 037's broad validation-offset-104 slice and pinned C4
   slice untouched until a checkpoint has been selected.

The initial coefficient is 0.5. The coefficient controls how rapidly the
inequality is repaired; unlike the old tail coefficient, it does not change
the objective after the floor is reached. Early checkpoint selection is the
trust region for this first probe. If the correction moves too far in a
single epoch, the next iteration will lower the learning rate or add an
explicit parameter-distance trust region rather than tuning against the final
gates.

## Implementation gates

Before the Gemma run:

- prove exact equality to conditional top-k loss and gradients whenever the
  mass floor is already satisfied;
- prove a finite, correctly directed gradient below the floor;
- cover token weighting and invalid ratio/weight inputs;
- bind objective version, floor ratio, coefficient, initializer global-tuning
  identity, calibration identity, and monitor token hash into the resumable
  checkpoint protocol;
- prove interrupted/resumed fixture execution reproduces an uninterrupted
  endpoint.

All five gates pass. The fixture verifies that thawing from a global-tuning
artifact reproduces every selected checkpoint tensor, and an interrupted
two-epoch correction commits the same final checkpoint as an uninterrupted
control.

This begins as an analysis-only objective. It should enter the canonical
configuration schema only if the retained-model result passes the quality
gates. That avoids declaring an unproven loss a general compression default.

## Acceptance sequence

The candidate must first pass the separate fit monitor. The selected durable
checkpoint must then, without further tuning:

- reach teacher-top-64 mass at least 0.75 on the broad 48x512 gate;
- improve broad NLL, full KL, and top-k-plus-tail KL over conditional KD;
- retain the conditional arm's six-task quality, with the 1,000-example
  inventory used to interpret small differences;
- improve both NLL and KL over conditional KD on the pinned C4 48x512 slice
  with paired confidence intervals;
- preserve effective BPW and represented payload bytes;
- materialize through the ordinary global-tuning artifact path, pass complete
  resident validation, and complete logical, packed, checkpoint, GGUF, and
  quality contracts before it can become a deployment candidate.

Failure on the fit monitor rejects this exact correction protocol cheaply.
Failure on either untouched final distribution does not authorize coefficient
or ratio tuning on that distribution.

## Result

The 0.5 and 1.0 coefficients failed the designated 16x512 fit monitor. The
first-epoch mass values were 0.73106 and 0.72838 respectively, below the
relative target 0.75870. Coefficient 2.0 produced the first survivor at epoch
1:

| Fit state | NLL | Full KL | Tail KL | Student top-64 mass |
| --- | ---: | ---: | ---: | ---: |
| Conditional initializer | 4.30468 | 1.69310 | 1.63195 | 0.50659 |
| Weight 2.0, epoch 1 | **4.19592** | **1.34764** | **1.28830** | **0.76165** |
| Relative target | - | - | - | 0.75870 |

Later checkpoints fell back below the floor, so the predeclared first-survivor
rule selected the durable epoch-1 checkpoint
`sha256-56dc4b560ed2c73965a02b5a4bb50945aa05fa6f62763171e2f486e207bbfa24`.

On the untouched validation-offset-104 48x512 slice, the selected checkpoint
improved conditional NLL from 4.47838 to 4.34548, full KL from 1.66403 to
1.28348, and tail KL from 1.59103 to 1.21332. Its mass generalized only from
0.76165 to 0.73890, however, and therefore failed the absolute 0.75 gate.
The token hash is the retained Experiment 037 value
`sha256:983ca15101666bef50ef4c1ccd44670a032e865e8f85230f942b08acc01e1b3d`.
The Hugging Face in-memory dataset fingerprint changed between the two runs,
but a fresh two-process audit reproduced that exact token hash; the evaluated
token inventory did not change.

No task, C4, materialization, or export gate was run for this rejected
checkpoint. A follow-up must be a newly declared protocol with a fit-only
generalization margin and a new untouched WikiText confirmation slice; it
cannot retune this candidate against offset 104.

Retained evidence:

- `evidence/038-source-validation.json`
- `evidence/038/experiment038-mass-floor-ratio0p8-weight0p5-correction/report.json`
- `evidence/038/experiment038-mass-floor-ratio0p8-weight1p0-correction/report.json`
- `evidence/038/experiment038-mass-floor-ratio0p8-weight2p0-correction/report.json`
- `evidence/038/experiment038-weight2-epoch1-validation104-48x512-kl.json`
