# Experiment 038: Conditional KD with a One-Sided Mass Floor

## Status

Fixture validation complete; retained Gemma correction pending. This
experiment reuses the validated Experiment 037 frozen state and conditional
checkpoint. It does not rerun factorization and does not resume Experiment
036.

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
