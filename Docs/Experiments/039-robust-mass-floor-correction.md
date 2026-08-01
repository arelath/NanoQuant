# Experiment 039: Robust Conditional Mass-Floor Correction

## Status

Complete and rejected on the strict paired C4-NLL gate. Experiment 038 is
closed separately; this protocol used a new untouched confirmation slice.

## Rationale

Experiment 038 proved that a short one-sided correction can improve NLL and
KL while moving the conditional endpoint toward a selected-mass constraint.
Its fit-selected checkpoint reached mass 0.76165, but only 0.73890 on the
previously untouched broad slice. The failure was a distribution-generalizing
margin, not a reversal of the distributional improvements.

Experiment 039 treats validation offset 104 as development evidence and does
not reuse it for acceptance. The relative training floor increases from 0.80
to 0.825. On the designated fit monitor this corresponds to mass about 0.7824,
providing approximately the 2.3 percentage points of headroom that Experiment
038 lacked. The loss remains zero above the batch floor and the candidate is
still warm-started from the same conditional global-tuning artifact.

## Fixed protocol

- initializer: Experiment 037 conditional global tuning
  `sha256-081190fcf25f0852c089c0d3265fcdfbd86d759eea1bc396d55ec14c14dfb134`;
- objective: normalized conditional top-k CE plus the one-sided batch
  log-odds mass deficit;
- teacher-mass ratio: 0.825;
- initial deficit coefficient: 2.0;
- at most four epochs of 32 batches, learning rate 1e-5, with durable epoch
  checkpoints;
- selection: first checkpoint meeting the 0.825 relative floor on the same
  validation-offset-56 16x512 fit monitor while improving fit NLL and full KL
  over the initializer;
- untouched WikiText confirmation: validation offset 200, 48x512, token hash
  `sha256:041f131775d952786366cb06021ffa799fa5ece70c45174d29673e05e1006a22`;
- pinned C4 remains a later confirmation only.

If coefficient 2.0 does not produce a fit survivor, coefficient changes remain
fit-only development. Once a checkpoint is selected, neither its ratio,
coefficient, epoch, nor learning rate may change in response to offset 200 or
C4.

## Gates

The selected checkpoint must reach absolute mass at least 0.75 and improve
NLL, full KL, and tail KL over the conditional initializer on offset 200.
Only then proceed to the complete task benchmark and pinned C4 comparison.
Acceptance still requires no task regression versus conditional KD, paired C4
NLL/KL improvement, unchanged BPW, ordinary materialization and validation,
and the complete export contract.

## Result

Coefficient 2.0 did not meet the fit floor. Fit-only searches at 4.0, 6.0,
7.0, and 8.0 bracketed the narrow region where mass and NLL could both pass.
Coefficient 7.2 produced the first qualifying epoch-1 checkpoint:

| Fit state | NLL | Full KL | Tail KL | Mass | Target |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conditional initializer | 4.30468 | 1.69310 | 1.63195 | 0.50659 | 0.78241 |
| Weight 7.2, epoch 1 | **4.28347** | **1.37214** | **1.31014** | **0.78384** | 0.78241 |

The selected checkpoint is
`sha256-4789b18325e81c64134d777e2a2565ac81fc666a60e7a950cb65237bb4d54d0c`.
On the untouched validation-offset-200 48x512 slice it reached mass 0.75672.
Against conditional KD it improved NLL by 0.069, full KL by 0.375, and tail
KL by 0.375. The confirmation token hash exactly matched the predeclared
`sha256:041f...6a22` inventory.

The complete 64x128 WikiText plus six-task/200 benchmark gave PPL 186.74
versus 187.52 for conditional KD, and task mean 0.46667 versus 0.47583. As in
Experiment 037, the larger 1,000-example diagnostic materially reduced the
task difference: candidate mean 0.45133 versus 0.45317, delta -0.00183. A
paired, task-stratified 10,000-resample comparison produced interval
[-0.00883, +0.00533], so it did not establish either regression or
improvement.

The checkpoint was materialized through the ordinary global-tuning path as
artifact
`sha256-114224e782489187e6772885e303a1c6afd95443cb56b4915f1719ff2bbe1e64`.
Fresh validation covered 708 resident artifacts and all 26 blocks at unchanged
effective BPW 1.024494712.

On pinned C4 48x512, candidate NLL was 4.94724 versus 4.95102 for conditional
KD. The point delta was beneficial (-0.00378), but its paired 95% interval
[-0.02806, +0.02047] crossed zero. KL improved from 1.28124 to 1.13457, delta
-0.14667 with interval [-0.17134, -0.12257]. The strict gate requires both NLL
and KL to improve with confidence, so Experiment 039 is rejected and no GGUF
is published.

This is still a better failure than the Experiment 037 calibrated candidate:
the mass floor holds, task quality is statistically unresolved rather than
clearly displaced, and C4 NLL is neutral rather than significantly harmful.
The next experiment should lower trained mass pressure and use only the small
fold needed to bridge the residual mass gap, with a new untouched WikiText
slice.

Retained evidence:

- `evidence/039/experiment039-mass-floor-ratio0p825-weight7p2-correction/report.json`
- `evidence/039/experiment039-weight7p2-epoch1-validation200-48x512-kl.json`
- `evidence/039/experiment039-conditional-epoch8-validation200-48x512-kl.json`
- `evidence/039/experiment039-weight7p2-epoch1-standard-quality.json`
- `evidence/039/experiment039-weight7p2-epoch1-tasklimit1000-quality.json`
- `evidence/039/experiment039-materialized-validation.json`
- `evidence/039/experiment039-matched-c4-validation104-48x512.json`
