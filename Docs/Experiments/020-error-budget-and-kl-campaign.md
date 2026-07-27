# Experiment 020: Error-budget and KL campaign

## Status

**Multi-arm diagnostic campaign.** Some arms completed and others were rejected or aborted; there was no single
canonical launcher or publishable model.

- Primary findings: [D2 findings](../ImprovementSuggestions/D2-findings.md)
- Synthesis: [error-budget-driven quality improvements](../33-error-budget-driven-quality-improvements.md)
- Retained evidence: [`evidence/020`](../../evidence/020/)

## Question

Which local error-budget signals and correction mechanisms actually predict held-out model quality at an equal bit
budget?

## What we did

We tested corrected D2/KL rank benefit, interaction/trust limits, bias correction, value-objective weighting, and
low-rank residual patches. We used splice tests before paying for full recompression and progressively tightened the
probe protocol.

## Results

- The old D2 implementation mixed absolute and normalized weighted squared error. Correcting and versioning the
  semantics made stale profiles fail closed.
- A 12-by-512-token probe was too noisy. On 48-by-512 tokens, corrected D2 improved KL from **2.81752** to **2.73799**
  (**-2.823%**, 95% CI **[-4.287%, -1.393%]**).
- A 0.25-trust candidate used **1.022967 BPW**, below Experiment 016's **1.025280**, but matched static Wiki NLL
  regressed from **7.07910** to **7.17178** (**+1.309%**).
- All-unit bias correction failed an `o_proj` splice: KL rose from **1.65693** to **2.01585** (**+21.66%**).
- A rank-4 residual patch improved local `o_proj` KL by **33.83%**, but its equal-budget model regressed KL from
  **1.40077** to **1.60032** (**+14.25%**).
- A preliminary doubled value objective looked promising, but regenerated Fisher inputs confounded the comparison.

## What we learned

Probe semantics, sample size, confidence intervals, and identical calibration inputs are part of the algorithm.
Corrected D2 had a real signal worth promoting to full experiments. Bias correction and residual patches could work
locally while failing globally or at equal cost. Cheap splice gates prevented several expensive but unpromising full
runs.

## Disposition

Promoted corrected exact-unit D2 to Experiments 021 and 022. Rejected global bias correction and the unfunded residual
patch; retained objective weighting as a hypothesis requiring controlled evaluation.
