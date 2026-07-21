
## 4. There is also a real flaw in the proposed D2 math

Suppose you have an unpushed implementation that does use `KL_u / E_w,u²`. There is still an objective mismatch.

Docs/30’s reconstruction allocator models **raw Frobenius squared error**:

[
D_u(r)=D_{u,0}\exp(-2B_u(r)).
]

The probe intentionally measures raw squared weight error. Production activation-weighted error is recorded only as a diagnostic, and allocation explicitly uses the raw error.

The domain allocator then computes marginal value as:

[
a_u,[D_u(r)-D_u(r+\Delta r)]
]

where `a_u` comes from the sensitivity scalar.

But D2 defines sensitivity using an **activation-space** error denominator, `E_w,u²`.

Therefore, simply replacing the sensitivity scalar produces a hybrid objective resembling:

[
\frac{\mathrm{KL}*u}{E*{\text{activation},u}^2}
\cdot
D_{\text{Frobenius},u}(r),
]

not:

[
\widehat{\mathrm{KL}}_u(r).
]

Unless activation error and raw Frobenius error have the same rank-response curve for every unit, this is dimensionally and methodologically inconsistent. The anatomy study itself says their rankings invert, so that assumption is particularly unsafe.

This is the most substantive flaw in the proposed D2 methodology.

## 5. The quadratic approximation is being used far outside a clearly local regime

D2 motivates:

[
s_u = \frac{\mathrm{KL}_u}{E_u^2}
]

using a small-perturbation quadratic approximation. The design itself acknowledges that the current operating point has large error and therefore treats the conversion as approximate.

The measured operating point was not mildly perturbed:

* BF16 perplexity: `53`
* compressed perplexity: `7262`
* whole-model KL: `4.675` nats/token

At that distance from the teacher:

* logits may cross decision boundaries;
* softmax curvature varies sharply by token;
* changing rank can change the activation distribution received by later blocks;
* one coefficient estimated at the baseline error is not necessarily valid after reallocating 20–40% of a unit’s rank.

So `KL/E²` is an average coefficient at one operating point, not necessarily the **marginal KL reduction per added rank quantum** that the allocator needs.

The useful allocation quantity is closer to:

[
\frac{
\mathrm{KL}_u(r)-\mathrm{KL}_u(r+\Delta r)
}{
\operatorname{bits}_u(r+\Delta r)-\operatorname{bits}_u(r)
}.
]

That needs at least two or three measured rank points per physical unit or cohort. A single splice measurement cannot identify it.

## 6. QKV grouping makes the D2 attribution less direct

The current model has one physical QKV factor owner. One added rank changes Q, K, and V simultaneously.

Docs/30 handles this by forming a physical group and aggregating member sensitivities geometrically.

But a D1 arm such as “quantize only V” measures a logical intervention that the grouped factor format cannot actually perform. Using independent Q/K/V KL values to price one shared QKV rank quantum can therefore over- or under-value the physical action.

For grouped units, the splice profile should measure the actual physical arm:

> replace the complete grouped QKV reconstruction at a given shared rank and measure its KL.

Member-level arms are still useful diagnostics, but they should not be treated as independently purchasable allocation choices.


## What I would change

### First, make the experiment genuinely controlled

Create the candidate by cloning one exact baseline configuration and modifying only:

* the sensitivity-profile artifact;
* the allocation strategy;
* any setting intrinsically required by that strategy.

Hold constant:

* outlier count, dtype, and charge policy;
* QKV topology;
* retry configuration and actual retry spend;
* rank floors and caps;
* ADMM settings and seeds;
* tuning and distillation;
* datasets and token windows;
* final **effective BPW**, not merely `target_bpw`.

I would also add a preflight that diffs the resolved candidate and baseline configs and fails unless every changed path appears in an explicit allow-list. Right now `BaselineRef` gives experiment lineage, but no matched-control guarantee.

### Second, make the predicted error basis match the KL coefficient

There are two defensible options.

**Lighter-weight option:** measure activation-space squared error at every rank-probe point and fit `E_activation²(r)` response curves. Then use:

[
\widehat{\mathrm{KL}}_u(r)
==========================

s_u,E_{\text{activation},u}^2(r)
]

with `s_u = KL_u/E_activation,u²` applied linearly.

**Stronger option:** directly measure rank-to-KL response around the baseline:

* `r − 32`
* `r`
* `r + 32`, or another aligned quantum

Then allocate from observed or fitted `ΔKL/Δbits`. This avoids pretending Frobenius response and functional response are interchangeable.

For QKV, do this at the complete physical group level.

### Third, separate the gates

A clean rollout would use:

1. **Planner integrity:** exact equal effective BPW and expected rank movement toward `up_proj` and blocks 0–10.
2. **D1 objective:** held-out whole-model teacher KL improves.
3. **Language modeling:** matched WikiText NLL/perplexity improves on a larger token set.
4. **Downstream tasks:** evaluate with enough examples or repeated deterministic subsets to distinguish a real shift from noise.

## Bottom line

**There is a concrete implementation/experiment problem:** the linked branch is not running the D2 described in the design, and Experiment 017’s baseline relationship does not produce an isolated, equal-capacity comparison.

**There is also a methodological flaw:** the proposed D2 coefficient is derived from activation-space error but is multiplied by a raw-Frobenius rank-response model. That does not actually predict marginal KL.

And the observed regression has a very plausible mechanism already demonstrated in the repository: **under a fixed budget, reconstruction improvements in protected Q/K/V/O/down cohorts were purchased by worsening gate/up, which matter more to end quality.**

So I would not interpret the current negative benchmark as “KL-guided allocation does not work.” I would interpret it as **D2 has not yet been cleanly implemented or cleanly tested.**
