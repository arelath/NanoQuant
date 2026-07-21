Taking **residual error** to mean the deployed approximation error

[
E = W-\widehat W,
]

the best compensation strategy for this implementation is:

> **Reallocate rank using functional/KL impact first, then remove the residual mean with an additive bias, then fit a tiny real-valued low-rank patch only where the centered residual is strongly structured—starting with `o_proj`.**

I would not make the existing residual-column outlier path much larger, and I would not prioritize GPTQ-style propagated-error calibration yet.

## 1. Add closed-form output-bias correction first

This is the cleanest residual compensation available: tiny storage cost, no extra matrix multiply, and your own measurements show that the residual has a meaningful constant component. For `o_proj`, the mean component accounts for about **22% of functional error energy on average and up to 33%** in the measured deep block; other projections still show smaller gains.

For a PyTorch-style linear layer,

[
y=xW^\top+b,
]

compute:

[
b_{\text{corr}}
= \mathbb E[x],(W-\widehat W_{\text{full}})^\top.
]

Then deploy:

[
\widehat y
=x\widehat W_{\text{full}}^\top+b_{\text{original}}+b_{\text{corr}}.
]

A minimal implementation:

```python
import torch


@torch.no_grad()
def fit_output_bias_correction(
    target_weight: torch.Tensor,
    deployed_weight: torch.Tensor,
    input_mean: torch.Tensor,
    original_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fit an additive output bias that zeroes the mean calibration error.

    Assumes PyTorch linear convention:
        output = input @ weight.T + bias
    """
    target = target_weight.float()
    deployed = deployed_weight.float()
    mean = input_mean.float()

    if target.shape != deployed.shape:
        raise ValueError("target and deployed weights must have the same shape")
    if mean.shape != (target.shape[1],):
        raise ValueError("input_mean must have shape (in_features,)")

    correction = mean @ (target - deployed).T

    if original_bias is not None:
        correction = correction + original_bias.float()

    return correction.to(deployed_weight.dtype).contiguous()
```

There is a small but important **sign trap** in the design notes. The anatomy document defines (\Delta W=\widehat W-W).  Therefore, an additive correction must be

[
-\mathbb E[x]\Delta W^\top
=\mathbb E[x](W-\widehat W)^\top,
]

not (+\mathbb E[x]\Delta W^\top). Otherwise the mean error is doubled rather than canceled.

### Where it should live

Add a `BiasCorrectionStage` immediately after `ScaleFitStage`. The scale-fit stage is already the last reconstruction-changing stage and can roll back a regression.

Crucially, `deployed_weight` must be the **exact final reconstruction**, including:

* binary factors;
* fitted pre/mid/post scales;
* restored and dequantized outlier columns;
* any other side correction already accepted.

Do not compute the bias against just the zeroed `residual_weight`, or the bias will partially compensate error that the outlier path already restores.

Accumulate `input_mean` during calibration using valid, non-padding tokens. Use an unclipped FP32 or FP64 running sum and an integer count; second-moment clipping rules should not be reused for the mean.

Your internal packed representation already has a separate additive bias field and validates its shape and storage dtype.  Global distillation also already selects `bias` alongside scales and outlier values.  The remaining deployment check is that the modified llama.cpp graph actually **adds** this tensor for normally bias-free Gemma linears, rather than merely serializing it.

## 2. Fit a low-rank patch to the centered residual

After removing the mean, the remaining residual is not white noise. Your measurements show it is strongly low-rank **in activation space**: a rank-16 real-valued correction removes roughly 25–54% of functional error energy, and `o_proj` is especially favorable—rank 4 removed as much as 46% in one measured block.

Use a side path

[
\widehat y
= x\widehat W^\top

* (xB^\top)A^\top
* b,
  ]

where:

* (A\in\mathbb R^{m\times r}),
* (B\in\mathbb R^{r\times n}),
* (r\in{4,8,16}).

That is essentially a frozen LoRA-style residual patch, fitted directly rather than trained.

### Fit it under the activation metric

Let

[
\mu = \mathbb E[x],\qquad
X_c=X-\mu,\qquad
E=W-\widehat W_{\text{full}}.
]

Fit (P=AB) by minimizing

[
\left|X_c(E-P)^\top\right|_F^2.
]

A covariance-aware closed-form construction is:

1. Compute the centered input covariance:

[
H=\frac{1}{N}X_c^\top X_c+\lambda I.
]

2. Let (H=LL^\top) be its Cholesky factorization.

3. Form the activation-whitened residual:

[
Z=EL.
]

4. Compute a truncated SVD:

[
Z\approx U_r\Sigma_rV_r^\top.
]

5. Set:

[
A=U_r\Sigma_r^{1/2},
\qquad
B=\Sigma_r^{1/2}V_r^\top L^{-1}.
]

A compact PyTorch sketch:

```python
import torch


@torch.no_grad()
def fit_activation_residual_patch(
    target_weight: torch.Tensor,
    deployed_weight: torch.Tensor,
    calibration_inputs: torch.Tensor,
    rank: int,
    damping_fraction: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit W - W_hat ≈ A @ B under centered calibration activations.

    Returns:
        A: (out_features, rank)
        B: (rank, in_features)
        input_mean: (in_features,)
    """
    if rank <= 0:
        raise ValueError("rank must be positive")

    weight_error = target_weight.float() - deployed_weight.float()
    inputs = calibration_inputs.reshape(-1, calibration_inputs.shape[-1]).float()

    if weight_error.shape[1] != inputs.shape[1]:
        raise ValueError("input feature dimension does not match weight")

    input_mean = inputs.mean(dim=0)
    centered = inputs - input_mean

    covariance = centered.T @ centered / max(1, centered.shape[0])
    damping = damping_fraction * covariance.diagonal().mean().clamp_min(1e-12)
    covariance.diagonal().add_(damping)

    chol = torch.linalg.cholesky(covariance)  # covariance = L @ L.T
    whitened_error = weight_error @ chol

    q = min(min(whitened_error.shape), rank + 8)
    u, singular, v = torch.svd_lowrank(
        whitened_error,
        q=q,
        niter=2,
    )
    u = u[:, :rank]
    singular = singular[:rank]
    v = v[:, :rank]

    root = singular.sqrt()
    a = u * root.unsqueeze(0)

    # Find B such that B @ chol = sqrt(S) @ V.T.
    transformed_b = root.unsqueeze(1) * v.T
    b = torch.linalg.solve_triangular(
        chol.T,
        transformed_b.T,
        upper=True,
    ).T

    return (
        a.to(target_weight.dtype).contiguous(),
        b.to(target_weight.dtype).contiguous(),
        input_mean.contiguous(),
    )
```

After fitting (P=AB), recompute the bias from the **final** residual:

[
b_{\text{corr}}
=\mu,(W-\widehat W_{\text{full}}-AB)^\top.
]

This matters because evaluating the patch on uncentered runtime inputs introduces its own mean output. Fitting bias before the patch is fine as an implementation milestone, but once patches exist, the bias stage should run again after patch fitting.

### Where to use it

Start with:

* `o_proj`, ranks 4, 8, and 16;
* perhaps only the blocks where held-out splice KL says the patch is valuable.

Do not initially patch every MLP matrix. Your measurements found `o_proj` had the strongest low-rank residual structure per added bit, while MLP-side patches had less attractive ceilings and substantially larger relative cost.

Charge its actual cost:

[
\text{patch bits}
=r(m+n)\times \text{storage bits}.
]

At a fixed total BPW, fund it by reducing binary rank elsewhere rather than treating it as free.

## 3. Reallocate binary rank before buying much residual storage

This is technically error prevention rather than compensation, but it appears to be the **largest quality-per-bit lever** in this repository.

Your end-to-end measurements show:

* MLP projections account for about 72% of the measured type-wise KL;
* `up_proj` is the largest individual contributor;
* blocks 0–10 account for about 65% of the block-wise damage;
* late blocks 18–24 account for only about 7.8%.

Use the splice profile to estimate:

[
s_u=\frac{\mathrm{KL}*u}{E*{\text{func},u}^2},
]

then allocate ranks to minimize predicted functional cost rather than weighted Frobenius error alone. The repository’s anatomy notes identify this as the largest measured, already-plumbed opportunity.

In practical terms, rank should generally move:

* from attention toward MLP, especially `up_proj`;
* from relatively harmless late blocks toward blocks 0–10;
* while preserving special protection for the final block.

This will usually beat storing a larger approximation of the residual because it preserves the fast binary representation instead of adding another runtime path.

## 4. Keep residual-column outliers, but treat them as the sparse tail

The existing implementation already does a sensible sparse correction:

1. run a probe factorization;
2. score columns by weighted residual;
3. zero selected columns before factorization;
4. store those columns separately;
5. restore them during reconstruction.

That is useful for a small number of genuinely bad, axis-aligned input columns. But the measured remainder is concentrated in **correlated activation directions**, not simply in an unimportant or unstructured tail: 63–93% of residual energy lies in the top activation subspace.

So I would use the two side paths for different jobs:

* **column outliers:** sparse, axis-aligned exceptions;
* **low-rank patch:** correlated residual structure;
* **bias:** residual mean.

Blindly increasing `outliers.fraction` can become an expensive dense side matrix in disguise. Select its size by marginal held-out KL improvement per added bit.

## 5. Finish with global continuous-parameter distillation

Once allocation, bias, and any selected patches are fixed, perform a short teacher-KL recovery pass over:

* pre/mid/post scales;
* outlier values;
* additive biases;
* normalization vectors;
* optionally patch factors.

Your current global distillation already freezes the binary factors and selects scales, outlier values, bias, and norm parameters.  That is the right final pass: continuous parameters can coordinate across layers without destabilizing the binary structure.

## Recommended implementation order

For this codebase, I would land the work in this order:

1. **KL-weighted rank allocation.**
2. **`BiasCorrectionStage` after final reconstruction.**
3. **Held-out rank-4/8/16 `o_proj` patch sweep at equal total BPW.**
4. **Global scales/outliers/bias distillation.**
5. Consider covariance-aware ADMM only after measuring how much headroom remains.

That ordering also matches the repository’s measured experiment priorities.

One thing I would explicitly deprioritize is sequential propagated-error calibration. The measured block errors were slightly **sub-additive**, with a whole-model-to-summed-block ratio around 0.90, so there is not currently evidence of runaway downstream compounding for it to correct.

### ADMM terminology note

The code also calls its primal and dual convergence gaps “residuals.” Those are a separate issue. For high ADMM primal/dual residuals, use residual-balanced adaptive (\rho), more iterations, or better stopping thresholds—not an output-side bias or low-rank patch. When changing (\rho) with scaled dual variables, rescale the dual state by (\rho_{\text{old}}/\rho_{\text{new}}); otherwise the penalty change also changes the represented unscaled multiplier. That modification would likely break exact legacy parity, so it should be evaluated as a separate factorizer experiment.
