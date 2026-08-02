Yeah—this looks like a real optimization gap. NanoQuant’s LB-ADMM is best understood as a fast **initializer for the binary factors**, not as an optimizer that gets especially close to the globally best binary product. It alternates continuous least-squares updates with SVID projections of the individual factors. Those projections minimize distance to each factor separately; they do not minimize the final error of the product of the two binarized factors. The paper itself describes direct binary optimization as nonconvex and combinatorial. ([arXiv][1])

The best practical replacement is:

> **Use NanoQuant ADMM as a warm start, then directly optimize the final binary signs using alternating binary quadratic optimization, exact scale refits, and small exact neighborhoods.**

That should be dramatically cheaper than full brute force while targeting the objective your brute-force oracle actually improves.

## First, determine exactly where the gap originates

There are two distinct possibilities.

### 1. The SVID projection itself is inaccurate

For one ADMM projection target (P), exact SVID should be

[
Z^*
===

\operatorname{sign}(P)\odot
\left(\sigma_1 u_1 v_1^\top\right),
]

where (\sigma_1,u_1,v_1) are the leading singular triplet of (|P|).

This is globally optimal for the sign-preserving rank-one-magnitude projection: once the rank-one magnitudes are fixed, every entry’s best sign is simply the sign of (P), reducing the problem to the ordinary best rank-one approximation of (|P|).

The checked version of your implementation uses a fresh random vector and only five power iterations for each SVID projection.  On tiny matrices, replace that temporarily with a full SVD:

```python
abs_p = p.abs().float()
u, s, vh = torch.linalg.svd(abs_p, full_matrices=False)

signs = torch.where(p >= 0, 1.0, -1.0)
z_exact = signs * (s[0] * u[:, :1] @ vh[:1, :])
```

Then rerun the oracle comparison.

* If exact SVID closes most of the gap, the issue is finite power iteration. In production, carry the previous singular vector forward as a warm start, use a nonnegative initial vector, and stop using a residual tolerance rather than a fixed five iterations.
* If brute force still wins by roughly 50%, it is optimizing the final binary factorization, not the SVID projection. That is the more interesting and likely result.

## Directly optimize the final binary product

Write the normalized target as (T). Your runtime representation is effectively

[
\widehat T
==========

D_p L D_a R D_q,
]

where

* (L\in{-1,+1}^{m\times r}),
* (R\in{-1,+1}^{r\times n}),
* (p,a,q) are post-, middle-, and pre-scale vectors.

For the paper’s simpler two-scale representation, set (D_a=I).

The complete problem is quartic in (L) and (R), but it becomes a collection of independent binary quadratic problems when either factor is fixed.

### Optimizing the left factor

Fix (R,a,q,p), and define

[
X=D_a R D_q.
]

For output row (i), let (x=L_{i,:}^{\top}) and (t=T_{i,:}^{\top}). Its objective is

[
|t-p_iX^\top x|_2^2.
]

Expanding it gives

[
f_i(x)
======

x^\top Q_i x-2c_i^\top x+\text{constant},
]

with

[
Q_i=p_i^2XX^\top,
\qquad
c_i=p_iXt.
]

So every output row is an independent (r)-variable QUBO.

Likewise, with (L) fixed, every column of (R) becomes an independent QUBO. Define

[
Y=D_p L D_a,
]

and for input column (j),

[
f_j(x)
======

## x^\top\left(q_j^2Y^\top Y\right)x

2\left(q_jY^\top T_{:,j}\right)^\top x
+\text{constant}.
]

This directly optimizes reconstruction error. It does not ask whether each factor is close to an SVID proxy.

## Exact and cheap one-bit updates

For

[
f(x)=x^\top Qx-2c^\top x,
\qquad x\in{-1,+1}^r,
]

the exact error change from flipping bit (k) is

[
\Delta_k
========

4\left[
Q_{kk}
------

x_k(Qx-c)_k
\right].
]

After accepting a flip, (Qx) can be updated in (O(r)):

[
Qx
\leftarrow
Qx-2x_k^{\text{old}}Q_{:,k}.
]

This means that after the initial Gram matrices and cross terms are computed with GEMMs, a step that selects at most one bit per row costs approximately (O(mr)), rather than another full matrix reconstruction.

On GPU, maintain all of these in batches:

* (G_L=LQ),
* cross terms (C=TX^\top),
* exact deltas for every (L_{ik}),
* one best accepted bit per row per iteration.

The right side can reuse the same implementation by transposing the problem.

Your covariance refiner already contains most of this machinery: exact bit deltas, bounded accepted updates, pairwise corrections for coupled batches, and alternating closed-form scale solves.  The important next step is to generalize that into a direct final-factor optimizer rather than treating it only as a covariance postprocessing screen.

## Eliminate the channel scale while choosing signs

You can make each sign update stronger by analytically optimizing the row scale (p_i) for every candidate sign vector.

With

[
Q=XX^\top,\qquad c=Xt,
]

the optimal scale for a particular (x) is

[
p_i^*(x)
========

\frac{c^\top x}{x^\top Qx}.
]

Substituting it into the objective gives

[
f_i^*(x)
========

## |t|_2^2

\frac{(c^\top x)^2}{x^\top Qx}.
]

This is a binary generalized Rayleigh quotient. A candidate flip can still be evaluated very cheaply. Maintain

[
\alpha=c^\top x,\qquad
\beta=x^\top Qx,\qquad
g=Qx.
]

Flipping bit (k) gives

[
\alpha'
=======

\alpha-2x_kc_k,
]

[
\beta'
======

\beta-4x_kg_k+4Q_{kk}.
]

The exact candidate error is then

[
|t|^2-\frac{\alpha'^2}{\beta'}.
]

That jointly chooses the bit and its best output-channel scale. Apply the symmetric version to (R) and (q_j).

For a dense activation covariance (C), the same math works after replacing

[
Q=XX^\top
\quad\text{with}\quad
Q=XCX^\top,
]

and

[
c=Xt
\quad\text{with}\quad
c=XCt.
]

## One-bit descent will not be enough by itself

A 50% oracle gap probably includes solutions separated by multi-bit barriers. Once no individual bit helps, use progressively larger neighborhoods.

### Pair flips

If (\Delta_i) and (\Delta_j) are the individual flip deltas, the exact joint delta is

[
\Delta_{ij}
===========

\Delta_i+\Delta_j+8x_ix_jQ_{ij}.
]

Search pairs among perhaps the 32–64 bits with the smallest single-bit margins. This gives a reasonably cheap 2-opt local optimum.

### Exact block search

Choose a block (S) of (b) bits and hold all other bits fixed. The resulting subproblem is

[
\min_{z\in{-1,+1}^{b}}
z^\top Q_{SS}z
--------------

2\left(c_S-Q_{S\bar S}x_{\bar S}\right)^\top z.
]

Now enumerate only this block:

* (b=8): 256 candidates
* (b=10): 1,024 candidates
* (b=12): 4,096 candidates
* (b=16): 65,536 candidates

This is **large-neighborhood search**, not full brute force. The exponential term is bounded by a constant block size rather than depending on the complete factor dimensions.

For selecting a block, mix:

* bits with the smallest positive single-flip deltas;
* strongly interacting bits according to (|Q_{ij}|);
* bits where the ADMM latent value was closest to zero;
* bits whose signs disagree with a continuous least-squares solution.

I would start with (b=10) or (12), only on the worst 5–20% of rows and columns.

## Add a move that changes both factors at once

Alternating (L) and (R) can still become trapped because every move holds one factor fixed. A useful nonlocal move is to replace one complete rank component.

Remove component (k) from the reconstruction, giving residual (E_k). With the boundary scales held fixed, find signs (u,v) that approximately maximize

[
\left|
u^\top D_pE_kD_qv
\right|,
\qquad
u\in{-1,+1}^{m},
\quad
v\in{-1,+1}^{n}.
]

Alternating updates are exact for each side:

[
u\leftarrow
\operatorname{sign}(D_pE_kD_qv),
]

[
v\leftarrow
\operatorname{sign}(D_qE_k^\top D_pu).
]

Then refit the component’s middle scale:

[
a_k
===

\frac{
u^\top D_pE_kD_qv
}{
|p|_2^2|q|_2^2
}.
]

This changes (m+n) signs together and can cross barriers that row/column bit flips cannot. Use the current component, a spectral sign initializer, and a few cheap perturbed starts—not complete ADMM restarts.

That distinction matters because your previous eight-seed ADMM sweep only improved the weighted error by about 0.026% while multiplying the full factorization cost.  Structurally different moves are much more promising than rerunning the same local solver from slightly different random states.

## A concrete solver I would implement

```text
1. Run ordinary NanoQuant ADMM.
2. On tiny tests, replace approximate SVID with exact SVD.
3. Extract binary factors and perform exact alternating scale fitting.
4. Optimize L:
      a. continuous least-squares sign candidate
      b. scale-eliminated one-bit descent
      c. pair search on ambiguous bits
      d. b=10 or b=12 exact block search on hard rows
5. Refit p, a, q.
6. Optimize R using the symmetric procedure.
7. Refit p, a, q.
8. Try one bounded rank-component replacement sweep.
9. Repeat steps 4–8 for 3–6 outer passes.
10. Retain the best state rather than the final state.
```

For the continuous sign candidate in step 4a, solve once per side:

[
w_i=(Q+\lambda I)^{-1}c_i,
]

then test

[
x_i=\operatorname{sign}(w_i)
]

against the current ADMM signs. Since (Q) is shared across all rows on that side, one Cholesky factorization serves every row. This is a cheap way to enter a basin substantially different from ADMM’s SVID signs.

Every accepted operation should be checked against the exact reconstruction objective. That gives monotonic behavior and makes debugging much easier.

## The oracle experiment should identify the missing neighborhood

For the tiny matrices where you have the true optimum, compare this ladder:

1. NanoQuant ADMM.
2. ADMM with exact SVID.
3. Exact SVID plus exact scale fitting.
4. Plus one-bit descent.
5. Plus pair descent.
6. Plus (b=8,10,12) block search.
7. Plus rank-component replacement.

Report

[
\text{gap closed}
=================

\frac{
E_{\text{ADMM}}-E_{\text{candidate}}
}{
E_{\text{ADMM}}-E_{\text{optimal}}
}.
]

That will tell you exactly what kind of failure the paper’s solver has:

* Exact SVID helps: inaccurate power iteration.
* Scale fitting helps: export/magnitude mismatch.
* One-bit descent helps: obvious bad sign choices remain.
* Pair or block search helps: multi-bit local barriers.
* Component replacement helps: alternating-factor basin is the binding problem.

Also canonicalize component permutations and paired sign flips before measuring Hamming distance to the oracle. Flipping both (L_{:,k}) and (R_{k,:}), or permuting rank components, leaves the reconstruction unchanged.

## Keep solver quality separate from model quality

Your tiny-matrix result is a clean **solver-optimality** question. It should initially use exactly the same objective for brute force and every approximate method.

That is distinct from whether the objective predicts language quality. Your covariance refinement demonstrated the danger: the static screen produced large covariance-error and KL improvements, but the complete pre-tuning experiment reduced the local covariance objective in all 104 eligible groups while regressing final WikiText perplexity by 19.26%.

So the sequence should be:

1. Prove that the new discrete solver closes the brute-force gap on identical objectives.
2. Test it on held-out layer reconstruction.
3. Test its placement relative to factorized tuning and block refitting.
4. Gate composition using held-out functional behavior representative of the actual target workload.

My strongest bet is that **scale-eliminated row/column sign descent will close the easy portion of the 50% gap**, while **small exact block neighborhoods and rank-component replacement will account for most of the remaining recoverable gap**. It directly attacks the final binary product, remains GPU-friendly, preserves the current packed format, and is far cheaper than trying to make nonconvex ADMM globally clever.

[1]: https://arxiv.org/html/2602.06694v3 "https://arxiv.org/html/2602.06694v3"
