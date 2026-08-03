Yes. Beyond one-bit descent, pair flips, and small exhaustive blocks, there are several worthwhile additions. The biggest opportunity is to exploit the fact that NanoQuant does **not** give you one arbitrary QUBO—it gives you a large batch of closely related QUBOs.

With (V) and the scales fixed, each row of (U) solves something of the form

[
\min_{x\in{-1,+1}^{r}}
f_i(x)
======

x^\top Qx-2c_i^\top x,
]

where

[
Q=A^\top A \succeq 0.
]

Every row shares the same (Q); only (c_i) changes. Optimizing (V) has the same structure after transposition. NanoQuant’s LB-ADMM stage initializes these latent factors but does not globally solve these final discrete subproblems. ([arXiv][1])

That shared-(Q) structure changes which optimizations are most attractive.

## 1. Use an elite solution codebook across rows

This is the most NanoQuant-specific optimization I would test.

Suppose local optimization produces (K) distinct sign vectors collected across all rows and restarts:

[
S=
\begin{bmatrix}
s_1^\top\
\vdots\
s_K^\top
\end{bmatrix}
\in{-1,+1}^{K\times r}.
]

Precompute each candidate’s quadratic cost:

[
q_k=s_k^\top Qs_k.
]

Then score every candidate against every row with

[
E_{ki}=q_k-2s_k^\top c_i.
]

In matrix form:

[
E=q\mathbf 1^\top-2SC^\top.
]

That is one ordinary GEMM. Each row selects its best codebook candidate and runs local search from it.

This works because each candidate’s energy is affine in (c_i):

[
f_{s}(c)=s^\top Qs-2c^\top s.
]

The exact solution as a function of (c) is the lower envelope of those affine functions. A pool of locally discovered patterns approximates that envelope. Nearby rows—or rows with similar continuous solutions (Q^\dagger c_i)—may reuse the same or nearby binary patterns.

A practical version would:

1. Solve every row from its ADMM initialization.
2. Deduplicate the resulting sign patterns.
3. Keep 128–512 diverse patterns per cluster of similar (Q^\dagger c_i).
4. Score the pool against all rows in the cluster.
5. Polish each selected pattern.
6. Add newly discovered patterns and repeat two or three times.

There is no quality risk because every transferred candidate is evaluated against the exact row objective before it replaces anything. The gain is uncertain, but the experiment is cheap and uniquely exploits your batch structure.

With the row scale analytically optimized, the same pool works using

[
f_i^*(s)
========

|t_i|^2-
\frac{(s^\top c_i)^2}{s^\top Qs}.
]

Again, (SC^\top) computes every numerator, while (s^\top Qs) is shared across rows.

## 2. Add variable-depth search or reactive tabu search

Pure bit-flip descent stops at a 1-optimum. Pair search stops at a 2-optimum. Neither can cross a barrier where the first few flips are harmful but the complete sequence is beneficial.

For the fixed-scale QUBO, remove the diagonal from (Q), because (x_i^2=1) makes it constant on binary points. Let

[
J=Q-\operatorname{diag}(Q).
]

Maintain the local field

[
g=Jx-c.
]

The cost change from flipping (x_k) is

[
\Delta_k=-4x_kg_k.
]

After accepting a flip, all fields update in (O(r)):

[
g\leftarrow g-2x_k^{\text{old}}J_{:,k}.
]

That delta cache supports two stronger searches.

### Variable-depth search

Construct a sequence of flips, temporarily allowing positive deltas:

1. Select the best currently unlocked flip.
2. Flip it and lock that bit for the remainder of the chain.
3. Record cumulative cost after every flip.
4. Continue for perhaps 16–64 flips.
5. Commit only the best improving prefix; otherwise roll back.

This is GPU-friendlier than a fully irregular metaheuristic because every row can execute a fixed-length chain with masks.

### Reactive tabu search

Allow uphill flips but prevent recently flipped variables from being immediately reversed. Preserve the best state seen, not merely the final state. Randomized or reactive tabu tenure avoids short cycles. Tabu search and adaptive-memory variants have historically been particularly effective for dense unconstrained binary quadratic problems. ([PubsOnline][2])

I would test:

* a short variable-depth chain first;
* then a small tabu tenure with jitter;
* after a stall, flip a handful of low-margin, strongly interacting bits and descend again.

Tune chain length and tenure against the tiny-matrix oracle rather than adopting generic QUBO defaults.

## 3. Use structured exact large-neighborhood search

Rather than choosing a block from the bits with the smallest individual deltas alone, build blocks from **ambiguous, strongly coupled variables**.

Start with a low-margin bit and greedily add variables according to something like

[
\operatorname{score}(j)
=======================

\frac{\sum_{k\in B}|J_{jk}|}
{\epsilon+\operatorname{margin}_j}.
]

This targets groups whose interactions can overcome their individual flip costs.

For a block (B), holding the other variables fixed gives

[
\min_{z\in{-1,+1}^{|B|}}
z^\top Q_{BB}z-2\widetilde c_B^\top z,
]

where

[
\widetilde c_B=c_B-Q_{B\bar B}x_{\bar B}.
]

If you already maintain (g=Qx-c), then

[
\widetilde c_B=Q_{BB}x_B-g_B,
]

so you do not need to multiply against the whole complement.

### Shared-block enumeration

Cluster the rank dimensions once using (|J_{ij}|), and use the same blocks across many rows. For each block:

* precompute all (2^b) assignments;
* precompute (z^\top Q_{BB}z) once;
* score the row-dependent linear terms with a GEMM.

This should make (b=12) routine and (b=16) feasible for selected hard rows. It is substantially more efficient than independently enumerating every row.

### Sphere decoding or branch-and-bound for larger blocks

Because the QUBO is binary least squares, block problems can also be solved as

[
\min_{z\in{-1,+1}^{b}}|A_Bz-y|^2.
]

A sphere decoder with Schnorr–Euchner-style ordering can use your current solution as a tight initial radius. The QR factorization of (A_B) is reusable across rows when the block is shared. Sphere decoding remains exponential in the worst case but can solve moderate binary least-squares instances much faster than enumeration when the incumbent is good. ([CaltechAUTHORS][3])

Another option is branch-and-bound with a convex box relaxation:

[
\min_{-1\le z\le 1}
z^\top Hz-2\widetilde c^\top z+\text{constant},
\qquad H\succeq0.
]

For exact pruning, use a dual-certified box-QP bound or a closed-form unconstrained quadratic lower bound—not merely the objective of an unfinished projected-gradient solve.

I would roughly use:

* exhaustive search for (b\le 12) everywhere;
* (b=16) for selected rows;
* bounded-node sphere decoding or box-QP branch-and-bound for (b\approx20)–(60).

The exact crossover will depend heavily on conditioning.

## 4. Tighten branch-and-bound with diagonal convexification

This is an easy detail to miss.

For binary (x), every diagonal quadratic term is constant:

[
x^\top \operatorname{diag}(d)x=\sum_i d_i.
]

Therefore, you may change the diagonal of the quadratic matrix without changing the binary problem, provided you adjust the constant.

Let

[
J=Q-\operatorname{diag}(Q).
]

Choose a diagonal matrix (D) such that

[
J+D\succeq0.
]

Then

[
x^\top Jx-2c^\top x
===================

x^\top(J+D)x-2c^\top x-\operatorname{tr}(D)
]

for every binary (x).

Different choices of (D) produce the same binary objective but different continuous box relaxations. The natural Gram diagonal from (A^\top A) may be much larger than needed, allowing fractional variables to achieve an unnecessarily low relaxed objective. A tighter convexification can materially improve branch-and-bound pruning.

A cheap starting point is the smallest uniform loading:

[
D=\left(-\lambda_{\min}(J)+\epsilon\right)I.
]

A more expensive version optimizes the diagonal perturbation, or uses several perturbations and takes the strongest bound. Quadratic convex reformulation methods use precisely this freedom to strengthen continuous relaxations. ([American Chemical Society Publications][4])

Because (J) is shared across all row QUBOs, computing good perturbations is amortized.

One important distinction:

* For binary local-search deltas, zero the diagonal.
* For scale-profiled objectives, retain the actual (Q) in (x^\top Qx).
* For convex lower bounds, use a deliberately chosen convexifying diagonal rather than casually zeroing it.

## 5. Use elite fusion and path relinking

Keep a small elite set—perhaps four to eight Hamming-diverse solutions per row—from:

* ADMM;
* continuous relaxation;
* tabu search;
* codebook transfer;
* perturbed linear terms;
* previous outer iterations.

Given two candidates (x^A) and (x^B), only optimize the bits on which they disagree. Write

[
x(z)=x^A+\operatorname{diag}(x^B-x^A)z,
\qquad z\in{0,1}^{d}.
]

Substitution produces another QUBO of dimension (d). If the two elites differ in 12–20 positions, solve the fusion exactly. If they differ more widely, use block search, branch-and-bound, or QPBO.

Path relinking is the cheaper alternative: move from (x^A) toward (x^B), at each step selecting the least damaging remaining disagreement flip, and locally polish the best intermediate point. Combining tabu search with path relinking has been effective on difficult QUBO instances, while fusion moves provide a general mechanism for combining two existing solutions through a subsidiary binary optimization. ([DOI][5])

This is more purposeful than unrelated random restarts: it searches the region between two known-good basins.

## 6. Use roof duality and QPBO primarily for reduction

Before invoking a heavier solver, apply cheap exact variable fixing.

For the zero-diagonal form, if

[
|c_i|>\sum_{j\ne i}|J_{ij}|,
]

then the field from the other variables can never overcome (c_i), so every global optimum has

[
x_i=\operatorname{sign}(c_i).
]

Fix such variables, fold their interactions into the remaining linear terms, and repeat.

Roof duality/QPBO can identify additional persistent variables and provide a lower bound through a max-flow-based relaxation. ([ScienceDirect][6])

I would not expect QPBO to solve every full dense NanoQuant row. The mixed-sign, dense interaction graph may leave many variables unresolved. Its better uses here are:

* preprocessing exact blocks;
* solving fusion subproblems;
* producing bounds in branch-and-bound;
* fixing variables before sphere decoding;
* operating on the hardest selected rows rather than all rows.

## 7. Use continuous relaxations as restart generators

Since the natural (Q=A^\top A) is PSD, the box relaxation

[
\min_{-1\le x\le1}x^\top Qx-2c^\top x
]

is convex. All rows can be solved simultaneously with batched projected gradient or accelerated coordinate descent:

[
x\leftarrow
\operatorname{clip}
\left(
x-\eta(Qx-c),-1,1
\right).
]

Use the result to:

* generate (\operatorname{sign}(x)) candidates;
* identify ambiguous variables with small (|x_i|);
* cluster rows by their relaxed solutions;
* initialize exact block search;
* generate randomized rounds before local polishing.

A continuation penalty can push the solution toward binary vertices:

[
F_\mu(x)
========

x^\top Qx-2c^\top x
+
\mu(r-|x|^2).
]

Start at (\mu=0), gradually increase it, and polish every rounded candidate using the exact binary objective. More formal MPEC-based approaches provide exact continuous penalty formulations once their penalty is sufficiently large, although they are more complex than the simple continuation above. ([AAAI Open Access Journals][7])

I would treat this as a source of structurally different starting points, not as the final solver.

## 8. Add a component-level bipartite search

Even a perfect solution of every row QUBO can remain trapped because it holds (V) fixed while optimizing (U), and vice versa.

Select one rank component (k), remove it, and form the residual

[
E_k
===

T-\sum_{\ell\ne k}a_\ell u_\ell v_\ell^\top.
]

Updating the missing component becomes approximately

[
\max_{u\in{-1,+1}^{m},,v\in{-1,+1}^{n}}
u^\top E_kv.
]

This is a bipartite Boolean quadratic problem. Given (v), the exact best response is

[
u=\operatorname{sign}(E_kv),
]

and symmetrically

[
v=\operatorname{sign}(E_k^\top u).
]

A very-large-neighborhood search can flip one side and then completely reoptimize the other side. Thus, what looks like a one-bit move can change hundreds or thousands of bits in the opposite factor. Specialized work on bipartite Boolean quadratic optimization finds that combining this kind of large-neighborhood search with tabu search is substantially stronger than either alone. ([arXiv][8])

This is likely more valuable than running an expensive SDP on every row because it attacks a failure mode that row-wise QUBOs fundamentally cannot address.

## 9. Profile the scale inside every search method

Even when using a QUBO to propose moves, I would rank candidates using the objective after analytically optimizing the row scale.

Maintain

[
a=c^\top x,
\qquad
b=x^\top Qx.
]

The profiled reconstruction error is

[
f^*(x)=|t|^2-\frac{a^2}{b}.
]

For a proposed bit flip (k),

[
a'=a-2x_kc_k,
]

[
b'=b-4x_k(Qx)*k+4Q*{kk}.
]

So exact scale-profiled evaluation still costs (O(1)) per candidate once (Qx) is cached. Tabu search, variable-depth search, fusion, and block enumeration can all optimize this objective instead of a stale fixed-scale surrogate.

That matters because a bit move which looks harmful with the current scale may be beneficial after the scale refits.

## The solver stack I would build

My preferred production sequence would be:

[
\text{ADMM}
\rightarrow
\text{exact scale fit}
\rightarrow
\text{batched 1-opt}
\rightarrow
\text{shared codebook transfer}
\rightarrow
\text{variable-depth/tabu}
\rightarrow
\text{2--4 structured block solves}
\rightarrow
\text{elite fusion}
\rightarrow
\text{component-level VLSN}
\rightarrow
\text{scale refit}.
]

Repeat the (U/V) alternation for a small number of outer passes, retaining the best complete state ever seen. Only the worst few percent of rows should receive sphere decoding, QPBO, or SDP-based treatment.

My implementation priority would be:

1. **Variable-depth/tabu search**, because it is the most reliable extension beyond 1-opt and 2-opt.
2. **Shared-(Q) candidate pools**, because they are cheap, batched, and unique to this problem family.
3. **Shared structured blocks with exact enumeration or bounded branch-and-bound**.
4. **Component-level bipartite VLSN**, to escape joint (U/V) basins.
5. **Roof duality and diagonal-convexified bounds**, for difficult blocks and exact certification.
6. **Low-rank SDP rounding**, mainly as an oracle and diagnostic rather than a universal production pass; scalable low-rank methods such as the Mixing method make these relaxations more practical, but solving one SDP per row is still expensive. ([arXiv][9])

The tiny brute-force tests can tell you exactly which additions matter. Track how much of the oracle gap is closed after variable-depth search, block sizes (8,12,16), elite fusion, and component updates. My expectation is that tabu or variable-depth search will remove the easy multi-flip barriers, structured exact blocks will remove most remaining row-level error, and any stubborn final gap will increasingly come from the alternating (U/V) decomposition rather than the individual QUBO solver.

[1]: https://arxiv.org/abs/2602.06694 "https://arxiv.org/abs/2602.06694"
[2]: https://pubsonline.informs.org/doi/10.1287/mnsc.44.3.336 "https://pubsonline.informs.org/doi/10.1287/mnsc.44.3.336"
[3]: https://authors.library.caltech.edu/records/xz0mk-7y361 "https://authors.library.caltech.edu/records/xz0mk-7y361"
[4]: https://pubs.acs.org/doi/abs/10.1021/acs.iecr.7b01270 "https://pubs.acs.org/doi/abs/10.1021/acs.iecr.7b01270"
[5]: https://doi.org/10.1007/S10732-018-9403-Z "https://doi.org/10.1007/S10732-018-9403-Z"
[6]: https://www.sciencedirect.com/science/article/pii/S1572528607000400 "https://www.sciencedirect.com/science/article/pii/S1572528607000400"
[7]: https://ojs.aaai.org/index.php/AAAI/article/view/10795 "https://ojs.aaai.org/index.php/AAAI/article/view/10795"
[8]: https://arxiv.org/abs/1305.5610 "https://arxiv.org/abs/1305.5610"
[9]: https://arxiv.org/abs/1706.00476 "https://arxiv.org/abs/1706.00476"
