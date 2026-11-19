## Bottom line

NanoQuant does **not** have an optimizer shortage. It has a scoreboard problem.

The repository already contains mature ADMM factorization, alternating scale fitting, multiple binary-search strategies, Fisher/KL-driven allocation, outlier sidecars, covariance refinement, block tuning, global distillation, product codebooks, and a fairly heroic experiment framework. But the early compression stages still primarily optimize a separable weighted matrix objective:

\[
\left\|D_{\text{out}}^{1/2}(\widehat W-W)D_{\text{in}}^{1/2}\right\|_F^2
\]

while actual success is judged using full-model NLL, teacher KL, perplexity, and downstream tasks. The experiment history repeatedly shows those objectives disagree:

- Covariance refinement reduced its local objective by roughly 39%, yet worsened matched perplexity by 19.3%. 
- Post-refit QKV covariance improvement still worsened perplexity by 5.5%. 
- Over-complete rank added capacity at almost unchanged BPW, nearly doubled block processing time, and worsened perplexity by 40%. 
- Increasing the binary learning rate improved block output and teacher KL even as the weighted matrix metric became worse. 

So, bluntly: **the next major quality gain probably will not come from a better binomial solver, more ADMM iterations, or more rank. It will come from putting model-level quality signals directly into sign selection, bit allocation, and end-to-end tuning.**

## Recommended priorities

| Priority | Direction | Confidence | Likely payoff | Engineering cost |
|---:|---|---|---|---|
| 1 | Complete the adaptive tail-aware correction campaign | High | High | Medium |
| 2 | Replace reconstruction-based allocation with functional utility | High | High | High |
| 3 | Add constrained end-to-end binary QAT | Medium-high | Potentially very high | High |
| 4 | Improve the compressed distillation target | High | Medium-high | Medium |
| 5 | Jointly allocate rank, outliers, and representation options | Medium-high | Medium-high | Medium-high |
| 6 | Test a grouped middle core instead of diagonal `scale_mid` | Medium | Unknown, potentially high | High |
| 7 | Improve held-out selection and numerical isolation | High | Modest but reliable | Low-medium |

---

# 1. Finish Experiment 048 before starting another format experiment

This is the highest-confidence next step because it is based on the strongest positive end-to-end result already in the repository.

Experiment 047 applied a capability-oriented correction after tail-aware KD and improved packed WikiText perplexity from **180.36 to 166.45**, at unchanged rank and BPW, without an established regression on the 1,000-example task suite. 

Experiment 048 correctly turns that one retained checkpoint into a transferable policy:

1. Produce the tail-aware endpoint.
2. Produce correction checkpoints at 32, 64, 96, and 128 steps.
3. Evaluate every checkpoint together on one calibration-only C4 slice.
4. Select the earliest checkpoint on the joint NLL/full-KL plateau.
5. Fall back to the uncorrected model when nothing clearly improves.
6. Confirm on untouched data and then run packed/GGUF and task gates.

The selector, exact reload checks, calibration-versus-capability separation, and campaign receipts are already largely implemented. The campaign is currently paused at a fail-closed numerical boundary after blocks 0–2, before either reserved quality slice was opened. 

### What I would change operationally

For this campaign, make one-block subprocess isolation the normal execution mode rather than a temporary workaround. The repository has repeatedly found process-local CUDA or optimizer contamination even when durable activation artifacts are finite and hash-valid. A clean process per block is ugly, but it preserves the causal experiment. Accuracy research conducted on a numerically haunted process is how one ends up debugging ghosts for three weeks.

The experiment should not copy “96 correction steps” into the production recipe. The important mechanism is the **adaptive checkpoint rule with a baseline escape path**, not the particular checkpoint that won on Experiment 044.

**This is the first thing I would complete.** It is much closer to a proven accuracy improvement than any new factor format.

---

# 2. Build one functional candidate scorer and use it everywhere

The most important architectural change would be to stop maintaining separate, weak proxies for:

- rank response;
- binary search;
- codebook option selection;
- outlier selection;
- covariance refinement;
- retry decisions.

Instead, introduce one reusable `FunctionalCandidateScorer`.

The current factorizer and scale fitter optimize weighted reconstruction of the weight matrix using separable input/output importance vectors.   That is useful for initialization, but the experiment record says it is not trustworthy enough to make final resource-allocation decisions.

## A better local objective

For a linear owner with inputs \(X\), let:

\[
\Delta Y = X(\widehat W-W)^\top
\]

Then approximate the downstream loss change around the model’s current operating point:

\[
\Delta \mathcal L
\approx
\langle G_Y,\Delta Y\rangle
+
\frac{1}{2}
\langle\Delta Y,H_Y\Delta Y\rangle
\]

where \(G_Y\) and \(H_Y\) are gradient and approximate curvature with respect to the owner’s output.

This can be implemented at several fidelity levels:

1. **Cheap screening:** diagonal output curvature plus cached activation inputs.
2. **Better screening:** low-rank or block-diagonal output curvature.
3. **Exact final gate:** install the candidate and run the downstream block/model suffix on held-out sequences, measuring NLL and teacher KL.

This is related to output-adaptive calibration methods, which are specifically motivated by the failure of input-only reconstruction objectives at very low precision. 

## Where it should be used

### Rank allocation

At each candidate rank, measure the improvement in held-out functional loss after a production-representative amount of tuning:

\[
U(c)=
\frac{
\mathcal L_{\text{baseline}}-\mathcal L_c
}{
\text{physical bits}_c-\text{physical bits}_{\text{baseline}}
}
\]

Then allocate using that utility rather than weighted residual reduction.

Experiment 055 is unusually strong evidence that this matters: the allocator found extra rank attractive under its response curves, but the resulting model became dramatically worse. The rank was real; the utility estimate was wrong. 

### Binary sign search

Keep ADMM as the initializer. But accept batches of sign changes only when they pass a functional trust-region score. The direct-search implementation is already extremely capable; the missing ingredient is not another move type but a better acceptance criterion. 

### Codebook options

A codebook changes storage efficiency, but it does not inherently change what errors matter. The checked-in product-codebook design measures candidate reconstruction response and keeps coded rows constrained during tuning.  That is sound structurally, but it will remain a small lever while option selection is based on the same proxy that misallocated over-complete rank.

Score the free/codebook mixture using held-out functional utility, not only reconstructed residual.

### Outlier selection

Select outliers according to marginal reduction in output-adaptive loss per stored bit, rather than weighted residual magnitude alone.

## A bounded first experiment

Use nine retained owners:

- blocks 0, 12, and 25;
- fused QKV, `gate_proj`, and `down_proj`.

For each owner, compare:

1. Current mature ADMM plus existing refinement.
2. The same incumbent plus a tightly bounded functional sign update.
3. Equal storage, identical calibration inputs, identical production-length tuning.

Require improvements on:

- held-out owner or block output;
- full-model teacher KL;
- next-token NLL;
- persisted-dtype reconstruction replay.

Only after that probe passes should the functional scorer be connected to the allocator.

---

# 3. Let the final end-to-end stage move binary signs

This is the biggest unused capacity in the current pipeline.

The global distillation loader explicitly creates trainable factorized layers with:

```python
immutable_binary_factors=True
```

and its parameter selection includes pre/mid/post scales, outlier values, biases, patches, and normalization parameters—but not the left or right binary factors. 

That means the only truly end-to-end model-level optimization stage is prohibited from adjusting the parameters that consume most of the representation’s useful capacity.

At approximately one bit per weight, that is a severe restriction.

## Proposed stage: coordinate binary QAT

Do not unfreeze every binary factor simultaneously. Use a coordinate schedule:

1. Freeze the complete model except one block or owner family.
2. Replace signs with FP32 shadow latents and a straight-through sign operator.
3. Optimize tail-aware teacher loss plus an anchor to the entry state.
4. Enforce a hard Hamming-distance budget.
5. Refit continuous scales after accepted binary movement.
6. Restore the best held-out checkpoint.
7. Freeze the block and advance.

Experiment 054 gives a useful starting scale for the trust region. Roughly **0.8% sign movement** passed its initial production canary, while approximately **2.3%** movement from a larger learning rate destabilized the propagated model. Those are not universal limits, but they suggest beginning with a sub-1% per-owner cap rather than allowing unconstrained movement. 

## Important safeguards

- Evaluate the actual persisted sign state, not the shadow variables.
- Use a separate held-out token set for checkpoint selection.
- Run each block in a fresh process until the contamination issue is genuinely resolved.
- Revert the complete owner atomically when NLL or full KL regresses.
- Do not accept movement merely because block MSE improves.
- Make the sign-change budget adaptive: owners that show no held-out gain keep their original signs.

The existing block-tuning machinery already proves binary latents can be optimized. This proposal moves that mechanism into the stage where the objective finally represents whole-model behavior.

Extreme-bit methods increasingly rely on QAT and distillation rather than expecting a purely local PTQ solve to recover all quality; confidence-aware distillation has also been shown to matter in this regime. 

I would expect this to have more upside than another factor-search tier, though it is also substantially riskier and more expensive.

---

# 4. Replace the single tail bucket and fixed coefficient with an adaptive target

The tail-aware KD work found a real and important defect in conditional top-64 distillation: the student could improve the renormalized top-64 objective while moving large amounts of probability mass outside the selected set.

The current implementation repairs this using:

- 64 individual teacher tokens;
- one aggregated “everything else” bucket;
- one global `tail_mass_weight`. 

That is much better than conditional-only KD, but it still throws away the shape of the omitted distribution.

The repository’s coefficient sweep shows a genuine tradeoff:

| Tail coefficient | WikiText PPL | Six-task mean |
|---:|---:|---:|
| 1.00 | 188.72 | 0.4550 |
| 0.50 | 178.36 | **0.4625** |
| 0.25 | **169.96** | 0.4550 |

So one global coefficient is being asked to balance very different token regimes. 

## Better alternatives

### Per-token tail weighting

Cache teacher confidence statistics with each token:

- top-64 probability mass;
- entropy or approximate entropy;
- top-1/top-2 margin;
- target-token rank.

Then make the mass coefficient token-dependent:

- high-confidence, high-top-64-mass tokens can emphasize conditional shape;
- low-confidence or diffuse tokens receive stronger tail preservation.

This should be predeclared as a deterministic function of teacher statistics—not learned against the evaluation benchmark.

### Multiple tail bands

Cache the teacher’s top 256 rather than top 64, but use a compact target:

- top 64 individually;
- ranks 65–256 as one aggregate bucket;
- remaining vocabulary as another bucket.

That adds one more degree of distributional shape without requiring full teacher logits during training.

A slightly richer option would divide ranks 65–256 into two logarithmic bands. I would stop there; turning it into a tiny histogramming thesis is probably not the best use of a GPU.

### Variable-\(k\) top-p targets

Cache up to a maximum \(k\), but select the smallest prefix reaching a teacher probability threshold. Confident tokens remain cheap; diffuse tokens retain more explicit alternatives.

### Add a small hard-label term

Teacher KL and true next-token CE are not identical objectives. A small supervised CE component may prevent KD from improving teacher matching while giving up useful task behavior. This should be tested as a fixed arm, not tuned repeatedly against the six-task suite.

## Selection rule

Do not select the endpoint using KD training loss. Use:

1. held-out NLL;
2. held-out full-vocabulary KL;
3. a distributional calibration diagnostic;
4. the larger task suite as a final guardrail.

Experiment 045 already showed that simple earlier stopping was not the missing answer for tail-aware KD: the later checkpoint genuinely remained better on its held-out NLL and KL metrics. 

---

# 5. Jointly allocate rank, outliers, scale precision, and format options

The current recipe fixes residual-selected outlier columns and excludes their cost from the main target-BPW allocation.  That makes experimental comparisons understandable, but it means rank and sidecar storage are not competing under one common rate-distortion problem.

They should.

## Per-owner candidate menu

For each owner, generate mutually exclusive options such as:

- rank \(r-32\), \(r\), \(r+32\);
- no outliers;
- several INT8 outlier counts;
- a smaller number of BF16 outliers;
- free binary factor;
- mixed codebook factor;
- 16-bit or compressed scale representation;
- grouped-middle-core option, discussed below.

Each option must include exact packed costs:

- binary factors;
- pre/mid/post scales;
- indices;
- padding and alignment;
- outlier quantization scales;
- codebook tables and assignments;
- runtime metadata.

Then solve a multiple-choice knapsack or Lagrangian allocation using functional utility.

## Improve the granularity of outliers

The current outlier path removes and restores entire input columns.  For a large output projection, one troublesome subset of rows can therefore force storage of a whole column.

Consider structured sparse sidecars:

- 16- or 32-row column segments;
- small aligned tiles such as 8×8 or 16×8;
- INT8 tile values plus one scale;
- output-adaptive saliency for selecting tiles.

This preserves vectorizable runtime structure without paying for unaffected rows.

The historical outlier results suggest there is useful capacity here, but not enough to justify a blanket increase:

- Moving to 0.2% INT8 outliers improved both perplexity and aggregate task quality at a modest bit increase. 
- Moving to 2% improved perplexity further but cost about 0.146 additional BPW and did not improve the task mean. 

That is exactly the shape of a resource that should be allocated marginally, owner by owner.

Sensitivity-based dense-and-sparse decomposition has also been effective in other low-bit LLM methods, but NanoQuant’s own experiments are already enough reason to test this. 

---

# 6. Test a grouped middle core

This is the most interesting untested representation change I found.

NanoQuant currently reconstructs approximately:

\[
\widehat W
=
D_{\text{post}} L D_{\text{mid}} R D_{\text{pre}}
\]

where \(D_{\text{mid}}\) is diagonal. Each left binary component therefore interacts only with the matching right binary component.

Replace the diagonal middle scale with a small block-diagonal core:

\[
\widehat W
=
D_{\text{post}} L M_{\text{grouped}} R D_{\text{pre}}
\]

using 2×2 or 4×4 blocks.

## Why it may be efficient

A 2×2 block has four coefficients where the existing representation has two diagonal coefficients. Thus it adds only one extra scalar per rank component.

At the roughly 111,000 total rank seen in recent campaigns, BF16 2×2 cores would add approximately **1.8 million bits**, or around **0.22 MB**, before any scale compression. That is generally only a few thousandths of a bit per source weight—far cheaper than adding enough binary rank to create equivalent new cross-component interactions. The recent over-complete-rank result suggests that simply buying more binary vectors is not an efficient capacity axis. 

## Implementation path

The current `_fit_mid` already constructs and solves a weighted linear system for all diagonal middle coefficients.  Generalize its basis from:

\[
L_i R_i
\]

to grouped cross-products:

\[
L_i R_j,\qquad i,j\in g
\]

For a 2×2 group, that is four basis matrices. Solve the grouped coefficients with fixed binary factors, alternating with the existing pre/post scale updates.

Pair components using one of:

- high residual correlation;
- strong off-diagonal Gram coupling;
- adjacent magnitude-sorted middle scales;
- a fixed deterministic grouping as the control.

## Do not build runtime support first

Run a dense splice probe:

1. Fit current diagonal middle scales.
2. Fit 2×2 grouped cores at the same binary rank.
3. Reduce rank slightly so total bits exactly match.
4. Materialize dense candidate weights.
5. Test held-out block output, full-model KL, and NLL.

Use fused QKV, gate, and down owners in blocks 0, 12, and 25. Only build packed and llama.cpp support if this survives the functional equal-bit gate.

This is speculative, but it is a genuinely different capacity mechanism—not another attempt to search harder inside the current representation.

---

# 7. Tighten held-out checkpoint selection and numerical isolation

There are some lower-risk improvements in the existing training pipeline.

The base recipe currently uses legacy training-loss semantics and disables best-state restoration, even though the tuning code supports full-evaluation epoch loss and restoring the best state.  

For new experiments, I would add:

- separate fit and selection activation sets;
- full block-output evaluation at epoch boundaries;
- persisted-dtype evaluation before accepting a state;
- best-state restoration by held-out functional loss;
- optional short suffix KL for high-impact owners;
- a maximum binary Hamming-distance constraint;
- production-horizon normalization of learning rates and step counts.

That final point matters. Experiment 054 found a learning rate that looked excellent on a smaller replay but moved three times farther during the production horizon and produced instability.  Candidate screens should be matched by **total optimizer excursion**, not just nominal learning rate.

For product-codebook option screens, use progressive fidelity:

1. cheap ADMM screen;
2. production-length factorization for survivors;
3. representative factorized tuning;
4. functional splice gate.

A 100-iteration response estimate can be useful for pruning, but it should not be trusted to rank close options against an 800-iteration production solution.

---

# What I would explicitly deprioritize

I would not spend the next campaign on:

- **More direct binary-search tiers under the same reconstruction objective.** Mature ADMM plus the existing search ladder leaves relatively small residual matrix gains, and broad tabu improvements did not reliably transfer functionally. 
- **Dense covariance refinement by itself.** Experiments 033 and 034 already falsified both pre-tuning and selected post-refit placements as default policies.  
- **Hadamard or randomized input rotations.** They worsened joint KL by roughly 47–57% at equal bits across every tested seed. 
- **Over-complete rank.** It was expensive and strongly harmful end to end. 
- **New broad MLP factor fusions.** Reciprocal MLP fusions improved matrix RMSE while badly worsening actual MLP output and teacher KL. QKV remains the only clearly supported fusion. 
- **Global bias correction or unfunded low-rank patches.** The rank-4 patch had a large local `o_proj` gain but worsened the equal-budget model; all-unit bias correction was also harmful. 
- **A larger fixed outlier fraction.** Outliers are useful, but their count should be selected through the same exact-bit functional allocator as rank.

## The experiment order I would use

1. **Stabilize and complete Experiment 048.**
2. **Implement `FunctionalCandidateScorer`** and validate it on nine retained owners.
3. **Use that scorer for a constrained binary-QAT probe** on one retained factorization.
4. **Use it to rebuild rank and outlier response curves.**
5. **Run a joint rank/INT8-sidecar/2×2-core allocation probe.**
6. Launch a new full compression campaign only after one of those candidates improves disjoint NLL and full KL at equal all-in BPW.

My strongest recommendation is therefore:

> **Finish the adaptive KD correction, then make functional model loss—not weighted matrix reconstruction—the common currency used to choose signs, rank, codebook coverage, and outlier bytes.**

That is where the repository’s own evidence points most consistently. I reviewed the current code and retained experiment history statically; I did not run a new full CUDA compression campaign, so the grouped-core and binary-QAT payoff estimates remain hypotheses rather than measured results.