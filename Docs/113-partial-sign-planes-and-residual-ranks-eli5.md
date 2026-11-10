# Partial sign planes and residual factor ranks in plain language

Date: 2026-11-10

Both methods help a small compressed model repair the mistakes in its first approximation. A **partial sign plane** stores little plus or minus correction instructions for selected columns. A **residual factor** learns another set of shared patterns that can repair errors across the matrix.

Think of making a small copy of a painting. First we paint the main picture with a limited set of tools. Then we spend the remaining paint on the details that are still wrong. These two methods spend that remaining paint differently.

## The first guess and its leftover error

A model contains large tables of numbers called weight matrices. The rows and columns connect one set of model signals to another. Compressing a matrix means storing a smaller recipe for recreating those numbers.

Our first recipe is already a useful approximation. We calculate what it missed:

```text
Leftover error = original numbers − first approximation

Improved approximation = first approximation + a compressed repair
```

That leftover is the **residual**. If the original number is 0.70 and our first guess is 0.60, the repair should add about 0.10. If our guess is too large, the repair should subtract something.

Both methods below compress this residual rather than starting the whole matrix over.

## A partial sign plane stores directions and shared sizes

Imagine the leftover errors form this tiny table. These numbers are an illustration:

| | Column A | Column B | Column C | Column D |
|---|---:|---:|---:|---:|
| Row 1 | +0.18 | +0.03 | -0.46 | -0.01 |
| Row 2 | -0.14 | +0.02 | +0.19 | 0.00 |

Suppose we can afford to repair only columns A and C. For each number in those columns, we keep one bit saying **add** or **subtract**:

| | Column A | Column C |
|---|---|---|
| Row 1 | Add | Subtract |
| Row 2 | Subtract | Add |

This sheet of signs is the **sign plane**. It is *partial* because some columns receive no correction from this sheet.

The bit tells us the direction, so we still need the size. We store one size knob for each row and one for each selected column. Multiplying the two knobs gives the correction's size at their intersection.

For example, row knobs of 1.0 and 0.5, and column knobs of 0.2 and 0.4, produce:

| | Column A repair | Column C repair |
|---|---:|---:|
| Row 1 | +0.20 | -0.40 |
| Row 2 | -0.10 | +0.20 |

Those repairs are close to the original leftovers. Each cell keeps its own sign, while neighboring cells share the size knobs. Sharing those knobs saves space, but limits how precisely we can choose every correction's magnitude.

In our implementation, we rank columns by their remaining error energy, weighted by estimated importance from sample text. We select the most important columns that fit the byte budget, keep their residual signs, and fit the row and column knobs. Unselected columns get zero correction from the sign plane. Signs can vary within a row or column; a whole column does not have to be positive or negative.

## Residual ranks store another recipe made of shared patterns

The residual-factor method uses the same kind of building blocks as the first approximation.

One building block has a sign pattern for the rows and a sign pattern for the columns. Combining them makes a pattern across the table. We add many such patterns together, with adjustable strengths, to approximate the desired numbers.

The factor's **rank** is the number of these building blocks. Rank 704 means that the repair recipe has 704 components. It does not mean we repair 704 columns: those components can contribute across all the ordinary columns.

The computer stores the row patterns and column patterns as two sign matrices. Their multiplication adds up all the components. There are also size knobs for input columns, individual components, and output rows.

The second factor gets **its own** knobs. It can use different sizes and patterns from the first factor because it is trying to reproduce the leftovers, rather than the original table.

```text
First factor recipe  → main approximation
Second factor recipe → approximation of the leftover error

Add their answers    → improved approximation
```

## How we implemented the residual repair

The residual-repair experiment uses three Gemma 3 1B matrices, each with 1,152 rows and 6,912 columns. For each matrix:

1. We fit a first factor with 1,152 components, using fresh importance measurements from sample text.
2. We subtract that approximation from the target to obtain the residual.
3. We choose the largest affordable residual rank in increments of 32. Under the current budget, that is **704**.
4. We fit a separate factor to the residual with the existing ADMM fitter. ADMM repeatedly adjusts the two factors while encouraging their stored entries to become plus or minus signs. We use 800 iterations, followed by two passes that refine the size knobs.
5. We pack the signs into actual bits, save the full payload, read it back, and reconstruct the repaired matrix before measuring it.

We try three fitting seeds per matrix and keep each method's best reconstruction before inspecting held-out model scores. Seven protected columns per matrix keep their original values through a separate path; they are restored after the two approximations are added.

For the residual stage alone, the two sign matrices have these shapes:

```text
1,152 × 704 row-side signs
  704 × 6,912 column-side signs

Their product describes a 1,152 × 6,912 repair.
```

The sign bits occupy 709,632 bytes. Its three vectors of FP32 size knobs add 35,072 bytes, making the extra factor **744,704 bytes** in this diagnostic format.

## Full rank still leaves room for another repair

An ordinary matrix with unrestricted real numbers can be represented exactly by a suitable full-rank factorization. Our factors have much stricter ingredients: their entries are signs, with shared size knobs. Reaching the first factor's 1,152-component cap therefore does not make its reconstruction exact.

Adding a second factor gives us another independently adjustable recipe. The two stages contain 1,152 and 704 components, but the resulting 1,152-row matrix still cannot have ordinary mathematical rank greater than 1,152. The additional capacity comes from storing more binary patterns and separate knobs.

We keep the two recipes separate because each has its own row and column knobs. Simply appending the new sign patterns to the first pair of factors would generally change their meaning. Our implementation adds the answers from two independently scaled stages.

## How the two repairs compare in our experiment

Both methods receive the same total cap of 1,990,656 bytes, or 2 bits per original matrix number. That budget includes the first factor, protected columns, size knobs, indices, header, and alignment.

| | Partial sign plane | Residual factor |
|---|---|---|
| What we store for the repair | A sign at each selected cell, plus row and column knobs | Two sign matrices, plus column, component, and row knobs |
| Where the repair applies | Selected columns | Across the ordinary matrix columns |
| What the current budget buys | 4,944 of 6,912 columns, about 71.5% | 704 shared components |
| Total stored bytes per matrix | 1,990,560 | 1,979,168 |

The sign plane can choose each selected cell's direction independently. The residual factor uses shared components whose sum determines the correction. Its shared structure can spend bits efficiently when many small patterns together describe the residual well.

We also compare against an INT8 column repair, which stores more precise numbers in fewer columns: 651 under this cap. This is our controlled residual-column comparison, rather than an exact recreation of the existing 060 recipe.

## What has worked so far

After fitting, we made the same small adjustment to the first factor's size knobs in each method, using the full model's predictions. The correction signs and correction scales stayed fixed during this adjustment.

On an independent 96-sequence C4 confirmation set, the residual-factor composition across the three matrices had **2.47% lower teacher KL** than unchanged 060 and **0.93% lower teacher KL** than the tuned sign-plane composition. Lower teacher KL means the compressed model's prediction probabilities are closer to the original model's. It also improved next-token NLL, which measures how well it predicts the actual text. These are prediction-quality improvements, rather than percentages of additional answers correct.

This result covers three replacement matrices in an otherwise unchanged model. Their residual-factor payloads together cost 279,240 bytes more than the corresponding 060 payloads. Our representation comparisons share a common cap, while 060 remains a smaller reference.

The repairs currently run through reconstructed dense weights for research. A deployed residual-factor implementation would evaluate two factor stages and add their outputs, preserving the protected-column handling. Packed-format and runtime support, inference-speed measurements, and whole-model task results remain later work.

For the implementation, see [the byte-budget fitter](../tools/probe_residual_stage_budget.py), [the multi-block fitting driver](../tools/probe_residual_multiblock_fit.py), and [the functional comparison](../tools/probe_residual_multiblock_functional.py). The complete measurements are in [the multi-block gate report](111-residual-stage-multiblock-gate.md).
