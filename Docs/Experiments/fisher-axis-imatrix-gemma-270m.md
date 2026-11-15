# Output-aware Fisher row/column IMatrix for Gemma 3 270M

Two compact, importable GGUF importance datasets were created from the
[512-target per-weight Fisher probe](weight-fisher-imatrix-gemma-270m.md).
Both cover all 126 projections of `unsloth/gemma-3-270m-it` revision
`23cf460f6bb16954176b3ddcc8d4f250501458a9`, the checkpoint used by Experiment 075.

| Dataset | Target used for the log-probability gradient | Size |
| --- | --- | ---: |
| `Datasets/gemma-3-270m-it-empirical-fisher-rows-512/imatrix.gguf` | Observed next token; empirical Fisher | 989,568 bytes |
| `Datasets/gemma-3-270m-it-model-fisher-rows-512/imatrix.gguf` | Token sampled from the teacher; model Fisher | 989,568 bytes |

These are measured-gradient importance vectors, not activation rows relabeled
as Fisher. They reuse the exact retained squared per-target weight gradients;
no backwards or calibration targets were repeated for this derivation.

## Aggregation

For each matrix, the source statistic is

\[
F_{ij}=\frac{1}{T}\sum_t
\left(\frac{\partial\log P(t\mid x)}{\partial W_{ij}}\right)^2.
\]

For an independently fitted projection, the exported axes are

\[
r_i=\operatorname{mean}_j F_{ij},\qquad
c_j=\frac{\operatorname{mean}_i F_{ij}}
{\operatorname{mean}_{ij}F_{ij}},\qquad
\widehat F_{ij}=r_i c_j.
\]

Thus the columns have mean one, every row's mean importance is retained, and
the reconstructed matrix has the original column marginals and overall mean.
Simply multiplying two unnormalized Fisher marginals would square the global
importance scale; the division above avoids that. No square root, tempering,
shrinkage coefficient, or evaluation-fitted parameter is added to the axes.
The resident objective takes its own square roots when preconditioning weights.

Experiment 075 fits Q/K/V with a shared input factor. Independent Fisher
column vectors would violate its shared-column contract. For each block, the
three Fisher matrices are therefore treated as a vertical stack: use one
normalized column marginal over the stack and each projection's own row means.
This preserves every row mean and the **stacked** column marginals, and makes
all three input vectors byte-identical. Other projections are fitted separately.
The existing recipe's additional v-projection multiplier of 2 is not baked into
these measured axes; it remains a separate downstream recipe choice.

This is a separable marginal approximation, not the complete diagonal Fisher
and not an SVD. It discards residual row/column interactions. Its full-Fisher
source already ignores cross-weight curvature interactions.

## Format, validation and provenance

The files use the existing `general.type=imatrix` GGUF layout and the normal
`blk.N.*.weight` mappings. `.in_sum2` and `.out_sum2` hold the derived column
and row scores, with counts of one. Those counts represent one derived
aggregate, not one observed activation or 512 independent activation slots.
Custom metadata and `source.json` explicitly record 512 gradient targets from
128 contexts of length 128, target distribution, source Fisher hash, model
hash, calibration token hash, method and producer hash. They do not claim
llama-imatrix activation collection or full-corpus coverage.

Validation passed the existing `load_imatrix` and `resolve_imatrix_axes` paths
against the actual safetensors model inventory, including the QKV group:
126 input vectors, 126 output vectors, correct dimensions, no unused entries,
and exact F32 vector round trips. A second build rederived the axes and audited
both files, reusing them unchanged. Semantic inputs and freshly derived vectors
must match exactly on reuse; the original producer hash remains provenance.

Pinned SHA-256 values:

- Empirical: `5eb23863172817afa4da9b02c7d63bc96bfa676a631111974a81d78a212964a6`
- Teacher-sampled: `95f11c507b9f6be977b1e1cbd9586473c955fc4057b7b519b1208f4ba8ab9565`

Use the `calibration.imatrix` object in each dataset's `source.json`; it includes
the path, hash, and `use_rows: true`. No production recipe was changed.

## Matched exploratory comparison

The unchanged scalar quantizer from the parent probe uses 8×8 tiles, 3- or
4-bit signed codes and one F16 scale per tile (logical projection payload
3.25 or 4.25 BPW). All 126 projections are perturbed. It is not the NanoQuant
binary factorizer and does not produce a deployment GGUF or whole-model BPW.

Evaluation reuses the exact same 24 already-retired WikiText windows and 768
sampled target positions. The clean teacher and both existing row/column
IMatrix controls reproduced the parent's per-sequence results **exactly**.
This is a paired exploratory extension, not an independent confirmation or a
fresh policy gate. Full-vocabulary KL and observed-token NLL are in nats/token;
lower values are better.

| Importance | 3-bit KL | 3-bit NLL | 4-bit KL | 4-bit NLL |
| --- | ---: | ---: | ---: | ---: |
| Existing row + column activation IMatrix | 2.851935 | 6.993601 | 0.755024 | 5.589291 |
| **Empirical Fisher rows + columns** | 3.496838 | 7.620285 | 0.710111 | 5.372279 |
| **Teacher-sampled Fisher rows + columns** | 3.362841 | 7.517334 | 0.681790 | 5.316254 |
| Full empirical Fisher (parent reference) | 3.646642 | 7.824960 | 0.688687 | 5.342509 |
| Full teacher-sampled Fisher (parent reference) | 3.462511 | 7.617947 | 0.685153 | 5.273036 |
| Matched-sample activation rows + columns (parent) | 2.694335 | 6.870793 | 0.676392 | 5.460212 |
| Existing activation columns only (parent) | 3.201177 | 7.610350 | 0.636724 | 5.494794 |

Teacher-sampled Fisher axes versus existing activation rows + columns:

- 3-bit KL is 17.9% worse: paired difference +0.510906, exploratory 95%
  sequence-bootstrap interval [+0.352091, +0.666027].
- 4-bit KL is 9.7% better: difference -0.073234, interval
  [-0.133300, -0.014683]. This still does not establish a KL advantage over
  the cheaper column-only or matched activation controls.
- Against full teacher-sampled Fisher, the 4-bit KL difference is -0.003363,
  interval [-0.020045, +0.013757]. The compact approximation is close on this
  screen, although that interval is not a predeclared noninferiority gate.

The empirical axes have a 4-bit KL improvement of 5.9% versus the existing
row/column IMatrix, but its paired interval [-0.106923, +0.016372] includes zero.
All intervals use the parent's 5,000 paired sequence-bootstrap draws without
multiple-comparison correction.

## Cost and conclusion

Deriving each pair of axes took approximately 0.29 CPU seconds. Each output is
about 0.99 MB, versus 401.09 MB for a full Fisher matrix: approximately 405×
less statistic storage, essentially the current IMatrix size.

This reduction does not erase collection cost: the retained empirical/model
Fisher collections took 203.7/269.2 CPU seconds versus 22.5 seconds for matched
activation statistics. Those historical timings used sparse target sampling
and a concurrent background workload, not a full-corpus GPU throughput test.
A future collector can accumulate row and column marginals immediately from
each squared weight gradient, avoiding permanent full-Fisher storage; backwards
would still be required. That collector is not implemented by this converter.

The requested compact datasets are ready for a controlled NanoQuant experiment.
The teacher-sampled version retains similar 4-bit KL to full Fisher at IMatrix
size, but neither variant justifies a general production replacement: both
lose at 3 bits, and cheaper activation controls remain competitive at 4 bits.
An actual factorization/export comparison and a fresh evaluation are required
before adopting either dataset as a production default.

## Reproduction and retained evidence

```powershell
# Build or hash-audit both compact files from the retained 512-target gradients.
.\.venv\Scripts\python.exe tools/build_fisher_axis_imatrix.py

# Replay the scalar comparison into a new immutable protocol directory.
.\.venv\Scripts\python.exe tools/build_fisher_axis_imatrix.py --evaluate `
  --output evidence/probes/gemma270m-fisher-axes-replay

.\.venv\Scripts\python.exe tools/inspect_imatrix.py `
  Datasets/gemma-3-270m-it-empirical-fisher-rows-512/imatrix.gguf
```

Small receipts, loader validation, per-sequence metrics, paired comparisons,
and the exact initial producer source are retained under
`Results/gemma-3-270m-it-fisher-axis-imatrix/`. The initial evaluation is under
`evidence/probes/gemma270m-fisher-axes/`. Evaluation protocol reuse is strict
about producer bytes; choose a new output directory after editing the tool.

Six new unit cases test marginal preservation, recovery of an exactly
separable matrix without squaring its magnitude, invalid-statistic rejection,
QKV shared columns, exact GGUF round trips, and overwrite rejection. The
focused suite including existing IMatrix unit and resident integration tests
passed all 28 tests. Repository Ruff and mypy also passed.
The full default suite subsequently passed 1,799 tests with 54 expected marker
deselections in 207.22 seconds. CUDA-marked tests and a complete NanoQuant
compression/export run were not included.
