# Experiment 057: matched Experiment 056 product-k16 encoding

## Question

Can the two-8-bit, no-flip product code improve the use of the 1.0-BPW budget
when every non-encoding choice and every quality gate matches completed
Experiment 056?

## Controlled change

Experiment 057 is defined directly against Experiment 056. The configuration
diff is restricted to the factorization implementation and its enabled product
codebook policy, plus the required experiment identity/output paths.

The unchanged control includes:

- pinned Gemma model/revision, calibration tokens, corrected-CCE Fisher state,
  and exact-unit KL profile;
- physical rank cap, 1.5x measured response range, global 1.0-BPW allocation,
  and V-member multiplier;
- residual-selected 0.1% BF16 outliers;
- factorization orientation and ADMM schedule;
- control-then-tabu binary search, layer/block tuning, post-block refit, and
  top-64 KL distillation; and
- the retained 64-sequence WikiText and six-task, limit-1,000 quality protocol.

The changed policy is `product-codebook-free-k16-v1` on eligible MLP right
factors. A coded 32-sign word stores two 8-bit selectors into learned 256 by 16
half-word tables. There is no correction or bit-flip stream. Each eligible
owner retains the ordinary free-factor option and adds 32-row-aligned mixed
free-prefix/coded-suffix options at the 056-selected rank, the midpoint to the
unchanged physical cap, and that cap. Candidate-specific measured response,
rather than Experiment 056's old rank anchors alone, drives the joint global
rank/encoding choice. The 0.1% BF16 outlier sidecars remain outside the charged
1.0-BPW ceiling exactly as they were in 056.

## Validity requirements

The run must not silently decode the codebook and tune every sign as a free
parameter. Coded rows remain on the code manifold throughout factorized tuning,
binary search, post-block refit, and distillation; only explicitly free rows and
already charged continuous axes may move. The committed artifact must retain
the tables, assignments, free-row count, and exact logical bit charge, and the
packed evaluator must execute that representation.

This is a semantic resident numerical-path change and therefore requires a
`RESIDENT_ALGORITHM_VERSION` increment when the resident implementation is
enabled. A tiny CPU regression must prove the constraint and artifact replay
before launching the real model.

## Promotion gate

Promotion requires a completed, freshly validated resident run and export,
effective BPW no greater than 1.0 except already declared Experiment 056
sidecar treatment, and protocol-matched quality that improves on Experiment
056. Reconstruction or KL-proxy gains alone are not sufficient.

## Current status

The numbered zero-argument definition, resident measured-option allocator,
immutable coded-suffix tuning/search path, persisted table/assignment state,
exact-cost retry accounting, and packed product overlay export are implemented.
The option screen is resumable and uses 100 outer ADMM iterations per candidate,
matching the role of 056's measured rank screen; production factorization keeps
the unchanged 800-iteration schedule. CPU/tiny tests prove constraint gradients,
plan allocation, exact bit accounting, and exact packed replay. The real CUDA
launch remains pending the complete repository
validation and single-worker/device safety check.
