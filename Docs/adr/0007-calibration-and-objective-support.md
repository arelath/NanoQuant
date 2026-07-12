# ADR 0007: Calibration and Hessian support tiers

Status: accepted

The release support tiers are:

| Capability | Tier |
| --- | --- |
| Online Fisher calibration | productized |
| Two-phase Fisher calibration | productized |
| Forward-only calibration | productized low-resource path |
| No calibration / precomputed statistics | productized replay path |
| DBF calibration | research-only |
| Diagonal objective | productized |
| Dense-Hessian objective | productized with an enforced workspace limit |
| Block-diagonal covariance | experimental until Milestone 5 equivalence gates pass |
| Low-rank-plus-diagonal covariance | experimental until Milestone 5 equivalence gates pass |

An executor may make an algorithm-preserving placement fallback. It may not switch calibration method or Hessian
representation without a plan revision, structured event, and explicit recipe policy.

