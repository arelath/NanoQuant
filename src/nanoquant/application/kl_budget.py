"""Persistent KL splice profiles and fail-closed planning sensitivities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from nanoquant.config.codec import canonical_json, from_dict, to_dict
from nanoquant.domain.models import ArtifactRef, ArtifactTypes
from nanoquant.ports.artifact_store import ArtifactStore

KL_BUDGET_EVALUATOR_VERSION = 2


@dataclass(frozen=True, slots=True)
class KlBudgetProvenance:
    model_source: str
    model_revision: str
    recipe_hash: str
    dataset_fingerprint: str
    dataset_slice_hash: str
    source_run_identity: str

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.model_source,
                self.model_revision,
                self.recipe_hash,
                self.dataset_fingerprint,
                self.dataset_slice_hash,
                self.source_run_identity,
            )
        ):
            raise ValueError("KL budget provenance fields must be non-empty")

    @property
    def profile_key(self) -> str:
        return "sha256:" + hashlib.sha256(canonical_json(self).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class KlBudgetArmResult:
    arm: str
    negative_log_likelihood: float
    kl_nats_per_token: float
    token_count: int
    weighted_error: float | None = None

    def __post_init__(self) -> None:
        if not self.arm or self.token_count <= 0:
            raise ValueError("KL budget arm requires an identity and positive token count")
        if not math.isfinite(self.negative_log_likelihood) or not math.isfinite(self.kl_nats_per_token):
            raise ValueError("KL budget arm metrics must be finite")
        if self.kl_nats_per_token < 0:
            raise ValueError("KL budget arm KL must not be negative")
        if self.weighted_error is not None and (
            not math.isfinite(self.weighted_error) or self.weighted_error <= 0
        ):
            raise ValueError("KL budget weighted error must be finite and positive")


@dataclass(frozen=True, slots=True)
class KlBudgetProfile:
    schema_version: int
    provenance: KlBudgetProvenance
    baseline_negative_log_likelihood: float
    arms: tuple[KlBudgetArmResult, ...]
    complete: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported KL budget profile schema")
        if not math.isfinite(self.baseline_negative_log_likelihood):
            raise ValueError("KL budget baseline NLL must be finite")
        names = tuple(arm.arm for arm in self.arms)
        if len(names) != len(set(names)):
            raise ValueError("KL budget profile contains duplicate arms")

    @property
    def profile_key(self) -> str:
        return self.provenance.profile_key


@dataclass(frozen=True, slots=True)
class PersistedKlBudgetProfile:
    reference: ArtifactRef
    profile: KlBudgetProfile


@dataclass(frozen=True, slots=True)
class KlBudgetRequest:
    provenance: KlBudgetProvenance
    arms: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.arms or len(self.arms) != len(set(self.arms)) or any(not arm for arm in self.arms):
            raise ValueError("KL budget request arms must be non-empty and unique")


class KlArmEvaluator(Protocol):
    def __call__(self, arm: str) -> KlBudgetArmResult: ...


class KlProfileCheckpoint(Protocol):
    def __call__(self, profile: KlBudgetProfile) -> None: ...


class KlBudgetWorkflow:
    """Evaluate missing splice arms and checkpoint after every completed arm."""

    def run(
        self,
        request: KlBudgetRequest,
        evaluator: KlArmEvaluator,
        *,
        baseline_negative_log_likelihood: float,
        resume: KlBudgetProfile | None = None,
        checkpoint: KlProfileCheckpoint | None = None,
    ) -> KlBudgetProfile:
        if not math.isfinite(baseline_negative_log_likelihood):
            raise ValueError("KL budget baseline NLL must be finite")
        completed: dict[str, KlBudgetArmResult] = {}
        if resume is not None:
            if resume.provenance.profile_key != request.provenance.profile_key:
                raise ValueError("KL budget checkpoint provenance differs from the request")
            if not set(arm.arm for arm in resume.arms).issubset(request.arms):
                raise ValueError("KL budget checkpoint contains an unrequested arm")
            completed = {arm.arm: arm for arm in resume.arms}
        for arm in request.arms:
            if arm in completed:
                continue
            result = evaluator(arm)
            if result.arm != arm:
                raise ValueError("KL arm evaluator returned a different arm identity")
            completed[arm] = result
            partial = KlBudgetProfile(
                1,
                request.provenance,
                baseline_negative_log_likelihood,
                tuple(completed[name] for name in request.arms if name in completed),
                len(completed) == len(request.arms),
            )
            if checkpoint is not None:
                checkpoint(partial)
        return KlBudgetProfile(
            1,
            request.provenance,
            baseline_negative_log_likelihood,
            tuple(completed[name] for name in request.arms),
            True,
        )


def causal_kl_nll_from_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    token_chunk_size: int = 128,
    teacher_is_log_probs: bool = False,
) -> tuple[float, float, int]:
    """Reduce next-token student NLL and teacher KL with bounded vocabulary temporaries."""

    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 3:
        raise ValueError("teacher and student logits must have the same rank-3 shape")
    if token_ids.shape != teacher_logits.shape[:2] or token_ids.ndim != 2:
        raise ValueError("KL token IDs must match the logits batch and sequence dimensions")
    if token_chunk_size <= 0:
        raise ValueError("KL token chunk size must be positive")
    valid = torch.ones_like(token_ids[:, 1:], dtype=torch.bool)
    if attention_mask is not None:
        if attention_mask.shape != token_ids.shape:
            raise ValueError("KL attention mask must match token IDs")
        valid = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    teacher = teacher_logits[:, :-1].reshape(-1, teacher_logits.shape[-1])
    student = student_logits[:, :-1].reshape(-1, student_logits.shape[-1])
    labels = token_ids[:, 1:].reshape(-1)
    valid_rows = valid.reshape(-1).nonzero(as_tuple=False).reshape(-1)
    if valid_rows.numel() == 0:
        raise ValueError("KL reduction has no valid next-token positions")
    total_nll = 0.0
    total_kl = 0.0
    for start in range(0, valid_rows.numel(), token_chunk_size):
        rows = valid_rows[start : start + token_chunk_size]
        teacher_values = teacher.index_select(0, rows).float()
        teacher_log_probs = teacher_values if teacher_is_log_probs else torch.log_softmax(teacher_values, dim=-1)
        student_log_probs = torch.log_softmax(student.index_select(0, rows).float(), dim=-1)
        teacher_probs = teacher_log_probs.exp()
        total_kl += float((teacher_probs * (teacher_log_probs - student_log_probs)).sum())
        selected_labels = labels.index_select(0, rows).reshape(-1, 1)
        total_nll -= float(student_log_probs.gather(1, selected_labels).sum())
    count = int(valid_rows.numel())
    return total_nll / count, total_kl / count, count


def persist_kl_budget_profile(profile: KlBudgetProfile, artifacts: ArtifactStore) -> PersistedKlBudgetProfile:
    with artifacts.begin_write(ArtifactTypes.KL_BUDGET_PROFILE) as writer:
        (writer.path / "kl-budget-profile.json").write_text(
            json.dumps(to_dict(profile), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    return PersistedKlBudgetProfile(
        ArtifactRef(ArtifactTypes.KL_BUDGET_PROFILE, descriptor.artifact_id, 1),
        profile,
    )


def load_kl_budget_profile(path: str | Path) -> KlBudgetProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("KL budget profile payload must be an object")
    return from_dict(KlBudgetProfile, payload, path="kl_budget_profile")


def validate_kl_budget_profile(
    profile: KlBudgetProfile,
    *,
    model_source: str,
    model_revision: str,
    expected_profile_key: str | None = None,
) -> None:
    if not profile.complete:
        raise ValueError("KL-calibrated planning requires a complete profile")
    if profile.provenance.model_source != model_source or profile.provenance.model_revision != model_revision:
        raise ValueError("KL budget profile model identity is stale")
    if expected_profile_key is not None and profile.profile_key != expected_profile_key:
        raise ValueError("KL budget profile key differs from the configured identity")


def kl_calibrated_sensitivities(
    profile: KlBudgetProfile,
    unit_ids: tuple[str, ...],
    *,
    use_exact_unit_arms: bool = True,
) -> tuple[tuple[str, float], ...]:
    """Return exact per-unit KL/E_w^2 values, with type×block fallback."""

    if not profile.complete:
        raise ValueError("KL-calibrated planning requires a complete profile")
    arms = {arm.arm: arm for arm in profile.arms}
    type_arms = {name[5:]: arm for name, arm in arms.items() if name.startswith("type:")}
    block_arms = {name[6:]: arm for name, arm in arms.items() if name.startswith("block:")}
    type_total = math.fsum(arm.kl_nats_per_token for arm in type_arms.values())
    block_total = math.fsum(arm.kl_nats_per_token for arm in block_arms.values())
    result: list[tuple[str, float]] = []
    for unit_id in unit_ids:
        exact = arms.get(f"unit:{unit_id}") if use_exact_unit_arms else None
        if exact is not None:
            if exact.weighted_error is None:
                raise ValueError(f"KL unit arm is missing weighted error: {unit_id}")
            sensitivity = exact.kl_nats_per_token / (exact.weighted_error**2)
        else:
            try:
                block, unit_type = unit_id.split(":", 1)
            except ValueError as exc:
                raise ValueError(f"invalid KL planning unit identity: {unit_id}") from exc
            type_result = type_arms.get(unit_type)
            block_result = block_arms.get(block)
            if type_result is None or block_result is None or type_total <= 0 or block_total <= 0:
                raise ValueError(f"KL profile cannot resolve planning unit {unit_id}")
            sensitivity = (type_result.kl_nats_per_token / type_total) * (
                block_result.kl_nats_per_token / block_total
            )
        if not math.isfinite(sensitivity) or sensitivity <= 0:
            raise ValueError(f"KL sensitivity is not positive for {unit_id}")
        result.append((unit_id, sensitivity))
    return tuple(result)


__all__ = [
    "KL_BUDGET_EVALUATOR_VERSION",
    "KlBudgetArmResult",
    "KlBudgetProfile",
    "KlBudgetProvenance",
    "KlBudgetRequest",
    "KlBudgetWorkflow",
    "PersistedKlBudgetProfile",
    "causal_kl_nll_from_logits",
    "kl_calibrated_sensitivities",
    "load_kl_budget_profile",
    "persist_kl_budget_profile",
    "validate_kl_budget_profile",
]
