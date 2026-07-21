"""Persistent KL splice profiles and fail-closed planning sensitivities."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from nanoquant.config.codec import canonical_json, from_dict, to_dict
from nanoquant.domain.models import ArtifactRef, ArtifactTypes
from nanoquant.ports.artifact_store import ArtifactStore

KL_BUDGET_EVALUATOR_VERSION = 3
KL_BUDGET_PROFILE_SCHEMA_VERSION = 2


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
class KlSequenceResult:
    negative_log_likelihood: float
    kl_nats_per_token: float
    token_count: int

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise ValueError("KL sequence result requires a positive token count")
        if not math.isfinite(self.negative_log_likelihood) or not math.isfinite(self.kl_nats_per_token):
            raise ValueError("KL sequence metrics must be finite")
        if self.kl_nats_per_token < 0:
            raise ValueError("KL sequence KL must not be negative")


@dataclass(frozen=True, slots=True)
class KlBudgetArmResult:
    arm: str
    negative_log_likelihood: float
    kl_nats_per_token: float
    token_count: int
    weighted_normalized_squared_error: float | None = None
    sequences: tuple[KlSequenceResult, ...] = ()

    def __post_init__(self) -> None:
        if not self.arm or self.token_count <= 0:
            raise ValueError("KL budget arm requires an identity and positive token count")
        if not math.isfinite(self.negative_log_likelihood) or not math.isfinite(self.kl_nats_per_token):
            raise ValueError("KL budget arm metrics must be finite")
        if self.kl_nats_per_token < 0:
            raise ValueError("KL budget arm KL must not be negative")
        if self.weighted_normalized_squared_error is not None and (
            not math.isfinite(self.weighted_normalized_squared_error)
            or self.weighted_normalized_squared_error <= 0
        ):
            raise ValueError("KL budget normalized weighted squared error must be finite and positive")
        if not self.sequences or math.fsum(item.token_count for item in self.sequences) != self.token_count:
            raise ValueError("KL budget arm sequence results must exactly cover its tokens")
        sequence_nll = math.fsum(item.negative_log_likelihood * item.token_count for item in self.sequences)
        sequence_kl = math.fsum(item.kl_nats_per_token * item.token_count for item in self.sequences)
        if not math.isclose(sequence_nll / self.token_count, self.negative_log_likelihood, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("KL budget arm NLL differs from its sequence results")
        if not math.isclose(sequence_kl / self.token_count, self.kl_nats_per_token, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("KL budget arm KL differs from its sequence results")


@dataclass(frozen=True, slots=True)
class KlBudgetProfile:
    schema_version: int
    provenance: KlBudgetProvenance
    baseline_negative_log_likelihood: float
    arms: tuple[KlBudgetArmResult, ...]
    complete: bool

    def __post_init__(self) -> None:
        if self.schema_version != KL_BUDGET_PROFILE_SCHEMA_VERSION:
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
                KL_BUDGET_PROFILE_SCHEMA_VERSION,
                request.provenance,
                baseline_negative_log_likelihood,
                tuple(completed[name] for name in request.arms if name in completed),
                len(completed) == len(request.arms),
            )
            if checkpoint is not None:
                checkpoint(partial)
        return KlBudgetProfile(
            KL_BUDGET_PROFILE_SCHEMA_VERSION,
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


def causal_kl_nll_per_sequence_from_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    token_chunk_size: int = 128,
    teacher_is_log_probs: bool = False,
) -> tuple[KlSequenceResult, ...]:
    """Reduce each sequence independently so profile uncertainty remains recoverable."""

    if teacher_logits.ndim != 3:
        raise ValueError("KL per-sequence reduction requires rank-3 logits")
    results = []
    for index in range(teacher_logits.shape[0]):
        mask = None if attention_mask is None else attention_mask[index : index + 1]
        nll, kl, count = causal_kl_nll_from_logits(
            teacher_logits[index : index + 1],
            student_logits[index : index + 1],
            token_ids[index : index + 1],
            attention_mask=mask,
            token_chunk_size=token_chunk_size,
            teacher_is_log_probs=teacher_is_log_probs,
        )
        results.append(KlSequenceResult(nll, kl, count))
    return tuple(results)


@dataclass(frozen=True, slots=True)
class KlBootstrapInterval:
    point_delta: float
    lower_delta: float
    upper_delta: float
    confidence: float
    resamples: int


def paired_bootstrap_kl_delta(
    before: KlBudgetArmResult,
    after: KlBudgetArmResult,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> KlBootstrapInterval:
    """Return a deterministic paired sequence-bootstrap interval for ``after - before`` KL."""

    if not 0 < confidence < 1 or resamples <= 0:
        raise ValueError("KL bootstrap confidence and resample count are invalid")
    if len(before.sequences) != len(after.sequences) or tuple(
        item.token_count for item in before.sequences
    ) != tuple(item.token_count for item in after.sequences):
        raise ValueError("paired KL bootstrap requires the same ordered sequence inventory")
    count = len(before.sequences)
    generator = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        indices = tuple(generator.randrange(count) for _index in range(count))
        tokens = math.fsum(before.sequences[index].token_count for index in indices)
        before_kl = math.fsum(
            before.sequences[index].kl_nats_per_token * before.sequences[index].token_count
            for index in indices
        ) / tokens
        after_kl = math.fsum(
            after.sequences[index].kl_nats_per_token * after.sequences[index].token_count
            for index in indices
        ) / tokens
        deltas.append(after_kl - before_kl)
    deltas.sort()

    def quantile(probability: float) -> float:
        position = probability * (len(deltas) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return deltas[lower]
        fraction = position - lower
        return deltas[lower] * (1 - fraction) + deltas[upper] * fraction

    tail = (1 - confidence) / 2
    return KlBootstrapInterval(
        after.kl_nats_per_token - before.kl_nats_per_token,
        quantile(tail),
        quantile(1 - tail),
        confidence,
        resamples,
    )


def persist_kl_budget_profile(profile: KlBudgetProfile, artifacts: ArtifactStore) -> PersistedKlBudgetProfile:
    with artifacts.begin_write(ArtifactTypes.KL_BUDGET_PROFILE, KL_BUDGET_PROFILE_SCHEMA_VERSION) as writer:
        (writer.path / "kl-budget-profile.json").write_text(
            json.dumps(to_dict(profile), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        descriptor = writer.commit()
    return PersistedKlBudgetProfile(
        ArtifactRef(
            ArtifactTypes.KL_BUDGET_PROFILE,
            descriptor.artifact_id,
            KL_BUDGET_PROFILE_SCHEMA_VERSION,
        ),
        profile,
    )


def load_kl_budget_profile(path: str | Path) -> KlBudgetProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("KL budget profile payload must be an object")
    if payload.get("schema_version") != KL_BUDGET_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported KL budget profile schema")
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
            if exact.weighted_normalized_squared_error is None:
                raise ValueError(f"KL unit arm is missing normalized weighted squared error: {unit_id}")
            sensitivity = exact.kl_nats_per_token / exact.weighted_normalized_squared_error
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


def measured_unit_kl_anchors(
    profile: KlBudgetProfile,
    unit_ids: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    """Return measured KL for complete physical-unit arms without an error proxy.

    These values anchor a separately measured, same-run relative rank-response
    curve.  They intentionally do not divide by a reconstruction error from a
    different factorization or operating point.
    """

    if not profile.complete:
        raise ValueError("measured-unit KL planning requires a complete profile")
    arms = {arm.arm: arm for arm in profile.arms}
    result: list[tuple[str, float]] = []
    for unit_id in unit_ids:
        arm = arms.get(f"unit:{unit_id}")
        if arm is None:
            raise ValueError(f"KL profile has no exact physical-unit arm: {unit_id}")
        value = arm.kl_nats_per_token
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"measured unit KL is not positive for {unit_id}")
        result.append((unit_id, value))
    return tuple(result)


__all__ = [
    "KL_BUDGET_EVALUATOR_VERSION",
    "KL_BUDGET_PROFILE_SCHEMA_VERSION",
    "KlBootstrapInterval",
    "KlBudgetArmResult",
    "KlBudgetProfile",
    "KlBudgetProvenance",
    "KlBudgetRequest",
    "KlBudgetWorkflow",
    "KlSequenceResult",
    "PersistedKlBudgetProfile",
    "causal_kl_nll_from_logits",
    "causal_kl_nll_per_sequence_from_logits",
    "kl_calibrated_sensitivities",
    "measured_unit_kl_anchors",
    "load_kl_budget_profile",
    "persist_kl_budget_profile",
    "paired_bootstrap_kl_delta",
    "validate_kl_budget_profile",
]
