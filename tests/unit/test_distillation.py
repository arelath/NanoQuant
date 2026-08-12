from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from nanoquant.application.distillation import (
    DistillationResumeState,
    TopKDistillationConfig,
    adaptive_topk_tail_distillation_loss,
    cache_topk_teacher_epoch,
    cache_topk_teacher_targets,
    distill_topk,
    multiband_tail_distillation_loss,
    teacher_topk_logits,
    teacher_topk_logits_with_normalizers,
    topk_distillation_loss,
    topk_mass_floor_distillation_loss,
    topk_tail_distillation_loss,
    topk_tail_with_hard_labels_loss,
    variable_top_p_tail_distillation_loss,
    vocabulary_logsumexp,
)
from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.infrastructure.profiling import Profiler


def test_adaptive_tail_matches_ordered_confidence_policy() -> None:
    head = nn.Linear(3, 6, bias=False)
    hidden = torch.randn(2, 3, generator=torch.Generator().manual_seed(101), requires_grad=True)
    logits = head(hidden.detach())
    values, indices = torch.topk(logits, 3, dim=-1)
    loss = adaptive_topk_tail_distillation_loss(
        hidden, values, indices, torch.logsumexp(logits, dim=-1), head,
        temperature=1.0, vocabulary_chunk_size=3, token_chunk_size=2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_multiband_and_variable_top_p_match_teacher_minimum() -> None:
    head = nn.Linear(4, 9, bias=False)
    hidden = torch.randn(3, 4, generator=torch.Generator().manual_seed(102), requires_grad=True)
    logits = head(hidden.detach())
    values, indices = torch.topk(logits, 6, dim=-1)
    normalizer = torch.logsumexp(logits, dim=-1)
    multiband = multiband_tail_distillation_loss(
        hidden, values, indices, normalizer, head, explicit_tokens=3,
        temperature=1.0, vocabulary_chunk_size=4, token_chunk_size=2,
    )
    variable = variable_top_p_tail_distillation_loss(
        hidden, values, indices, normalizer, head, probability=0.7,
        temperature=1.0, vocabulary_chunk_size=4, token_chunk_size=2,
    )
    assert torch.isfinite(multiband)
    assert torch.isfinite(variable)
    (multiband + variable).backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_hard_label_blend_uses_true_vocabulary_labels() -> None:
    head = nn.Linear(3, 7, bias=False)
    hidden = torch.randn(4, 3, generator=torch.Generator().manual_seed(103), requires_grad=True)
    logits = head(hidden.detach())
    values, indices = torch.topk(logits, 3, dim=-1)
    loss = topk_tail_with_hard_labels_loss(
        hidden, values, indices, torch.logsumexp(logits, dim=-1),
        torch.tensor([1, 2, 3, 4]), head, hard_label_weight=0.1,
        hard_label_mask=torch.tensor([True, True, False, True]),
        temperature=1.0, vocabulary_chunk_size=3, token_chunk_size=2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


class ToyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 6)
        self.projection = nn.Linear(6, 6, bias=False)
        self.norm = nn.LayerNorm(6)
        self.lm_head = nn.Linear(6, 17, bias=False)

    def hidden_states(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(self.embedding(token_ids)))


def _hidden(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    assert isinstance(model, ToyLanguageModel)
    return model.hidden_states(token_ids)


def test_distillation_config_uses_legacy_zero_weight_decay() -> None:
    assert TopKDistillationConfig().objective == "top_k"
    assert TopKDistillationConfig().weight_decay == 0.0
    assert TopKDistillationConfig().maximum_batches_per_epoch is None
    assert TopKDistillationConfig().scheduler_total_steps is None
    assert TopKDistillationConfig().tail_mass_weight == 1.0
    assert TopKDistillationConfig().minimum_teacher_mass_ratio == 0.8
    assert TopKDistillationConfig().mass_floor_weight == 1.0
    assert TopKDistillationConfig().sampling_version == "legacy-python-device-rng-v1"


def test_distillation_config_validates_mass_floor_policy() -> None:
    config = TopKDistillationConfig(
        objective="top_k_mass_floor",
        minimum_teacher_mass_ratio=0.75,
        mass_floor_weight=2.0,
    )
    assert config.objective == "top_k_mass_floor"

    with pytest.raises(ValueError, match="teacher mass ratio"):
        TopKDistillationConfig(minimum_teacher_mass_ratio=0.0)
    with pytest.raises(ValueError, match="mass floor weight"):
        TopKDistillationConfig(mass_floor_weight=float("nan"))
    with pytest.raises(ValueError, match="scheduler total steps"):
        TopKDistillationConfig(scheduler_total_steps=0)


def test_teacher_cache_plan_matches_legacy_python_and_device_rng() -> None:
    torch.manual_seed(6)
    teacher = ToyLanguageModel()
    tokens = torch.arange(20).remainder(17).reshape(4, 5)
    config = TopKDistillationConfig(
        epochs=2,
        batch_size=1,
        top_k=3,
        vocabulary_chunk_size=8,
        token_chunk_size=4,
        maximum_tokens_per_batch=2,
        seed=7,
    )

    first, _ = cache_topk_teacher_epoch(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        epoch_index=0,
        device="cpu",
        pad_token_id=None,
    )
    second, _ = cache_topk_teacher_epoch(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        epoch_index=1,
        device="cpu",
        pad_token_id=None,
    )

    assert [batch.sample_indices for batch in first] == [(3,), (1,), (0,), (2,)]
    assert [batch.token_indices.tolist() for batch in first] == [[0, 1], [3, 4], [3, 2], [3, 2]]
    assert [batch.sample_indices for batch in second] == [(1,), (0,), (2,), (3,)]
    assert [batch.token_indices.tolist() for batch in second] == [[1, 0], [1, 0], [3, 1], [0, 3]]


def test_chunked_teacher_topk_matches_dense_logits() -> None:
    generator = torch.Generator().manual_seed(4)
    head = nn.Linear(5, 19, bias=True)
    hidden = torch.randn(7, 5, generator=generator)

    values, indices = teacher_topk_logits(
        hidden,
        head,
        top_k=6,
        vocabulary_chunk_size=4,
        temperature=0.7,
    )
    expected_values, expected_indices = torch.topk(head(hidden) / 0.7, 6, dim=-1)

    assert torch.allclose(values, expected_values)
    assert torch.equal(indices, expected_indices)

    summary_values, summary_indices, log_normalizers = teacher_topk_logits_with_normalizers(
        hidden,
        head,
        top_k=6,
        vocabulary_chunk_size=4,
        token_chunk_size=3,
        temperature=0.7,
    )
    assert torch.equal(summary_values, values)
    assert torch.equal(summary_indices, indices)
    assert torch.allclose(
        log_normalizers,
        torch.logsumexp(head(hidden).float() / 0.7, dim=-1),
        atol=1e-6,
    )


def test_tail_cache_batch_cap_matches_prefix_of_uncapped_rng_plan() -> None:
    torch.manual_seed(61)
    teacher = ToyLanguageModel()
    tokens = torch.arange(20).remainder(17).reshape(4, 5)
    uncapped_config = TopKDistillationConfig(
        objective="top_k_tail",
        epochs=2,
        batch_size=1,
        top_k=3,
        vocabulary_chunk_size=8,
        token_chunk_size=4,
        maximum_tokens_per_batch=2,
        seed=7,
    )
    capped_config = TopKDistillationConfig(
        objective="top_k_tail",
        epochs=2,
        batch_size=1,
        top_k=3,
        vocabulary_chunk_size=8,
        token_chunk_size=4,
        maximum_tokens_per_batch=2,
        maximum_batches_per_epoch=2,
        tail_mass_weight=0.5,
        seed=7,
    )

    uncapped = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        uncapped_config,
        device="cpu",
        pad_token_id=None,
    )
    capped = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        capped_config,
        device="cpu",
        pad_token_id=None,
    )

    assert tuple(len(epoch) for epoch in capped.epochs) == (2, 2)
    for capped_epoch, uncapped_epoch in zip(capped.epochs, uncapped.epochs, strict=True):
        for capped_batch, uncapped_batch in zip(capped_epoch, uncapped_epoch[:2], strict=True):
            assert capped_batch.sample_indices == uncapped_batch.sample_indices
            assert torch.equal(capped_batch.token_indices, uncapped_batch.token_indices)
            assert torch.equal(capped_batch.top_values, uncapped_batch.top_values)
            assert torch.equal(capped_batch.top_indices, uncapped_batch.top_indices)
            assert capped_batch.teacher_log_normalizers is not None
            assert uncapped_batch.teacher_log_normalizers is not None
            assert torch.equal(
                capped_batch.teacher_log_normalizers,
                uncapped_batch.teacher_log_normalizers,
            )
    expected_bytes = sum(
        tensor.numel() * tensor.element_size()
        for epoch in capped.epochs
        for batch in epoch
        for tensor in (
            batch.token_indices,
            batch.top_values,
            batch.top_indices,
            batch.teacher_log_normalizers,
        )
        if tensor is not None
    )
    assert capped.bytes == expected_bytes


def test_chunked_vocabulary_logsumexp_matches_dense_value_and_gradient() -> None:
    generator = torch.Generator().manual_seed(5)
    head = nn.Linear(5, 19, bias=True)
    hidden = torch.randn(7, 5, generator=generator, requires_grad=True)
    observed = vocabulary_logsumexp(
        hidden,
        head,
        vocabulary_chunk_size=4,
        token_chunk_size=3,
        temperature=0.7,
    )
    expected = torch.logsumexp(head(hidden).float() / 0.7, dim=-1)

    assert torch.allclose(observed, expected, atol=1e-6)
    observed.sum().backward(retain_graph=True)
    observed_gradient = hidden.grad.detach().clone()
    hidden.grad = None
    expected.sum().backward()
    assert torch.allclose(observed_gradient, hidden.grad, atol=1e-6)


def test_topk_tail_loss_detects_mass_shift_hidden_from_conditional_loss() -> None:
    head = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(4))
    teacher_hidden = torch.tensor([[3.0, 2.0, 0.0, -1.0]])
    shifted_student = torch.tensor([[1.0, 0.0, 0.0, -1.0]], requires_grad=True)
    values, indices = torch.topk(teacher_hidden, 2, dim=-1)
    teacher_normalizer = torch.logsumexp(teacher_hidden, dim=-1)
    teacher_loss = topk_tail_distillation_loss(
        teacher_hidden,
        values,
        indices,
        teacher_normalizer,
        head,
        temperature=1.0,
        vocabulary_chunk_size=2,
        token_chunk_size=1,
    )
    shifted_loss = topk_tail_distillation_loss(
        shifted_student,
        values,
        indices,
        teacher_normalizer,
        head,
        temperature=1.0,
        vocabulary_chunk_size=2,
        token_chunk_size=1,
    )
    teacher_probabilities = torch.softmax(teacher_hidden, dim=-1)
    shifted_log_probabilities = torch.log_softmax(shifted_student, dim=-1)
    expected_shifted = -(
        teacher_probabilities[:, :2] * shifted_log_probabilities[:, :2]
    ).sum() - teacher_probabilities[:, 2:].sum() * torch.log(
        shifted_log_probabilities[:, 2:].exp().sum()
    )
    conditional_teacher = topk_distillation_loss(
        teacher_hidden,
        values,
        indices,
        head,
        temperature=1.0,
        token_chunk_size=1,
    )
    conditional_shifted = topk_distillation_loss(
        shifted_student,
        values,
        indices,
        head,
        temperature=1.0,
        token_chunk_size=1,
    )

    assert float(conditional_shifted.detach()) == pytest.approx(
        float(conditional_teacher.detach()),
        abs=1e-7,
    )
    assert shifted_loss > teacher_loss
    assert float(shifted_loss.detach()) == pytest.approx(float(expected_shifted.detach()), abs=1e-6)
    shifted_loss.backward()
    assert shifted_student.grad is not None
    assert torch.isfinite(shifted_student.grad).all()


def test_topk_tail_mass_weight_strengthens_mass_calibration_without_moving_optimum() -> None:
    head = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(4))
    teacher_hidden = torch.tensor([[3.0, 2.0, 0.0, -1.0]])
    shifted_student = torch.tensor([[1.0, 0.0, 0.0, -1.0]])
    values, indices = torch.topk(teacher_hidden, 2, dim=-1)
    teacher_normalizer = torch.logsumexp(teacher_hidden, dim=-1)

    def loss(hidden: torch.Tensor, weight: float) -> torch.Tensor:
        return topk_tail_distillation_loss(
            hidden,
            values,
            indices,
            teacher_normalizer,
            head,
            temperature=1.0,
            vocabulary_chunk_size=2,
            token_chunk_size=1,
            mass_loss_weight=weight,
        )

    base_excess = loss(shifted_student, 1.0) - loss(teacher_hidden, 1.0)
    weighted_excess = loss(shifted_student, 4.0) - loss(teacher_hidden, 4.0)

    assert weighted_excess > base_excess


def test_topk_mass_floor_is_exactly_conditional_above_floor() -> None:
    head = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(4))
    teacher_hidden = torch.tensor([[3.0, 2.0, 0.0, -1.0]])
    student_hidden = teacher_hidden.detach().clone().requires_grad_(True)
    values, indices = torch.topk(teacher_hidden, 2, dim=-1)
    normalizer = torch.logsumexp(teacher_hidden, dim=-1)

    conditional = topk_distillation_loss(
        student_hidden,
        values,
        indices,
        head,
        temperature=1.0,
        token_chunk_size=1,
    )
    conditional_gradient = torch.autograd.grad(conditional, student_hidden, retain_graph=True)[0]
    constrained = topk_mass_floor_distillation_loss(
        student_hidden,
        values,
        indices,
        normalizer,
        head,
        temperature=1.0,
        vocabulary_chunk_size=2,
        token_chunk_size=1,
        minimum_teacher_mass_ratio=0.8,
        mass_loss_weight=0.5,
    )
    constrained_gradient = torch.autograd.grad(constrained, student_hidden)[0]

    assert torch.equal(constrained.detach(), conditional.detach())
    assert torch.equal(constrained_gradient, conditional_gradient)


def test_topk_mass_floor_only_pushes_selected_mass_when_below_floor() -> None:
    head = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(4))
    teacher_hidden = torch.tensor([[3.0, 2.0, 0.0, -1.0]])
    # Preserve the selected-logit difference while shifting probability mass
    # from selected entries to the aggregated tail.
    student_hidden = torch.tensor([[1.0, 0.0, 1.0, 0.0]], requires_grad=True)
    values, indices = torch.topk(teacher_hidden, 2, dim=-1)
    normalizer = torch.logsumexp(teacher_hidden, dim=-1)

    conditional = topk_distillation_loss(
        student_hidden,
        values,
        indices,
        head,
        temperature=1.0,
        token_chunk_size=1,
    )
    conditional_gradient = torch.autograd.grad(conditional, student_hidden, retain_graph=True)[0]
    constrained = topk_mass_floor_distillation_loss(
        student_hidden,
        values,
        indices,
        normalizer,
        head,
        temperature=1.0,
        vocabulary_chunk_size=2,
        token_chunk_size=1,
        minimum_teacher_mass_ratio=0.8,
        mass_loss_weight=0.5,
    )
    constrained_gradient = torch.autograd.grad(constrained, student_hidden)[0]

    assert constrained > conditional
    assert torch.isfinite(constrained_gradient).all()
    mass_gradient = constrained_gradient - conditional_gradient
    assert torch.all(mass_gradient[0, :2] < 0)
    assert torch.all(mass_gradient[0, 2:] > 0)


def test_topk_mass_floor_validates_policy_and_token_weights() -> None:
    head = nn.Linear(2, 3, bias=False)
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values, indices = torch.topk(head(hidden), 2, dim=-1)
    normalizers = torch.logsumexp(head(hidden), dim=-1)
    common = {
        "student_hidden_states": hidden,
        "teacher_top_values": values,
        "teacher_top_indices": indices,
        "teacher_log_normalizers": normalizers,
        "lm_head": head,
        "temperature": 1.0,
        "vocabulary_chunk_size": 2,
        "token_chunk_size": 1,
        "minimum_teacher_mass_ratio": 0.8,
    }

    with pytest.raises(ValueError, match="ratio"):
        topk_mass_floor_distillation_loss(**(common | {"minimum_teacher_mass_ratio": 0.0}))
    with pytest.raises(ValueError, match="weight"):
        topk_mass_floor_distillation_loss(**common, mass_loss_weight=float("nan"))
    with pytest.raises(ValueError, match="positive value"):
        topk_mass_floor_distillation_loss(**common, token_weights=torch.zeros(2))
    with pytest.raises(ValueError, match="match selected tokens"):
        topk_mass_floor_distillation_loss(**common, token_weights=torch.ones(1))


def test_topk_loss_matches_selected_teacher_cross_entropy() -> None:
    generator = torch.Generator().manual_seed(5)
    head = nn.Linear(4, 13, bias=False)
    teacher_hidden = torch.randn(8, 4, generator=generator)
    student_hidden = torch.randn(8, 4, generator=generator)
    teacher_values, teacher_indices = torch.topk(head(teacher_hidden), 5, dim=-1)

    actual = topk_distillation_loss(
        student_hidden,
        teacher_values,
        teacher_indices,
        head,
        temperature=1.0,
        token_chunk_size=3,
    )
    selected_weights = head.weight.index_select(0, teacher_indices.reshape(-1)).view(8, 5, 4)
    selected_student_logits = torch.bmm(selected_weights, student_hidden.unsqueeze(-1)).squeeze(-1)
    expected = -(
        torch.softmax(teacher_values, dim=-1) * torch.log_softmax(selected_student_logits, dim=-1)
    ).sum(dim=-1).mean()

    assert float(actual.detach()) == pytest.approx(float(expected.detach()))


def test_target_mask_and_weights_control_cached_tokens_and_weighted_loss() -> None:
    torch.manual_seed(51)
    teacher = ToyLanguageModel()
    tokens = torch.arange(10).remainder(17).reshape(2, 5)
    target_mask = torch.tensor(
        ((False, True, False, True, False), (True, False, False, False, True))
    )
    weights = torch.tensor(((0.0, 1.0, 0.0, 3.0, 0.0), (2.0, 0.0, 0.0, 0.0, 4.0)))
    config = TopKDistillationConfig(
        epochs=1,
        batch_size=2,
        top_k=4,
        maximum_tokens_per_batch=None,
    )

    cache = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        device="cpu",
        pad_token_id=None,
        target_mask=target_mask,
        target_weights=weights,
    )

    batch = cache.epochs[0][0]
    selected_samples = torch.tensor(batch.sample_indices)
    expected_mask = target_mask.index_select(0, selected_samples).reshape(-1)
    expected_weights = weights.index_select(0, selected_samples).reshape(-1)[expected_mask]
    assert batch.token_indices.tolist() == expected_mask.nonzero().flatten().tolist()
    assert batch.token_weights is not None
    assert torch.equal(batch.token_weights, expected_weights)


def test_cached_topk_distillation_is_bounded_deterministic_and_improves_student() -> None:
    torch.manual_seed(7)
    teacher = ToyLanguageModel()
    student = deepcopy(teacher)
    with torch.no_grad():
        student.projection.weight.add_(
            0.35 * torch.randn(student.projection.weight.shape, generator=torch.Generator().manual_seed(8))
        )
    tokens = torch.randint(1, 17, (8, 7), generator=torch.Generator().manual_seed(9))
    tokens[0, -2:] = 0
    config = TopKDistillationConfig(
        epochs=8,
        batch_size=2,
        learning_rate=0.04,
        top_k=8,
        vocabulary_chunk_size=5,
        token_chunk_size=4,
        maximum_tokens_per_batch=6,
        weight_decay=0.0,
        seed=10,
    )

    cache_progress: list[tuple[str, dict[str, object]]] = []
    cache = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        device="cpu",
        pad_token_id=0,
        progress=lambda event, fields: cache_progress.append((event, dict(fields))),
    )
    repeated = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        device="cpu",
        pad_token_id=0,
    )
    resumed_epoch, resumed_bytes = cache_topk_teacher_epoch(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        epoch_index=5,
        device="cpu",
        pad_token_id=0,
    )
    untouched_embedding = student.embedding.weight.detach().clone()
    training_progress: list[tuple[str, dict[str, object]]] = []
    metrics = distill_topk(
        student,
        tokens,
        student.lm_head,
        _hidden,
        cache,
        config,
        lambda name, _parameter: name == "projection.weight",
        device="cpu",
        progress=lambda event, fields: training_progress.append((event, dict(fields))),
    )

    assert cache.bytes > 0
    assert resumed_bytes == sum(
        value.numel() * value.element_size()
        for batch in cache.epochs[5]
        for value in (batch.token_indices, batch.top_values, batch.top_indices)
    )
    assert all(
        torch.equal(left.token_indices, right.token_indices)
        and torch.equal(left.top_values, right.top_values)
        and torch.equal(left.top_indices, right.top_indices)
        for left, right in zip(cache.epochs[5], resumed_epoch, strict=True)
    )
    assert all(target.token_indices.numel() <= 6 for epoch in cache.epochs for target in epoch)
    assert all(
        torch.equal(left.token_indices, right.token_indices)
        and torch.equal(left.top_values, right.top_values)
        and torch.equal(left.top_indices, right.top_indices)
        for left_epoch, right_epoch in zip(cache.epochs, repeated.epochs, strict=True)
        for left, right in zip(left_epoch, right_epoch, strict=True)
    )
    assert metrics.steps_completed == 32
    assert metrics.selected_parameter_count == 1
    assert metrics.epoch_losses[-1] < metrics.epoch_losses[0]
    assert torch.equal(student.embedding.weight, untouched_embedding)
    assert [event for event, _fields in cache_progress].count(
        "teacher_cache.batch_completed"
    ) == 32
    assert cache_progress[0] == (
        "teacher_cache.epoch_started",
        {"epoch": 0, "epochs": 8, "total_batches": 4},
    )
    assert cache_progress[-1][0] == "teacher_cache.epoch_completed"
    assert [event for event, _fields in training_progress].count(
        "training.batch_completed"
    ) == 32
    assert training_progress[0][0] == "training.started"
    assert training_progress[-1] == (
        "training.completed",
        {
            "completed_steps": 32,
            "total_steps": 32,
            "final_epoch_loss": metrics.epoch_losses[-1],
        },
    )


@pytest.mark.parametrize("objective", ["top_k", "top_k_tail", "top_k_mass_floor"])
def test_topk_distillation_resume_restores_adam_and_scheduler_exactly(
    objective: str,
) -> None:
    torch.manual_seed(11)
    teacher = ToyLanguageModel()
    initial_student = deepcopy(teacher)
    with torch.no_grad():
        initial_student.projection.weight.add_(0.2)
    tokens = torch.randint(1, 17, (6, 5), generator=torch.Generator().manual_seed(12))
    config = TopKDistillationConfig(
        objective=objective,
        epochs=4,
        batch_size=2,
        learning_rate=0.025,
        top_k=6,
        vocabulary_chunk_size=5,
        token_chunk_size=3,
        maximum_tokens_per_batch=7,
        gradient_checkpointing=False,
        weight_decay=0.0,
        seed=13,
    )
    cache = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        device="cpu",
        pad_token_id=None,
    )
    control = deepcopy(initial_student)
    control_metrics = distill_topk(
        control,
        tokens,
        control.lm_head,
        _hidden,
        cache,
        config,
        lambda name, _parameter: name == "projection.weight",
        device="cpu",
    )
    checkpoints: list[DistillationResumeState] = []

    def interrupt(checkpoint: DistillationResumeState) -> None:
        checkpoints.append(checkpoint)
        if checkpoint.completed_epochs == 2:
            raise InterruptedError("test interruption")

    interrupted = deepcopy(initial_student)
    with pytest.raises(InterruptedError, match="test interruption"):
        distill_topk(
            interrupted,
            tokens,
            interrupted.lm_head,
            _hidden,
            cache,
            config,
            lambda name, _parameter: name == "projection.weight",
            device="cpu",
            checkpoint_sink=interrupt,
        )
    resumed = deepcopy(initial_student)
    resumed_metrics = distill_topk(
        resumed,
        tokens,
        resumed.lm_head,
        _hidden,
        cache,
        config,
        lambda name, _parameter: name == "projection.weight",
        device="cpu",
        resume=checkpoints[-1],
    )

    assert resumed_metrics.epoch_losses == pytest.approx(control_metrics.epoch_losses, abs=1e-8)
    assert resumed_metrics.steps_completed == control_metrics.steps_completed
    assert torch.equal(resumed.projection.weight, control.projection.weight)


def test_short_correction_can_retain_a_longer_cosine_schedule_horizon() -> None:
    torch.manual_seed(31)
    teacher = ToyLanguageModel()
    initial_student = deepcopy(teacher)
    with torch.no_grad():
        initial_student.projection.weight.add_(0.2)
    tokens = torch.randint(1, 17, (6, 5), generator=torch.Generator().manual_seed(32))
    long_config = TopKDistillationConfig(
        objective="top_k_mass_floor",
        epochs=4,
        batch_size=2,
        learning_rate=0.025,
        top_k=6,
        vocabulary_chunk_size=5,
        token_chunk_size=3,
        maximum_tokens_per_batch=7,
        maximum_batches_per_epoch=2,
        gradient_checkpointing=False,
        seed=33,
    )
    long_cache = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        long_config,
        device="cpu",
        pad_token_id=None,
    )
    checkpoints: list[DistillationResumeState] = []

    def stop_after_first_epoch(state: DistillationResumeState) -> None:
        checkpoints.append(state)
        raise InterruptedError("first epoch retained")

    with pytest.raises(InterruptedError, match="first epoch retained"):
        distill_topk(
            deepcopy(initial_student),
            tokens,
            initial_student.lm_head,
            _hidden,
            long_cache,
            long_config,
            lambda name, _parameter: name == "projection.weight",
            device="cpu",
            checkpoint_sink=stop_after_first_epoch,
        )

    short_config = replace(
        long_config,
        epochs=1,
        scheduler_total_steps=sum(len(epoch) for epoch in long_cache.epochs),
    )
    short_cache = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        short_config,
        device="cpu",
        pad_token_id=None,
    )
    short_student = deepcopy(initial_student)
    short_metrics = distill_topk(
        short_student,
        tokens,
        short_student.lm_head,
        _hidden,
        short_cache,
        short_config,
        lambda name, _parameter: name == "projection.weight",
        device="cpu",
    )

    retained = checkpoints[-1]
    assert short_metrics.epoch_losses == retained.epoch_losses
    assert torch.equal(
        short_student.projection.weight,
        dict(retained.parameter_values)["projection.weight"],
    )


def test_distillation_micro_profile_preserves_cache_training_and_parameters() -> None:
    torch.manual_seed(21)
    teacher = ToyLanguageModel()
    initial_student = deepcopy(teacher)
    with torch.no_grad():
        initial_student.projection.weight.add_(0.15)
    tokens = torch.randint(1, 17, (4, 5), generator=torch.Generator().manual_seed(22))
    config = TopKDistillationConfig(
        epochs=2,
        batch_size=2,
        learning_rate=0.02,
        top_k=5,
        vocabulary_chunk_size=6,
        token_chunk_size=3,
        maximum_tokens_per_batch=6,
        gradient_checkpointing=False,
        seed=23,
    )
    control_cache = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        device="cpu",
        pad_token_id=None,
    )
    profiler = Profiler(
        ProfilingConfig(level=ProfilingLevel.MICRO, emit_span_events=False),
        run_id="distillation-micro",
    )
    profiled_cache = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        _hidden,
        config,
        device="cpu",
        pad_token_id=None,
        recorder=profiler,
    )
    assert profiled_cache.bytes == control_cache.bytes
    for profiled_epoch, control_epoch in zip(profiled_cache.epochs, control_cache.epochs, strict=True):
        for profiled_batch, control_batch in zip(profiled_epoch, control_epoch, strict=True):
            assert profiled_batch.sample_indices == control_batch.sample_indices
            assert torch.equal(profiled_batch.token_indices, control_batch.token_indices)
            assert torch.equal(profiled_batch.top_values, control_batch.top_values)
            assert torch.equal(profiled_batch.top_indices, control_batch.top_indices)

    control_student = deepcopy(initial_student)
    profiled_student = deepcopy(initial_student)
    control_metrics = distill_topk(
        control_student,
        tokens,
        control_student.lm_head,
        _hidden,
        control_cache,
        config,
        lambda name, _parameter: name == "projection.weight",
        device="cpu",
    )
    profiled_metrics = distill_topk(
        profiled_student,
        tokens,
        profiled_student.lm_head,
        _hidden,
        profiled_cache,
        config,
        lambda name, _parameter: name == "projection.weight",
        device="cpu",
        recorder=profiler,
    )

    assert profiled_metrics == control_metrics
    assert torch.equal(profiled_student.projection.weight, control_student.projection.weight)
    payload = profiler.snapshot()
    phase_paths = {str(phase["path"]) for phase in payload["phases"]}  # type: ignore[index]
    assert {
        "planning",
        "h2d",
        "forward",
        "topk",
        "d2h",
        "epoch/h2d",
        "epoch/forward",
        "epoch/loss",
        "epoch/backward",
        "epoch/optimizer_step",
    } <= phase_paths
    counters = {str(counter["name"]): counter for counter in payload["counters"]}  # type: ignore[index]
    assert counters["distillation.steps"]["total"] == profiled_metrics.steps_completed
    assert counters["distillation.teacher_cache_bytes"]["total"] == profiled_cache.bytes
