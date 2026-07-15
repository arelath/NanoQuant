from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nanoquant.application.evaluation import EvaluatorRegistry
from nanoquant.application.task_evaluation import (
    MultipleChoiceEvaluationRequest,
    MultipleChoiceExample,
    MultipleChoiceTaskSpec,
    MultipleChoiceTokenizerIdentity,
    evaluate_multiple_choice,
    pinned_legacy_multiple_choice_tasks,
    prepare_multiple_choice_inputs,
    register_pinned_multiple_choice_evaluators,
    render_pinned_task_document,
    tokenize_multiple_choice_example,
)


def _task(name: str) -> MultipleChoiceTaskSpec:
    return next(task for task in pinned_legacy_multiple_choice_tasks() if task.task_name == name)


def _known_logits(tokens: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    del attention_mask
    result = torch.full((*tokens.shape, 32), -5.0)
    for row in range(tokens.shape[0]):
        first = int(tokens[row, 0])
        if first == 1:
            result[row, 0, 2] = 5.0
        elif first == 4:
            result[row, 0, 6] = 0.0
        elif first == 5:
            result[row, 0, 6] = 5.0
    return result


def _examples() -> tuple[MultipleChoiceExample, ...]:
    return (
        MultipleChoiceExample("first", ((1,), (1,)), ((2,), (3,)), 0),
        MultipleChoiceExample("second", ((4,), (5,)), ((6,), (6,)), 1),
    )


def test_supported_tasks_pin_legacy_task_prompt_and_dataset_revisions() -> None:
    tasks = pinned_legacy_multiple_choice_tasks()

    assert [task.task_name for task in tasks] == [
        "piqa",
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "boolq",
    ]
    assert all(len(task.dataset_revision) == 40 for task in tasks)
    assert all(task.prompt_revision.startswith("lm-eval-harness@3ba40d3:") for task in tasks)
    assert len({task.prompt_hash for task in tasks}) == len(tasks)
    assert len({task.semantic_key for task in tasks}) == len(tasks)
    assert all(task.few_shot_count == 0 for task in tasks)
    assert [task.split for task in tasks] == [
        "validation",
        "test",
        "test",
        "validation",
        "validation",
        "validation",
    ]
    assert [task.primary_metric for task in tasks] == [
        "acc_norm",
        "acc_norm",
        "acc_norm",
        "acc_norm",
        "acc",
        "acc",
    ]


def test_task_renderers_match_retained_lm_eval_prompt_arguments() -> None:
    piqa = render_pinned_task_document(
        _task("piqa"),
        {"goal": "Open the jar?", "sol1": "Use a lid grip.", "sol2": "Freeze it.", "label": 0},
        sample_id="piqa-0",
    )
    arc = render_pinned_task_document(
        _task("arc_easy"),
        {
            "question": "What is 1+1?",
            "choices": {"label": ["A", "B"], "text": ["one", "two"]},
            "answerKey": "B",
        },
        sample_id="arc-0",
    )
    hella = render_pinned_task_document(
        _task("hellaswag"),
        {
            "activity_label": "Roof removal",
            "ctx_a": "A man is on a roof.",
            "ctx_b": "he",
            "endings": ["pulls tiles.", " [title] falls [noise] down."],
            "label": "0",
        },
        sample_id="hella-0",
    )
    wino = render_pinned_task_document(
        _task("winogrande"),
        {
            "sentence": "Sarah was better than Maria so _ got the cases.",
            "option1": "Sarah",
            "option2": "Maria",
            "answer": "2",
        },
        sample_id="wino-0",
    )
    boolq = render_pinned_task_document(
        _task("boolq"),
        {"passage": "The retained passage.", "question": "is this retained", "label": 1},
        sample_id="boolq-0",
    )

    assert piqa.contexts == ("Question: Open the jar?\nAnswer:",) * 2
    assert piqa.continuations == (" Use a lid grip.", " Freeze it.")
    assert arc.contexts == ("Question: What is 1+1?\nAnswer:",) * 2
    assert arc.continuations == (" one", " two") and arc.correct_choice == 1
    assert hella.contexts == ("Roof removal: A man is on a roof. He",) * 2
    assert hella.continuations == (" pulls tiles.", "  falls down.")
    assert wino.contexts == (
        "Sarah was better than Maria so Sarah",
        "Sarah was better than Maria so Maria",
    )
    assert wino.continuations == (" got the cases.",) * 2 and wino.correct_choice == 1
    assert boolq.contexts == ("The retained passage.\nQuestion: is this retained?\nAnswer:",) * 2
    assert boolq.continuations == (" no", " yes") and boolq.correct_choice == 1


def test_few_shot_rendering_and_pair_tokenization_are_explicit() -> None:
    zero_shot = _task("piqa")
    demonstration = render_pinned_task_document(
        zero_shot,
        {"goal": "Demo?", "sol1": "right", "sol2": "wrong", "label": 0},
        sample_id="demo",
    )
    one_shot = replace(zero_shot, few_shot_count=1, selection_seed=19)
    rendered = render_pinned_task_document(
        one_shot,
        {"goal": "Query?", "sol1": "first", "sol2": "second", "label": 1},
        sample_id="query",
        demonstrations=(demonstration,),
    )

    assert rendered.contexts[0] == "Question: Demo?\nAnswer: right\n\nQuestion: Query?\nAnswer:"
    tokenized = tokenize_multiple_choice_example(
        rendered,
        lambda context, continuation: (
            tuple(ord(character) for character in context),
            tuple(ord(character) for character in continuation),
        ),
    )
    assert tokenized.correct_choice == 1
    assert tokenized.contexts[0][-1] == ord(":")
    assert tokenized.continuations[0][0] == ord(" ")


def test_prepared_inputs_bind_documents_tokens_prompts_and_tokenizer_behavior() -> None:
    task = _task("piqa")
    documents = (
        {"goal": "First?", "sol1": "yes", "sol2": "no", "label": 0},
        {"goal": "Second?", "sol1": "left", "sol2": "right", "label": 1},
    )
    tokenizer = MultipleChoiceTokenizerIdentity(
        "google/gemma-3-1b-it",
        "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "sha256:" + "1" * 64,
        (("add_special_tokens", True), ("pair_encoding", "lm-eval-causal-pair-v1")),
    )
    tokenize_pair = lambda context, continuation: (  # noqa: E731
        tuple(ord(character) for character in context),
        tuple(ord(character) for character in continuation),
    )

    prepared = prepare_multiple_choice_inputs(
        task,
        documents,
        tokenize_pair,
        tokenizer,
        maximum_samples=1,
    )
    changed = prepare_multiple_choice_inputs(
        task,
        ({**documents[0], "sol1": "changed"},),
        tokenize_pair,
        tokenizer,
        maximum_samples=1,
    )

    assert len(prepared.examples) == len(prepared.text_examples) == 1
    assert prepared.partition.item_hashes == (prepared.examples[0].content_hash,)
    assert prepared.cache_identity.partition_content_hash == prepared.partition.content_hash
    assert prepared.cache_identity.prompt_template_hash == task.prompt_hash
    assert prepared.cache_identity.dataset_revision == task.dataset_revision
    assert prepared.cache_identity.tokenizer_parameters == tokenizer.parameters
    assert prepared.cache_identity.semantic_key != changed.cache_identity.semantic_key


def test_known_logits_batching_primary_metric_and_sample_limiting() -> None:
    task = _task("piqa")
    serial = evaluate_multiple_choice(
        MultipleChoiceEvaluationRequest(task, _examples(), batch_size=1),
        _known_logits,
    )
    batched = evaluate_multiple_choice(
        MultipleChoiceEvaluationRequest(task, _examples(), batch_size=3),
        _known_logits,
    )
    limited = evaluate_multiple_choice(
        MultipleChoiceEvaluationRequest(task, _examples(), batch_size=4, maximum_samples=1),
        _known_logits,
    )

    assert batched == serial
    assert serial.sample_count == serial.raw_correct_count == serial.normalized_correct_count == 2
    assert serial.accuracy == serial.normalized_accuracy == serial.primary_value == 1.0
    assert serial.primary_metric == "acc_norm"
    assert limited.sample_count == 1 and limited.examples[0].sample_id == "first"
    assert all(
        normalized == pytest.approx(raw)
        for result in serial.examples
        for raw, normalized in zip(
            result.choice_log_likelihoods,
            result.choice_mean_log_likelihoods,
            strict=True,
        )
    )


def test_left_truncation_preserves_all_choice_tokens_and_is_reported() -> None:
    task = replace(_task("boolq"), maximum_length=4)
    example = MultipleChoiceExample(
        "long",
        ((7, 8, 9, 10, 1), (7, 8, 9, 10, 1)),
        ((2,), (3,)),
        0,
    )
    result = evaluate_multiple_choice(
        MultipleChoiceEvaluationRequest(task, (example,), batch_size=2),
        _known_logits,
    )

    assert result.truncated_choice_count == 2
    assert result.sample_count == 1


def test_registry_dispatches_only_the_exact_pinned_task_contract() -> None:
    registry = EvaluatorRegistry()
    register_pinned_multiple_choice_evaluators(registry, _known_logits)
    task = _task("piqa")
    request = MultipleChoiceEvaluationRequest(task, _examples())

    result = registry.evaluate(task.evaluator_spec.name, task.evaluator_spec.version, request)

    assert result == evaluate_multiple_choice(request, _known_logits)
    assert len(registry.specifications_for_tier("quick")) == 6
    with pytest.raises(ValueError, match="does not match"):
        registry.evaluate(
            task.evaluator_spec.name,
            task.evaluator_spec.version,
            replace(request, task=replace(task, selection_seed=1)),
        )
