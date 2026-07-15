"""Pinned zero/few-shot multiple-choice task rendering and evaluation."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch

from nanoquant.application.evaluation import EvaluationPartition, EvaluatorRegistry, EvaluatorSpec, LogitsFunction
from nanoquant.application.evaluation_cache import TaskInputCacheIdentity
from nanoquant.config.codec import canonical_json

TokenizePair = Callable[[str, str], tuple[tuple[int, ...], tuple[int, ...]]]


def _hash(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MultipleChoiceTaskSpec:
    task_name: str
    task_version: str
    dataset_name: str
    dataset_config: str | None
    dataset_revision: str
    split: str
    prompt_revision: str
    prompt_template: str
    primary_metric: str
    few_shot_count: int = 0
    selection_seed: int = 0
    maximum_length: int = 2048

    def __post_init__(self) -> None:
        required = (
            self.task_name,
            self.task_version,
            self.dataset_name,
            self.dataset_revision,
            self.split,
            self.prompt_revision,
            self.prompt_template,
        )
        if any(not value for value in required):
            raise ValueError("multiple-choice task identity fields must be non-empty")
        if len(self.dataset_revision) != 40:
            raise ValueError("multiple-choice dataset revision must be a pinned 40-character commit")
        try:
            int(self.dataset_revision, 16)
        except ValueError as exc:
            raise ValueError("multiple-choice dataset revision must be hexadecimal") from exc
        if self.primary_metric not in {"acc", "acc_norm"}:
            raise ValueError("multiple-choice primary metric must be acc or acc_norm")
        if type(self.few_shot_count) is not int or self.few_shot_count < 0:
            raise ValueError("multiple-choice few-shot count must be a non-negative integer")
        if type(self.selection_seed) is not int:
            raise ValueError("multiple-choice selection seed must be an integer")
        if type(self.maximum_length) is not int or self.maximum_length < 2:
            raise ValueError("multiple-choice maximum length must be at least two")

    @property
    def prompt_hash(self) -> str:
        return _hash((self.prompt_revision, self.prompt_template))

    @property
    def semantic_key(self) -> str:
        return _hash(self)

    @property
    def evaluator_spec(self) -> EvaluatorSpec:
        return EvaluatorSpec(
            f"multiple-choice-{self.task_name}",
            f"lm-eval-0.4.12-3ba40d3-task-{self.task_version}",
            "quick",
            (
                ("task_semantic_key", self.semantic_key),
                ("primary_metric", self.primary_metric),
            ),
        )


@dataclass(frozen=True, slots=True)
class MultipleChoiceTokenizerIdentity:
    name: str
    revision: str
    content_hash: str
    parameters: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.revision:
            raise ValueError("multiple-choice tokenizer name and revision are required")
        if not self.content_hash.startswith("sha256:") or len(self.content_hash) != 71:
            raise ValueError("multiple-choice tokenizer content hash must be a sha256 semantic hash")
        try:
            int(self.content_hash[7:], 16)
        except ValueError as exc:
            raise ValueError("multiple-choice tokenizer content hash must be a sha256 semantic hash") from exc
        names = [name for name, _value in self.parameters]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("multiple-choice tokenizer parameters must have unique non-empty names")
        canonical_json(tuple(sorted(self.parameters, key=lambda item: item[0])))


@dataclass(frozen=True, slots=True)
class MultipleChoiceTextExample:
    sample_id: str
    contexts: tuple[str, ...]
    continuations: tuple[str, ...]
    correct_choice: int

    def __post_init__(self) -> None:
        if (
            not self.sample_id
            or len(self.contexts) < 2
            or len(self.contexts) != len(self.continuations)
            or any(not value for value in (*self.contexts, *self.continuations))
            or self.correct_choice < 0
            or self.correct_choice >= len(self.contexts)
        ):
            raise ValueError("invalid multiple-choice text example")

    @property
    def content_hash(self) -> str:
        return _hash(self)


@dataclass(frozen=True, slots=True)
class MultipleChoiceExample:
    sample_id: str
    contexts: tuple[tuple[int, ...], ...]
    continuations: tuple[tuple[int, ...], ...]
    correct_choice: int

    def __post_init__(self) -> None:
        if (
            not self.sample_id
            or len(self.contexts) < 2
            or len(self.contexts) != len(self.continuations)
            or any(not value for value in (*self.contexts, *self.continuations))
            or any(token < 0 for values in (*self.contexts, *self.continuations) for token in values)
            or self.correct_choice < 0
            or self.correct_choice >= len(self.contexts)
        ):
            raise ValueError("invalid tokenized multiple-choice example")

    @property
    def content_hash(self) -> str:
        return _hash(self)


@dataclass(frozen=True, slots=True)
class MultipleChoiceEvaluationRequest:
    task: MultipleChoiceTaskSpec
    examples: tuple[MultipleChoiceExample, ...]
    batch_size: int = 1
    maximum_samples: int | None = None
    pad_token_id: int = 0
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class MultipleChoiceExampleResult:
    sample_id: str
    correct_choice: int
    raw_prediction: int
    normalized_prediction: int
    choice_log_likelihoods: tuple[float, ...]
    choice_mean_log_likelihoods: tuple[float, ...]
    raw_correct: bool
    normalized_correct: bool
    raw_tie: bool
    normalized_tie: bool


@dataclass(frozen=True, slots=True)
class MultipleChoiceEvaluationResult:
    task_semantic_key: str
    task_name: str
    task_version: str
    prompt_hash: str
    sample_count: int
    raw_correct_count: int
    normalized_correct_count: int
    accuracy: float
    normalized_accuracy: float
    primary_metric: str
    primary_value: float
    truncated_choice_count: int
    tie_count: int
    examples: tuple[MultipleChoiceExampleResult, ...]


@dataclass(frozen=True, slots=True)
class PreparedMultipleChoiceInputs:
    task: MultipleChoiceTaskSpec
    text_examples: tuple[MultipleChoiceTextExample, ...]
    examples: tuple[MultipleChoiceExample, ...]
    partition: EvaluationPartition
    cache_identity: TaskInputCacheIdentity
    dataset_content_hash: str


def pinned_legacy_multiple_choice_tasks() -> tuple[MultipleChoiceTaskSpec, ...]:
    harness = "lm-eval-harness@3ba40d3"
    return (
        MultipleChoiceTaskSpec(
            "piqa",
            "1.0",
            "baber/piqa",
            None,
            "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
            "validation",
            f"{harness}:tasks/piqa/piqa.yaml",
            "Question: {{goal}}\nAnswer: || {{sol1, sol2}}",
            "acc_norm",
        ),
        MultipleChoiceTaskSpec(
            "arc_easy",
            "1.0",
            "allenai/ai2_arc",
            "ARC-Easy",
            "210d026faf9955653af8916fad021475a3f00453",
            "test",
            f"{harness}:tasks/arc/arc_easy.yaml",
            "Question: {{question}}\nAnswer: || {{choices.text}}",
            "acc_norm",
        ),
        MultipleChoiceTaskSpec(
            "arc_challenge",
            "1.0",
            "allenai/ai2_arc",
            "ARC-Challenge",
            "210d026faf9955653af8916fad021475a3f00453",
            "test",
            f"{harness}:tasks/arc/arc_challenge.yaml+arc_easy.yaml",
            "Question: {{question}}\nAnswer: || {{choices.text}}",
            "acc_norm",
        ),
        MultipleChoiceTaskSpec(
            "hellaswag",
            "1.0",
            "Rowan/hellaswag",
            None,
            "218ec52e09a7e7462a5400043bb9a69a41d06b76",
            "validation",
            f"{harness}:tasks/hellaswag/hellaswag.yaml+utils.py",
            "preprocess(activity_label + ': ' + ctx_a + ' ' + capitalize(ctx_b)) || preprocess(endings)",
            "acc_norm",
        ),
        MultipleChoiceTaskSpec(
            "winogrande",
            "1.0",
            "allenai/winogrande",
            "winogrande_xl",
            "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
            "validation",
            f"{harness}:tasks/winogrande/default.yaml+preprocess_winogrande.py",
            "sentence[:blank] + option || ' ' + sentence[blank+1:].strip()",
            "acc",
        ),
        MultipleChoiceTaskSpec(
            "boolq",
            "2.0",
            "aps/super_glue",
            "boolq",
            "3de24cf8022e94f4ee4b9d55a6f539891524d646",
            "validation",
            f"{harness}:tasks/super_glue/boolq/default.yaml",
            "{{passage}}\nQuestion: {{question}}?\nAnswer: || {{no, yes}}",
            "acc",
        ),
    )


def _text(document: Mapping[str, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"task document requires non-empty text field: {key}")
    return value


def _integer(document: Mapping[str, object], key: str) -> int:
    value = document.get(key)
    if type(value) is int:
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    raise ValueError(f"task document requires integer field: {key}")


def _hellaswag_text(value: str) -> str:
    result = value.strip().replace(" [title]", ". ")
    result = re.sub(r"\[.*?\]", "", result)
    return result.replace("  ", " ")


def _common_context(context: str, choices: tuple[str, ...], sample_id: str, correct: int) -> MultipleChoiceTextExample:
    return MultipleChoiceTextExample(sample_id, (context,) * len(choices), choices, correct)


def render_pinned_task_document(
    task: MultipleChoiceTaskSpec,
    document: Mapping[str, object],
    *,
    sample_id: str,
    demonstrations: tuple[MultipleChoiceTextExample, ...] = (),
) -> MultipleChoiceTextExample:
    if len(demonstrations) != task.few_shot_count:
        raise ValueError("few-shot demonstrations do not match the pinned task count")
    if task.task_name == "piqa":
        context = f"Question: {_text(document, 'goal')}\nAnswer:"
        piqa_choices = (" " + _text(document, "sol1"), " " + _text(document, "sol2"))
        result = _common_context(context, piqa_choices, sample_id, _integer(document, "label"))
    elif task.task_name in {"arc_easy", "arc_challenge"}:
        arc_choices = document.get("choices")
        if not isinstance(arc_choices, Mapping):
            raise ValueError("ARC document requires choices")
        labels = arc_choices.get("label")
        texts = arc_choices.get("text")
        if not isinstance(labels, list) or not isinstance(texts, list) or len(labels) != len(texts):
            raise ValueError("ARC choices must have aligned labels and text")
        answer = document.get("answerKey")
        if answer not in labels or not all(isinstance(value, str) and value for value in texts):
            raise ValueError("ARC answer key or choice text is invalid")
        text_values = tuple(value for value in texts if isinstance(value, str))
        result = _common_context(
            f"Question: {_text(document, 'question')}\nAnswer:",
            tuple(" " + value for value in text_values),
            sample_id,
            labels.index(answer),
        )
    elif task.task_name == "hellaswag":
        endings = document.get("endings")
        if not isinstance(endings, list) or len(endings) < 2 or not all(isinstance(value, str) for value in endings):
            raise ValueError("HellaSwag document requires endings")
        ending_values = tuple(value for value in endings if isinstance(value, str))
        context = _hellaswag_text(
            f"{_text(document, 'activity_label')}: {_text(document, 'ctx_a')} "
            f"{_text(document, 'ctx_b').capitalize()}"
        )
        result = _common_context(
            context,
            tuple(" " + _hellaswag_text(value) for value in ending_values),
            sample_id,
            _integer(document, "label"),
        )
    elif task.task_name == "winogrande":
        sentence = _text(document, "sentence")
        if sentence.count("_") != 1 or document.get("answer") not in {"1", "2"}:
            raise ValueError("WinoGrande sentence or answer is invalid")
        prefix, suffix = sentence.split("_")
        options = (_text(document, "option1"), _text(document, "option2"))
        result = MultipleChoiceTextExample(
            sample_id,
            tuple(prefix + option for option in options),
            (" " + suffix.strip(),) * 2,
            int(str(document["answer"])) - 1,
        )
    elif task.task_name == "boolq":
        result = _common_context(
            f"{_text(document, 'passage')}\nQuestion: {_text(document, 'question')}?\nAnswer:",
            (" no", " yes"),
            sample_id,
            _integer(document, "label"),
        )
    else:
        raise ValueError(f"unsupported pinned multiple-choice task: {task.task_name}")
    if not demonstrations:
        return result
    prefixes = []
    for demonstration in demonstrations:
        if demonstration.correct_choice >= len(demonstration.contexts):
            raise ValueError("few-shot demonstration answer is invalid")
        prefixes.append(
            demonstration.contexts[demonstration.correct_choice]
            + demonstration.continuations[demonstration.correct_choice]
        )
    prefix = "\n\n".join(prefixes) + "\n\n"
    return MultipleChoiceTextExample(
        result.sample_id,
        tuple(prefix + context for context in result.contexts),
        result.continuations,
        result.correct_choice,
    )


def tokenize_multiple_choice_example(
    example: MultipleChoiceTextExample,
    tokenize_pair: TokenizePair,
) -> MultipleChoiceExample:
    pairs = tuple(
        tokenize_pair(context, continuation)
        for context, continuation in zip(example.contexts, example.continuations, strict=True)
    )
    return MultipleChoiceExample(
        example.sample_id,
        tuple(context for context, _continuation in pairs),
        tuple(continuation for _context, continuation in pairs),
        example.correct_choice,
    )


def prepare_multiple_choice_inputs(
    task: MultipleChoiceTaskSpec,
    documents: Iterable[Mapping[str, object]],
    tokenize_pair: TokenizePair,
    tokenizer: MultipleChoiceTokenizerIdentity,
    *,
    maximum_samples: int | None = None,
    partition_name: str | None = None,
    partition_version: str = "legacy-ordered-limit-v1",
    demonstrations: tuple[MultipleChoiceTextExample, ...] = (),
    preprocessing_version: str = "nanoquant-multiple-choice-v1",
) -> PreparedMultipleChoiceInputs:
    if maximum_samples is not None and (type(maximum_samples) is not int or maximum_samples <= 0):
        raise ValueError("multiple-choice maximum samples must be a positive integer")
    if not partition_version or not preprocessing_version:
        raise ValueError("multiple-choice partition and preprocessing versions are required")
    if len(demonstrations) != task.few_shot_count:
        raise ValueError("few-shot demonstrations do not match the pinned task count")
    selected: list[tuple[str, Mapping[str, object]]] = []
    for index, document in enumerate(documents):
        if maximum_samples is not None and len(selected) >= maximum_samples:
            break
        selected.append((f"{task.task_name}:{task.split}:{index}", dict(document)))
    if not selected:
        raise ValueError("multiple-choice task preparation requires documents")
    text_examples = tuple(
        render_pinned_task_document(
            task,
            document,
            sample_id=sample_id,
            demonstrations=demonstrations,
        )
        for sample_id, document in selected
    )
    examples = tuple(tokenize_multiple_choice_example(example, tokenize_pair) for example in text_examples)
    partition = EvaluationPartition.build(
        partition_name or f"{task.task_name}-{task.split}",
        partition_version,
        examples,
    )
    dataset_content_hash = _hash(
        (
            task.dataset_name,
            task.dataset_config,
            task.dataset_revision,
            task.split,
            selected,
        )
    )
    cache_identity = TaskInputCacheIdentity.build(
        task.evaluator_spec,
        partition,
        task_name=task.task_name,
        task_revision=task.task_version,
        dataset_name=task.dataset_name,
        dataset_revision=task.dataset_revision,
        dataset_content_hash=dataset_content_hash,
        split=task.split,
        tokenizer_name=tokenizer.name,
        tokenizer_revision=tokenizer.revision,
        tokenizer_content_hash=tokenizer.content_hash,
        tokenizer_parameters=tokenizer.parameters,
        prompt_template_revision=task.prompt_revision,
        prompt_template_hash=task.prompt_hash,
        few_shot_count=task.few_shot_count,
        few_shot_item_hashes=tuple(example.content_hash for example in demonstrations),
        selection_seed=task.selection_seed,
        preprocessing_version=preprocessing_version,
    )
    return PreparedMultipleChoiceInputs(
        task,
        text_examples,
        examples,
        partition,
        cache_identity,
        dataset_content_hash,
    )


def _prediction(values: tuple[float, ...]) -> tuple[int, bool]:
    maximum = max(values)
    winners = tuple(index for index, value in enumerate(values) if value == maximum)
    return winners[0], len(winners) > 1


def evaluate_multiple_choice(
    request: MultipleChoiceEvaluationRequest,
    logits: LogitsFunction,
) -> MultipleChoiceEvaluationResult:
    if not request.examples:
        raise ValueError("multiple-choice evaluation requires examples")
    if type(request.batch_size) is not int or request.batch_size <= 0:
        raise ValueError("multiple-choice batch size must be positive")
    if request.maximum_samples is not None and (
        type(request.maximum_samples) is not int or request.maximum_samples <= 0
    ):
        raise ValueError("multiple-choice maximum samples must be a positive integer")
    if type(request.pad_token_id) is not int or request.pad_token_id < 0:
        raise ValueError("multiple-choice pad token ID must be non-negative")
    examples = request.examples[: request.maximum_samples]
    candidates: list[tuple[int, int, tuple[int, ...], int, int]] = []
    truncated = 0
    for example_index, example in enumerate(examples):
        for choice_index, (context, continuation) in enumerate(
            zip(example.contexts, example.continuations, strict=True)
        ):
            if len(continuation) > request.task.maximum_length:
                raise ValueError("multiple-choice continuation does not fit the task maximum length")
            # Match lm-eval's causal window exactly: the final continuation token is
            # a prediction target and is therefore not part of the model input.
            retained_context = context[-(request.task.maximum_length + 1 - len(continuation)) :]
            truncated += int(len(retained_context) != len(context))
            sequence = (*retained_context, *continuation)
            candidates.append((example_index, choice_index, sequence, len(retained_context), len(continuation)))
    raw_scores = [[0.0] * len(example.contexts) for example in examples]
    mean_scores = [[0.0] * len(example.contexts) for example in examples]
    for start in range(0, len(candidates), request.batch_size):
        selected = candidates[start : start + request.batch_size]
        width = max(len(candidate[2]) - 1 for candidate in selected)
        tokens = torch.full(
            (len(selected), width),
            request.pad_token_id,
            dtype=torch.long,
            device=request.device,
        )
        mask = torch.zeros((len(selected), width), dtype=torch.long, device=request.device)
        for row, (_example, _choice, sequence, _context_length, _choice_length) in enumerate(selected):
            model_input = sequence[:-1]
            tokens[row, : len(model_input)] = torch.tensor(model_input, dtype=torch.long, device=request.device)
            mask[row, : len(model_input)] = 1
        prediction = logits(tokens, mask)
        if prediction.ndim != 3 or prediction.shape[:2] != tokens.shape:
            raise ValueError("multiple-choice evaluator logits have an invalid shape")
        log_probabilities = torch.nn.functional.log_softmax(prediction.float(), dim=-1)
        for row, (example_index, choice_index, sequence, context_length, choice_length) in enumerate(selected):
            targets = torch.tensor(sequence[context_length:], dtype=torch.long, device=prediction.device)
            if targets.numel() != choice_length or int(targets.max()) >= prediction.shape[-1]:
                raise ValueError("multiple-choice target token exceeds the logits vocabulary")
            positions = torch.arange(
                context_length - 1,
                context_length - 1 + choice_length,
                device=prediction.device,
            )
            score = float(log_probabilities[row, positions, targets].sum())
            if not math.isfinite(score):
                raise ValueError("multiple-choice evaluator produced a non-finite score")
            raw_scores[example_index][choice_index] = score
            mean_scores[example_index][choice_index] = score / choice_length
    example_results = []
    for example, raw, normalized in zip(examples, raw_scores, mean_scores, strict=True):
        raw_values = tuple(raw)
        normalized_values = tuple(normalized)
        raw_prediction, raw_tie = _prediction(raw_values)
        normalized_prediction, normalized_tie = _prediction(normalized_values)
        example_results.append(
            MultipleChoiceExampleResult(
                example.sample_id,
                example.correct_choice,
                raw_prediction,
                normalized_prediction,
                raw_values,
                normalized_values,
                raw_prediction == example.correct_choice,
                normalized_prediction == example.correct_choice,
                raw_tie,
                normalized_tie,
            )
        )
    results = tuple(example_results)
    raw_correct = sum(result.raw_correct for result in results)
    normalized_correct = sum(result.normalized_correct for result in results)
    accuracy = raw_correct / len(results)
    normalized_accuracy = normalized_correct / len(results)
    primary = accuracy if request.task.primary_metric == "acc" else normalized_accuracy
    return MultipleChoiceEvaluationResult(
        request.task.semantic_key,
        request.task.task_name,
        request.task.task_version,
        request.task.prompt_hash,
        len(results),
        raw_correct,
        normalized_correct,
        accuracy,
        normalized_accuracy,
        request.task.primary_metric,
        primary,
        truncated,
        sum(result.raw_tie or result.normalized_tie for result in results),
        results,
    )


def register_pinned_multiple_choice_evaluators(
    registry: EvaluatorRegistry,
    logits: LogitsFunction,
) -> None:
    for task in pinned_legacy_multiple_choice_tasks():
        def evaluate(request: object, *, expected: MultipleChoiceTaskSpec = task) -> object:
            if not isinstance(request, MultipleChoiceEvaluationRequest) or request.task != expected:
                raise ValueError("multiple-choice request does not match the registered pinned task")
            return evaluate_multiple_choice(request, logits)

        registry.register(task.evaluator_spec, evaluate)
