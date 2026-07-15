"""Deterministic, metadata-explicit batched autoregressive generation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch

from nanoquant.runtime.backend import WorkloadKind


class GenerationError(ValueError):
    """Raised when a generation request or model result violates the runtime contract."""


@dataclass(frozen=True, slots=True)
class GenerationStep:
    """One model invocation result with an opaque, model-owned cache handle."""

    logits: torch.Tensor
    cache: object | None


@runtime_checkable
class GenerationModel(Protocol):
    """Minimal model-shell boundary consumed by the deployment generation engine."""

    def forward_step(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        cache: object | None,
        max_cache_length: int,
        workload: WorkloadKind,
        deterministic: bool,
    ) -> GenerationStep: ...


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Device-side token selection policy."""

    mode: Literal["greedy", "sample"] = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("greedy", "sample"):
            raise GenerationError(f"NQ-GEN-SAMPLING unsupported sampling mode: {self.mode}")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise GenerationError("NQ-GEN-SAMPLING temperature must be finite and positive")
        if self.top_k is not None and self.top_k <= 0:
            raise GenerationError("NQ-GEN-SAMPLING top_k must be positive when configured")
        if self.top_p is not None and (
            not math.isfinite(self.top_p) or not 0 < self.top_p <= 1
        ):
            raise GenerationError("NQ-GEN-SAMPLING top_p must be in (0, 1]")
        if self.seed is not None and self.seed < 0:
            raise GenerationError("NQ-GEN-SAMPLING seed must be non-negative")
        if self.mode == "greedy" and (
            self.temperature != 1.0
            or self.top_k is not None
            or self.top_p is not None
            or self.seed is not None
        ):
            raise GenerationError(
                "NQ-GEN-SAMPLING greedy mode does not accept sampling-only settings"
            )
        if self.mode == "sample" and self.seed is None:
            raise GenerationError(
                "NQ-GEN-SAMPLING deterministic sampling requires an explicit seed"
            )


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A rectangular, left-padded prompt batch and deterministic stopping policy."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    max_new_tokens: int
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    deterministic: bool = True
    stop_token_sequences: tuple[tuple[int, ...], ...] = ()
    sampling: SamplingConfig = SamplingConfig()
    stopping_check_interval: int = 1
    prefill_chunk_size: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Rectangular generated tokens plus per-row logical lengths and stop reasons."""

    token_ids: torch.Tensor
    lengths: tuple[int, ...]
    stop_reasons: tuple[str, ...]
    prefill_forward_count: int
    decode_forward_count: int
    maximum_cache_length: int
    stopping_sync_count: int
    terminal_sync_count: int


def batch_prompts(
    prompts: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the left-padded rectangular prompt tensors used by generation.

    Left padding keeps the final input column valid for every sequence, which lets
    the model shell request only the final prompt logit row. The generation engine
    supplies physical cache-aligned positions; padded slots are masked separately.
    """

    if pad_token_id < 0:
        raise GenerationError("NQ-GEN-TOKEN-ID the pad token ID must be non-negative")
    if not prompts:
        raise GenerationError("NQ-GEN-EMPTY prompt batches must be non-empty")
    normalized = tuple(tuple(int(token) for token in prompt) for prompt in prompts)
    if any(not prompt for prompt in normalized):
        raise GenerationError("NQ-GEN-EMPTY every prompt row must contain at least one token")
    if any(token < 0 for prompt in normalized for token in prompt):
        raise GenerationError("NQ-GEN-TOKEN-ID prompt token IDs must be non-negative")

    width = max(len(prompt) for prompt in normalized)
    tokens = torch.full(
        (len(normalized), width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros((len(normalized), width), dtype=torch.bool, device=device)
    for row, prompt in enumerate(normalized):
        start = width - len(prompt)
        tokens[row, start:] = torch.tensor(prompt, dtype=torch.long, device=tokens.device)
        mask[row, start:] = True
    return tokens, mask


def _validate_request(request: GenerationRequest) -> torch.Tensor:
    tokens = request.input_ids
    mask = request.attention_mask
    if tokens.ndim != 2 or mask.ndim != 2 or tokens.shape != mask.shape:
        raise GenerationError("NQ-GEN-SHAPE input IDs and attention mask must have one equal 2D shape")
    if tokens.dtype != torch.long:
        raise GenerationError("NQ-GEN-TOKEN-DTYPE input IDs must use torch.long")
    if tokens.device != mask.device:
        raise GenerationError("NQ-GEN-DEVICE input IDs and attention mask must share a device")
    if tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
        raise GenerationError("NQ-GEN-EMPTY prompt batches and prompt widths must be non-empty")
    if request.max_new_tokens <= 0:
        raise GenerationError("NQ-GEN-LIMIT max_new_tokens must be positive")
    if not request.eos_token_ids or len(set(request.eos_token_ids)) != len(request.eos_token_ids):
        raise GenerationError("NQ-GEN-EOS EOS token IDs must be non-empty and unique")
    if request.pad_token_id < 0 or min(request.eos_token_ids) < 0:
        raise GenerationError("NQ-GEN-TOKEN-ID pad and EOS token IDs must be non-negative")
    if bool(torch.any(tokens < 0)):
        raise GenerationError("NQ-GEN-TOKEN-ID prompt token IDs must be non-negative")
    if len(set(request.stop_token_sequences)) != len(request.stop_token_sequences):
        raise GenerationError("NQ-GEN-STOP stop token sequences must be unique")
    if any(not sequence for sequence in request.stop_token_sequences):
        raise GenerationError("NQ-GEN-STOP stop token sequences must be non-empty")
    if any(token < 0 for sequence in request.stop_token_sequences for token in sequence):
        raise GenerationError("NQ-GEN-TOKEN-ID stop token IDs must be non-negative")
    if not request.deterministic:
        raise GenerationError(
            "NQ-GEN-MODE the generation engine requires deterministic model execution"
        )
    if request.stopping_check_interval <= 0:
        raise GenerationError("NQ-GEN-SYNC stopping_check_interval must be positive")
    if request.prefill_chunk_size is not None and (
        type(request.prefill_chunk_size) is not int or request.prefill_chunk_size <= 0
    ):
        raise GenerationError("NQ-GEN-PREFILL prefill_chunk_size must be positive when configured")

    bool_mask = mask.bool()
    if not bool(torch.all(mask == bool_mask)):
        raise GenerationError("NQ-GEN-MASK attention mask values must be boolean or zero/one")
    lengths = bool_mask.sum(dim=1)
    if bool(torch.any(lengths == 0)):
        raise GenerationError("NQ-GEN-EMPTY every prompt row must contain at least one token")
    # Left padding makes the final prompt position valid for every row. This allows
    # model shells to compute only one vocabulary logit row per sequence.
    if not bool(torch.all(bool_mask[:, -1])):
        raise GenerationError("NQ-GEN-PADDING prompt batches must use left padding")
    if bool_mask.shape[1] > 1 and bool(torch.any(bool_mask[:, :-1] & ~bool_mask[:, 1:])):
        raise GenerationError("NQ-GEN-PADDING attention masks must be contiguous left padding")
    return bool_mask


def _validate_step(
    step: GenerationStep,
    batch_size: int,
    workload: WorkloadKind,
    expected_device: torch.device,
) -> None:
    logits = step.logits
    if logits.ndim != 3 or logits.shape[0] != batch_size or logits.shape[1] != 1:
        raise GenerationError(
            f"NQ-GEN-LOGITS runtime {workload} must return logits shaped [batch, 1, vocabulary]"
        )
    if logits.shape[2] <= 0 or not logits.dtype.is_floating_point:
        raise GenerationError("NQ-GEN-LOGITS model logits must have a floating vocabulary dimension")
    if logits.device != expected_device:
        raise GenerationError(
            f"NQ-GEN-DEVICE runtime {workload} logits must remain on {expected_device}"
        )


def _eos_matches(tokens: torch.Tensor, eos_token_ids: torch.Tensor) -> torch.Tensor:
    return torch.any(tokens.unsqueeze(1) == eos_token_ids.unsqueeze(0), dim=1)


def _stop_sequence_matches(
    generated: torch.Tensor,
    output_index: int,
    stop_token_sequences: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    matches = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
    emitted_count = output_index + 1
    for sequence in stop_token_sequences:
        if sequence.numel() > emitted_count:
            continue
        actual = generated[:, emitted_count - sequence.numel() : emitted_count]
        matches |= torch.all(actual == sequence.unsqueeze(0), dim=1)
    return matches


def _sample_tokens(
    logits: torch.Tensor,
    config: SamplingConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if config.mode == "greedy":
        return torch.argmax(logits, dim=-1)
    filtered = logits.float() / config.temperature
    if config.top_k is not None:
        keep = min(config.top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, keep, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
    if config.top_p is not None and config.top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > config.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        filtered = torch.full_like(filtered, -torch.inf).scatter(
            -1,
            sorted_indices,
            sorted_logits,
        )
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


def generate(request: GenerationRequest, model: GenerationModel) -> GenerationResult:
    """Generate a deterministic greedy batch with explicit attention/cache metadata."""

    mask = _validate_request(request)
    tokens = request.input_ids
    batch_size, prompt_width = tokens.shape
    maximum_cache_length = prompt_width + request.max_new_tokens
    prompt_positions = torch.arange(
        prompt_width,
        dtype=torch.long,
        device=tokens.device,
    ).expand(batch_size, -1).clone()
    prompt_positions.masked_fill_(~mask, 0)
    prompt_cache_positions = torch.arange(prompt_width, dtype=torch.long, device=tokens.device)
    decode_cache_positions = torch.arange(
        prompt_width,
        maximum_cache_length,
        dtype=torch.long,
        device=tokens.device,
    )
    full_attention_mask = torch.zeros(
        (batch_size, maximum_cache_length),
        dtype=torch.bool,
        device=tokens.device,
    )
    full_attention_mask[:, :prompt_width] = mask
    eos_token_ids = torch.tensor(
        request.eos_token_ids,
        dtype=torch.long,
        device=tokens.device,
    )
    stop_token_sequences = tuple(
        torch.tensor(sequence, dtype=torch.long, device=tokens.device)
        for sequence in request.stop_token_sequences
    )
    sampling_generator: torch.Generator | None = None
    if request.sampling.mode == "sample":
        sampling_generator = torch.Generator(device=tokens.device)
        assert request.sampling.seed is not None
        sampling_generator.manual_seed(request.sampling.seed)

    with torch.inference_mode():
        prefill_chunk_size = request.prefill_chunk_size or prompt_width
        cache: object | None = None
        step: GenerationStep | None = None
        prefill_forward_count = 0
        for start in range(0, prompt_width, prefill_chunk_size):
            end = min(start + prefill_chunk_size, prompt_width)
            step = model.forward_step(
                input_ids=tokens[:, start:end],
                attention_mask=full_attention_mask[:, :end],
                position_ids=prompt_positions[:, start:end],
                cache_position=prompt_cache_positions[start:end],
                cache=cache,
                max_cache_length=maximum_cache_length,
                workload="prefill",
                deterministic=request.deterministic,
            )
            _validate_step(step, batch_size, "prefill", tokens.device)
            cache = step.cache
            prefill_forward_count += 1
        assert step is not None

        generated = torch.full(
            (batch_size, request.max_new_tokens),
            request.pad_token_id,
            dtype=torch.long,
            device=tokens.device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=tokens.device)
        ended_by_eos = torch.zeros_like(finished)
        ended_by_stop_sequence = torch.zeros_like(finished)
        cache = step.cache
        decode_forward_count = 0
        stopping_sync_count = 0

        for output_index in range(request.max_new_tokens):
            next_tokens = _sample_tokens(
                step.logits[:, -1, :],
                request.sampling,
                sampling_generator,
            )
            active = ~finished
            emitted = torch.where(active, next_tokens, request.pad_token_id)
            generated[:, output_index] = emitted
            lengths += active.long()
            new_eos = active & _eos_matches(next_tokens, eos_token_ids)
            new_stop_sequence = (
                active
                & ~new_eos
                & _stop_sequence_matches(
                    generated,
                    output_index,
                    stop_token_sequences,
                )
            )
            ended_by_eos |= new_eos
            ended_by_stop_sequence |= new_stop_sequence
            finished |= new_eos | new_stop_sequence

            if output_index + 1 == request.max_new_tokens:
                break
            if (output_index + 1) % request.stopping_check_interval == 0:
                stopping_sync_count += 1
                if bool(torch.all(finished)):
                    break
            if cache is None:
                raise GenerationError("NQ-GEN-CACHE model prefill returned no cache for decode")

            decode_active = ~finished
            decode_tokens = torch.where(decode_active, next_tokens, request.pad_token_id).unsqueeze(1)
            attention_width = prompt_width + output_index + 1
            full_attention_mask[:, attention_width - 1] = decode_active
            decode_positions = torch.where(
                decode_active,
                torch.full(
                    (batch_size,),
                    prompt_width + output_index,
                    dtype=torch.long,
                    device=tokens.device,
                ),
                torch.zeros(batch_size, dtype=torch.long, device=tokens.device),
            ).unsqueeze(1)
            step = model.forward_step(
                input_ids=decode_tokens,
                attention_mask=full_attention_mask[:, :attention_width],
                position_ids=decode_positions,
                cache_position=decode_cache_positions[output_index : output_index + 1],
                cache=cache,
                max_cache_length=maximum_cache_length,
                workload="decode",
                deterministic=request.deterministic,
            )
            _validate_step(step, batch_size, "decode", tokens.device)
            cache = step.cache
            decode_forward_count += 1

    host_metadata = torch.stack(
        (lengths, ended_by_eos.long(), ended_by_stop_sequence.long()),
        dim=1,
    ).cpu()
    host_lengths = tuple(int(value) for value in host_metadata[:, 0].tolist())
    host_eos = tuple(bool(value) for value in host_metadata[:, 1].tolist())
    host_stop_sequence = tuple(bool(value) for value in host_metadata[:, 2].tolist())
    return GenerationResult(
        generated,
        host_lengths,
        tuple(
            "eos" if eos else "stop_sequence" if stop else "max_new_tokens"
            for eos, stop in zip(host_eos, host_stop_sequence, strict=True)
        ),
        prefill_forward_count,
        decode_forward_count,
        maximum_cache_length,
        stopping_sync_count,
        1,
    )
