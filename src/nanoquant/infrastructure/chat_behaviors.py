"""Qwen3 chat-template integration with strict reasoning-mode invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from nanoquant.config.schema import ReasoningMode
from nanoquant.ports.chat_behavior import (
    ChatBehaviorPort,
    ReasoningModeId,
    RenderedBehaviorRecord,
    TokenRole,
)


def _token_ids(value: object) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("chat template did not return token IDs")
    return tuple(int(item) for item in value)


def _subsequence(values: tuple[int, ...], needle: tuple[int, ...], start: int = 0) -> int:
    if not needle:
        raise ValueError("delimiter tokenization is empty")
    for index in range(start, len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


class Qwen3ChatBehavior:
    family = "qwen3"

    @property
    def supported_modes(self) -> tuple[ReasoningMode, ...]:
        return (ReasoningMode.THINKING, ReasoningMode.NON_THINKING)

    def policy_identity(self, tokenizer: Any) -> str:
        payload = {
            "implementation": "qwen3-chat-behavior-v1",
            "chat_template": getattr(tokenizer, "chat_template", None),
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
            "thinking_argument": "enable_thinking",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def _apply(
        self,
        tokenizer: Any,
        messages: list[dict[str, object]],
        mode: ReasoningMode,
        *,
        add_generation_prompt: bool,
    ) -> tuple[int, ...]:
        if mode not in self.supported_modes:
            raise ValueError(f"Qwen3 chat rendering does not support mode {mode.value!r}")
        return _token_ids(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                truncation=False,
                enable_thinking=mode is ReasoningMode.THINKING,
            )
        )

    def render_generation_prompt(
        self,
        tokenizer: Any,
        messages: list[dict[str, object]],
        mode: ReasoningMode,
    ) -> tuple[int, ...]:
        if not messages or messages[-1].get("role") == "assistant":
            raise ValueError("generation prompt must end in a non-assistant message")
        rendered = self._apply(tokenizer, messages, mode, add_generation_prompt=True)
        if mode is ReasoningMode.NON_THINKING:
            opening = _token_ids(tokenizer.encode("<think>", add_special_tokens=False))
            closing = _token_ids(tokenizer.encode("</think>", add_special_tokens=False))
            open_index = _subsequence(rendered, opening)
            close_index = _subsequence(rendered, closing, max(0, open_index + len(opening)))
            empty_span = (
                ""
                if open_index < 0 or close_index < 0
                else str(tokenizer.decode(rendered[open_index + len(opening) : close_index])).strip()
            )
            if open_index < 0 or close_index < 0 or empty_span:
                raise ValueError("Qwen3 non-thinking prompt does not contain the required empty thinking span")
        return rendered

    def render_completed(
        self,
        tokenizer: Any,
        messages: list[dict[str, object]],
        mode: ReasoningMode,
        *,
        assistant_target_weight: float,
        prompt_target_weight: float,
    ) -> RenderedBehaviorRecord:
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError("completed Qwen3 record must end in an assistant message")
        prompt = messages[:-1]
        response = messages[-1]
        answer = str(response.get("content") or "").strip()
        reasoning = str(response.get("reasoning_content") or "").strip()
        if not answer:
            raise ValueError("completed Qwen3 record has an empty final answer")
        if mode is ReasoningMode.THINKING and not reasoning:
            raise ValueError("thinking record has no non-empty reasoning content")
        if mode is ReasoningMode.NON_THINKING and reasoning:
            raise ValueError("non-thinking record contains non-empty reasoning content")

        prefix = self.render_generation_prompt(tokenizer, prompt, mode)
        rendered = self._apply(tokenizer, messages, mode, add_generation_prompt=False)
        if rendered[: len(prefix)] != prefix:
            raise ValueError("completed Qwen3 record does not preserve the pinned generation prefix")
        if len(rendered) <= len(prefix):
            raise ValueError("completed Qwen3 record contains no response tokens")

        roles = [int(TokenRole.PROMPT)] * len(rendered)
        for index in range(len(prefix), len(rendered)):
            roles[index] = int(TokenRole.ANSWER)
        opening = _token_ids(tokenizer.encode("<think>", add_special_tokens=False))
        closing = _token_ids(tokenizer.encode("</think>", add_special_tokens=False))
        open_index = _subsequence(rendered, opening)
        close_index = _subsequence(rendered, closing, max(0, open_index + len(opening)))
        if open_index < 0 or close_index < 0:
            raise ValueError("completed Qwen3 record is missing thinking delimiters")
        for index in range(open_index, open_index + len(opening)):
            roles[index] = int(TokenRole.DELIMITER)
        for index in range(close_index, close_index + len(closing)):
            roles[index] = int(TokenRole.DELIMITER)
        if mode is ReasoningMode.THINKING:
            reasoning_start = open_index + len(opening)
            rendered_reasoning = str(tokenizer.decode(rendered[reasoning_start:close_index])).strip()
            if close_index <= reasoning_start or not rendered_reasoning:
                raise ValueError("thinking record rendered an empty reasoning span")
            for index in range(reasoning_start, close_index):
                roles[index] = int(TokenRole.REASONING)
        elif str(tokenizer.decode(rendered[open_index + len(opening) : close_index])).strip():
            raise ValueError("non-thinking record rendered a non-empty reasoning span")

        # Qwen's end-of-turn token is structural rather than answer content.
        end_token = getattr(tokenizer, "convert_tokens_to_ids", lambda _value: None)("<|im_end|>")
        if isinstance(end_token, int):
            for index in range(len(prefix), len(rendered)):
                if rendered[index] == end_token:
                    roles[index] = int(TokenRole.DELIMITER)

        mode_id = (
            int(ReasoningModeId.THINKING)
            if mode is ReasoningMode.THINKING
            else int(ReasoningModeId.NON_THINKING)
        )
        modes = [mode_id] * len(rendered)
        targets = [False] * len(rendered)
        weights = [0.0] * len(rendered)
        for predicted_position in range(1, len(rendered)):
            loss_position = predicted_position - 1
            if predicted_position >= len(prefix):
                targets[loss_position] = assistant_target_weight > 0
                weights[loss_position] = assistant_target_weight
            elif prompt_target_weight > 0:
                targets[loss_position] = True
                weights[loss_position] = prompt_target_weight
        return RenderedBehaviorRecord(
            rendered,
            tuple(roles),
            tuple(modes),
            tuple(targets),
            tuple(weights),
        )


def chat_behavior_for_snapshot(snapshot: str | Path) -> ChatBehaviorPort:
    config = json.loads((Path(snapshot) / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError("configured reasoning modes require a Qwen3 model adapter")
    return Qwen3ChatBehavior()


__all__ = ["Qwen3ChatBehavior", "chat_behavior_for_snapshot"]
