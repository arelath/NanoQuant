from __future__ import annotations

from nanoquant.config.schema import ReasoningMode
from nanoquant.infrastructure.chat_behaviors import Qwen3ChatBehavior
from nanoquant.ports.chat_behavior import TokenRole


class CharacterTokenizer:
    chat_template = "fixture-qwen3-template"
    special_tokens_map = {"eos_token": "<|im_end|>"}

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(value) for value in text]

    @staticmethod
    def decode(values: object) -> str:
        return "".join(chr(int(value)) for value in values)  # type: ignore[arg-type]

    @staticmethod
    def convert_tokens_to_ids(value: str) -> int:
        # The character fixture cannot represent the multi-character special
        # token as one ID, so delimiter classification relies on think tags.
        assert value == "<|im_end|>"
        return -1

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        truncation: bool,
        enable_thinking: bool,
    ) -> list[int]:
        assert tokenize and not truncation
        rendered = ""
        for message in messages:
            role = str(message["role"])
            rendered += f"<|im_start|>{role}\n"
            if role == "assistant" and enable_thinking:
                rendered += f"<think>\n{message.get('reasoning_content', '')}\n</think>\n\n"
            elif role == "assistant":
                rendered += "<think>\n\n</think>\n\n"
            rendered += str(message.get("content") or "") + "<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
            if not enable_thinking:
                rendered += "<think>\n\n</think>\n\n"
        return self.encode(rendered)


def test_qwen3_renderer_enforces_both_completed_modes_and_aligns_response_targets() -> None:
    tokenizer = CharacterTokenizer()
    behavior = Qwen3ChatBehavior()
    prompt = [{"role": "user", "content": "2+2?"}]

    thinking = behavior.render_completed(
        tokenizer,
        [*prompt, {"role": "assistant", "reasoning_content": "add two", "content": "4"}],
        ReasoningMode.THINKING,
        assistant_target_weight=1.0,
        prompt_target_weight=0.0,
    )
    non_thinking = behavior.render_completed(
        tokenizer,
        [*prompt, {"role": "assistant", "content": "4"}],
        ReasoningMode.NON_THINKING,
        assistant_target_weight=1.0,
        prompt_target_weight=0.0,
    )

    thinking_text = tokenizer.decode(thinking.input_ids)
    non_thinking_text = tokenizer.decode(non_thinking.input_ids)
    assert "<think>\nadd two\n</think>" in thinking_text
    assert "<think>\n\n</think>" in non_thinking_text
    assert int(TokenRole.REASONING) in thinking.token_role_ids
    assert int(TokenRole.REASONING) not in non_thinking.token_role_ids
    thinking_prefix = behavior.render_generation_prompt(tokenizer, prompt, ReasoningMode.THINKING)
    first_response_token = len(thinking_prefix)
    assert thinking.distillation_target_mask[first_response_token - 1]
    assert not thinking.distillation_target_mask[first_response_token - 2]


def test_qwen3_renderer_rejects_empty_thinking_and_reasoning_in_non_thinking_mode() -> None:
    tokenizer = CharacterTokenizer()
    behavior = Qwen3ChatBehavior()
    prompt = [{"role": "user", "content": "question"}]

    import pytest

    with pytest.raises(ValueError, match="no non-empty reasoning"):
        behavior.render_completed(
            tokenizer,
            [*prompt, {"role": "assistant", "content": "answer"}],
            ReasoningMode.THINKING,
            assistant_target_weight=1.0,
            prompt_target_weight=0.0,
        )
    with pytest.raises(ValueError, match="contains non-empty reasoning"):
        behavior.render_completed(
            tokenizer,
            [*prompt, {"role": "assistant", "reasoning_content": "secret", "content": "answer"}],
            ReasoningMode.NON_THINKING,
            assistant_target_weight=1.0,
            prompt_target_weight=0.0,
        )
