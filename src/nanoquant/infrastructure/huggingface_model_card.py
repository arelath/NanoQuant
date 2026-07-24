"""Deterministic Hugging Face model cards for published NanoQuant GGUFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData

from nanoquant.infrastructure.io_utils import atomic_write_text

_MODEL_CARD_TAGS = ("gguf", "nanoquant", "quantized")


@dataclass(frozen=True, slots=True)
class HuggingFaceModelCardMetadata:
    """Source identity and inherited card fields for one quantized derivative."""

    base_model: str
    base_model_revision: str
    languages: tuple[str, ...] = ()
    license: str | None = None
    license_name: str | None = None
    license_link: str | None = None

    def __post_init__(self) -> None:
        if not self.base_model or self.base_model != self.base_model.strip():
            raise ValueError("Hugging Face model-card base model must be a trimmed non-empty ID")
        if not self.base_model_revision or self.base_model_revision != self.base_model_revision.strip():
            raise ValueError("Hugging Face model-card base revision must be trimmed and non-empty")
        if any(not language or language != language.strip() for language in self.languages):
            raise ValueError("Hugging Face model-card languages must be trimmed and non-empty")


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"source model card {field} must be a non-empty string")
    return value.strip()


def _languages(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(language, str) or not language.strip() for language in values
    ):
        raise ValueError("source model card language must be a string or list of strings")
    return tuple(language.strip() for language in values)


def load_huggingface_model_card_metadata(
    base_model: str,
    base_model_revision: str,
    snapshot: str | Path,
) -> HuggingFaceModelCardMetadata:
    """Load transferable language/license fields from a pinned source snapshot."""

    source_card = Path(snapshot) / "README.md"
    if not source_card.is_file():
        return HuggingFaceModelCardMetadata(base_model, base_model_revision)
    card = ModelCard.load(source_card)
    return HuggingFaceModelCardMetadata(
        base_model,
        base_model_revision,
        _languages(card.data.get("language")),
        _optional_string(card.data.get("license"), "license"),
        _optional_string(card.data.get("license_name"), "license_name"),
        _optional_string(card.data.get("license_link"), "license_link"),
    )


def huggingface_model_card_output(gguf_output: str | Path) -> Path:
    """Return the stable local model-card path associated with one GGUF."""

    return Path(gguf_output).with_suffix(".model-card.md")


def _without_existing_metadata(markdown: str) -> str:
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    if normalized.startswith("---\n"):
        return ModelCard(normalized).text.lstrip("\n")
    return normalized


def render_huggingface_model_card(
    metadata: HuggingFaceModelCardMetadata,
    *,
    model_name: str,
    body: str = "",
    pipeline_tag: str = "text-generation",
) -> str:
    """Render and validate one complete model-card README."""

    if not model_name or model_name != model_name.strip():
        raise ValueError("Hugging Face model-card name must be trimmed and non-empty")
    if pipeline_tag not in {"text-generation", "image-text-to-text"}:
        raise ValueError("Hugging Face model-card pipeline tag is unsupported")
    markdown_body = _without_existing_metadata(body).strip()
    if not markdown_body:
        markdown_body = (
            f"# {model_name}\n\n"
            f"This repository contains a NanoQuant GGUF quantization of "
            f"[`{metadata.base_model}`](https://huggingface.co/{metadata.base_model}) at revision "
            f"`{metadata.base_model_revision}`.\n\n"
            "Machine-readable evaluation and provenance are provided in `quality.json` when available."
        )
    card_data = ModelCardData(
        base_model=metadata.base_model,
        language=list(metadata.languages) or None,
        license=metadata.license,
        license_name=metadata.license_name,
        license_link=metadata.license_link,
        pipeline_tag=pipeline_tag,
        tags=list(_MODEL_CARD_TAGS),
        base_model_relation="quantized",
    )
    yaml_metadata = card_data.to_yaml(line_break="\n")
    rendered = f"---\n{yaml_metadata}\n---\n\n{markdown_body}\n"
    card = ModelCard(rendered)
    card.validate(repo_type="model")
    return str(card).rstrip() + "\n"


def write_huggingface_model_card(
    metadata: HuggingFaceModelCardMetadata,
    output: str | Path,
    *,
    model_name: str,
    body_source: str | Path | None = None,
    pipeline_tag: str = "text-generation",
) -> Path:
    """Create a validated card atomically, optionally retaining a report body."""

    body = "" if body_source is None else Path(body_source).read_text(encoding="utf-8")
    destination = Path(output).resolve()
    atomic_write_text(
        destination,
        render_huggingface_model_card(
            metadata,
            model_name=model_name,
            body=body,
            pipeline_tag=pipeline_tag,
        ),
    )
    return destination


__all__ = [
    "HuggingFaceModelCardMetadata",
    "huggingface_model_card_output",
    "load_huggingface_model_card_metadata",
    "render_huggingface_model_card",
    "write_huggingface_model_card",
]
