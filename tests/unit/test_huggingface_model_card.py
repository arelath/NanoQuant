from __future__ import annotations

from pathlib import Path

from huggingface_hub import ModelCard

from nanoquant.infrastructure.huggingface_model_card import (
    load_huggingface_model_card_metadata,
    write_huggingface_model_card,
)
from tools.render_huggingface_model_card import main


def test_model_card_inherits_source_metadata_and_retains_report_body(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "README.md").write_text(
        "---\n"
        "language:\n"
        "- en\n"
        "- de\n"
        "license: llama3.2\n"
        "---\n\n"
        "# Source\n",
        encoding="utf-8",
    )
    report = tmp_path / "quality.md"
    report.write_text("# Quality\n\nMeasured results.\n", encoding="utf-8")
    output = tmp_path / "model-card.md"

    metadata = load_huggingface_model_card_metadata(
        "meta-llama/Llama-3.2-1B-Instruct",
        "pinned-revision",
        snapshot,
    )
    write_huggingface_model_card(
        metadata,
        output,
        model_name="Llama-3.2-1B-Instruct-nanoquant-GGUF",
        body_source=report,
    )

    card = ModelCard.load(output)
    assert card.data.to_dict() == {
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "language": ["en", "de"],
        "license": "llama3.2",
        "pipeline_tag": "text-generation",
        "tags": ["gguf", "nanoquant", "quantized"],
        "base_model_relation": "quantized",
    }
    assert card.text.strip() == "# Quality\n\nMeasured results."
    assert report.read_text(encoding="utf-8") == "# Quality\n\nMeasured results.\n"


def test_reusable_model_card_script_writes_a_valid_standalone_card(tmp_path: Path) -> None:
    output = tmp_path / "README.md"

    assert (
        main(
            (
                "--base-model",
                "owner/base-model",
                "--base-revision",
                "revision",
                "--model-name",
                "quantized-model",
                "--output",
                str(output),
            )
        )
        == 0
    )

    card = ModelCard.load(output)
    assert card.data.get("base_model") == "owner/base-model"
    assert card.data.get("base_model_relation") == "quantized"
    assert "# quantized-model" in card.text
