from pathlib import Path

import pytest
from recipes import (
    BASE_COMPRESSION_TEMPLATE,
    GEMMA_3_1B_PARITY_TEMPLATE,
    CompressionExportPolicy,
    HuggingFaceUploadConfig,
)

from tests.support.experiments import load_experiment


def test_base_compression_templates_are_visible_and_unnumbered() -> None:
    experiment_001 = load_experiment(1)
    experiment_003 = load_experiment(3)

    assert BASE_COMPRESSION_TEMPLATE.intent.experiment_number is None
    assert BASE_COMPRESSION_TEMPLATE.intent.name == "unnamed-run"
    assert BASE_COMPRESSION_TEMPLATE.allocation.maximum_rank_layer_patterns == (
        "self_attn.v_proj",
        "self_attn.k_proj",
    )
    assert tuple(
        (item.pattern, item.multiplier)
        for item in BASE_COMPRESSION_TEMPLATE.allocation.layer_budget_multipliers
    ) == (("self_attn.q_proj", 1.25),)
    assert not GEMMA_3_1B_PARITY_TEMPLATE.allocation.maximum_rank_layer_patterns
    assert not GEMMA_3_1B_PARITY_TEMPLATE.allocation.layer_budget_multipliers
    assert experiment_001.config.factorization == BASE_COMPRESSION_TEMPLATE.factorization
    assert experiment_003.config.factorization == BASE_COMPRESSION_TEMPLATE.factorization


def test_compression_templates_use_the_pinned_wikitext_ultrachat_mix() -> None:
    sources = BASE_COMPRESSION_TEMPLATE.dataset.sources

    assert tuple((source.name, source.split, source.subset, source.weight) for source in sources) == (
        ("HuggingFaceH4/ultrachat_200k", "train_sft", None, 0.5),
        ("Salesforce/wikitext", "train", "wikitext-2-raw-v1", 0.5),
    )
    assert load_experiment(1).config.dataset == BASE_COMPRESSION_TEMPLATE.dataset
    assert load_experiment(3).config.dataset == BASE_COMPRESSION_TEMPLATE.dataset


def test_experiment_layout_derives_safe_numbered_export_outputs() -> None:
    experiment = load_experiment(3)
    export = experiment.workflow.export

    assert export.logical_output == Path("outputs/003/logical")
    assert export.packed_output == Path("outputs/003/packed")
    assert export.gguf_output == Path("outputs/003/gemma-3-4b-it-nanoquant.gguf")
    assert export.token_embedding_type == "q8_0"
    with pytest.raises(ValueError, match="release name"):
        CompressionExportPolicy(release_name="../escape")


def test_export_policy_accepts_explicit_huggingface_destination() -> None:
    destination = HuggingFaceUploadConfig(
        "owner/gemma-3-4b-it-nanoquant-GGUF",
        private=True,
        commit_message="Publish Experiment 008",
    )

    policy = CompressionExportPolicy(huggingface=destination)

    assert policy.huggingface is destination
