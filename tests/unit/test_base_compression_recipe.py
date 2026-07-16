from pathlib import Path

import pytest
from recipes import (
    BASE_COMPRESSION_CONFIG,
    EXPERIMENT_001,
    EXPERIMENT_001_CONFIG,
    EXPERIMENT_003,
    EXPERIMENT_003_CONFIG,
    compression_export_recipe,
)


def test_base_compression_recipe_is_visible_and_unnumbered() -> None:
    assert BASE_COMPRESSION_CONFIG.intent.experiment_number is None
    assert BASE_COMPRESSION_CONFIG.intent.name == "base-compression-gemma-3"
    assert BASE_COMPRESSION_CONFIG.allocation.maximum_rank_layer_patterns == (
        "self_attn.v_proj",
    )
    assert not EXPERIMENT_001_CONFIG.allocation.maximum_rank_layer_patterns
    assert not EXPERIMENT_003_CONFIG.allocation.maximum_rank_layer_patterns
    assert EXPERIMENT_001_CONFIG.factorization == BASE_COMPRESSION_CONFIG.factorization
    assert EXPERIMENT_003_CONFIG.factorization == BASE_COMPRESSION_CONFIG.factorization
    assert EXPERIMENT_001.export == compression_export_recipe(1, "gemma-3-1b-it")
    assert EXPERIMENT_003.export == compression_export_recipe(3, "gemma-3-4b-it")


def test_base_compression_export_recipe_requires_safe_numbered_outputs() -> None:
    export = compression_export_recipe(3, "gemma-3-4b-it")

    assert export.logical_output == Path("outputs/003-gemma-3-4b-it/logical")
    assert export.packed_output == Path("outputs/003-gemma-3-4b-it/packed")
    assert export.gguf_output == Path("outputs/003-gemma-3-4b-it/gemma-3-4b-it-nanoquant.gguf")
    assert export.token_embedding_type == "q8_0"
    assert compression_export_recipe(3, "gemma-3-4b-it", token_embedding_type="Q4_K").token_embedding_type == "q4_k"
    with pytest.raises(ValueError, match="safe path"):
        compression_export_recipe(3, "../escape")
    with pytest.raises(ValueError, match="unsupported token embedding"):
        compression_export_recipe(3, "gemma-3-4b-it", token_embedding_type="bf16")
