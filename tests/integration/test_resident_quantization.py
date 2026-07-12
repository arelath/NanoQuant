import math
from dataclasses import replace
from pathlib import Path

import pytest
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from nanoquant.config.schema import ADMMConfig
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.progress import ProgressJournal
from nanoquant.resident_quantization import ResidentQuantizationRequest, run_resident_quantization


def test_resident_quantization_commits_complete_transformers_model(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    Gemma3ForCausalLM(config).save_pretrained(snapshot, safe_serialization=True)
    output = tmp_path / "run"

    result = run_resident_quantization(
        ResidentQuantizationRequest(
            snapshot,
            output,
            "fixture/gemma3",
            "pinned-test-revision",
            ((1, 2, 3, 4),),
            device="cpu",
            target_bpw=8.0,
            rank_multiple=1,
            admm=ADMMConfig(outer_iterations=2, inner_iterations=1),
        )
    )

    assert len(result.blocks) == 1
    assert len(result.blocks[0].layers) == 7
    assert result.frozen_model.effective_bpw <= 8.0
    assert result.frozen_model.actual_total_bits > 0
    assert math.isfinite(result.reference_nll)
    assert math.isfinite(result.compressed_nll)
    assert math.isfinite(result.logit_mse)
    assert 0 <= result.argmax_agreement <= 1
    assert result.peak_host_bytes > 0
    assert result.artifact_bytes > 0
    artifacts = LocalArtifactStore(output / "artifacts")
    artifacts.validate(result.report.artifact_id)
    discovery = ProgressJournal(output / "state", "resident-quantization", artifacts).discover(
        result.plan, result.identity
    )
    assert discovery.first_incomplete is None

    resumed_output = tmp_path / "resumed"
    interrupted_request = ResidentQuantizationRequest(
        snapshot,
        resumed_output,
        "fixture/gemma3",
        "pinned-test-revision",
        ((1, 2, 3, 4),),
        device="cpu",
        target_bpw=8.0,
        rank_multiple=1,
        admm=ADMMConfig(outer_iterations=2, inner_iterations=1),
        interrupt_after_layer_commits=3,
    )
    with pytest.raises(InterruptedError, match="after 3"):
        run_resident_quantization(interrupted_request)
    resumed = run_resident_quantization(replace(interrupted_request, interrupt_after_layer_commits=None))

    assert resumed.reused_commit_count == 3
    assert resumed.plan == result.plan
    assert resumed.frozen_model.actual_total_bits == result.frozen_model.actual_total_bits
    assert [layer.factorization for layer in resumed.blocks[0].layers] == [
        layer.factorization for layer in result.blocks[0].layers
    ]
    assert resumed.compressed_nll == pytest.approx(result.compressed_nll)
