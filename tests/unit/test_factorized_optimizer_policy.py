from dataclasses import replace

from nanoquant.config.schema import FactorizedOptimizerConfig
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    _factorized_optimizer_backend,
    _resident_config_hash,
    _resident_manifest_config,
)


def test_optimizer_policy_preserves_old_boundary_and_limits_projection_scope(tmp_path):
    source = ResidentQuantizationRequest(tmp_path, tmp_path, "fixture", "revision", ((1, 2),), device="cpu")
    policy = FactorizedOptimizerConfig("fused_adamw_scales_fp32", 19, ("mlp.gate_proj", "mlp.down_proj"), "1" * 64)
    candidate = replace(source, factorized_tuning_optimizer=policy)
    assert _factorized_optimizer_backend(candidate, 18, "mlp.gate_proj") == "parity"
    assert _factorized_optimizer_backend(candidate, 19, "mlp.gate_proj") == "fused_adamw_scales_fp32"
    assert _factorized_optimizer_backend(candidate, 20, "mlp.up_proj") == "parity"
    assert _factorized_optimizer_backend(candidate, 20, "self_attn.q_proj") == "parity"
    assert _resident_config_hash(source) != _resident_config_hash(candidate)
    assert _resident_config_hash(candidate) != _resident_config_hash(replace(
        candidate, factorized_tuning_optimizer=replace(policy, continuation_journal_sha256="2" * 64),
    ))
    assert "factorized_tuning_optimizer" not in _resident_manifest_config(source, "resident-quantization")
    assert "factorized_tuning_optimizer" in _resident_manifest_config(candidate, "resident-quantization")
