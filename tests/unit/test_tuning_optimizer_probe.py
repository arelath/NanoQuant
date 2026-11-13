import profile_gemma_block_tuning as probe
import pytest
import torch


@pytest.mark.parametrize("backend", ["torch-fused", "torch-fused-fp32", "torch-fused-scales-fp32"])
def test_fused_probe_preserves_optimizer_settings_and_restores_factory(backend):
    original = probe.tuning_module.ParityAdamW
    with pytest.raises(RuntimeError, match="probe failure"):
        with probe.optimizer_backend(backend):
            parameter = torch.nn.Parameter(torch.ones(2))
            optimizer = probe.tuning_module.ParityAdamW(
                [{"params": [parameter], "lr": 3e-5}], lr=1e-5, weight_decay=0.0,
            )
            assert optimizer.param_groups[0]["lr"] == 3e-5
            assert optimizer.defaults["betas"] == (0.9, 0.99)
            assert optimizer.defaults["eps"] == 1e-6
            assert optimizer.defaults["fused"] is True
            raise RuntimeError("probe failure")
    assert probe.tuning_module.ParityAdamW is original


def test_compiled_probe_restores_update_function(monkeypatch):
    original = probe.adamw_module._foreach_update
    calls = []
    received = []
    def sentinel(*args, **kwargs):
        received.append(kwargs)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: calls.append((fn, kwargs)) or sentinel)
    with probe.optimizer_backend("compiled-parity"):
        probe.adamw_module._foreach_update(beta1=0.9, beta2=0.99, learning_rate=1e-5)
        assert all(value.dtype == torch.float64 for value in received[0].values())
    assert probe.adamw_module._foreach_update is original
    assert calls[0][0] is original
    assert calls[0][1]["options"]["emulate_precision_casts"]


@pytest.mark.parametrize("ambiguous", [False, True])
def test_admm_initialization_loads_continuous_stages_and_rejects_ambiguity(tmp_path, ambiguous):
    store = probe.LocalArtifactStore(tmp_path / "artifacts")
    factors = {
        "left_latent": torch.tensor([[0.1, -0.2], [0.3, -0.4]]),
        "right_latent": torch.tensor([[0.5, -0.6], [0.7, -0.8]]),
    }
    scales = {name: torch.ones(2) for name in ("scale_pre", "scale_mid", "scale_post")}
    identifiers = []
    for stage in (1, 2):
        for kind, values in (("factorization-attempt", factors), ("scale-fit", scales)):
            with store.begin_write(kind) as writer:
                probe.SAFETENSORS.save({key: value * stage for key, value in values.items()},
                                      writer.path / "tensors.safetensors")
                identifiers.append(writer.commit().artifact_id)
    if ambiguous:
        with store.begin_write("factorization-attempt") as writer:
            (writer.path / "extra.json").write_text("{}")
            writer.commit()
    with store.begin_write("layer-result") as writer:
        (writer.path / "layer.json").write_text("{}")
        committed = writer.commit()
    layer = {"factorization": {"artifact_id": identifiers[0]},
             "frozen_state": {"residual_stages": [{}]}}
    records = [{"kind": "layer", "block": 0, "layer": "mlp.gate_proj", "artifact_id": committed.artifact_id}]
    if ambiguous:
        with pytest.raises(ValueError, match="uniquely recover"):
            probe.load_admm_gate(store, layer, records)
    else:
        module, sources = probe.load_admm_gate(store, layer, records)
        assert sources == identifiers
        assert torch.equal(module.left_latent, factors["left_latent"].bfloat16())
        assert torch.equal(module.residual_terms[0].left_latent, (factors["left_latent"] * 2).bfloat16())
        assert torch.equal(module.residual_terms[0].scale_pre, torch.full((2,), 2, dtype=torch.bfloat16))
