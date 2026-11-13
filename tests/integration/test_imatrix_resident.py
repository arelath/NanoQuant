import json
import struct
from dataclasses import replace

import numpy as np
import pytest
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from nanoquant.config.codec import from_dict
from nanoquant.config.schema import ADMMConfig, ProfilingConfig, ProfilingLevel, SharedInputGroupConfig
from nanoquant.domain.models import ArtifactRef, CalibrationStats
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.imatrix import imatrix_config
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    load_completed_resident_quantization,
    run_resident_quantization,
)


def test_imatrix_weights_reach_fitting_and_resume(tmp_path, monkeypatch):
    torch.manual_seed(7)
    snapshot = tmp_path / 'snapshot'
    config = Gemma3TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=4)
    Gemma3ForCausalLM(config).save_pretrained(snapshot, safe_serialization=True)
    entries = [('attn_q', 16), ('attn_k', 16), ('attn_v', 16), ('attn_output', 16),
               ('ffn_gate', 16), ('ffn_up', 16), ('ffn_down', 32)]
    payload = struct.pack('<i', len(entries))
    for stem, width in entries:
        name = f'blk.0.{stem}.weight'.encode()
        payload += struct.pack('<i', len(name)) + name + struct.pack('<ii', 2, width)
        payload += (np.arange(1, width + 1, dtype='<f4') * 2).tobytes()
    path = tmp_path / 'i.dat'
    path.write_bytes(payload)
    base = ResidentQuantizationRequest(snapshot, tmp_path / 'base', 'fixture/gemma3', 'pinned-test',
                                       ((1, 2, 3, 4),), device='cpu', target_bpw=8., rank_multiple=1,
                                       admm=ADMMConfig(outer_iterations=2, inner_iterations=1),
                                       profiling=ProfilingConfig(level=ProfilingLevel.OFF),
                                       registry_root=tmp_path / 'runs')
    baseline = run_resident_quantization(base)
    import nanoquant.application.quantization_stages as stages
    original_fit = stages.factorize_admm_with_parameters
    observed_inputs = []

    def capture_fit(weight, input_importance, *args, **kwargs):
        observed_inputs.append(input_importance.detach().clone())
        return original_fit(weight, input_importance, *args, **kwargs)

    monkeypatch.setattr(stages, 'factorize_admm_with_parameters', capture_fit)
    request = replace(base, output=tmp_path / 'weighted', imatrix=imatrix_config(path),
                      shared_input_groups=(SharedInputGroupConfig('self_attn.attn_qkv',
                          ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj')),))
    with pytest.raises(InterruptedError):
        run_resident_quantization(replace(request, interrupt_after_layer_commits=1))
    result = run_resident_quantization(request)
    assert result.reused_commit_count >= 1
    assert any(torch.equal(value, torch.arange(1, 17).float()) for value in observed_inputs)
    assert any(torch.equal(value, torch.arange(1, 33).float()) for value in observed_inputs)
    assert result.identity != baseline.identity
    assert load_completed_resident_quantization(request).identity == result.identity
    stores = [LocalArtifactStore(item.output / 'artifacts') for item in (base, request)]
    stats = []
    for item, store in zip((base, request), stores, strict=True):
        state = json.loads((item.output / 'state' / 'preprocessing.json').read_text())
        ref = from_dict(ArtifactRef, state['calibration'])
        stats.append(from_dict(CalibrationStats, json.loads(
            (store.path_for(ref.artifact_id) / 'stats.json').read_text())))
    before = {layer.layer: layer for layer in stats[0].layers}
    tensors = LocalTensorStore(stores[1])
    for layer in stats[1].layers:
        assert layer.output_summary == before[layer.layer].output_summary
        with tensors.read(layer.input_importance, 'cpu') as value:
            expected = torch.arange(1, value.numel() + 1).float()
            assert torch.equal(value, expected)
    report = json.loads((request.output / 'imatrix.json').read_text())
    assert len(report['tensor_map']) == 7
    # Explicit old preprocessing cannot silently bypass the new objective.
    state = json.loads((base.output / 'state' / 'preprocessing.json').read_text())
    from nanoquant.application.calibration_artifacts import PersistedCalibration, PersistedObjectives
    from nanoquant.domain.models import ObjectiveSpec
    from nanoquant.resident_quantization import _validate_imatrix_objectives
    calibration = from_dict(ArtifactRef, state['calibration'])
    objectives = from_dict(ArtifactRef, state['objectives'])
    specs = tuple(from_dict(ObjectiveSpec, value) for value in json.loads(
        (stores[0].path_for(objectives.artifact_id) / 'objectives.json').read_text()))
    with pytest.raises(ValueError, match='differs from pinned imatrix'):
        _validate_imatrix_objectives(request, stores[0], baseline.inventory,
                                    PersistedCalibration(calibration, stats[0]), PersistedObjectives(objectives, specs))
