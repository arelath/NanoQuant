import hashlib
import struct
from dataclasses import replace

import numpy as np
import pytest
import torch
from gguf import GGUFWriter

from nanoquant.config.codec import config_hash, from_dict, to_dict
from nanoquant.config.schema import ImatrixConfig, ImatrixTensorMapping, ModelConfig, RunConfig, SharedInputGroupConfig
from nanoquant.config.validation import validate
from nanoquant.domain.models import BlockId, BlockInventory, LayerId, LayerInventory, ModelInventory
from nanoquant.infrastructure.imatrix import imatrix_config, load_imatrix, resolve_imatrix
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    _calibration_config_hash,
    _resident_config_hash,
    _resident_manifest_config,
    _validate_imatrix_source,
)


def write_legacy(path, entries, trailer=True):
    payload = struct.pack('<i', len(entries))
    for name, values, count in entries:
        encoded = name.encode()
        payload += struct.pack('<i', len(encoded)) + encoded + struct.pack('<ii', count, len(values))
        payload += np.array(values, dtype='<f4').tobytes()
    if trailer:
        payload += struct.pack('<ii', 7, 4) + b'test'
    path.write_bytes(payload)
    return imatrix_config(path)


def write_gguf(path, values, counts, *, kind='imatrix', omit_counts=False):
    writer = GGUFWriter(str(path), 'imatrix')
    writer.add_string('general.type', kind)
    writer.add_uint32('imatrix.chunk_count', 7)
    writer.add_uint32('imatrix.chunk_size', 512)
    writer.add_array('imatrix.datasets', ['test'])
    writer.add_tensor('blk.0.attn_q.weight.in_sum2', np.array(values, dtype=np.float32))
    if not omit_counts:
        writer.add_tensor('blk.0.attn_q.weight.counts', np.array(counts, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def inventory(paths=('self_attn.q_proj',)):
    # Only source_key is consumed by the resolver; the real inventory contract
    # is exercised by the resident integration test.
    from types import SimpleNamespace
    block = BlockId(0)
    layers = tuple(LayerInventory(LayerId(block, path), SimpleNamespace(source_key=f'model.layers.0.{path}.weight'),
                                  None, 3, 2) for path in paths)
    return ModelInventory(1, None, (BlockInventory(block, (), layers),), (), 0)


@pytest.mark.parametrize('trailer', [True, False])
def test_legacy_normalization_matches_llama_cpp(tmp_path, trailer):
    path = tmp_path / 'importance.dat'
    config = write_legacy(path, [('blk.0.attn_q.weight', [2, 8, 18], 2)], trailer)
    data = load_imatrix(path, config.sha256)
    assert torch.equal(data.entries['blk.0.attn_q.weight'], torch.tensor([[1., 4., 9.]]))
    vectors, report = resolve_imatrix(config, inventory())
    assert torch.allclose(next(iter(vectors.values())), torch.tensor([1., 4., 9.]))
    assert report['tensor_map'] == {'0:self_attn.q_proj': 'blk.0.attn_q.weight'}


def test_gguf_matches_legacy_and_preserves_expert_axis(tmp_path):
    path = tmp_path / 'importance.gguf'
    write_gguf(path, [[2, 8, 18]], [[2]])
    data = load_imatrix(path)
    assert data.format == 'gguf'
    assert torch.equal(data.entries['blk.0.attn_q.weight'], torch.tensor([[1., 4., 9.]]))
    write_gguf(path, [[2, 8, 18], [4, 8, 12]], [[2], [4]])
    data = load_imatrix(path)
    assert torch.equal(data.entries['blk.0.attn_q.weight'][1], torch.tensor([1., 2., 3.]))
    with pytest.raises(ValueError, match='shape mismatch'):
        resolve_imatrix(imatrix_config(path), inventory())


@pytest.mark.parametrize('values,count', [([0, 0, 0], 1), ([1, -1, 3], 1), ([1, float('nan'), 3], 1),
                                         ([1, float('inf'), 3], 1), ([1, 2, 3], 0), ([1, 2, 3], -2)])
def test_reject_invalid_legacy_statistics(tmp_path, values, count):
    with pytest.raises(ValueError, match='imatrix'):
        write_legacy(tmp_path / 'bad.dat', [('blk.0.attn_q.weight', values, count)])


def test_reject_malformed_files_and_wrong_kind(tmp_path):
    path = tmp_path / 'bad.dat'
    path.write_bytes(struct.pack('<ii', 1, 2**30))
    with pytest.raises(ValueError):
        load_imatrix(path)
    write_gguf(path, [[1, 2, 3]], [[1]], kind='model')
    with pytest.raises(ValueError, match='general.type'):
        load_imatrix(path)
    write_gguf(path, [[1, 2, 3]], [[1]], omit_counts=True)
    with pytest.raises(ValueError, match='matching'):
        load_imatrix(path)


def test_missing_ambiguous_shape_and_group_rejected(tmp_path):
    path = tmp_path / 'i.dat'
    config = write_legacy(path, [('blk.0.attn_q.weight', [1, 2, 3], 1)])
    with pytest.raises(ValueError, match='missing'):
        resolve_imatrix(config, inventory(('self_attn.k_proj',)))
    config = write_legacy(path, [('blk.0.attn_q.weight', [1, 2], 1)])
    with pytest.raises(ValueError, match='shape mismatch'):
        resolve_imatrix(config, inventory())
    config = write_legacy(path, [('blk.0.attn_q.weight', [1, 2, 3], 1), ('blk.0.attn_q', [1, 2, 3], 1)])
    with pytest.raises(ValueError, match='ambiguous'):
        resolve_imatrix(config, inventory())
    config = write_legacy(path, [('blk.0.attn_q.weight', [1, 2, 3], 1), ('blk.0.attn_k.weight', [3, 2, 1], 1)])
    with pytest.raises(ValueError, match='shared-input'):
        resolve_imatrix(config, inventory(('self_attn.q_proj', 'self_attn.k_proj')),
                        (SharedInputGroupConfig('qk', ('self_attn.q_proj', 'self_attn.k_proj')),))


def test_hash_pinning_config_roundtrip_and_identities(tmp_path):
    path = tmp_path / 'i.dat'
    pinned = write_legacy(path, [('blk.0.attn_q.weight', [1, 2, 3], 1)])
    config = RunConfig(ModelConfig('fixture'))
    altered = replace(config, calibration=replace(config.calibration, imatrix=pinned))
    assert from_dict(RunConfig, to_dict(altered)) == altered
    assert not [issue for issue in validate(altered) if issue.code.startswith('CFG25')]
    assert config_hash(config) != config_hash(altered)
    request = ResidentQuantizationRequest(tmp_path, tmp_path / 'out', 'fixture', 'rev', ((1, 2),), device='cpu')
    changed = replace(request, imatrix=pinned)
    assert _resident_config_hash(request) != _resident_config_hash(changed)
    assert _calibration_config_hash(request) != _calibration_config_hash(changed)
    assert 'imatrix' not in _resident_manifest_config(request, 'resident-quantization')
    _validate_imatrix_source(changed)
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(ValueError, match='SHA-256'):
        _validate_imatrix_source(changed)
    with pytest.raises(ValueError, match='SHA-256'):
        load_imatrix(path, pinned.sha256)
    assert pinned.sha256 != hashlib.sha256(path.read_bytes()).hexdigest()


def test_schema_rejects_unpinned_source():
    config = RunConfig(ModelConfig('fixture'))
    config = replace(config, calibration=replace(config.calibration, imatrix=ImatrixConfig('file.dat', '')))
    assert any(issue.code == 'CFG254' for issue in validate(config))


def test_explicit_mapping_and_duplicate_legacy_names(tmp_path):
    path = tmp_path / 'i.dat'
    config = write_legacy(path, [('custom.q', [1, 4, 9], 1)])
    config = replace(config, tensor_map=(ImatrixTensorMapping('0:self_attn.q_proj', 'custom.q'),))
    vectors, report = resolve_imatrix(config, inventory())
    assert torch.equal(next(iter(vectors.values())), torch.tensor([1., 4., 9.]))
    assert report['tensor_map'] == {'0:self_attn.q_proj': 'custom.q'}
    with pytest.raises(ValueError, match='duplicate'):
        write_legacy(path, [('a', [1], 1), ('a', [2], 1)])
