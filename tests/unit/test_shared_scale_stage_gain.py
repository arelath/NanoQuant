"""Gain must fold into ordinary middle scales without changing BF16 execution."""

import torch
from probe_shared_scale_stage_gain import SharedStageGain, tune
from safetensors.torch import load_file, save_file


def test_unit_gain_replays_parent_and_nonunit_gain_folds(tmp_path):
    payload = {
        "stage0.left_binary": torch.tensor([[1, -1, 1], [-1, 1, 1]], dtype=torch.int8),
        "stage0.right_binary": torch.tensor([[1, -1], [1, 1], [-1, 1]], dtype=torch.int8),
        "stage0.scale_pre": torch.tensor([0.5, 2], dtype=torch.bfloat16),
        "stage0.scale_mid": torch.tensor([1, 0.5, 0.25], dtype=torch.bfloat16),
        "stage0.scale_post": torch.tensor([1.5, 0.75], dtype=torch.bfloat16),
    }
    control = SharedStageGain(payload, 1, "scales", "cpu")
    candidate = SharedStageGain(payload, 1, "gain_only", "cpu")
    x = torch.tensor([[1.5, 2.5], [-3, 0.5]], dtype=torch.bfloat16)
    assert torch.equal(control(x), candidate(x))
    with torch.no_grad():
        candidate.gain.fill_(0.873)
    assert not torch.equal(control(x), candidate(x))
    file = tmp_path / "folded.safetensors"
    save_file(candidate.folded_payload(), str(file))
    folded = load_file(str(file))
    assert folded.keys() == payload.keys()
    assert torch.equal(folded["stage0.scale_mid"][:1], payload["stage0.scale_mid"][:1])
    assert torch.equal(folded["stage0.scale_pre"], payload["stage0.scale_pre"])
    assert torch.equal(folded["stage0.scale_post"], payload["stage0.scale_post"])
    assert torch.equal(candidate(x), SharedStageGain(folded, 1, "scales", "cpu")(x))
    before = float(candidate.gain.detach())
    after = tune(candidate, [(x, control(x).detach().float())], 8)
    assert after > before
    assert torch.equal(candidate.scale_mid.detach().bfloat16(), payload["stage0.scale_mid"])
