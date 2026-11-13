from __future__ import annotations

import json
from types import SimpleNamespace

import measure_block_kl_sensitivity as sensitivity
import pytest
import torch
from torch import nn


def test_calibration_windows_are_seeded_disjoint_and_bos_prefixed() -> None:
    corpus = list(range(10, 100))
    tokens = sensitivity.calibration_windows(corpus, bos=2, samples=4, length=6, seed=1)
    assert torch.equal(tokens, sensitivity.calibration_windows(corpus, bos=2, samples=4, length=6, seed=1))
    assert tokens.shape == (4, 6)
    assert tokens[:, 0].tolist() == [2] * 4
    assert len(set(tokens[:, 1:].flatten().tolist())) == 20
    assert torch.all(tokens[:, 2:] - tokens[:, 1:-1] == 1)
    with pytest.raises(ValueError, match="complete windows"):
        sensitivity.calibration_windows(corpus, bos=2, samples=20, length=6, seed=1)


def test_perturbation_scales_and_restores_fp16_weights_even_when_scoring_fails() -> None:
    first = torch.ones((16, 16), dtype=torch.float16)
    second = first.clone() * 4
    weights = {"0:mlp.up_proj": first, "0:mlp.down_proj": second}
    with pytest.raises(RuntimeError, match="scoring failed"), sensitivity.perturb_weights(
        weights, level=0.10, seed=3,
    ) as metrics:
        assert not first.equal(torch.ones_like(first))
        for values in metrics.values():
            assert values["relative_frobenius_error"] == pytest.approx(0.10, abs=0.0002)
        raise RuntimeError("scoring failed")
    assert torch.all(first == 1)
    assert torch.all(second == 4)


class TinyModel(nn.Module):
    def __init__(self, granularity: str) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(*(nn.Linear(4, 4, bias=False, dtype=torch.float16) for _ in range(2)))
            for _ in range(2)
        ])
        self.model = SimpleNamespace(layers=self.blocks)
        self.original = torch.eye(4, dtype=torch.float16)
        with torch.no_grad():
            for weight in self.parameters():
                weight.copy_(self.original)
        self.changed_counts: list[int] = []
        self.granularity = granularity

    def forward(self, input_ids: torch.Tensor, *, use_cache: bool) -> SimpleNamespace:
        assert not use_cache
        changed_per_block = [sum(not torch.equal(weight, self.original) for weight in block.parameters())
                             for block in self.blocks]
        changed = sum(changed_per_block)
        assert changed == 0 or changed == (1 if self.granularity == "projection" else 2)
        assert sum(count > 0 for count in changed_per_block) <= 1
        self.changed_counts.append(changed)
        logits = nn.functional.one_hot(input_ids, 4).to(torch.float16)
        for block in self.blocks:
            logits = block(logits)
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("granularity", ["projection", "block"])
def test_sweep_scores_cached_teacher_and_resumes_completed_trials(tmp_path, monkeypatch, granularity) -> None:
    model = TinyModel(granularity).eval()
    tokens = torch.tensor([[0, 1, 2, 3]])
    root = tmp_path / "teacher"
    adapter = SimpleNamespace(
        attention_implementation="eager",
        get_decoder_layers=lambda model: model.blocks,
        quantizable_layers=lambda _block, _index: [SimpleNamespace(path="0"), SimpleNamespace(path="1")],
    )
    monkeypatch.setattr(sensitivity, "adapter_for_config", lambda _: adapter)
    monkeypatch.setattr(sensitivity, "vocabulary_size_from_config", lambda _: 4)
    monkeypatch.setattr(sensitivity, "hash_hf_tokenizer_snapshot", lambda _: "test-tokenizer")
    monkeypatch.setattr(sensitivity, "_load_tokens", lambda *_: (tokens, 0))
    args = SimpleNamespace(
        output=tmp_path / "reports", blocks=None, granularity=granularity,
        teacher_dataset=root, device="cpu", token_chunk_size=2,
        error_levels=list(sensitivity.DEFAULT_ERROR_LEVELS), seeds=[0, 1],
        text_column="content", calibration_seed=0, samples=1, sequence_length=4,
    )
    dataset = sensitivity._teacher_dataset(args, tmp_path, model, {}, root, "test-data")
    reused = sensitivity._teacher_dataset(args, tmp_path, model, {}, root, "test-data")
    assert dataset.reference == reused.reference
    with pytest.raises(ValueError, match="different calibration input"):
        sensitivity._teacher_dataset(args, tmp_path, model, {}, root, "another-source")
    score_model = sensitivity.score_model

    def interrupt_after_first_trial(*args, **kwargs):
        if len(model.changed_counts) == 3:  # Teacher, control, first trial.
            raise KeyboardInterrupt
        return score_model(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sensitivity, "score_model", interrupt_after_first_trial)
        with pytest.raises(KeyboardInterrupt):
            sensitivity._sweep(args, {}, model, dataset)
    checkpoint, = root.glob("measurements/*/results.json")
    partial = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert len(partial["trials"]) == 1
    assert all(torch.equal(weight, model.original) for weight in model.parameters())
    sensitivity._sweep(args, {}, model, dataset)
    report = json.loads((args.output / "results.json").read_text(encoding="utf-8"))
    assert report["clean_baseline"]["kl_nats_per_token"] == pytest.approx(0, abs=1e-7)
    if granularity == "projection":
        targets = {"0:0", "0:1", "1:0", "1:1"}
        changed_per_trial = 1
    else:
        targets = {"block 0", "block 1"}
        changed_per_trial = 2
    trials = len(targets) * 4 * 2
    assert report["expected_trials"] == len(report["trials"]) == trials
    assert {row["target"] for row in report["summary"]} == targets
    assert {row["requested_error"] for row in report["trials"]} == {0.05, 0.10, 0.20, 0.35}
    assert all(row["kl_nats_per_token"] > 0 for row in report["trials"])
    assert all(row["token_count"] == 3 for row in report["trials"])
    assert all(torch.equal(weight, model.original) for weight in model.parameters())
    assert f"Measured {trials} of {trials} trials." in (args.output / "results.md").read_text(encoding="utf-8")
    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert stored["trials"] == report["trials"]
    assert stored["protocol"]["teacher_artifact"] == dataset.reference.artifact_id
    args.output = tmp_path / "replacement-reports"
    sensitivity._sweep(args, {}, model, dataset)
    assert model.changed_counts == [0, 0] + [changed_per_trial] * trials
    assert json.loads((args.output / "results.json").read_text(encoding="utf-8")) == report
