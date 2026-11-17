from __future__ import annotations

import json
from types import SimpleNamespace

import kl_sensitivity_dataset as sensitivity
import pytest
import torch
from torch import nn

from nanoquant.infrastructure.io_utils import hash_file


def test_local_curated_text_preserves_special_tokens_and_rejects_changed_bytes(tmp_path, monkeypatch) -> None:
    path = tmp_path / "calibration.txt"
    text = "<bos>prose<eos>\n<start_of_turn>user\nquestion<end_of_turn>\n"
    path.write_text(text, encoding="utf-8")
    calls = []

    class Tokenizer:
        bos_token_id = 2
        eos_token_id = 1

        def encode(self, value, *, add_special_tokens, verbose):
            calls.append((value, add_special_tokens, verbose))
            return list(range(10, 34))

    monkeypatch.setattr(sensitivity.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    args = sensitivity.make_parser(sensitivity.GEMMA_1B, dataset_reports=True).parse_args([
        "--calibration-text", str(path), "--samples", "4", "--sequence-length", "5",
    ])
    digest = hash_file(path)
    tokens, bos = sensitivity._load_tokens(args, tmp_path, digest)
    assert calls == [(text, False, False)]
    assert bos == 2 and tokens.shape == (4, 5) and tokens[:, 0].tolist() == [2] * 4
    assert len(set(tokens[:, 1:].flatten().tolist())) == 16
    path.write_text(text + "changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        sensitivity._load_tokens(args, tmp_path, digest)


def test_4b_defaults_use_distinct_model_and_dataset_reports() -> None:
    spec = sensitivity.GEMMA_4B
    args = sensitivity.make_parser(spec, dataset_reports=True).parse_args([])
    assert spec.source == "google/gemma-3-4b-it"
    assert spec.blocks == 34
    assert len(spec.weights_identity) == 2
    assert args.output is None and args.plots
    assert (args.samples, args.sequence_length) == (16, 256)
    legacy = sensitivity.make_parser(sensitivity.GEMMA_1B, dataset_reports=False).parse_args([])
    assert legacy.output.name == "gemma-3-1b-it-weight-sensitivity"
    assert not legacy.plots
    assert sensitivity.GEMMA_1B.weights_identity == sensitivity.MODEL_WEIGHTS_SHA256


def test_270m_measures_in_fp32_in_its_own_dataset() -> None:
    import build_gemma_3_270m_sensitivity_dataset as launcher

    spec = launcher.GEMMA_270M
    assert spec.dtype == torch.float32 and spec.precision == "fp32"
    assert sensitivity.GEMMA_1B.precision == "fp16"
    assert sensitivity.ModelSpec("a/b", "r", "t", 1, (), dtype=torch.bfloat16).precision == "bf16"


def test_local_sharded_weights_verify_every_shard(tmp_path) -> None:
    shards = [tmp_path / name for name in ("a.safetensors", "b.safetensors")]
    for path in shards:
        path.write_bytes(path.name.encode())
    spec = sensitivity.ModelSpec("test/model", "revision", "gemma3", 34,
                                 tuple((path.name, hash_file(path)) for path in shards))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"first": shards[0].name, "second": shards[1].name}}))
    sensitivity.verify_local_weights(tmp_path, spec)
    shards[1].write_bytes(b"wrong")
    with pytest.raises(ValueError, match="b.safetensors"):
        sensitivity.verify_local_weights(tmp_path, spec)


def test_completed_sweep_plots_and_fit_stay_inside_dataset(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    checkpoint = tmp_path / "dataset" / "measurements" / "sweep" / "results.json"
    checkpoint.parent.mkdir(parents=True)
    levels = [0.0, 0.05, 0.10, 0.20, 0.35]
    trials = [{"target": "33:mlp.up_proj", "requested_error": level, "seed": 0,
               "actual_error": level, "kl_nats_per_token": 0.001 + level**2} for level in levels]
    report = {"protocol": {"teacher": {"source": sensitivity.GEMMA_4B.source}, "teacher_artifact": "test",
                           "targets": ["33:mlp.up_proj"], "levels": levels, "seeds": [0]},
              "clean_baseline": {"kl_nats_per_token": 0.001}, "trials": trials, "expected_trials": len(trials)}
    checkpoint.write_text(json.dumps(report), encoding="utf-8")
    output = checkpoint.parent / "reports"
    sensitivity._write_plots(checkpoint, report, output, "projection")
    assert {p.name for p in output.glob("*.png")} == {
        "kl-sensitivity-heatmaps.png", "kl-error-response.png",
        "equal-kl-target-error-heatmap-0.001-nats.png",
        "equal-kl-target-error-heatmap-0.0025-nats.png",
        "equal-kl-target-error-heatmap-0.005-nats.png",
    }
    fit = json.loads(checkpoint.with_name("kl-response-fits.json").read_text())
    curve = fit["curves"]["33:mlp.up_proj"]
    assert curve["exponent"] == pytest.approx(2.0)
    assert fit["kl_targets"] == [0.001, 0.0025, 0.005]
    # KL above control is level ** 2, so each target inverts to its square root.
    for item, kl in zip(curve["kl_targets"], fit["kl_targets"], strict=True):
        assert item["kl_above_control"] == kl
        assert item["target_error"] == pytest.approx(kl ** 0.5)
    assert [item["range_status"] for item in curve["kl_targets"]] == [
        "below_measured", "within_measured", "within_measured",
    ]


def test_kl_targets_must_be_positive_and_unique(tmp_path) -> None:
    from fit_kl_sensitivity import fit_dataset

    path = tmp_path / "results.json"
    path.write_text("{}", encoding="utf-8")
    for targets in ((), (0.0,), (float("nan"),), (0.001, 0.001)):
        with pytest.raises(ValueError, match="KL targets"):
            fit_dataset(path, targets)


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
    trials = len(targets) * len(sensitivity.DEFAULT_ERROR_LEVELS) * 2
    assert report["expected_trials"] == len(report["trials"]) == trials
    assert {row["target"] for row in report["summary"]} == targets
    assert {row["requested_error"] for row in report["trials"]} == {0.01, 0.025, 0.05, 0.10, 0.20, 0.35}
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


def _tiny_sweep(tmp_path, monkeypatch, samples=1):
    model = TinyModel("projection").eval()
    adapter = SimpleNamespace(
        attention_implementation="eager",
        get_decoder_layers=lambda model: model.blocks,
        quantizable_layers=lambda _block, _index: [SimpleNamespace(path="0"), SimpleNamespace(path="1")],
    )
    monkeypatch.setattr(sensitivity, "adapter_for_config", lambda _: adapter)
    monkeypatch.setattr(sensitivity, "vocabulary_size_from_config", lambda _: 4)
    monkeypatch.setattr(sensitivity, "hash_hf_tokenizer_snapshot", lambda _: "test-tokenizer")
    monkeypatch.setattr(sensitivity, "_load_tokens",
                        lambda *_: (torch.tensor([[0, 1, 2, 3]]).repeat(samples, 1), 0))
    args = SimpleNamespace(
        output=tmp_path / "reports", blocks=None, granularity="projection",
        teacher_dataset=tmp_path / "teacher", device="cpu", token_chunk_size=2,
        error_levels=[0.05, 0.10], seeds=[0, 1],
        text_column="content", calibration_seed=0, samples=samples, sequence_length=4,
    )
    dataset = sensitivity._teacher_dataset(args, tmp_path, model, {}, args.teacher_dataset, "test-data")
    return model, args, dataset


def test_teacher_passes_resume_partial_work_and_match_full_scores(tmp_path, monkeypatch) -> None:
    model, args, dataset = _tiny_sweep(tmp_path, monkeypatch, samples=5)
    sensitivity._sweep(args, {}, model, dataset)
    checkpoint, = (args.teacher_dataset / "measurements").glob("*/results.json")
    reference = json.loads(checkpoint.read_text(encoding="utf-8"))
    partial = {**reference, "trials": reference["trials"][:1]}
    checkpoint.write_text(json.dumps(partial), encoding="utf-8")
    args.output = tmp_path / "pass-reports"
    args.teacher_passes = 4
    args.teacher_gpu_cache_gib = 1
    scorer = sensitivity.score_model
    calls = []

    def interrupted(*values, **keywords):
        calls.append(None)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return scorer(*values, **keywords)

    with monkeypatch.context() as patch:
        patch.setattr(sensitivity, "score_model", interrupted)
        with pytest.raises(KeyboardInterrupt):
            sensitivity._sweep(args, {}, model, dataset)
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["trials"] == reference["trials"][:1]
    pass_file = checkpoint.parent / "passes-4" / "000.json"
    assert len(json.loads(pass_file.read_text(encoding="utf-8"))["trials"]) == 3
    resumed_calls = []

    def resumed(*values, **keywords):
        resumed_calls.append(None)
        return scorer(*values, **keywords)

    with monkeypatch.context() as patch:
        patch.setattr(sensitivity, "score_model", resumed)
        sensitivity._sweep(args, {}, model, dataset)
    final = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert final == reference
    assert len(resumed_calls) == (reference["expected_trials"] - 1) * 4 - 3
    assert all(torch.equal(weight, model.original) for weight in model.parameters())


def test_added_error_levels_reuse_trials_measured_at_shared_levels(tmp_path, monkeypatch) -> None:
    model, args, dataset = _tiny_sweep(tmp_path, monkeypatch)
    sensitivity._sweep(args, {}, model, dataset)
    first, = (args.teacher_dataset / "measurements").glob("*/results.json")
    original = json.loads(first.read_text(encoding="utf-8"))
    measured = len(model.changed_counts)

    args.error_levels = [0.01, 0.05, 0.10]
    sensitivity._sweep(args, {}, model, dataset)

    # Only the eight new 1% trials run; the clean control and shared trials are imported.
    assert model.changed_counts[measured:] == [1] * 8
    second, = (path for path in (args.teacher_dataset / "measurements").glob("*/results.json") if path != first)
    extended = json.loads(second.read_text(encoding="utf-8"))
    assert extended["clean_baseline"] == original["clean_baseline"]
    assert extended["imported_trials"] == [
        {"sweep_id": first.parent.name, "sha256": sensitivity.hash_file(first), "trials": 16},
    ]
    assert extended["trials"][:16] == original["trials"]
    assert len(extended["trials"]) == extended["expected_trials"] == 24
    assert {row["requested_error"] for row in extended["trials"][16:]} == {0.01}
    exported = json.loads((args.output / "results.json").read_text(encoding="utf-8"))
    assert exported["protocol"]["levels"] == [0.01, 0.05, 0.10]
    # The earlier sweep keeps its own measurements.
    assert json.loads(first.read_text(encoding="utf-8")) == original


def test_trials_are_not_imported_from_a_different_protocol(tmp_path, monkeypatch) -> None:
    model, args, dataset = _tiny_sweep(tmp_path, monkeypatch)
    sensitivity._sweep(args, {}, model, dataset)
    measured = len(model.changed_counts)

    args.seeds, args.error_levels = [0, 2], [0.01, 0.05, 0.10]
    args.output = tmp_path / "other-reports"
    sensitivity._sweep(args, {}, model, dataset)

    assert model.changed_counts[measured:] == [0] + [1] * 24
    assert all("imported_trials" not in json.loads(path.read_text(encoding="utf-8"))
               for path in (args.teacher_dataset / "measurements").glob("*/results.json"))
