import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytest import MonkeyPatch

from nanoquant.application.kl_budget import KlSequenceResult
from nanoquant.application.temperature_calibration import (
    TemperatureNllStatistics,
    TemperatureSequenceMetrics,
    fit_logit_temperature,
)
from nanoquant.infrastructure.temperature_fit_checkpoint import (
    complete_temperature_fit_receipt,
)
from tools.probe_non_wikitext_kd_quality import (
    _arm_result,
    _c4_slice_reservation,
    _contiguous_token_windows,
    _evaluate_temperature_arm,
    _kl_result_from_temperature,
    _load_c4_tokens,
    _load_temperature_receipts,
    _paired_temperature_metric,
    _parse_arm,
    _parse_temperature_receipt,
    _parser,
    _sequence_from_checkpoint,
    _temperature_bootstrap,
)


class _Tokenizer:
    bos_token_id = 2

    def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
        assert text == "first second"
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": True}
        return SimpleNamespace(input_ids=torch.arange(18).unsqueeze(0))


class _LogitModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("values", logits)

    def forward(self, *, input_ids: torch.Tensor, use_cache: bool) -> SimpleNamespace:
        assert not use_cache
        return SimpleNamespace(logits=self.values.expand(input_ids.shape[0], -1, -1))


def test_c4_arm_reports_temperature_invariant_top1_agreement() -> None:
    result = _arm_result(
        "candidate",
        (
            KlSequenceResult(4.0, 1.0, 3, 2 / 3),
            KlSequenceResult(5.0, 2.0, 1, 0.0),
        ),
    )

    assert result.teacher_top1_agreement == pytest.approx(0.5)


def test_c4_checkpoint_resume_preserves_top1_agreement() -> None:
    restored = _sequence_from_checkpoint(
        {
            "negative_log_likelihood": 4.0,
            "kl_nats_per_token": 1.0,
            "token_count": 3,
            "teacher_top1_agreement": 2 / 3,
        }
    )

    assert restored.teacher_top1_agreement == pytest.approx(2 / 3)


def test_c4_checkpoint_resume_accepts_pre_top1_checkpoint() -> None:
    restored = _sequence_from_checkpoint(
        {
            "negative_log_likelihood": 4.0,
            "kl_nats_per_token": 1.0,
            "token_count": 3,
        }
    )

    assert restored.teacher_top1_agreement is None


def test_c4_temperature_pass_keeps_raw_metrics_and_top1_ordering() -> None:
    tokens = torch.tensor([[0, 1, 0]])
    teacher = _LogitModel(torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]]))
    student = _LogitModel(torch.tensor([[[0.0, 0.5], [0.5, 0.0], [0.0, 0.0]]]))

    sequences = _evaluate_temperature_arm(
        "candidate",
        teacher,
        student,
        tokens,
        logit_scale=1.5,
        top_k=1,
        device="cpu",
        token_chunk_size=2,
    )
    raw = _kl_result_from_temperature("candidate", sequences, fitted=False)
    fitted = _kl_result_from_temperature("candidate", sequences, fitted=True)

    assert fitted.negative_log_likelihood < raw.negative_log_likelihood
    assert fitted.kl_nats_per_token < raw.kl_nats_per_token
    assert fitted.teacher_top1_agreement == raw.teacher_top1_agreement == 1.0


def test_arm_parser_keeps_windows_drive_colons_inside_the_run_path() -> None:
    assert _parse_arm(r"candidate=postkd;D:\evidence\candidate") == (
        "candidate",
        "postkd",
        Path(r"D:\evidence\candidate"),
        None,
        None,
    )
    assert _parse_arm(
        r"conditional=tuning;D:\evidence\candidate;D:\evidence\primary.json"
    ) == (
        "conditional",
        "tuning",
        Path(r"D:\evidence\candidate"),
        Path(r"D:\evidence\primary.json"),
        None,
    )


def test_checkpoint_arm_parser_keeps_windows_paths_and_epoch() -> None:
    assert _parse_arm(r"candidate=checkpoint;D:\frozen;D:\checkpoints;4") == (
        "candidate",
        "checkpoint",
        Path(r"D:\frozen"),
        Path(r"D:\checkpoints"),
        4,
    )


def test_temperature_receipts_bind_primary_arms_and_share_calibration_slice(
    tmp_path: Path,
) -> None:
    result = fit_logit_temperature(
        lambda scale: TemperatureNllStatistics(8, 3.0, scale - 1.2, 1.0)
    )
    entries = []
    for name, role in (("baseline", "baseline"), ("candidate", "selected")):
        output = tmp_path / f"{name}.json"
        protocol = {
            "solver": {
                "version": 1,
                "initial_logit_scale": 1.0,
                "minimum_logit_scale": 0.5,
                "maximum_logit_scale": 1.5,
                "maximum_update_passes": 4,
                "convergence_tolerance": 1e-4,
                "hessian_floor": 1e-12,
            },
            "selection": {
                "decision": "decision.json",
                "decision_sha256": "sha256:decision",
                "selected_arm": "candidate",
                "role": role,
            },
            "dataset": {"name": "allenai/c4"},
            "slice": {"token_hash": "sha256:calibration"},
            "model": {"revision": "revision", "frozen_identity": {"model_hash": "m"}},
            "arm": {"name": name, "steps_completed": 96},
        }
        complete_temperature_fit_receipt(
            output,
            tmp_path / f"{name}.checkpoint.json",
            protocol,
            result,
        )
        entries.append((name, output))

    protocols, results, summaries = _load_temperature_receipts(
        tuple(entries),
        baseline="baseline",
        candidate="candidate",
        current_token_hash="sha256:held-out",
    )

    assert set(protocols) == set(results) == set(summaries) == {"baseline", "candidate"}
    assert summaries["candidate"]["logit_scale"] == pytest.approx(1.2)
    assert _parse_temperature_receipt("candidate=fit.json") == (
        "candidate",
        Path("fit.json"),
    )


def test_temperature_report_bootstraps_mass_and_paired_effects() -> None:
    baseline = (
        TemperatureSequenceMetrics(4.0, 1.0, 3.8, 0.8, 0.5, 0.9, 0.7, 0.75, 3),
        TemperatureSequenceMetrics(5.0, 2.0, 4.8, 1.8, 1.0, 0.9, 0.8, 0.85, 1),
    )
    candidate = (
        TemperatureSequenceMetrics(3.5, 0.8, 3.4, 0.7, 0.5, 0.9, 0.75, 0.8, 3),
        TemperatureSequenceMetrics(4.5, 1.8, 4.4, 1.7, 1.0, 0.9, 0.85, 0.9, 1),
    )

    absolute = _temperature_bootstrap(
        baseline,
        "raw_student_teacher_topk_mass",
        resamples=100,
    )
    paired = _paired_temperature_metric(
        baseline,
        candidate,
        "raw_student_teacher_topk_mass",
        resamples=100,
        higher_is_better=True,
    )

    assert absolute["mean"] == pytest.approx(0.725)
    assert paired["point_delta"] == pytest.approx(0.05)
    assert paired["improved_with_confidence"] is True


def test_temperature_receipts_reject_reuse_of_final_slice(tmp_path: Path) -> None:
    result = fit_logit_temperature(
        lambda scale: TemperatureNllStatistics(8, 3.0, scale - 1.2, 1.0)
    )
    entries = []
    for name, role in (("baseline", "baseline"), ("candidate", "selected")):
        output = tmp_path / f"{name}.json"
        protocol = {
            "solver": {
                "version": 1,
                "initial_logit_scale": 1.0,
                "minimum_logit_scale": 0.5,
                "maximum_logit_scale": 1.5,
                "maximum_update_passes": 4,
                "convergence_tolerance": 1e-4,
                "hessian_floor": 1e-12,
            },
            "selection": {"selected_arm": "candidate", "role": role},
            "dataset": {},
            "slice": {"token_hash": "sha256:same"},
            "model": {},
            "arm": {"name": name},
        }
        complete_temperature_fit_receipt(
            output,
            tmp_path / f"{name}.checkpoint.json",
            protocol,
            result,
        )
        entries.append((name, output))

    with pytest.raises(ValueError, match="data role"):
        _load_temperature_receipts(
            tuple(entries),
            baseline="baseline",
            candidate="candidate",
            current_token_hash="sha256:same",
        )


def test_contiguous_c4_windows_apply_offset_after_tokenization() -> None:
    tokens, bos_token_id = _contiguous_token_windows(
        ["first", "second"],
        _Tokenizer(),
        offset=1,
        samples=2,
        sequence_length=6,
    )

    assert bos_token_id == 2
    assert torch.equal(tokens, torch.arange(6, 18).reshape(2, 6))


def test_primary_comparison_is_explicit_in_the_cli_protocol() -> None:
    args = _parser().parse_args(
        [
            "--snapshot",
            "snapshot",
            "--output",
            "output.json",
            "--arm",
            "baseline=postkd;baseline-run",
            "--arm",
            "candidate=postkd;candidate-run",
            "--primary-baseline",
            "baseline",
            "--primary-candidate",
            "candidate",
            "--expected-steps",
            "baseline=256",
            "--expected-steps",
            "candidate=256",
            "--slice-registry",
            "registry.json",
            "--slice-id",
            "candidate-slice",
            "--temperature-fit-receipt",
            "baseline=baseline-fit.json",
            "--temperature-fit-receipt",
            "candidate=candidate-fit.json",
        ]
    )

    assert args.primary_baseline == "baseline"
    assert args.primary_candidate == "candidate"
    assert args.temperature_fit_receipt == [
        ("baseline", Path("baseline-fit.json")),
        ("candidate", Path("candidate-fit.json")),
    ]


def test_c4_slice_reservation_rejects_retired_overlap(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slices": [
                    {
                        "id": "old",
                        "dataset": "allenai/c4",
                        "split": "validation",
                        "offset": 0,
                        "samples": 48,
                        "sequence_length": 512,
                        "token_start": 0,
                        "token_end": 24576,
                        "token_hash": "old",
                        "status": "retired",
                    },
                    {
                        "id": "new",
                        "dataset": "allenai/c4",
                        "split": "validation",
                        "offset": 24,
                        "samples": 48,
                        "sequence_length": 512,
                        "token_start": 12288,
                        "token_end": 36864,
                        "token_hash": "new",
                        "status": "reserved",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlaps"):
        _c4_slice_reservation(
            registry,
            "new",
            offset=24,
            samples=48,
            sequence_length=512,
            token_hash="new",
        )


def test_local_c4_shard_uses_packaged_json_loader(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    shard = tmp_path / "c4.json.gz"
    shard.write_bytes(b"retained shard")
    calls: list[tuple[object, ...]] = []

    class _Dataset:
        _fingerprint = "local-fingerprint"

        def __len__(self) -> int:
            return 2

        def __getitem__(self, item: slice) -> dict[str, list[str]]:
            assert item == slice(None, 2)
            return {"text": ["first", "second"]}

    def fake_load_dataset(*args: object, **kwargs: object) -> _Dataset:
        calls.append(args + (kwargs,))
        return _Dataset()

    monkeypatch.setattr(
        "tools.probe_non_wikitext_kd_quality.load_dataset",
        fake_load_dataset,
    )
    monkeypatch.setattr(
        "tools.probe_non_wikitext_kd_quality.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )

    tokens, fingerprint, bos_token_id = _load_c4_tokens(
        tmp_path,
        revision="revision",
        data_file=str(shard),
        documents=2,
        offset=1,
        samples=2,
        sequence_length=6,
        local_files_only=True,
    )

    assert calls == [
        (
            "json",
            {
                "data_files": {"validation": str(shard.resolve())},
                "split": "validation",
            },
        )
    ]
    assert fingerprint == "local-fingerprint"
    assert bos_token_id == 2
    assert torch.equal(tokens, torch.arange(6, 18).reshape(2, 6))


def test_local_c4_arrow_cache_opens_without_dataset_rebuild(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    arrow = tmp_path / "c4-validation.arrow"
    arrow.write_bytes(b"retained arrow")

    class _Dataset:
        _fingerprint = "arrow-fingerprint"

        def __len__(self) -> int:
            return 2

        def __getitem__(self, item: slice) -> dict[str, list[str]]:
            assert item == slice(None, 2)
            return {"text": ["first", "second"]}

    observed: list[str] = []
    monkeypatch.setattr(
        "tools.probe_non_wikitext_kd_quality.Dataset.from_file",
        lambda path: observed.append(path) or _Dataset(),
    )
    monkeypatch.setattr(
        "tools.probe_non_wikitext_kd_quality.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )

    tokens, fingerprint, bos_token_id = _load_c4_tokens(
        tmp_path,
        revision="revision",
        data_file=str(arrow),
        documents=2,
        offset=1,
        samples=2,
        sequence_length=6,
        local_files_only=True,
    )

    assert observed == [str(arrow.resolve())]
    assert fingerprint == "arrow-fingerprint"
    assert bos_token_id == 2
    assert torch.equal(tokens, torch.arange(6, 18).reshape(2, 6))
