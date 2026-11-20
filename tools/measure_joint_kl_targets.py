"""Perturb every projection at once to its equal-KL target error and measure the joint KL.

Each projection's power-law KL response is fitted from a completed projection sweep and inverted
at every KL target, exactly as the equal-KL heatmaps are. All projections are then perturbed at
the same time, each to its own target error with the same seeded Gaussian direction the sweep
used, and full-vocabulary KL against the sweep's teacher is measured. If isolated effects were
additive, the joint KL above the clean control would be the number of projections times the target.

Usage:
    .venv\\Scripts\\python.exe tools\\measure_joint_kl_targets.py ^
        Datasets\\<dataset>\\measurements\\<sweep-id>\\results.json --local-files-only
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import torch
from build_gemma_3_270m_sensitivity_dataset import GEMMA_270M
from fit_kl_sensitivity import DEFAULT_KL_TARGETS, fit_dataset
from huggingface_hub import snapshot_download
from kl_sensitivity_dataset import (
    GEMMA_1B,
    GEMMA_4B,
    ModelSpec,
    perturb_weights,
    score_model,
    verify_local_weights,
)
from torch import nn

from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file
from nanoquant.infrastructure.kl_teacher_dataset import KlTeacherDataset, load_kl_teacher_dataset
from nanoquant.infrastructure.model_adapters import adapter_for_config

SPECS = (GEMMA_1B, GEMMA_270M, GEMMA_4B, replace(GEMMA_4B, dtype=torch.float16))
# Two-sided 95% Student-t critical values for 1-30 degrees of freedom.
_T95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("a confidence interval needs at least two runs")
    if degrees_of_freedom <= len(_T95):
        return _T95[degrees_of_freedom - 1]
    z, df = 1.959964, degrees_of_freedom  # Cornish-Fisher expansion; within 1e-3 above 30 df.
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def confidence_interval_95(values: list[float]) -> tuple[float, float] | None:
    """Two-sided 95% Student-t interval for the mean of independent runs."""
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    half_width = t_critical_95(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def spec_for_protocol(protocol: Mapping) -> ModelSpec:
    teacher = protocol["teacher"]
    for spec in SPECS:
        if (spec.source, spec.revision, str(spec.dtype)) == (
            teacher["source"], teacher["revision"], teacher["model_dtype"],
        ):
            return spec
    raise ValueError(f"no pinned sensitivity model matches {teacher['source']} in {teacher['model_dtype']}")


def target_errors(fit: Mapping, index: int) -> dict[str, float]:
    """Return every projection's target relative error for one KL target, refusing extrapolation."""
    errors = {}
    for target, curve in fit["curves"].items():
        item = curve["kl_targets"][index]
        if item["range_status"] != "within_measured":
            raise ValueError(
                f"{target} needs {item['target_error']:.3%} error for {item['kl_above_control']:g} nats/token, "
                "outside its measured range; extend the sweep before measuring this target jointly"
            )
        errors[target] = item["target_error"]
    return errors


@contextmanager
def perturb_jointly(
    weights: Mapping[str, torch.Tensor], errors: Mapping[str, float], *, seed: int,
) -> Iterator[dict[str, dict[str, float | int]]]:
    """Perturb every named weight to its own relative error with the sweep's per-target noise draw."""
    if set(weights) != set(errors):
        raise ValueError("joint perturbation needs exactly one target error per weight")
    metrics: dict[str, dict[str, float | int]] = {}
    with ExitStack() as stack:
        for name, weight in weights.items():
            metrics.update(stack.enter_context(perturb_weights({name: weight}, level=errors[name], seed=seed)))
        yield metrics


def projection_weights(model: nn.Module, adapter, targets: list[str]) -> dict[str, torch.Tensor]:
    layers = adapter.get_decoder_layers(model)
    weights = {}
    for target in targets:
        block, separator, path = target.partition(":")
        if not separator:
            raise ValueError("joint KL targets require a projection-granularity sweep")
        weights[target] = layers[int(block)].get_submodule(path).weight
    return weights


def measure(
    model: nn.Module, dataset: KlTeacherDataset, weights: Mapping[str, torch.Tensor], fit: Mapping,
    *, seeds: list[int], device: str, token_chunk_size: int, include_control: bool = True, log=print,
) -> dict:
    clean = score_model(model, dataset, device=device, token_chunk_size=token_chunk_size)
    curves = fit["curves"]
    results = []
    # The control row runs the same joint path at zero error, so it measures repeat-run noise.
    cases = [(0.0, dict.fromkeys(curves, 0.0))] if include_control else []
    cases += [(kl_target, target_errors(fit, index)) for index, kl_target in enumerate(fit["kl_targets"])]
    for kl_target, errors in cases:
        runs = []
        for seed in seeds:
            started = time.perf_counter()
            with perturb_jointly(weights, errors, seed=seed) as metrics:
                score = score_model(model, dataset, device=device, token_chunk_size=token_chunk_size)
            actual = {name: float(row["relative_frobenius_error"]) for name, row in metrics.items()}
            predicted = math.fsum(
                curves[name]["coefficient"] * actual[name] ** curves[name]["exponent"] for name in actual
            )
            kl_above = score["kl_nats_per_token"] - clean["kl_nats_per_token"]
            ratio = kl_above / predicted if predicted > 0 else None
            runs.append({
                "seed": seed,
                "kl_nats_per_token": score["kl_nats_per_token"],
                "kl_above_control": kl_above,
                "nll_nats_per_token": score["nll_nats_per_token"],
                "nll_above_control": score["nll_nats_per_token"] - clean["nll_nats_per_token"],
                "additive_prediction": predicted,
                "measured_over_additive": ratio,
                "actual_errors": actual,
                "sequences": score["sequences"],
                "seconds": time.perf_counter() - started,
            })
            log(f"KL target {kl_target:g}, seed {seed}: joint KL above control {kl_above:.6g} nats/token "
                f"(additive prediction {predicted:.6g}"
                + ("" if ratio is None else f", ratio {ratio:.3f}") + ")")
        values = [run["kl_above_control"] for run in runs]
        sd = statistics.stdev(values) if len(values) > 1 else None
        ratios = [run["measured_over_additive"] for run in runs]
        results.append({
            "kl_target": kl_target,
            "projection_count": len(errors),
            "isolated_sum": kl_target * len(errors),
            "target_errors": errors,
            "mean_kl_above_control": statistics.fmean(values),
            "kl_seed_sd": sd,
            "kl_standard_error": None if sd is None else sd / math.sqrt(len(values)),
            "kl_confidence_interval_95": confidence_interval_95(values),
            "mean_additive_prediction": statistics.fmean(run["additive_prediction"] for run in runs),
            "mean_measured_over_additive": None if None in ratios else statistics.fmean(ratios),
            "ratio_confidence_interval_95": None if None in ratios else confidence_interval_95(ratios),
            "mean_nll_above_control": statistics.fmean(run["nll_above_control"] for run in runs),
            "runs": runs,
        })
    return {"clean": clean, "results": results}


def _error_summary(errors: Mapping[str, float]) -> str:
    values = sorted(errors.values())
    if values[-1] == 0:
        return "0 (unperturbed control)"
    return f"{statistics.median(values):.2%} (range {values[0]:.2%}–{values[-1]:.2%})"


def _nats(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign}.2e}" if value != 0 and abs(value) < 1e-3 else f"{value:{sign}.4f}"


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _interval(bounds, formatter) -> str:
    return "n/a" if bounds is None else f"{formatter(bounds[0])} – {formatter(bounds[1])}"


def render_markdown(report: Mapping) -> str:
    protocol = report["protocol"]
    clean = report["clean"]
    lines = [
        f"# Joint equal-KL perturbation: {protocol['teacher']['source']}",
        "",
        f"Measured on {report['measured_at']} from sweep `{report['measurement_sweep']}`.",
        f"Model precision `{protocol['teacher']['model_dtype']}`, {clean['token_count']:,} scored next-token "
        f"positions per run, {len(report['seeds'])} runs per target (noise seeds "
        f"{report['seeds'][0]}–{report['seeds'][-1]}).",
        "",
        "Every projection is perturbed at the same time to the relative weight error that its isolated",
        "power-law fit predicts would add the KL target by itself. The additive prediction sums each",
        "projection's fitted KL at its measured error, which is about the projection count times the",
        "target. A ratio above 1 means the errors compound; below 1, they partly cancel or saturate.",
        "The 0 row is the control: the same joint path with zero error, run as many times as the others.",
        "Each run uses a different noise seed, so the SD is the spread across noise draws and the",
        "standard error is the uncertainty of the mean. The 95% confidence intervals are two-sided",
        "Student-t intervals for the mean over the runs, treating noise draws as independent.",
        "",
        f"Clean model: {clean['kl_nats_per_token']:.6g} nats/token KL against the stored teacher, "
        f"{clean['nll_nats_per_token']:.6f} nats/token NLL.",
        "",
        "| KL target per projection | Projections | Median target error | Additive prediction | "
        "Mean joint KL above control | 95% CI of mean KL | SD | Standard error | Mean measured / additive | "
        "95% CI of ratio | Mean NLL change |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["results"]:
        lines.append(
            f"| {item['kl_target']:g} | {item['projection_count']} | {_error_summary(item['target_errors'])} | "
            f"{_nats(item['mean_additive_prediction'])} | {_nats(item['mean_kl_above_control'])} | "
            f"{_interval(item.get('kl_confidence_interval_95'), _nats)} | "
            f"{_nats(item['kl_seed_sd'])} | {_nats(item.get('kl_standard_error'))} | "
            f"{_ratio(item['mean_measured_over_additive'])} | "
            f"{_interval(item.get('ratio_confidence_interval_95'), _ratio)} | "
            f"{_nats(item['mean_nll_above_control'], signed=True)} |"
        )
    lines += ["", "## Runs", "", "| KL target | Seed | Joint KL above control | Additive prediction | Ratio | "
              "NLL change |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in report["results"]:
        for run in item["runs"]:
            lines.append(
                f"| {item['kl_target']:g} | {run['seed']} | {_nats(run['kl_above_control'])} | "
                f"{_nats(run['additive_prediction'])} | {_ratio(run['measured_over_additive'])} | "
                f"{_nats(run['nll_above_control'], signed=True)} |"
            )
    lines += [
        "",
        "## Notes",
        "",
        "- KL is full-vocabulary KL(teacher ‖ perturbed) per next-token position against the sweep's teacher.",
        "- Each projection reuses the sweep's seeded noise direction, so a single-projection run reproduces",
        "  that projection's isolated sweep trial at the same error.",
        "- Noise is isotropic Gaussian. NanoQuant fitting errors have other structure and can interact differently.",
        f"- Full per-run errors and per-window scores: `{report['json']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measurements", type=Path, help="a completed projection sweep's measurements/<id>/results.json")
    parser.add_argument("--kl-target", type=float, action="append",
                        help=f"nats/token per projection; repeatable (default {DEFAULT_KL_TARGETS})")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(15)),
                        help="noise seeds, one run per seed for every target (default 0-14)")
    parser.add_argument("--no-control", action="store_true", help="skip the zero-error control row")
    parser.add_argument("--output", type=Path, help="Markdown report (default joint-kl-targets.md beside the sweep)")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    measurements = args.measurements.resolve()
    sweep = json.loads(measurements.read_text(encoding="utf-8"))
    protocol = sweep["protocol"]
    if len(sweep["trials"]) != sweep["expected_trials"]:
        raise ValueError("the sensitivity sweep is incomplete")
    spec = spec_for_protocol(protocol)
    fit = fit_dataset(measurements, tuple(args.kl_target or DEFAULT_KL_TARGETS))
    for index in range(len(fit["kl_targets"])):
        target_errors(fit, index)  # Refuse extrapolated targets before loading the model.
    root = measurements.parent.parent.parent
    output = (args.output or measurements.with_name("joint-kl-targets.md")).resolve()
    snapshot = Path(snapshot_download(
        repo_id=spec.source, revision=spec.revision, local_files_only=args.local_files_only,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    ))
    verify_local_weights(snapshot, spec)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    if spec.dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = protocol["device"]
    with acquire_device_lease(device):
        dataset = load_kl_teacher_dataset(root)
        if dataset.reference.artifact_id != protocol["teacher_artifact"]:
            raise ValueError("the teacher dataset does not match the sweep's teacher artifact")
        model = load_causal_language_model(
            snapshot, torch_dtype=spec.dtype, attention_implementation=adapter.attention_implementation,
            local_files_only=True,
        ).to(device).eval()
        weights = projection_weights(model, adapter, protocol["targets"])
        measured = measure(
            model, dataset, weights, fit, seeds=args.seeds, device=device,
            token_chunk_size=protocol["token_chunk_size"], include_control=not args.no_control,
        )
    report = {
        "schema_version": 1,
        "measured_at": time.strftime("%Y-%m-%d %H:%M"),
        "measurement_sweep": measurements.parent.name,
        "measurement_sha256": hash_file(measurements),
        "protocol": protocol,
        "seeds": args.seeds,
        "fit": {"model": fit["model"], "kl_targets": fit["kl_targets"]},
        "json": str(output.with_suffix(".json")),
        **measured,
    }
    atomic_write_json(output.with_suffix(".json"), report)
    atomic_write_text(output, render_markdown(report))
    print(f"Report written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
