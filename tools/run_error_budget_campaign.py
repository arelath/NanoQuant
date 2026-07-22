"""Run the resumable, operating-point-ordered Gemma-3-270M Doc 33 campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401

from nanoquant.application.kl_budget import (
    KL_BUDGET_EVALUATOR_VERSION,
    load_kl_budget_profile,
    paired_bootstrap_kl_delta,
)
from nanoquant.config.schema import KlSensitivityGranularity
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.runtime_export import load_frozen_run_rank_inventory

ROOT = Path(__file__).resolve().parent.parent
SOURCE = "unsloth/gemma-3-270m-it"
REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"
QKV_ARM = "type:self_attn.attn_qkv"
O_ARM = "type:self_attn.o_proj"
MIN_MATERIAL_RELATIVE_KL_IMPROVEMENT = 0.01
# A single aligned rank step can leave a few thousand unusable bits.  Reserving
# 0.01% of the factor budget for bias/patch arms keeps their *actual* BPW below
# the 1.0-target Experiment 016 artifact instead of spending that alignment tail.
SIDECAR_TARGET_BPW = 0.9999
DEFAULT_BASELINE_SUMMARY = (
    ROOT / "Results/016/016-compress-and-benchmark-gemma-3-270m-it-summary.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-source", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--baseline-profile-key")
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _profile_key(profile: Path) -> str:
    artifact = _read(profile / "artifact.json")
    if artifact.get("complete") is not True:
        raise ValueError(f"KL profile is incomplete: {profile}")
    if artifact.get("evaluator_version") != KL_BUDGET_EVALUATOR_VERSION:
        raise ValueError(f"KL profile evaluator identity is stale: {profile}")
    key = artifact.get("profile_key")
    if not isinstance(key, str) or not key:
        raise ValueError(f"KL profile has no key: {profile}")
    return key


def _arm_kl(profile: Path, arm: str) -> float:
    payload = _read(profile / "kl-budget-profile.json")
    for result in cast(list[dict[str, object]], payload["arms"]):
        if result.get("arm") == arm:
            return float(cast(float, result["kl_nats_per_token"]))
    raise ValueError(f"KL profile has no {arm!r} arm: {profile}")


def _reported_arm_kls(profile: Path) -> dict[str, object]:
    """Return the phase metrics that were intentionally measured for a profile."""

    payload = _read(profile / "kl-budget-profile.json")
    aliases = {"full": "full", QKV_ARM: "qkv", O_ARM: "o"}
    result: dict[str, object] = {"profile_key": _profile_key(profile)}
    for arm in cast(list[dict[str, object]], payload["arms"]):
        name = aliases.get(str(arm.get("arm")))
        if name is not None:
            result[name] = float(cast(float, arm["kl_nats_per_token"]))
    return result


def _require_at_or_below_budget(output: Path, effective_bpw: float, baseline_bpw: float) -> None:
    if effective_bpw > baseline_bpw + 1e-12:
        raise ValueError(
            f"candidate exceeds the Experiment 016 budget: {output}: "
            f"{effective_bpw} > {baseline_bpw}"
        )


def _require_candidate_sidecars(
    output: Path,
    summary: dict[str, Any],
    *,
    bias: bool,
    patch_rank: int,
) -> None:
    factor_owners = int(summary["factor_owners"])
    bias_owners = int(summary.get("bias_owner_count", -1))
    bias_bits = int(summary.get("actual_bias_bits", -1))
    expected_bias_owners = factor_owners if bias else 0
    if bias_owners != expected_bias_owners or (bias_bits > 0) != bias:
        raise ValueError(
            f"candidate bias inventory does not match its configured arm: {output}: "
            f"owners={bias_owners}/{expected_bias_owners}, bits={bias_bits}"
        )
    patch_owners = int(summary.get("patch_owner_count", -1))
    patch_bits = int(summary.get("actual_patch_bits", -1))
    if patch_rank == 0 and (patch_owners != 0 or patch_bits != 0):
        raise ValueError(f"rank-zero candidate unexpectedly froze patch tensors: {output}")
    if patch_rank > 0 and (patch_owners > 0) != (patch_bits > 0):
        raise ValueError(f"candidate patch owners and actual patch bits disagree: {output}")


def _select_winning_alpha(qkv_kl_by_alpha: dict[int, float]) -> int:
    if set(qkv_kl_by_alpha) != {1, 2, 4}:
        raise ValueError("D4 selection requires alpha_v arms 1, 2, and 4")
    return min(qkv_kl_by_alpha, key=qkv_kl_by_alpha.__getitem__)


def _select_winning_patch(
    o_kl_by_rank: dict[int, float],
    *,
    eligible_ranks: set[int] | None = None,
) -> int:
    if set(o_kl_by_rank) != {0, 4, 8, 16}:
        raise ValueError("D5 selection requires patch ranks 0, 4, 8, and 16")
    eligible = {0, 4, 8, 16} if eligible_ranks is None else set(eligible_ranks)
    if 0 not in eligible or not eligible.issubset(o_kl_by_rank):
        raise ValueError("D5 eligible ranks must include zero and name only measured ranks")
    no_patch_kl = o_kl_by_rank[0]
    return next(
        (rank for rank in (4, 8, 16) if rank in eligible and o_kl_by_rank[rank] < no_patch_kl),
        0,
    )


def _improvement_gate(before: float, after: float) -> dict[str, float | bool]:
    delta = after - before
    return {
        "before": before,
        "after": after,
        "delta": delta,
        "relative_delta": delta / before if before != 0 else float("nan"),
        "improved": after < before,
    }


def _profile_improvement_gate(before_profile: Path, after_profile: Path, arm: str) -> dict[str, float | bool]:
    before = load_kl_budget_profile(before_profile / "kl-budget-profile.json")
    after = load_kl_budget_profile(after_profile / "kl-budget-profile.json")
    if (
        before.provenance.model_source != after.provenance.model_source
        or before.provenance.model_revision != after.provenance.model_revision
        or before.provenance.dataset_fingerprint != after.provenance.dataset_fingerprint
        or before.provenance.dataset_slice_hash != after.provenance.dataset_slice_hash
    ):
        raise ValueError("paired KL gate profiles do not share model and dataset identities")
    before_arm = next((result for result in before.arms if result.arm == arm), None)
    after_arm = next((result for result in after.arms if result.arm == arm), None)
    if before_arm is None or after_arm is None:
        raise ValueError(f"paired KL gate profiles do not both contain {arm!r}")
    interval = paired_bootstrap_kl_delta(before_arm, after_arm)
    gate = _improvement_gate(before_arm.kl_nats_per_token, after_arm.kl_nats_per_token)
    gate.update(
        {
            "bootstrap_confidence": interval.confidence,
            "bootstrap_resamples": float(interval.resamples),
            "lower_delta": interval.lower_delta,
            "upper_delta": interval.upper_delta,
            "lower_relative_delta": interval.lower_delta / before_arm.kl_nats_per_token,
            "upper_relative_delta": interval.upper_delta / before_arm.kl_nats_per_token,
        }
    )
    return gate


def _material_improvement_passed(
    gate: dict[str, float | bool],
    minimum_relative_improvement: float = MIN_MATERIAL_RELATIVE_KL_IMPROVEMENT,
) -> bool:
    if not 0 < minimum_relative_improvement < 1:
        raise ValueError("material KL improvement threshold must be between zero and one")
    measured_delta = float(gate.get("upper_relative_delta", gate["relative_delta"]))
    return bool(gate["improved"]) and measured_delta <= -minimum_relative_improvement


def _phase_kl(phase_kls: dict[str, dict[str, object]], phase: str, arm: str) -> float:
    return float(cast(float, phase_kls[phase][arm]))


def _rank_redistribution(
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    baseline_by_id = {str(entry["unit_id"]): entry for entry in baseline}
    candidate_by_id = {str(entry["unit_id"]): entry for entry in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("D2 rank inventory differs from the Experiment 016 unit inventory")

    scopes: dict[str, Callable[[int, str], bool]] = {
        "mlp": lambda block, name: name.startswith("mlp."),
        "attention": lambda block, name: name.startswith("self_attn."),
        "early_blocks_0_10": lambda block, name: block <= 10,
        "late_blocks_11_17": lambda block, name: block >= 11,
    }
    result: dict[str, dict[str, int]] = {}
    for scope, selected in scopes.items():
        baseline_entries = [
            entry
            for entry in baseline
            if selected(int(cast(int, entry["block"])), str(entry["name"]))
        ]
        candidate_entries = [candidate_by_id[str(entry["unit_id"])] for entry in baseline_entries]
        baseline_rank = sum(int(cast(int, entry["rank"])) for entry in baseline_entries)
        candidate_rank = sum(int(cast(int, entry["rank"])) for entry in candidate_entries)
        baseline_bits = sum(int(cast(int, entry["factor_bits"])) for entry in baseline_entries)
        candidate_bits = sum(int(cast(int, entry["factor_bits"])) for entry in candidate_entries)
        result[scope] = {
            "baseline_rank": baseline_rank,
            "candidate_rank": candidate_rank,
            "rank_delta": candidate_rank - baseline_rank,
            "baseline_factor_bits": baseline_bits,
            "candidate_factor_bits": candidate_bits,
            "factor_bit_delta": candidate_bits - baseline_bits,
        }
    return result


def _rank_redistribution_gate(
    redistribution: dict[str, dict[str, int]],
) -> dict[str, bool]:
    directions = {
        "mlp_gained": (
            redistribution["mlp"]["rank_delta"] > 0
            and redistribution["mlp"]["factor_bit_delta"] > 0
        ),
        "attention_drained": (
            redistribution["attention"]["rank_delta"] < 0
            and redistribution["attention"]["factor_bit_delta"] < 0
        ),
        "early_gained": (
            redistribution["early_blocks_0_10"]["rank_delta"] > 0
            and redistribution["early_blocks_0_10"]["factor_bit_delta"] > 0
        ),
        "late_drained": (
            redistribution["late_blocks_11_17"]["rank_delta"] < 0
            and redistribution["late_blocks_11_17"]["factor_bit_delta"] < 0
        ),
    }
    return {**directions, "passed": all(directions.values())}


def _rank_inventory_for_run(
    run: Path,
    summary: dict[str, Any],
    *,
    expected_blocks: int = 18,
) -> list[dict[str, object]]:
    """Read a persisted inventory, or reconstruct it from validated frozen commits.

    Candidate summaries created before rank-inventory reporting was added remain valid
    campaign checkpoints.  Their committed frozen states are the authoritative fallback;
    malformed inventories still fail closed rather than silently changing evidence.
    """

    raw_inventory = summary.get("rank_inventory")
    if raw_inventory is not None:
        if not isinstance(raw_inventory, list) or any(
            not isinstance(entry, dict) for entry in raw_inventory
        ):
            raise ValueError(f"candidate has a malformed rank inventory: {run}")
        return cast(list[dict[str, object]], raw_inventory)
    return [
        {
            "unit_id": entry.unit_id,
            "block": entry.block,
            "name": entry.name,
            "rank": entry.rank,
            "factor_bits": entry.factor_bits,
        }
        for entry in load_frozen_run_rank_inventory(
            run.resolve(),
            expected_blocks,
            fresh_validation=True,
        )
    ]


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root: Path = Path(args.output_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.env = os.environ.copy()
        cache = ROOT / ".triton-cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.env["TRITON_CACHE_DIR"] = str(cache)
        self.commands: list[dict[str, object]] = []
        self.checkpoint_fields: dict[str, object] = {}
        baseline_summary = _read(Path(args.baseline_summary).resolve())
        compression = cast(dict[str, object], baseline_summary["compression"])
        self.baseline_effective_bpw = float(cast(float, compression["effective_bpw"]))
        self.baseline_rank_inventory = _rank_inventory_for_run(
            Path(args.calibration_source),
            baseline_summary,
        )

    @staticmethod
    def _marker_complete(marker: Path) -> bool:
        if not marker.exists():
            return False
        if marker.suffix == ".xml":
            try:
                root = ET.parse(marker).getroot()
                suites = tuple(root.iter("testsuite"))
                tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
                failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
                errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
                skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
            except (ET.ParseError, OSError, ValueError):
                return False
            return tests > 0 and failures == 0 and errors == 0 and skipped == 0
        if marker.suffix != ".json":
            return True
        try:
            payload = _read(marker)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if marker.name in {"candidate-summary.json", "distillation-summary.json"}:
            return payload.get("status") == "completed"
        if marker.name == "artifact.json":
            return (
                payload.get("complete") is True
                and payload.get("evaluator_version") == KL_BUDGET_EVALUATOR_VERSION
            )
        if marker.name == "resident-validation.json":
            return payload.get("complete") is True
        if marker.name.startswith("quality-comparison-"):
            return "candidate_evaluation" in payload
        return True

    def _run(self, name: str, command: list[str], marker: Path) -> None:
        if self._marker_complete(marker):
            self.commands.append({"step": name, "status": "reused", "marker": str(marker)})
            return
        rendered = subprocess.list2cmdline(command)
        self.commands.append({"step": name, "status": "planned", "command": rendered})
        self._checkpoint()
        if self.args.dry_run:
            print(rendered)
            return
        log = self.root / f"{name}.log"
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nCOMMAND: {rendered}\n")
            stream.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=self.env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"campaign step failed ({completed.returncode}): {name}; see {log}")
        if not self._marker_complete(marker):
            raise RuntimeError(f"campaign step produced no completion marker: {name}: {marker}")
        self.commands[-1] = {"step": name, "status": "completed", "marker": str(marker), "log": str(log)}
        self._checkpoint()

    def _checkpoint(self, **fields: object) -> None:
        self.checkpoint_fields.update(fields)
        atomic_write_json(
            self.root / "campaign-state.json",
            {
                "schema_version": 1,
                "commands": self.commands,
                **self.checkpoint_fields,
            },
        )

    def compress(
        self,
        name: str,
        profile: Path,
        key: str,
        *,
        bias: bool,
        alpha_v: float,
        patch_rank: int,
        target_bpw: float | None = None,
    ) -> Path:
        output = self.root / name
        command = [
            sys.executable,
            str(ROOT / "tools/run_error_budget_gemma270.py"),
            "--output",
            str(output),
            "--snapshot",
            str(self.args.snapshot.resolve()),
            "--calibration-source",
            str(self.args.calibration_source.resolve()),
            "--kl-profile",
            str(profile.resolve()),
            "--kl-profile-key",
            key,
            "--kl-granularity",
            KlSensitivityGranularity.EXACT.value,
            "--v-multiplier",
            str(alpha_v),
            "--patch-rank",
            str(patch_rank),
            "--device",
            self.args.device,
        ]
        if target_bpw is not None:
            command.extend(("--target-bpw", str(target_bpw)))
        if not bias:
            command.append("--no-bias-correction")
        self._run(name, command, output / "candidate-summary.json")
        if not self.args.dry_run:
            summary = _read(output / "candidate-summary.json")
            if summary.get("status") != "completed":
                raise ValueError(f"candidate is not complete: {output}")
            _require_candidate_sidecars(
                output,
                summary,
                bias=bias,
                patch_rank=patch_rank,
            )
            effective_bpw = float(summary["effective_bpw"])
            _require_at_or_below_budget(output, effective_bpw, self.baseline_effective_bpw)
            validation = output / "resident-validation.json"
            self._run(
                f"{name}-validate",
                [
                    sys.executable,
                    str(ROOT / "tools/validate_resident_run.py"),
                    "--run-output",
                    str(output),
                    "--expected-blocks",
                    "18",
                    "--require-complete",
                    "--output",
                    str(validation),
                ],
                validation,
            )
        return output

    def profile(
        self,
        name: str,
        run: Path,
        *,
        tuned: bool = False,
        arms: tuple[str, ...] = (),
        samples: int = 12,
        persistent_teacher_cache: bool = True,
    ) -> Path:
        if samples <= 0:
            raise ValueError("KL profile sample count must be positive")
        output = self.root / name
        command = [
            sys.executable,
            "-m",
            "nanoquant.cli.main",
            "kl-budget",
            "--run-output",
            str(run),
            "--snapshot",
            str(self.args.snapshot.resolve()),
            "--source",
            SOURCE,
            "--revision",
            REVISION,
            "--profile-output",
            str(output),
            "--device",
            self.args.device,
            "--wikitext-samples",
            str(samples),
            "--sequence-length",
            "512",
            "--batch-size",
            "1",
            "--token-chunk-size",
            "128",
            "--local-files-only",
        ]
        if persistent_teacher_cache:
            command.extend(("--teacher-cache-root", str(self.root / "teacher-cache")))
        else:
            command.extend(("--teacher-cache-mode", "on_the_fly"))
        if tuned:
            command.append("--use-global-tuning")
        for arm in arms:
            command.extend(("--arm", arm))
        self._run(name, command, output / "artifact.json")
        return output

    def distill(self, run: Path) -> None:
        self._run(
            f"{run.name}-distill",
            [
                sys.executable,
                str(ROOT / "tools/run_error_budget_distillation.py"),
                "--candidate-run",
                str(run),
                "--snapshot",
                str(self.args.snapshot.resolve()),
                "--calibration-source",
                str(self.args.calibration_source.resolve()),
                "--device",
                self.args.device,
            ],
            run / "distillation-summary.json",
        )

    def quality(self, run: Path) -> None:
        self._run(
            f"{run.name}-quality",
            [
                sys.executable,
                str(ROOT / "tools/run_error_budget_quality.py"),
                "--candidate-run",
                str(run),
                "--device",
                self.args.device,
                "--use-global-tuning",
                "--local-files-only",
            ],
            run / "quality-comparison-tuned.json",
        )

    def cuda_sidecar(self) -> Path:
        marker = self.root / "cuda-packed-bias-patch.xml"
        self._run(
            "cuda-packed-bias-patch",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "-m",
                "cuda",
                (
                    "tests/unit/test_runtime_cuda_backend.py::"
                    "test_cuda_packed_backend_executes_low_rank_patch_sidecars"
                ),
                "--junitxml",
                str(marker),
            ],
            marker,
        )
        return marker

    def execute(self) -> None:
        baseline_profile = self.args.baseline_profile.resolve()
        observed_baseline_key = _profile_key(baseline_profile)
        if (
            self.args.baseline_profile_key is not None
            and self.args.baseline_profile_key != observed_baseline_key
        ):
            raise ValueError("configured baseline KL profile key differs from its artifact")
        baseline_key = self.args.baseline_profile_key or observed_baseline_key
        baseline_d2_gate_profile = self.profile(
            "baseline-d2-full-gate-48",
            Path(self.args.calibration_source),
            arms=("full",),
            samples=48,
            persistent_teacher_cache=False,
        )

        d2 = self.compress(
            "d2-kl-exact-static",
            baseline_profile,
            baseline_key,
            bias=False,
            alpha_v=1,
            patch_rank=0,
        )
        d2_profile = self.profile("d2-kl-exact-profile", d2)
        d2_gate_profile = self.profile(
            "d2-kl-exact-full-gate-48",
            d2,
            arms=("full",),
            samples=48,
            persistent_teacher_cache=False,
        )
        if self.args.dry_run:
            return

        def d2_arm_report(run: Path, profile: Path, gate_profile: Path) -> dict[str, object]:
            candidate_summary = _read(run / "candidate-summary.json")
            redistribution = _rank_redistribution(
                self.baseline_rank_inventory,
                _rank_inventory_for_run(run, candidate_summary),
            )
            kl_gate = _profile_improvement_gate(baseline_d2_gate_profile, gate_profile, "full")
            rank_gate = _rank_redistribution_gate(redistribution)
            return {
                "run": str(run),
                "profile": str(profile),
                "gate_profile": str(gate_profile),
                "full_kl_gate": kl_gate,
                "rank_redistribution": redistribution,
                "rank_redistribution_diagnostic": rank_gate,
                "passed": _material_improvement_passed(kl_gate),
            }

        selected_d2 = d2_arm_report(d2, d2_profile, d2_gate_profile)
        if not bool(selected_d2["passed"]):
            self._checkpoint(
                d2_arm=selected_d2,
                d2_selection={
                    "status": "failed",
                    "reason": "exact-unit measured-response D2 did not pass the paired material KL gate",
                },
            )
            raise ValueError("exact-unit measured-response D2 did not pass its adoption gate")
        self._checkpoint(
            d2_arm=selected_d2,
            d2_selection={
                "status": "selected",
                "granularity": KlSensitivityGranularity.EXACT.value,
                "reason": "exact-unit measured-response KL passed the paired material gate",
            },
        )

        d3_control = self.compress(
            "d3-no-bias-control-static",
            d2_profile,
            _profile_key(d2_profile),
            bias=False,
            alpha_v=1,
            patch_rank=0,
        )
        d3_control_profile = self.profile(
            "d3-no-bias-control-kl-profile",
            d3_control,
            arms=(O_ARM,),
        )
        d3 = self.compress(
            "d3-bias-budget-guard-static",
            d2_profile,
            _profile_key(d2_profile),
            bias=True,
            alpha_v=1,
            patch_rank=0,
            target_bpw=SIDECAR_TARGET_BPW,
        )
        d3_profile = self.profile("d3-bias-kl-profile", d3)

        alpha_runs: dict[int, tuple[Path, Path]] = {}
        for alpha in (1, 2, 4):
            run = self.compress(
                f"d4-v{alpha}-static",
                d3_profile,
                _profile_key(d3_profile),
                bias=True,
                alpha_v=alpha,
                patch_rank=0,
                target_bpw=SIDECAR_TARGET_BPW,
            )
            alpha_runs[alpha] = (
                run,
                self.profile(
                    f"d4-v{alpha}-kl-profile",
                    run,
                    arms=(QKV_ARM,),
                ),
            )
        winning_alpha = _select_winning_alpha(
            {alpha: _arm_kl(profile, QKV_ARM) for alpha, (_run, profile) in alpha_runs.items()}
        )
        winning_d4_run = alpha_runs[winning_alpha][0]
        winning_d4_profile = self.profile(
            f"d4-v{winning_alpha}-full-kl-profile",
            winning_d4_run,
        )

        patch_runs: dict[int, tuple[Path, Path]] = {}
        for rank in (0, 4, 8, 16):
            run = self.compress(
                f"d5-k{rank}-static",
                winning_d4_profile,
                _profile_key(winning_d4_profile),
                bias=True,
                alpha_v=winning_alpha,
                patch_rank=rank,
                target_bpw=SIDECAR_TARGET_BPW,
            )
            patch_runs[rank] = (
                run,
                self.profile(
                    f"d5-k{rank}-kl-profile",
                    run,
                    arms=(O_ARM,),
                ),
            )
        patch_summaries = {
            rank: _read(run / "candidate-summary.json")
            for rank, (run, _profile) in patch_runs.items()
        }
        winning_patch = _select_winning_patch(
            {rank: _arm_kl(profile, O_ARM) for rank, (_run, profile) in patch_runs.items()},
            eligible_ranks={0}
            | {
                rank
                for rank, summary in patch_summaries.items()
                if rank > 0 and int(cast(int, summary.get("patch_owner_count", 0))) > 0
            },
        )
        final_run = patch_runs[winning_patch][0]
        final_static_profile = self.profile(
            f"d5-k{winning_patch}-full-kl-profile",
            final_run,
        )

        self.distill(final_run)
        tuned_profile = self.profile(f"{final_run.name}-tuned-kl-profile", final_run, tuned=True)
        self.quality(final_run)
        cuda_sidecar = self.cuda_sidecar()

        phase_profiles = {
            "baseline": baseline_profile,
            "d2": d2_profile,
            "d3_control": d3_control_profile,
            "d3": d3_profile,
            **{f"d4_v{alpha}": profile for alpha, (_run, profile) in alpha_runs.items()},
            "d4_winner_full": winning_d4_profile,
            **{f"d5_k{rank}": profile for rank, (_run, profile) in patch_runs.items()},
            "final_static": final_static_profile,
            "final_tuned": tuned_profile,
        }
        phase_kls = {name: _reported_arm_kls(profile) for name, profile in phase_profiles.items()}
        budget_runs = {
            d2,
            d3_control,
            d3,
            *(run for run, _profile in alpha_runs.values()),
            *(run for run, _profile in patch_runs.values()),
        }
        budgets = {
            run.name: float(_read(run / "candidate-summary.json")["effective_bpw"])
            for run in budget_runs
        }
        quality = _read(final_run / "quality-comparison-tuned.json")
        d2_rank_redistribution = cast(
            dict[str, dict[str, int]],
            selected_d2["rank_redistribution"],
        )
        phase_gates = {
            "d2_full_kl": _improvement_gate(
                _phase_kl(phase_kls, "baseline", "full"),
                _phase_kl(phase_kls, "d2", "full"),
            ),
            "d3_o_kl": _improvement_gate(
                _phase_kl(phase_kls, "d3_control", "o"),
                _phase_kl(phase_kls, "d3", "o"),
            ),
            "d4_qkv_kl": {
                **_improvement_gate(
                    _phase_kl(phase_kls, "d4_v1", "qkv"),
                    _phase_kl(phase_kls, f"d4_v{winning_alpha}", "qkv"),
                ),
                "adopted": winning_alpha != 1,
            },
            "d5_o_kl": {
                **_improvement_gate(
                    _phase_kl(phase_kls, "d5_k0", "o"),
                    _phase_kl(phase_kls, f"d5_k{winning_patch}", "o"),
                ),
                "adopted": winning_patch > 0,
            },
            "distillation_full_kl": _improvement_gate(
                _phase_kl(phase_kls, "final_static", "full"),
                _phase_kl(phase_kls, "final_tuned", "full"),
            ),
        }
        d2_rank_gate = _rank_redistribution_gate(d2_rank_redistribution)
        phase_claims = {
            "d2_full_kl_materially_improved": _material_improvement_passed(
                phase_gates["d2_full_kl"]
            ),
            "d3_o_kl_improved": bool(phase_gates["d3_o_kl"]["improved"]),
            "d4_multiplier_adopted_and_improved": bool(
                phase_gates["d4_qkv_kl"]["adopted"]
                and phase_gates["d4_qkv_kl"]["improved"]
            ),
            "d5_optional_decision_valid": bool(
                winning_patch == 0 or phase_gates["d5_o_kl"]["improved"]
            ),
            "distillation_full_kl_improved": bool(
                phase_gates["distillation_full_kl"]["improved"]
            ),
            "cuda_packed_bias_patch_passed": self._marker_complete(cuda_sidecar),
        }
        all_candidates_at_budget = all(
            value <= self.baseline_effective_bpw + 1e-12 for value in budgets.values()
        )
        quality_improved_at_same_budget = bool(quality["quality_improved_at_same_budget"])
        summary = {
            "schema_version": 1,
            "status": "completed",
            "winning_alpha_v": winning_alpha,
            "winning_patch_rank": winning_patch,
            "d2_sensitivity_granularity": KlSensitivityGranularity.EXACT.value,
            "sidecar_target_bpw": SIDECAR_TARGET_BPW,
            "d2_arm": selected_d2,
            "patch_acceptance": {
                str(rank): {
                    "accepted_owner_count": int(cast(int, candidate.get("patch_owner_count", 0))),
                    "actual_patch_bits": int(cast(int, candidate.get("actual_patch_bits", 0))),
                }
                for rank, candidate in patch_summaries.items()
            },
            "final_run": str(final_run),
            "phase_kls": phase_kls,
            "phase_gates": phase_gates,
            "material_relative_kl_improvement_threshold": (
                MIN_MATERIAL_RELATIVE_KL_IMPROVEMENT
            ),
            "phase_claims": {**phase_claims, "passed": all(phase_claims.values())},
            "d2_rank_redistribution": d2_rank_redistribution,
            "d2_rank_redistribution_gate": d2_rank_gate,
            "baseline_effective_bpw": self.baseline_effective_bpw,
            "effective_bpw": budgets,
            "all_candidates_at_or_below_baseline_budget": all_candidates_at_budget,
            "quality_improved_at_same_budget": quality_improved_at_same_budget,
            "design_claims_verified": (
                all_candidates_at_budget
                and quality_improved_at_same_budget
                and all(phase_claims.values())
            ),
            "quality_comparison": str(final_run / "quality-comparison-tuned.json"),
            "cuda_sidecar": str(cuda_sidecar),
            "commands": self.commands,
        }
        atomic_write_json(self.root / "campaign-summary.json", summary)
        self._checkpoint(**summary)


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    Campaign(args).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
