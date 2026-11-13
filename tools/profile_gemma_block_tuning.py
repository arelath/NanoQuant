"""Profile block-zero tuning using the saved run's batches, epochs and calibration.

Loads source weights on CPU and moves one decoder block to CUDA. The retained
frozen gate factors initialize independent residual stages; remaining projections
use BF16 source weights. This is a post-fit replay, not the original tuning entry
state. Compiled mode resets weights after a full cold invocation before timing a
warm invocation. Micro mode includes synchronized per-phase profiling overhead.
"""

import argparse
import copy
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401
import torch
from safetensors.torch import save_file

from nanoquant.application.layers import LayerFreezer, TrainableFactorizedLinear, thaw_residual_stages
from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.application.tuning import (
    FactorizedTuningLearningRates,
    TuningRequest,
    tune_factorized,
    tune_non_factorized,
)
from nanoquant.config.codec import from_dict
from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.domain.models import ArtifactRef, FrozenNanoQuantState, TensorRef
from nanoquant.domain.profiling import NULL_RECORDER
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.profiling import Profiler
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.resident_quantization import (
    _forward_metadata_to_device,
    _legacy_cuda_numerics,
    _run_block_batched,
    _run_prefix_batched,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "compiled", "profile", "micro"), default="eager")
    parser.add_argument("--stage", choices=("factorized", "nonfactorized", "both"), default="both")
    args = parser.parse_args()
    run, snapshot, out = args.run_output, args.snapshot, args.output
    out.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((run / "manifest.json").read_text())["resolved_config"]["canonical_run_config"]
    tuning = recipe["block_tuning"]
    if tuning["epoch_loss_mode"] != "legacy_training" or tuning["restore_best_state"]:
        raise ValueError("This diagnostic requires legacy training loss without best-state restoration")

    with acquire_device_lease("cuda"), _legacy_cuda_numerics():
        tokens = load_pinned_calibration(
            run,
            ArtifactRef(
                "calibration-dataset-manifest",
                json.loads((run / "calibration-input.json").read_text())["artifact_id"],
                1,
            ),
        ).input_ids
        print("tokens loaded", flush=True)
        model = load_causal_language_model(
            snapshot, local_files_only=True, torch_dtype=torch.bfloat16, attention_implementation="eager",
        ).eval()
        adapter = adapter_for_config(model.config.to_dict())
        block = model.model.layers[0]
        metadata = capture_prefix_invocations(
            block, (lambda model=model: adapter.run_decoder_forward(model, tokens[:1]),)
        )[0].keyword
        inputs = _run_prefix_batched(adapter, model, tokens, 1, "cpu").detach()
        block = copy.deepcopy(block).cuda()
        metadata = _forward_metadata_to_device(metadata, "cuda")
        del model
        gc.collect()
        targets = _run_block_batched(adapter, block, inputs, metadata, 1, "cpu").detach()
        block = block.cpu()
        torch.cuda.empty_cache()
        store = LocalArtifactStore(run / "artifacts", use_persistent_validation_cache=False)
        records = [json.loads(line) for line in (run / "state/journal.jsonl").read_text().splitlines()]
        record = next(row for row in reversed(records) if row.get("kind") == "block" and row.get("block") == 0)
        store.validate(record["artifact_id"])
        result = json.loads((store.path_for(record["artifact_id"]) / "block-result.json").read_text())
        layer = next(x for x in result["layers"] if x["layer"]["path"] == "mlp.gate_proj")

        state = from_dict(FrozenNanoQuantState, layer["frozen_state"])
        frozen = LayerFreezer().load(state, LocalTensorStore(store)).module
        block.mlp.gate_proj = TrainableFactorizedLinear(
            frozen.left_binary,
            frozen.right_binary,
            frozen.scale_pre,
            frozen.scale_mid,
            frozen.scale_post,
            residual_stages=thaw_residual_stages(frozen),
        ).to(dtype=torch.bfloat16)
        del frozen
        calibration_ref = json.loads((run / "state/preprocessing.json").read_text())["calibration"]["artifact_id"]
        store.validate(calibration_ref)
        stats = json.loads((store.path_for(calibration_ref) / "stats.json").read_text())
        importance_ref = next(
            x["output_importance"]
            for x in stats["layers"]
            if x["layer"] == {"block": {"index": 0}, "path": "mlp.down_proj"}
        )
        with LocalTensorStore(store).read(from_dict(TensorRef, importance_ref), "cpu") as value:
            importance = value.clone()

        def forward(module, values):
            return adapter.run_block(module, values, **metadata)

        (out / "protocol.json").write_text(
            json.dumps(
                {
                    "run_output": str(run.resolve()),
                    "snapshot": str(snapshot.resolve()),
                    "block_artifact": record["artifact_id"],
                    "calibration_artifact": calibration_ref,
                    "input_shape": list(inputs.shape),
                    "tuning_config": tuning,
                    "seed": 0,
                    "initialization": "retained frozen gate factors with BF16 remaining block projections",
                    "torch": str(torch.__version__),
                    "gpu": torch.cuda.get_device_name(),
                },
                indent=2,
            )
        )
        if tuning["compile_forward"] and args.mode != "eager":
            forward = torch.compile(forward, options={"emulate_precision_casts": True, "triton.cudagraphs": False})
        results = []
        for kind in (args.stage,) if args.stage != "both" else ("factorized", "nonfactorized"):
            working = copy.deepcopy(block).cuda()
            recorder = (
                Profiler(
                    ProfilingConfig(
                        level=ProfilingLevel.MICRO,
                        cuda_timing=True,
                        cuda_sample_every=1,
                        raw_samples_per_phase=1024,
                        emit_span_events=False,
                    ),
                    run_id=kind,
                )
                if args.mode == "micro"
                else NULL_RECORDER
            )
            cold_seconds = None
            stage_config = tuning["factorized" if kind == "factorized" else "non_factorized"]
            epochs = (
                stage_config["loop"]["epochs"] if kind == "factorized"
                else stage_config["epochs_by_layer_position"][0]
            )
            learning_rate = (
                stage_config["learning_rates"]["scale"] if kind == "factorized"
                else stage_config["optimizer"]["learning_rate"]
            )
            request = TuningRequest(
                inputs,
                targets,
                epochs,
                stage_config["loop"]["batch_size"],
                learning_rate,
                seed=0,
                output_importance=importance,
                microbatch_size=tuning["microbatch_size"],
                restore_best_state=False,
                epoch_loss_mode="legacy_training",
            )

            def execute(kind=kind, working=working, request=request, recorder=recorder):
                if kind == "factorized":
                    return tune_factorized(
                        working,
                        "mlp.gate_proj",
                        request,
                        forward,
                        recorder,
                        learning_rates=FactorizedTuningLearningRates(
                            tuning["factorized"]["learning_rates"]["binary"],
                            tuning["factorized"]["learning_rates"]["scale"],
                            tuning["factorized"]["learning_rates"]["outlier"] or request.learning_rate,
                            tuning["factorized"]["learning_rates"]["bias"],
                        ),
                    )
                return tune_non_factorized(working, request, forward, recorder)

            if args.mode == "compiled":
                cold_start = time.perf_counter()
                execute()
                torch.cuda.synchronize()
                cold_seconds = time.perf_counter() - cold_start
                print("cold", kind, cold_seconds, flush=True)
                working.load_state_dict(block.state_dict())
            torch.cuda.synchronize()
            start = time.perf_counter()
            if args.mode == "profile":
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as profiler:
                    metrics = execute()
                profiler.export_chrome_trace(str(out / f"{kind}-trace.json"))
                (out / f"{kind}-operators.txt").write_text(
                    profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=45)
                )
            else:
                metrics = execute()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if isinstance(recorder, Profiler):
                (out / f"{kind}-phases.json").write_text(json.dumps(recorder.snapshot(), indent=2))
            print(kind, seconds, metrics, flush=True)
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in working.named_parameters()},
                out / f"{args.mode}-{kind}.safetensors",
            )
            results.append(dict(kind=kind, seconds=seconds, cold_seconds=cold_seconds, metrics=asdict(metrics)))
            (out / f"{args.mode}.json").write_text(json.dumps(results, indent=2))
            del working
            del execute
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
