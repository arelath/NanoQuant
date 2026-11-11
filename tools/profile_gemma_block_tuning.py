"""Profile Gemma block zero with a retained Experiment 070 gate projection.

The block's other projections start from BF16 weights. The retained gate's frozen
binary factors and independent residual stages initialize the tuning module.
Both stages run independently from the same state with 256x2048 calibration
tokens, eight epochs, batch size eight, seed zero and the recorded output weights.
Compiled mode measures the first invocation, resets all model state, and measures
a warmed invocation with a fresh optimizer. Profile mode includes trace overhead.
The initialization is a post-fit replay of the first projection in the block.
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
from transformers import AutoModelForCausalLM

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
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.profiling import Profiler
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.resident_quantization import _legacy_cuda_numerics, _run_block_batched, _run_prefix_batched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "compiled", "profile", "micro"), default="eager")
    args = parser.parse_args()
    run, snapshot, out = args.run_output, args.snapshot, args.output
    out.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((run / "manifest.json").read_text())["resolved_config"]["canonical_run_config"]
    tuning = recipe["block_tuning"]
    if (
        tuning["factorized"]["loop"]["epochs"] != 8
        or tuning["factorized"]["loop"]["batch_size"] != 8
        or tuning["microbatch_size"] != 8
        or tuning["epoch_loss_mode"] != "legacy_training"
        or tuning["restore_best_state"]
    ):
        raise ValueError("This diagnostic implements the Experiment 070 eight-epoch tuning protocol")

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
        model = (
            AutoModelForCausalLM.from_pretrained(
                snapshot, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
        )
        adapter = adapter_for_config(model.config.to_dict())
        block = model.model.layers[0]
        metadata = capture_prefix_invocations(
            block, (lambda model=model: model.model(input_ids=tokens[:1].cuda(), use_cache=False),)
        )[0].keyword
        inputs = _run_prefix_batched(adapter, model, tokens.cuda(), 8, "cpu").detach()
        targets = _run_block_batched(adapter, block, inputs, metadata, 8, "cpu").detach()
        block = copy.deepcopy(block).cpu()
        del model
        gc.collect()
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
                    "batch_size": 8,
                    "epochs": 8,
                    "seed": 0,
                    "initialization": "retained frozen gate factors with BF16 remaining block projections",
                    "torch": str(torch.__version__),
                    "gpu": torch.cuda.get_device_name(),
                },
                indent=2,
            )
        )
        if args.mode == "compiled":
            forward = torch.compile(forward, options={"emulate_precision_casts": True, "triton.cudagraphs": False})
        results = []
        for kind in ("factorized", "nonfactorized"):
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
            request = TuningRequest(
                inputs,
                targets,
                8,
                8,
                3e-5 if kind == "factorized" else 1e-4,
                seed=0,
                output_importance=importance,
                microbatch_size=8,
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
                        learning_rates=FactorizedTuningLearningRates(3e-5, 1e-5, 1e-5, 1e-5),
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
