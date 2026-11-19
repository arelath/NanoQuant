"""Confirm a preselected rank split with full KL/NLL in the unchanged 070 context.

Compare each freshly fitted split to its freshly fitted default, seed matched.
The unchanged distilled 070 model is an additional reference, not the matched
control. Uses a disjoint WikiText validation window, never the final test set.
Run validate_resident_run.py --require-complete before this tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_corrected_codebook_splice import _paired_payload
from probe_mlp_overlays_kl import _split_tokens
from probe_real_block_tabu import _load_cases
from probe_residual_rank_split import payload_weight
from probe_residual_stage_functional import aggregate
from safetensors.torch import load_file

from nanoquant.application.kl_budget import causal_kl_nll_per_sequence_from_logits
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash


def selected_pairs(rows, block, owner, ranks):
    selected = [r for r in rows if r["block"] == block and r["owner"] == owner]
    pairs = {}
    for seed in sorted({r["seed"] for r in selected}):
        control = next(r for r in selected if r["seed"] == seed and r["is_baseline"])
        candidate = next(r for r in selected if r["seed"] == seed and r["ranks"] == ranks)
        if candidate["bits"] > control["bits"] or candidate["is_baseline"]:
            raise ValueError("candidate must differ from control and not cost more")
        pairs[seed] = (control, candidate)
    if not pairs:
        raise ValueError("no matching completed probes")
    return pairs


def run(args):
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.probe / "summary.json").read_text())
    protocol = json.loads((args.probe / "protocol.json").read_text())
    if source["status"] != "complete_local_probe":
        raise ValueError("local probe must be complete before confirmation selection")
    run_output = Path(protocol["source_run"])
    _, cases = _load_cases(run_output, args.snapshot / "model.safetensors", (args.block,), (args.owner,), 26)
    case = cases[0]
    pairs = selected_pairs(source["rows"], args.block, args.owner, args.ranks)
    length = protocol["sequence_length"]
    screen_start = protocol["tokens"]["validation_offset"]
    screen_end = screen_start + protocol["fit_samples"]
    if args.offset < screen_end and args.offset + args.samples > screen_start:
        raise ValueError("confirmation overlaps split-selection window")
    tokens, fingerprint, _ = _split_tokens(args.snapshot, split="validation",
        samples=args.offset + args.samples, sequence_length=length, local_files_only=True)
    tokens = tokens[args.offset:]
    receipt = dict(probe_summary_sha256=hash_file(args.probe / "summary.json"),
                   tool_sha256=hash_file(Path(__file__)), pairs=pairs, block=args.block, owner=args.owner,
                   validation_offset=args.offset, samples=args.samples, sequence_length=length,
                   token_hash=_token_hash(tokens), fingerprint=fingerprint,
                   scope="single-owner dense splice in globally tuned 070 context, not native deployment",
                   hypothesis="candidate minus fresh default split; lower KL and non-regressing NLL")
    atomic_write_json(args.output / "protocol.json", receipt)
    with acquire_device_lease(args.device):
        loaded = load_frozen_run(run_output, args.snapshot, source_name="google/gemma-3-1b-it",
            revision="dcc83ea841ab6100d6b47a070329e1ba4cf78752", device=args.device,
            verify_hashes=True, backend="factorized", use_global_tuning=True)
        receipt["frozen_identity"] = to_dict(loaded.identity)
        receipt["global_tuning"] = to_dict(loaded.global_tuning)
        atomic_write_json(args.output / "protocol.json", receipt)
        student = loaded.model.eval()
        block = student.model.layers[args.block]
        originals = {member.path: block.get_submodule(member.path) for member in case.members}
        arms = {"unchanged070": {}}
        for seed, pair in pairs.items():
            for label, row in zip(("control", "candidate"), pair, strict=True):
                path = args.probe / row["payload"]
                if hash_file(path) != row["sha256"]:
                    raise ValueError("probe payload hash mismatch")
                weight = payload_weight(load_file(str(path)), "cpu")
                replacements = {}
                for member, part in zip(case.members, weight.split(case.member_rows), strict=True):
                    module = torch.nn.Linear(part.shape[1], part.shape[0], bias=False,
                                             device=args.device, dtype=torch.bfloat16)
                    with torch.no_grad():
                        module.weight.copy_(part.to(args.device))
                    replacements[member.path] = module
                arms[f"seed{seed}_{label}"] = replacements
        config = json.loads((args.snapshot / "config.json").read_text())
        teacher = load_causal_language_model(args.snapshot, torch_dtype=torch.bfloat16, local_files_only=True,
            attention_implementation=adapter_for_config(config).attention_implementation).to(args.device).eval()
        rows = []
        with torch.no_grad():
            for index, tokens_row in enumerate(tokens):
                batch = tokens_row[None].to(args.device)
                teacher_logits = teacher(input_ids=batch, use_cache=False).logits
                row = {}
                for name, replacements in arms.items():
                    try:
                        for path, module in replacements.items():
                            parent, child = path.rsplit(".", 1)
                            setattr(block.get_submodule(parent), child, module)
                        logits = student(input_ids=batch, use_cache=False).logits
                        row[name] = to_dict(causal_kl_nll_per_sequence_from_logits(
                            teacher_logits, logits, batch, token_chunk_size=32)[0])
                    finally:
                        for path, module in originals.items():
                            parent, child = path.rsplit(".", 1)
                            setattr(block.get_submodule(parent), child, module)
                rows.append(row)
                atomic_write_json(args.output / "progress.json", dict(rows=rows))
                print(json.dumps(dict(completed_sequence=index + 1, total=len(tokens))), flush=True)
        results = {name: aggregate(name, rows) for name in arms}
        comparisons = {}
        for seed in pairs:
            before, after = results[f"seed{seed}_control"], results[f"seed{seed}_candidate"]
            comparisons[str(seed)] = dict(kl=_paired_payload(before, after, seed=0),
                                         nll=_paired_metric_payload(before, after, "negative_log_likelihood"))
    atomic_write_json(args.output / "summary.json", dict(status="completed",
        results={k: to_dict(v) for k, v in results.items()}, paired_candidate_minus_control=comparisons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=25)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--offset", type=int, default=128)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
