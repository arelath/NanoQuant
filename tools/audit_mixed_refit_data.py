"""Record exact token-window separation for the fixed mixed-refit continuation."""

import argparse
import json
from pathlib import Path

import numpy as np
from benchmark_mixed_gguf_wikitext import verify_tokens
from refit_mixed_gguf import token_digest
from safetensors.torch import load_file

from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.quality_evaluation import _prepare_wikitext_windows


def exact_matches(corpus, probes):
    hits = []
    for p, probe in enumerate(probes):
        needle = np.asarray(probe)
        for r, row in enumerate(corpus):
            row = np.asarray(row)
            for start in np.flatnonzero(row == needle[0]):
                if np.array_equal(row[start:start + len(needle)], needle):
                    hits.append({"probe": p, "row": r, "start": int(start)})
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    run = Path(plan["run_output"])
    receipt = json.loads((run / "calibration-input.json").read_text())
    calibration = load_pinned_calibration(run, ArtifactRef("calibration-dataset-manifest", receipt["artifact_id"], 1))
    train = calibration.input_ids[plan["train_rows"], :plan["sequence_length"]]
    validation = calibration.input_ids[plan["validation_rows"], :plan["sequence_length"]]
    artifact = json.loads((args.teacher_root / "dataset.json").read_text())["artifact"]["artifact_id"]
    path = args.teacher_root / "artifacts" / artifact.removeprefix("sha256-")[:2] / artifact / "tokens.safetensors"
    sensitivity = load_file(path)["token_ids"]
    quality = json.loads(args.quality.read_text())
    wiki, _, _, mask = _prepare_wikitext_windows(Path(plan["snapshot"]), samples=64,
                                                sequence_length=128, local_files_only=True)
    wiki_hash = verify_tokens(wiki, mask, quality["protocol"]["wikitext_token_hash"])
    checks = {
        "validation_content_in_training": exact_matches(train.numpy(), validation[:, 1:].numpy()),
        "sensitivity_content_in_training": exact_matches(train.numpy(), sensitivity[:, 1:].numpy()),
        "sensitivity_content_in_validation": exact_matches(validation.numpy(), sensitivity[:, 1:].numpy()),
        "wikitext_content_in_training": exact_matches(train.numpy(), wiki[:, 1:].numpy()),
        "wikitext_content_in_validation": exact_matches(validation.numpy(), wiki[:, 1:].numpy()),
    }
    report = {"plan_sha256": hash_file(args.plan), "code_sha256": hash_file(Path(__file__)),
              "calibration_receipt": receipt, "train_shape": list(train.shape),
              "validation_shape": list(validation.shape), "train_token_sha256": token_digest(train),
              "validation_token_sha256": token_digest(validation),
              "sensitivity_token_sha256": token_digest(sensitivity),
              "wikitext_retained_token_hash": wiki_hash, "exact_overlap_checks": checks,
              "passed": not any(checks.values()),
              "limit": "Exact contiguous window checks; not a proof of disjoint broader source corpora."}
    atomic_write_json(args.output, report)
    if not report["passed"]:
        raise ValueError("refit training/evaluation content overlaps")
    print("All exact window separation checks passed; retained WikiText token hash matches.")


if __name__ == "__main__":
    main()
