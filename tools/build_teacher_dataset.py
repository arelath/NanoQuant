"""Build or resume a reusable teacher-response dataset.

Run without arguments for the interactive menu, or pass explicit arguments for
an automation-friendly immutable run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nanoquant.config.schema import (
    LLAMACPP_TEACHER_TRACE_IMPLEMENTATION,
    ReasoningMode,
)
from nanoquant.infrastructure.environment import load_repository_dotenv
from nanoquant.teacher_dataset import (
    DEFAULT_SAMPLES_PER_MODE,
    TEACHER_DATASET_SETTINGS_NAME,
    ULTRACHAT_DATASET,
    ULTRACHAT_REVISION,
    ULTRACHAT_SPLIT,
    TeacherDatasetGeneration,
    TeacherDatasetUpload,
    TeacherModel,
    TeacherPromptSource,
    execute_teacher_dataset,
    new_teacher_dataset_settings,
    resolve_dataset_revision,
    resolve_gguf_filename,
    resolve_model_revision,
    run_interactive_teacher_dataset,
    write_teacher_dataset_settings,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="show the menu explicitly (the default when no arguments are supplied)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume from a run directory or its settings.yaml",
    )
    parser.add_argument("--output", type=Path, help="new run directory")
    parser.add_argument("--teacher-model", help="Hugging Face teacher model ID")
    parser.add_argument("--teacher-revision", help="teacher commit; resolved when omitted")
    parser.add_argument(
        "--teacher-tokenizer",
        help="matching tokenizer repository; defaults to the teacher without a -GGUF suffix",
    )
    parser.add_argument(
        "--teacher-tokenizer-revision",
        help="tokenizer commit; resolved when omitted",
    )
    parser.add_argument(
        "--teacher-gguf-file",
        help="UD-Q8_K_XL GGUF entrypoint; auto-detected for a -GGUF teacher repository",
    )
    parser.add_argument(
        "--source-dataset",
        default=ULTRACHAT_DATASET,
        help="Hugging Face conversational prompt dataset",
    )
    parser.add_argument(
        "--source-revision",
        help="prompt dataset commit; the pinned UltraChat commit is the default",
    )
    parser.add_argument("--source-config", help="optional prompt dataset configuration")
    parser.add_argument(
        "--source-split",
        default=ULTRACHAT_SPLIT,
        help="prompt dataset split",
    )
    parser.add_argument(
        "--messages-column",
        default="messages",
        help="column containing role/content or from/value messages",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "thinking", "non-thinking"),
        default="both",
    )
    parser.add_argument(
        "--samples-per-mode",
        type=int,
        default=DEFAULT_SAMPLES_PER_MODE,
        help="accepted records in each requested mode",
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--maximum-new-tokens", type=int, default=1536)
    parser.add_argument("--minimum-new-tokens", type=int, default=16)
    parser.add_argument("--maximum-attempt-multiplier", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("llamacpp", "transformers"),
        default="llamacpp",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--hub-repo",
        help="upload to this Hugging Face dataset repository after local completion",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="make a new Hugging Face dataset repository public (private is the default)",
    )
    parser.add_argument(
        "--commit-message",
        default="Publish generated teacher-response dataset",
    )
    return parser


def _modes(value: str) -> tuple[ReasoningMode, ...]:
    if value == "both":
        return ReasoningMode.THINKING, ReasoningMode.NON_THINKING
    if value == "thinking":
        return (ReasoningMode.THINKING,)
    return (ReasoningMode.NON_THINKING,)


def _resume_path(value: Path) -> Path:
    path = value.resolve()
    if path.is_dir():
        path = path / TEACHER_DATASET_SETTINGS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"teacher dataset settings are missing: {path}")
    return path


def _create_settings(args: argparse.Namespace) -> Path:
    if args.output is None:
        raise ValueError("--output is required for a new non-interactive run")
    if not args.teacher_model:
        raise ValueError("--teacher-model is required for a new non-interactive run")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"teacher dataset output already exists: {output}; use --resume to continue it"
        )
    source_revision = args.source_revision
    if source_revision is None and args.source_dataset == ULTRACHAT_DATASET:
        source_revision = ULTRACHAT_REVISION
    source_revision = resolve_dataset_revision(args.source_dataset, source_revision)
    teacher_revision = resolve_model_revision(args.teacher_model, args.teacher_revision)
    default_tokenizer = (
        args.teacher_model[:-5]
        if args.teacher_model.lower().endswith("-gguf")
        else args.teacher_model
    )
    tokenizer_source = args.teacher_tokenizer or default_tokenizer
    tokenizer_revision = resolve_model_revision(
        tokenizer_source,
        args.teacher_tokenizer_revision,
    )
    gguf_filename = resolve_gguf_filename(
        args.teacher_model,
        teacher_revision,
        args.teacher_gguf_file,
    )
    implementation = (
        LLAMACPP_TEACHER_TRACE_IMPLEMENTATION
        if args.backend == "llamacpp"
        else "hf-greedy-qwen3-v1"
    )
    upload = (
        None
        if args.hub_repo is None
        else TeacherDatasetUpload(
            args.hub_repo,
            private=not args.public,
            commit_message=args.commit_message,
        )
    )
    settings = new_teacher_dataset_settings(
        prompt_source=TeacherPromptSource(
            args.source_dataset,
            source_revision,
            args.source_split,
            args.source_config,
            args.messages_column,
        ),
        teacher=TeacherModel(
            args.teacher_model,
            teacher_revision,
            tokenizer_source,
            tokenizer_revision,
            gguf_filename,
            implementation,
            args.device,
        ),
        generation=TeacherDatasetGeneration(
            modes=_modes(args.mode),
            samples_per_mode=args.samples_per_mode,
            sequence_length=args.sequence_length,
            maximum_new_tokens=args.maximum_new_tokens,
            minimum_new_tokens=args.minimum_new_tokens,
            maximum_attempt_multiplier=args.maximum_attempt_multiplier,
            seed=args.seed,
        ),
        upload=upload,
    )
    output.mkdir(parents=True, exist_ok=False)
    settings_path = output / TEACHER_DATASET_SETTINGS_NAME
    write_teacher_dataset_settings(settings_path, settings)
    print(f"Settings written: {settings_path}", flush=True)
    return settings_path


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = list(argv) if argv is not None else None
    args = parser.parse_args(arguments)
    repository_root = Path(__file__).resolve().parent.parent
    load_repository_dotenv(repository_root)
    supplied = arguments if arguments is not None else sys.argv[1:]
    try:
        if args.interactive or not supplied:
            catalog = repository_root / "tools" / "teacher_dataset_models.yaml"
            return run_interactive_teacher_dataset(repository_root, catalog)
        if args.resume is not None:
            explicit_options = {
                token.split("=", 1)[0]
                for token in supplied
                if token.startswith("--")
            }
            if explicit_options != {"--resume"}:
                raise ValueError("--resume cannot be combined with any new-run settings")
            settings_path = _resume_path(args.resume)
        else:
            settings_path = _create_settings(args)
        return execute_teacher_dataset(settings_path)
    except KeyboardInterrupt:
        print(
            "\nTeacher dataset generation interrupted. Run with --resume or use the menu to continue.",
            flush=True,
        )
        return 130
    except BaseException as exc:
        print(f"Teacher dataset generation failed: {type(exc).__name__}: {exc}", flush=True)
        print("Persisted settings and completed response journals remain resumable.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
