from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import evaluator
from app.common.models import (
    CLI_MODEL_KEYS,
    clear_model_overrides,
    selected_embedding_model,
    set_model_overrides,
)

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_FILE = APP_ROOT / "multirun.jsonl"

# Only embedding + chat vary per run; preprocess/rewrite/rerank/judge are constants.
MODEL_FIELDS = (
    "embedding_model",
    "chat_model",
)
# Stages each run executes, in order. Override per run with a "stages" list.
STAGES = ("chunk", "ingest", "evaluate")
# Chunking knobs forwarded to chunk() (both pipelines: plain recursive splitting).
CHUNK_FIELDS = ("chunk_size", "chunk_overlap")
# Fields consumed directly by the runner (everything else would be unexpected).
RUN_FIELDS = {
    "name",
    "pipeline",
    "stages",
    "limit",
    "k",
    "include_answers",
    *CHUNK_FIELDS,
    *MODEL_FIELDS,
}


def pipeline_modules(pipeline: str):
    """Import the chunking + ingest modules (lazily, on demand).

    Both are pipeline-agnostic now; the pipeline is passed to their functions.
    """
    from app.common import chunking, ingest

    return chunking, ingest


def resolve_stages(record: dict[str, Any]) -> list[str]:
    stages = record.get("stages", list(STAGES))
    if isinstance(stages, str):
        stages = [stages]
    invalid = [stage for stage in stages if stage not in STAGES]
    if invalid:
        raise ValueError(f"unknown stage(s) {invalid}; choose from {list(STAGES)}")
    # Preserve canonical order regardless of how they were listed.
    return [stage for stage in STAGES if stage in stages]


def run_chunking(pipeline: str, chunking: Any, record: dict[str, Any]) -> None:
    kwargs = {
        field: int(record[field]) for field in CHUNK_FIELDS if field in record
    }
    chunking.chunk(pipeline, **kwargs)


def run_ingest(pipeline: str, ingest: Any) -> None:
    ingest.ingest(pipeline, embedding_model=selected_embedding_model(pipeline))


def load_runs(path: Path) -> list[dict[str, Any]]:
    """Parse a multirun JSONL file (skipping blank and //- or #-comment lines)."""
    if not path.exists():
        raise FileNotFoundError(f"Multirun file not found: {path}")
    runs: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{lineno}: each run must be a JSON object")
        runs.append(record)
    return runs


def model_overrides_from(record: dict[str, Any]) -> dict[str, str]:
    """Collect model overrides from a run record (canonical names + CLI aliases)."""
    overrides = {field: record[field] for field in MODEL_FIELDS if record.get(field)}
    for alias, canonical in CLI_MODEL_KEYS.items():
        if record.get(alias) and canonical not in overrides:
            overrides[canonical] = record[alias]
    return overrides


def run_one(record: dict[str, Any], index: int, total: int) -> Path | None:
    pipeline = record.get("pipeline")
    if pipeline not in ("basic", "advanced"):
        raise ValueError(
            f"run {index}: 'pipeline' must be 'basic' or 'advanced', got {pipeline!r}"
        )

    unknown = set(record) - RUN_FIELDS - set(CLI_MODEL_KEYS)
    if unknown:
        print(f"[warn] run {index}: ignoring unknown field(s): {sorted(unknown)}")

    stages = resolve_stages(record)
    limit = record.get("limit")
    limit = None if limit in (None, "") else int(limit)
    k = int(record.get("k", 10))
    include_answers = bool(record.get("include_answers", True))
    overrides = model_overrides_from(record)
    name = record.get("name") or pipeline

    # Reset first so a previous run's overrides never leak into this one.
    clear_model_overrides(pipeline)
    set_model_overrides(pipeline, overrides)

    print(
        f"\n{'=' * 70}\n"
        f"Run {index}/{total}: {name}  "
        f"(pipeline={pipeline}, stages={stages}, k={k}, limit={limit}, "
        f"answers={include_answers})\n"
        f"    overrides: {overrides or 'pipeline defaults'}\n"
        f"{'=' * 70}"
    )

    chunking, ingest = pipeline_modules(pipeline)

    if "chunk" in stages:
        print(f"\n[{name}] chunking ({pipeline}) ...")
        run_chunking(pipeline, chunking, record)

    if "ingest" in stages:
        print(f"\n[{name}] ingesting ({pipeline}) ...")
        run_ingest(pipeline, ingest)

    if "evaluate" not in stages:
        print(f"\n[{name}] evaluation skipped (stages={stages}).")
        return None

    print(f"\n[{name}] evaluating ({pipeline}) ...")
    questions = evaluator.stratified_sample(evaluator.load_eval_questions(), limit)
    # evaluate_pipeline prints the retrieval summary as soon as retrieval finishes,
    # before the slow answer judging — labelled with this run's name.
    frames = evaluator.evaluate_pipeline(
        pipeline, questions, k=k, include_answers=include_answers, label=name
    )

    answers = evaluator.answer_summary(frames["answer"])
    if not answers.empty:
        print(f"\n{name} answer summary")
        print(answers.to_string(index=False))

    extra_config: dict[str, Any] = {"name": name, "stages": stages}
    for field in CHUNK_FIELDS:
        if field in record:
            extra_config[field] = record[field]
    return evaluator.save_results(
        pipeline, frames, k=k, limit=limit, extra_config=extra_config
    )


def run_all(path: Path) -> None:
    runs = load_runs(path)
    if not runs:
        print(f"No runs found in {path}.")
        return
    print(f"Loaded {len(runs)} run(s) from {path}")

    saved: list[tuple[str, str]] = []
    for index, record in enumerate(runs, start=1):
        name = record.get("name") or record.get("pipeline", f"run-{index}")
        try:
            run_dir = run_one(record, index, len(runs))
            saved.append((str(name), str(run_dir) if run_dir else "(no eval saved)"))
        except Exception as exc:  # keep going so one bad run can't abort the batch
            print(f"\n[error] run {index} ({name}) failed: {exc}")
            saved.append((str(name), f"FAILED: {exc}"))

    print(f"\n{'=' * 70}\nMultirun complete: {len(runs)} run(s)")
    for name, location in saved:
        print(f"  - {name}: {location}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a batch of evaluations described in a JSONL file, "
        "saving each into app/results/<n>/."
    )
    parser.add_argument(
        "runs_file",
        nargs="?",
        default=str(DEFAULT_RUNS_FILE),
        help=f"Path to the multirun JSONL file (default: {DEFAULT_RUNS_FILE}).",
    )
    args = parser.parse_args()
    run_all(Path(args.runs_file))


if __name__ == "__main__":
    main()
