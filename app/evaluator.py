from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from tqdm import tqdm

from app.advanced import implementation as advanced_implementation
from app.basic import implementation as basic_implementation
from app.common.chat import chat_structured
from app.common.models import (
    JUDGE_MODEL,
    selected_pipeline_models,
    set_model_overrides,
    split_model_overrides,
)
from app.common.paths import DATA_DIR

APP_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = APP_ROOT / "results"
RETRIEVAL_SUMMARY_METRICS = ["mrr", "ndcg", "keyword_coverage"]
ANSWER_SUMMARY_METRICS = [
    "accuracy",
    "completeness",
    "relevance",
    "answer_keyword_coverage",
    "overall",
]
EVAL_QUESTION_PATHS = (
    DATA_DIR / "eval-questions.json",
    DATA_DIR / "eval_questions.json",
    DATA_DIR / "original_eval_questions.json",
)
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass
class EvalQuestion:
    id: str
    difficulty: str
    category: str
    question: str
    keywords: list[str]
    reference_answer: str
    expected_patient_ids: list[str]
    expected_source_tables: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalQuestion":
        return cls(
            id=str(data["id"]),
            difficulty=str(data["difficulty"]),
            category=str(data["category"]),
            question=str(data["question"]),
            keywords=[str(keyword) for keyword in data.get("keywords", [])],
            reference_answer=str(data["reference_answer"]),
            expected_patient_ids=[
                str(pid) for pid in data.get("expected_patient_ids", [])
            ],
            expected_source_tables=[
                str(table) for table in data.get("expected_source_tables", [])
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "difficulty": self.difficulty,
            "category": self.category,
            "question": self.question,
            "keywords": self.keywords,
            "reference_answer": self.reference_answer,
            "expected_patient_ids": self.expected_patient_ids,
            "expected_source_tables": self.expected_source_tables,
        }


class AnswerEval(BaseModel):
    feedback: str = Field(description="Concise feedback comparing answer to reference")
    accuracy: float = Field(description="Factual correctness from 1 to 5")
    completeness: float = Field(description="Completeness from 1 to 5")
    relevance: float = Field(description="Direct relevance from 1 to 5")


def eval_questions_path() -> Path:
    for path in EVAL_QUESTION_PATHS:
        if path.exists():
            return path
    candidates = ", ".join(str(path) for path in EVAL_QUESTION_PATHS)
    raise FileNotFoundError(f"No evaluation question file found. Checked: {candidates}")


def load_eval_questions(path: Path | None = None) -> list[EvalQuestion]:
    payload = json.loads((path or eval_questions_path()).read_text(encoding="utf-8"))
    questions = payload["questions"] if isinstance(payload, dict) else payload
    return [EvalQuestion.from_dict(item) for item in questions]


def keyword_reciprocal_rank(keyword: str, docs: list[Document]) -> float:
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def keyword_ndcg(keyword: str, docs: list[Document], k: int) -> float:
    keyword_lower = keyword.lower()
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in docs[:k]
    ]
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal)
    return dcg(relevances) / ideal_dcg if ideal_dcg else 0.0


def deterministic_answer_coverage(
    question: EvalQuestion, generated_answer: str
) -> float:
    answer_lower = generated_answer.lower()
    hits = [keyword for keyword in question.keywords if keyword.lower() in answer_lower]
    return len(hits) / len(question.keywords) if question.keywords else 0.0


def judge_answer(
    question: EvalQuestion,
    generated_answer: str,
    *,
    judge_model: str,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator assessing answer quality. Compare the generated "
                "answer to the reference answer. Only give 5/5 scores for perfect answers."
            ),
        },
        {
            "role": "user",
            "content": f"""Question:
            {question.question}

            Generated Answer:
            {generated_answer}

            Reference Answer:
            {question.reference_answer}

            Evaluate the generated answer on:
            1. Accuracy: factual correctness vs reference answer.
            2. Completeness: coverage of all reference information.
            3. Relevance: how directly it answers without extra information.

            Provide concise feedback and scores from 1 to 5. If the answer is wrong, accuracy must be 1.""",
        },
    ]
    try:
        result = chat_structured(messages, AnswerEval, model=judge_model)
        return {
            "judge_feedback": result.feedback,
            "accuracy": float(result.accuracy),
            "completeness": float(result.completeness),
            "relevance": float(result.relevance),
            "judge_error": "",
        }
    except Exception as exc:
        return {
            "judge_feedback": "",
            "accuracy": np.nan,
            "completeness": np.nan,
            "relevance": np.nan,
            "judge_error": str(exc),
        }


def evaluate_retrieval_question(
    question: EvalQuestion,
    implementation: Any,
    *,
    k: int = 10,
) -> dict[str, Any]:
    docs = implementation.fetch_context(question.question, k=k)
    mrr_scores = [
        keyword_reciprocal_rank(keyword, docs) for keyword in question.keywords
    ]
    ndcg_scores = [keyword_ndcg(keyword, docs, k) for keyword in question.keywords]
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(question.keywords)
    return {
        **question.as_dict(),
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "ndcg": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "keywords_found": keywords_found,
        "total_keywords": total_keywords,
        "keyword_coverage": keywords_found / total_keywords if total_keywords else 0.0,
        "retrieved_doc_types": [doc.metadata.get("doc_type", "") for doc in docs],
        "retrieved_patient_ids": [doc.metadata.get("patient_id", "") for doc in docs],
    }


def evaluate_answer_question(
    question: EvalQuestion,
    implementation: Any,
    *,
    judge_model: str,
) -> dict[str, Any]:
    generated_answer, docs = implementation.answer_question(question.question)
    return {
        **question.as_dict(),
        "generated_answer": generated_answer,
        "answer_keyword_coverage": deterministic_answer_coverage(
            question, generated_answer
        ),
        "retrieved_sources": [doc.metadata.get("source_tables", "") for doc in docs],
        **judge_answer(question, generated_answer, judge_model=judge_model),
    }


def stratified_sample(
    eval_questions: list[EvalQuestion], limit: int | None
) -> list[EvalQuestion]:
    if limit is None or limit >= len(eval_questions):
        return eval_questions
    if limit <= 0:
        return []
    buckets: dict[str, list[EvalQuestion]] = {
        difficulty: [] for difficulty in DIFFICULTIES
    }
    for question in eval_questions:
        buckets.setdefault(question.difficulty, []).append(question)

    selected: list[EvalQuestion] = []
    cursors = {difficulty: 0 for difficulty in buckets}
    while len(selected) < limit:
        progressed = False
        for difficulty, bucket in buckets.items():
            if len(selected) >= limit:
                break
            cursor = cursors[difficulty]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                cursors[difficulty] = cursor + 1
                progressed = True
        if not progressed:
            break
    return selected


def implementation_for(pipeline: str) -> Any:
    if pipeline == "basic":
        return basic_implementation
    if pipeline == "advanced":
        return advanced_implementation
    raise ValueError("pipeline must be one of: basic, advanced")


def summarize(
    results: pd.DataFrame,
    metrics: list[str],
    *,
    group_by: str = "difficulty",
    order: tuple[str, ...] = DIFFICULTIES,
) -> pd.DataFrame:
    """Mean each metric within each ``group_by`` value (difficulty or category).

    ``order`` lists known group values to sort first; any unlisted values
    (e.g. arbitrary categories) sort alphabetically after them.
    """
    if results.empty or group_by not in results:
        return pd.DataFrame()
    aggregations = {metric: (metric, "mean") for metric in metrics if metric in results}
    summary = results.groupby(group_by, as_index=False).agg(
        questions=("id", "count"),
        **aggregations,
    )
    rank = {value: index for index, value in enumerate(order)}
    summary["_rank"] = summary[group_by].map(lambda value: rank.get(value, len(rank)))
    summary = summary.sort_values(["_rank", group_by]).drop(columns="_rank")
    return summary.reset_index(drop=True)


def print_retrieval_summary(label: str, retrieval: pd.DataFrame) -> None:
    """Print the retrieval (MRR/nDCG/coverage) summary per difficulty and category."""
    print(f"\n{label} retrieval summary (by difficulty)", flush=True)
    print(
        summarize(retrieval, RETRIEVAL_SUMMARY_METRICS).to_string(index=False),
        flush=True,
    )
    by_category = summarize(
        retrieval, RETRIEVAL_SUMMARY_METRICS, group_by="category", order=()
    )
    if not by_category.empty:
        print(f"\n{label} retrieval summary (by category)", flush=True)
        print(by_category.to_string(index=False), flush=True)


def evaluate_pipeline(
    pipeline: str,
    questions: list[EvalQuestion],
    *,
    k: int = 10,
    include_answers: bool = True,
    label: str | None = None,
) -> dict[str, pd.DataFrame]:
    implementation = implementation_for(pipeline)
    retrieval_rows = [
        evaluate_retrieval_question(question, implementation, k=k)
        for question in tqdm(questions, desc=f"{pipeline} retrieval evals")
    ]
    retrieval = pd.DataFrame(retrieval_rows)

    # Show retrieval metrics immediately, before the slow answer judging runs.
    print_retrieval_summary(label or pipeline, retrieval)

    answer = pd.DataFrame()
    if include_answers:
        judge_model = JUDGE_MODEL
        answer_rows = [
            evaluate_answer_question(
                question,
                implementation,
                judge_model=judge_model,
            )
            for question in tqdm(questions, desc=f"{pipeline} answer evals")
        ]
        answer = pd.DataFrame(answer_rows)
        _warn_on_judge_errors(pipeline, judge_model, answer)

    return {"retrieval": retrieval, "answer": answer}


def _warn_on_judge_errors(
    pipeline: str, judge_model: str, answer: pd.DataFrame
) -> None:
    """Surface judge failures so NaN answer scores are diagnosable, not silent."""
    if "judge_error" not in answer:
        return
    failed = answer[answer["judge_error"].astype(bool)]
    if failed.empty:
        return
    sample = str(failed["judge_error"].iloc[0])[:300]
    print(
        f"\n[warn] {pipeline}: judge model {judge_model!r} failed on "
        f"{len(failed)}/{len(answer)} question(s); their answer scores are NaN. "
        f"First error: {sample}"
    )


def evaluate(
    pipeline: str = "both",
    limit: int | None = None,
    k: int = 10,
    include_answers: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    questions = stratified_sample(load_eval_questions(), limit)
    pipelines = ["basic", "advanced"] if pipeline == "both" else [pipeline]
    return {
        name: evaluate_pipeline(name, questions, k=k, include_answers=include_answers)
        for name in pipelines
    }


def answer_summary(
    answer: pd.DataFrame,
    *,
    group_by: str = "difficulty",
    order: tuple[str, ...] = DIFFICULTIES,
) -> pd.DataFrame:
    """Answer summary (grouped by difficulty or category) with an ``overall`` mean."""
    if answer.empty:
        return pd.DataFrame()
    scored = answer.copy()
    scored["overall"] = scored[["accuracy", "completeness", "relevance"]].mean(axis=1)
    return summarize(scored, ANSWER_SUMMARY_METRICS, group_by=group_by, order=order)


def _json_number(column: str, value: Any) -> float | int | None:
    """Coerce a summary cell to a JSON-safe number (NaN -> None for valid JSON)."""
    if column == "questions":
        return int(value)
    number = float(value)
    return None if math.isnan(number) else round(number, 6)


def _summary_records(
    summary: pd.DataFrame, *, key: str = "difficulty"
) -> dict[str, dict[str, Any]]:
    """Convert a grouped summary frame into a JSON-friendly nested dict keyed by ``key``."""
    if summary.empty:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in summary.to_dict(orient="records"):
        group = row.pop(key)
        records[group] = {
            column: _json_number(column, value) for column, value in row.items()
        }
    return records


def _next_run_dir(results_root: Path) -> Path:
    """Return ``results_root/<n>`` for the next unused integer ``n`` (1-based)."""
    results_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name)
        for path in results_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    return results_root / str(max(existing, default=0) + 1)


def save_results(
    pipeline: str,
    frames: dict[str, pd.DataFrame],
    *,
    k: int,
    limit: int | None,
    results_root: Path = RESULTS_ROOT,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    """Persist one evaluation run into ``app/results/<n>`` (config.json + evals.json).

    Mirrors the root ``results/`` layout: a numbered run folder holding the resolved
    model config and per-difficulty retrieval/answer summaries. ``extra_config`` is
    merged into config.json (e.g. a run name from the multirun runner).
    """
    retrieval = frames["retrieval"]
    answer = frames["answer"]

    run_dir = _next_run_dir(Path(results_root))
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        **(extra_config or {}),
        "pipeline": pipeline,
        **selected_pipeline_models(pipeline),
        "k": k,
        "limit": limit,
        "questions": int(retrieval.shape[0]),
        "db_dir": f"vector_db/{pipeline}",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    evals = {
        "retrieval": _summary_records(summarize(retrieval, RETRIEVAL_SUMMARY_METRICS)),
        "answer": _summary_records(answer_summary(answer)),
        "retrieval_by_category": _summary_records(
            summarize(
                retrieval, RETRIEVAL_SUMMARY_METRICS, group_by="category", order=()
            ),
            key="category",
        ),
        "answer_by_category": _summary_records(
            answer_summary(answer, group_by="category", order=()),
            key="category",
        ),
    }
    (run_dir / "evals.json").write_text(json.dumps(evals, indent=2))

    print(f"\nSaved {pipeline} results to {run_dir}/ (config.json, evals.json)")
    return run_dir


def main() -> None:
    overrides, cli_args = split_model_overrides(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="Evaluate app RAG pipelines. Override models with "
        "key=value args, e.g. chat_model='deepseek-r1:1.5b' judge_model='qwen3:4b'."
    )
    parser.add_argument(
        "pipeline",
        choices=["basic", "advanced"],
        help="Pipeline to evaluate.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--skip-answers", action="store_true")
    args = parser.parse_args(cli_args)

    set_model_overrides(args.pipeline, overrides)

    results = evaluate(
        pipeline=args.pipeline,
        limit=args.limit,
        k=args.k,
        include_answers=not args.skip_answers,
    )
    for pipeline_name, frames in results.items():
        # Retrieval summary was already printed by evaluate_pipeline (before the
        # answer judging). Here we just add the answer summary and persist.
        answers = answer_summary(frames["answer"])
        if not answers.empty:
            print(f"\n{pipeline_name.upper()} answer summary (by difficulty)")
            print(answers.to_string(index=False))
            by_category = answer_summary(
                frames["answer"], group_by="category", order=()
            )
            if not by_category.empty:
                print(f"\n{pipeline_name.upper()} answer summary (by category)")
                print(by_category.to_string(index=False))

        save_results(pipeline_name, frames, k=args.k, limit=args.limit)


if __name__ == "__main__":
    main()
