"""Build a self-contained ``rag-benchmark.html`` dashboard from evaluation results.

Scans every ``<root>/results/<n>/`` folder (``run-*/results`` at the repo root, plus
``app/results``) for the ``config.json`` + ``evals.json`` a run produces, aggregates the
per-difficulty scores into overall metrics, and embeds everything into a single HTML
file. The page needs no server and no network (data is inlined) — open it directly.

Re-run after any new evaluation to refresh the dashboard::

    uv run python -m app.build_benchmark
    uv run python -m app.build_benchmark --output rag-benchmark.html
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RETRIEVAL_METRICS = ("mrr", "ndcg", "keyword_coverage")
ANSWER_METRICS = (
    "accuracy",
    "completeness",
    "relevance",
    "answer_keyword_coverage",
    "overall",
)


# Folders that may contain run-set subdirectories (each holding a results/ tree).
_SET_CONTAINERS = ("all-results", "all-runs")


def _run_set_dirs() -> list[Path]:
    """Each "run set" folder: holds a ``results/<n>/`` tree plus a top-level ``config.json``.

    A run set is any directory with a ``results/`` subfolder. We look directly under the
    repo root (legacy ``run-*``), inside grouping folders (``all-results/``, ``all-runs/``),
    and at ``app`` (the evaluator's live output dir) — so curated archives and fresh runs
    both show, wherever the user keeps them.
    """
    parents: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if (path / "results").is_dir() and resolved not in seen:
            seen.add(resolved)
            parents.append(path)

    containers = [PROJECT_ROOT, *(PROJECT_ROOT / name for name in _SET_CONTAINERS)]
    for container in containers:
        if container.is_dir():
            for child in sorted(container.iterdir()):
                if child.is_dir():
                    add(child)
    add(PROJECT_ROOT / "app")
    return parents


def _run_set_constants(parent: Path) -> dict[str, str] | None:
    """The shared/constant models for a run set (from its top-level ``config.json``)."""
    config_path = parent / "config.json"
    if not config_path.exists():
        return None
    data = json.loads(config_path.read_text(encoding="utf-8"))
    constants = data.get("constant_models")
    return {str(k): str(v) for k, v in constants.items()} if isinstance(constants, dict) else None


def _weighted_overall(buckets: dict[str, dict[str, Any]], metrics) -> dict[str, float]:
    """Collapse per-difficulty (or per-category) buckets into question-weighted means."""
    overall: dict[str, float] = {}
    for metric in metrics:
        weighted_sum = 0.0
        weight = 0
        for bucket in buckets.values():
            value = bucket.get(metric)
            count = bucket.get("questions") or 0
            if value is None or count == 0:
                continue
            weighted_sum += float(value) * count
            weight += count
        if weight:
            overall[metric] = round(weighted_sum / weight, 6)
    return overall


def _load_run(source: str, run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "config.json"
    evals_path = run_dir / "evals.json"
    if not (config_path.exists() and evals_path.exists()):
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    evals = json.loads(evals_path.read_text(encoding="utf-8"))

    retrieval = evals.get("retrieval", {})
    answer = evals.get("answer", {})

    def rel(path: Path) -> str:
        return path.relative_to(PROJECT_ROOT).as_posix()

    viz = {
        "d2": rel(run_dir / "visualise_2d.html")
        if (run_dir / "visualise_2d.html").exists()
        else None,
        "d3": rel(run_dir / "visualise_3d.html")
        if (run_dir / "visualise_3d.html").exists()
        else None,
    }

    return {
        "id": f"{source}/{run_dir.name}",
        "source": source,
        "name": str(config.get("name", run_dir.name)),
        "pipeline": config.get("pipeline", "unknown"),
        "embedding_model": config.get("embedding_model", "—"),
        "chat_model": config.get("chat_model", "—"),
        "judge_model": config.get("judge_model", "—"),
        "rerank_model": config.get("rerank_model", "—"),
        "preprocess_model": config.get("preprocess_model", "—"),
        "rewrite_model": config.get("rewrite_model", "—"),
        "k": config.get("k"),
        "questions": config.get("questions"),
        "limit": config.get("limit"),
        "retrieval": {
            "overall": _weighted_overall(retrieval, RETRIEVAL_METRICS),
            "byDifficulty": retrieval,
            "byCategory": evals.get("retrieval_by_category", {}),
        },
        "answer": {
            "overall": _weighted_overall(answer, ANSWER_METRICS),
            "byDifficulty": answer,
            "byCategory": evals.get("answer_by_category", {}),
        },
        "viz": viz,
    }


def collect(min_questions: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (runs, run_sets); skip smoke tests below ``min_questions`` (logged, never silent)."""
    runs: list[dict[str, Any]] = []
    run_sets: list[dict[str, Any]] = []
    skipped: list[str] = []
    for parent in _run_set_dirs():
        source = parent.name  # e.g. "run-1" or "app"
        results_root = parent / "results"
        set_count = 0
        for run_dir in sorted(
            results_root.iterdir(),
            key=lambda path: (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name),
        ):
            if not run_dir.is_dir():
                continue
            run = _load_run(source, run_dir)
            if run is None:
                continue
            questions = run.get("questions")
            if isinstance(questions, int) and questions < min_questions:
                skipped.append(f"{run['id']} ({questions}q)")
                continue
            runs.append(run)
            set_count += 1
        if set_count:
            run_sets.append(
                {"source": source, "count": set_count, "constants": _run_set_constants(parent)}
            )
    if skipped:
        print(
            f"[build-benchmark] skipped {len(skipped)} run(s) below "
            f"--min-questions={min_questions}: {', '.join(skipped)}"
        )
    return runs, run_sets


def build_payload(min_questions: int = 0) -> dict[str, Any]:
    runs, run_sets = collect(min_questions)
    return {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "runs": runs,
        "runSets": run_sets,
        "pipelines": sorted({run["pipeline"] for run in runs}),
        "embeddings": sorted({run["embedding_model"] for run in runs}),
        "chats": sorted({run["chat_model"] for run in runs}),
    }


def render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, indent=None, ensure_ascii=False)
    return _TEMPLATE.replace("/*__DATA__*/null", data_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG benchmark dashboard HTML.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "rag-benchmark.html"),
        help="Where to write the dashboard (default: rag-benchmark.html at repo root).",
    )
    parser.add_argument(
        "--min-questions",
        type=int,
        default=5,
        help="Skip runs evaluated on fewer than this many questions (smoke tests). Default: 5.",
    )
    args = parser.parse_args()

    payload = build_payload(args.min_questions)
    if not payload["runs"]:
        print(
            "[build-benchmark] no runs found. Looked under run-*/, all-results/*, "
            "all-runs/*, and app/results."
        )
    html = render_html(payload)
    output = Path(args.output)
    output.write_text(html, encoding="utf-8")
    print(
        f"[build-benchmark] wrote {output} "
        f"({len(payload['runs'])} run(s), {len(html):,} bytes)"
    )


# The dashboard template. Data is injected by replacing the `/*__DATA__*/null` token.
# Pure HTML/CSS/JS, no external assets — works offline from file://.
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="dark" />
<title>RAGnosis · Benchmark</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg: #0e1116;
    --bg-soft: #161b22;
    --panel: #1b222c;
    --panel-2: #212a36;
    --border: #2a3441;
    --text: #e6edf3;
    --muted: #9aa6b6;
    --faint: #717d8d;
    --accent: #4dd0c4;
    --accent-2: #7c9cff;
    --basic: #7c9cff;
    --advanced: #4dd0c4;
    --good: #38b48b;
    --mid: #d8b24a;
    --bad: #e08585;
    --shadow: 0 1px 0 rgba(255,255,255,.03), 0 8px 28px rgba(0,0,0,.45);
    --radius: 14px;
    --sans: "Fira Sans", ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --mono: "Fira Code", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
    --focus: 0 0 0 2px var(--bg), 0 0 0 4px var(--accent);
    --t-fast: 130ms;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  /* Crisp tabular figures for all data; mono ligatures off so digits read cleanly. */
  .num, .heat, .kpi .value, .cv, .mval, .bk-row .lab, .kv .v, .run-id,
  thead th, td { font-feature-settings: "tnum" 1, "calt" 0; }
  body {
    background:
      radial-gradient(1200px 600px at 80% -10%, rgba(124,156,255,.08), transparent 60%),
      radial-gradient(900px 500px at -10% 10%, rgba(77,208,196,.07), transparent 55%),
      var(--bg);
    color: var(--text);
    font: 14px/1.5 var(--sans);
    -webkit-font-smoothing: antialiased;
    padding: 0 0 80px;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .wrap { max-width: 1240px; margin: 0 auto; padding: 0 24px; }

  /* Data values in mono for tabular alignment (Data-Dense Dashboard style). */
  .heat, .kpi .value, .cv, .mval, .kv .v, tbody td { font-family: var(--mono); }
  /* Visible keyboard focus on every interactive element (a11y: focus-states). */
  :focus { outline: none; }
  :focus-visible { outline: none; box-shadow: var(--focus); border-radius: 8px; }
  thead th:focus-visible { box-shadow: inset var(--focus); }
  tbody tr:focus-visible { box-shadow: inset 0 0 0 2px var(--accent); }
  /* Honour reduced-motion: keep meaning, drop movement (a11y: reduced-motion). */
  @media (prefers-reduced-motion: reduce) {
    * { transition-duration: .01ms !important; animation-duration: .01ms !important; scroll-behavior: auto !important; }
  }

  header.top {
    position: sticky; top: 0; z-index: 20;
    backdrop-filter: blur(10px);
    background: linear-gradient(180deg, rgba(14,17,22,.92), rgba(14,17,22,.72));
    border-bottom: 1px solid var(--border);
  }
  .top-inner { display: flex; align-items: center; gap: 16px; padding: 16px 24px; max-width: 1240px; margin: 0 auto; }
  .logo {
    width: 34px; height: 34px; border-radius: 9px; flex: none;
    background: conic-gradient(from 200deg, var(--accent), var(--accent-2), var(--accent));
    box-shadow: 0 0 0 1px rgba(255,255,255,.08), 0 6px 18px rgba(77,208,196,.25);
    position: relative;
  }
  .logo::after {
    content: ""; position: absolute; inset: 8px; border-radius: 5px;
    background: var(--bg); box-shadow: inset 0 0 0 2px rgba(255,255,255,.06);
  }
  h1 { font-size: 17px; margin: 0; letter-spacing: .2px; }
  .sub { color: var(--muted); font-size: 12.5px; }
  .top-spacer { flex: 1; }
  .stamp { color: var(--faint); font-size: 12px; text-align: right; }

  section { margin-top: 28px; }
  .section-title { font-size: 12px; text-transform: uppercase; letter-spacing: .14em; color: var(--muted); margin: 0 0 12px; }

  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 22px; }
  .kpi {
    background: linear-gradient(180deg, var(--panel), var(--bg-soft));
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px 16px 14px; box-shadow: var(--shadow); position: relative; overflow: hidden;
    transition: border-color var(--t-fast), transform var(--t-fast);
  }
  .kpi:not(:first-child)::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(var(--accent), var(--accent-2)); opacity: .65;
  }
  .kpi:hover { border-color: var(--faint); transform: translateY(-1px); }
  .kpi .label { color: var(--muted); font-size: 12px; }
  .kpi .value { font-size: 26px; font-weight: 600; margin-top: 6px; letter-spacing: .3px; }
  .kpi .meta { color: var(--faint); font-size: 11.5px; margin-top: 4px; }
  .kpi .spark { position: absolute; right: 14px; top: 14px; font-size: 11px; color: var(--accent); }

  .toolbar {
    display: flex; flex-wrap: wrap; gap: 16px 22px; align-items: flex-end;
    background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px 16px; box-shadow: var(--shadow);
  }
  .filter-group { display: flex; flex-direction: column; gap: 7px; }
  .filter-group > .gl { font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--faint); }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    border: 1px solid var(--border); background: var(--panel-2); color: var(--muted);
    padding: 5px 11px; border-radius: 999px; font-size: 12.5px; cursor: pointer;
    user-select: none; transition: border-color var(--t-fast), color var(--t-fast), background var(--t-fast);
    white-space: nowrap; font-family: inherit; line-height: 1.4;
  }
  .chip:hover { border-color: var(--faint); color: var(--text); }
  .chip.on { background: rgba(77,208,196,.16); border-color: var(--accent); color: var(--text); }
  .chip.on.pipe-basic { background: rgba(124,156,255,.18); border-color: var(--basic); }
  .chip.on.pipe-advanced { background: rgba(77,208,196,.18); border-color: var(--advanced); }
  .toolbar .right { margin-left: auto; display: flex; gap: 16px; align-items: flex-end; }
  select, .reset {
    background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
    border-radius: 9px; padding: 7px 10px; font-size: 13px; cursor: pointer;
    font-family: inherit; transition: border-color var(--t-fast);
  }
  select:hover { border-color: var(--faint); }
  .reset:hover { border-color: var(--faint); }

  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden;
  }
  .card .card-head { display: flex; align-items: center; gap: 12px; padding: 14px 16px; border-bottom: 1px solid var(--border); }
  .card .card-head h3 { margin: 0; font-size: 14px; }
  .count-pill { font-size: 12px; color: var(--muted); background: var(--panel-2); border: 1px solid var(--border); border-radius: 999px; padding: 3px 9px; }
  .seg { display: inline-flex; background: var(--panel-2); border: 1px solid var(--border); border-radius: 9px; padding: 2px; gap: 2px; }
  .seg button { background: transparent; border: 0; color: var(--muted); padding: 5px 12px; border-radius: 7px; font-size: 12.5px; cursor: pointer; font-weight: 600; }
  .seg button:hover { color: var(--text); }
  .seg button.on { background: rgba(77,208,196,.18); color: var(--text); box-shadow: inset 0 0 0 1px var(--accent); }

  table { width: 100%; border-collapse: collapse; }
  thead th {
    position: sticky; top: 0; background: var(--bg-soft); z-index: 5;
    text-align: right; padding: 10px 12px; font-size: 11px; font-weight: 600;
    color: var(--muted); border-bottom: 1px solid var(--border); cursor: pointer;
    white-space: nowrap; text-transform: uppercase; letter-spacing: .05em;
    transition: color var(--t-fast), background var(--t-fast); user-select: none;
  }
  thead th[data-key]:hover { color: var(--text); background: var(--panel-2); }
  thead th.lt { text-align: left; cursor: default; }
  thead th .grp { display: block; font-size: 9.5px; color: var(--faint); letter-spacing: .08em; }
  thead th.sorted { color: var(--accent); }
  thead th .arrow { opacity: .8; font-size: 10px; color: var(--accent); }
  tbody td { padding: 9px 12px; text-align: right; border-bottom: 1px solid rgba(42,52,65,.6); font-variant-numeric: tabular-nums; }
  tbody td.lt { text-align: left; }
  tbody tr.run-row { cursor: pointer; transition: background var(--t-fast); }
  tbody tr.run-row:hover { background: rgba(124,156,255,.08); }
  tbody tr.run-row:hover .heat { filter: brightness(1.06); }
  tbody tr.run-row.open { background: rgba(124,156,255,.11); }
  .runtag { display: inline-flex; align-items: center; gap: 8px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
  .dot.basic { background: var(--basic); }
  .dot.advanced { background: var(--advanced); }
  .run-id { font-weight: 600; }
  .run-src { color: var(--faint); font-size: 11.5px; }
  .pill {
    display: inline-block; font-size: 11.5px; padding: 2px 8px; border-radius: 6px;
    background: var(--panel-2); border: 1px solid var(--border); color: var(--muted);
  }
  .pill.basic { color: var(--basic); border-color: rgba(124,156,255,.4); }
  .pill.advanced { color: var(--advanced); border-color: rgba(77,208,196,.4); }
  .heat { border-radius: 6px; padding: 4px 8px; display: inline-block; min-width: 52px; color: #0c0f14; font-weight: 600; transition: filter var(--t-fast); }
  .best-col { box-shadow: inset 0 0 0 1.5px var(--accent); border-radius: 6px; }
  .heat.err { background: rgba(211,107,107,.16); color: var(--bad); box-shadow: inset 0 0 0 1px var(--bad); font-weight: 700; letter-spacing: .04em; cursor: help; }

  .detail td { background: var(--bg-soft); padding: 0; }
  .detail-inner { padding: 18px 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }
  .detail-inner h4 { margin: 0 0 12px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }
  .bk-row { display: grid; grid-template-columns: 92px 1fr; align-items: center; gap: 10px; margin-bottom: 9px; }
  .bk-row .lab { font-size: 12px; color: var(--muted); text-transform: capitalize; }
  .bars { display: flex; flex-direction: column; gap: 5px; }
  .bar-line { display: grid; grid-template-columns: 64px 1fr 42px; align-items: center; gap: 8px; font-size: 11.5px; }
  .bar-line .mlab { color: var(--faint); text-align: right; }
  .track { height: 8px; background: var(--panel-2); border-radius: 6px; overflow: hidden; }
  .fill { display: block; height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--accent-2), var(--accent)); }
  .bar-line .mval { color: var(--text); text-align: right; font-variant-numeric: tabular-nums; }
  .bar-line .mval.err { color: var(--bad); font-weight: 700; cursor: help; }
  .detail-meta { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: center; padding-top: 6px; border-top: 1px dashed var(--border); margin-top: 4px; color: var(--muted); font-size: 12px; }
  .detail-meta b { color: var(--text); font-weight: 600; }
  .viz-links { display: flex; gap: 8px; margin-left: auto; }
  .viz-links a {
    border: 1px solid var(--accent); color: var(--accent); border-radius: 8px;
    padding: 5px 11px; font-size: 12px; font-weight: 600;
  }
  .viz-links a:hover { background: rgba(77,208,196,.14); text-decoration: none; }
  .viz-links a.off { border-color: var(--border); color: var(--faint); cursor: not-allowed; pointer-events: none; }

  .compare { display: grid; grid-template-columns: 1fr; gap: 10px; padding: 16px; }
  .cmp-row { display: grid; grid-template-columns: 220px 1fr 56px; align-items: center; gap: 12px; }
  .cmp-row .cl { font-size: 12.5px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cmp-row .ct { height: 18px; background: var(--panel-2); border-radius: 7px; overflow: hidden; }
  .cmp-row .cf { display: block; height: 100%; border-radius: 7px; min-width: 2px; }
  .cmp-row .cv { text-align: right; font-variant-numeric: tabular-nums; font-size: 12.5px; }

  .constants { display: flex; flex-direction: column; gap: 10px; padding: 16px; }
  .const-set {
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px;
    padding: 11px 13px; border: 1px solid var(--border); border-radius: 11px;
    background: linear-gradient(180deg, var(--panel-2), var(--bg-soft));
  }
  .const-set .set-name { font-weight: 650; font-size: 13px; display: inline-flex; align-items: center; gap: 8px; }
  .const-set .set-name .badge { font-size: 11px; color: var(--muted); background: var(--panel); border: 1px solid var(--border); border-radius: 999px; padding: 2px 8px; font-weight: 500; }
  .const-set .sep { width: 1px; height: 18px; background: var(--border); margin: 0 4px; }
  .kv { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; }
  .kv .k { color: var(--faint); text-transform: uppercase; letter-spacing: .06em; font-size: 10.5px; }
  .kv .v { color: var(--text); background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 2px 8px; font-variant-numeric: tabular-nums; }
  .const-set .none { color: var(--faint); font-size: 12px; font-style: italic; }
  .empty { padding: 40px; text-align: center; color: var(--muted); }
  .footnote { color: var(--faint); font-size: 12px; margin-top: 14px; line-height: 1.7; }
  @media (max-width: 920px) {
    .kpis { grid-template-columns: repeat(2,1fr); }
    .detail-inner { grid-template-columns: 1fr; }
    .cmp-row { grid-template-columns: 140px 1fr 50px; }
  }
</style>
</head>
<body>
<header class="top">
  <div class="top-inner">
    <div class="logo"></div>
    <div>
      <h1>RAGnosis Benchmark</h1>
      <div class="sub">Retrieval &amp; answer quality across pipeline / embedding / chat configurations</div>
    </div>
    <div class="top-spacer"></div>
    <div class="stamp" id="stamp"></div>
  </div>
</header>

<div class="wrap">
  <section class="kpis" id="kpis"></section>

  <section>
    <div class="toolbar" id="toolbar"></div>
  </section>

  <section id="constSection">
    <div class="card">
      <div class="card-head">
        <h3>Constant configuration</h3>
        <span class="sub" style="margin-left:auto">Models held fixed across each run set (preprocess / rewrite / judge / rerank)</span>
      </div>
      <div class="constants" id="constants"></div>
    </div>
  </section>

  <section>
    <div class="card">
      <div class="card-head">
        <h3>Leaderboard</h3>
        <span class="count-pill" id="count"></span>
        <div class="seg" id="viewToggle">
          <button data-view="metrics" class="on">Metrics</button>
          <button data-view="percent">Percent + RAG Index</button>
        </div>
        <span class="sub" style="margin-left:auto">Click a row for difficulty / category breakdown &amp; vector plots</span>
      </div>
      <div style="overflow:auto">
        <table id="table"></table>
      </div>
    </div>
  </section>

  <section>
    <div class="card">
      <div class="card-head">
        <h3>Compare</h3>
        <select id="cmpMetric"></select>
        <span class="sub" style="margin-left:auto">Sorted best → worst on the selected metric</span>
      </div>
      <div class="compare" id="compare"></div>
    </div>
    <div class="footnote" id="foot"></div>
  </section>
</div>

<script>
const DATA = /*__DATA__*/null;

const RETRIEVAL = [
  {key:"mrr", label:"MRR", group:"Retrieval", domain:[0,1], dec:3},
  {key:"ndcg", label:"nDCG", group:"Retrieval", domain:[0,1], dec:3},
  {key:"keyword_coverage", label:"Kw Cov", group:"Retrieval", domain:[0,1], dec:3},
];
const ANSWER = [
  {key:"accuracy", label:"Acc", group:"Answer", domain:[1,5], dec:2},
  {key:"completeness", label:"Compl", group:"Answer", domain:[1,5], dec:2},
  {key:"relevance", label:"Rel", group:"Answer", domain:[1,5], dec:2},
  {key:"answer_keyword_coverage", label:"Ans Kw", group:"Answer", domain:[0,1], dec:3},
  {key:"overall", label:"Overall", group:"Answer", domain:[1,5], dec:2},
];
const COLS = [...RETRIEVAL, ...ANSWER];
const COLMAP = Object.fromEntries(COLS.map(c => [c.key, c]));

// Aggregate "raw percentage" columns + a single blended RAG Index.
// Each metric is taken as a percent of its scale max (0-1 -> x100, 1-5 score -> /5*100).
const COMPUTED = [
  {key:"retrievalPct", label:"Ret %", group:"Summary", dec:1, computed:true},
  {key:"answerPct", label:"Ans %", group:"Summary", dec:1, computed:true},
  {key:"ragIndex", label:"RAG Index", group:"Summary", dec:1, computed:true},
];
const COMPMAP = Object.fromEntries(COMPUTED.map(c => [c.key, c]));
function avg(xs){ xs = xs.filter(x => x != null); return xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : null; }
function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
// Percent of a metric's scale max. Errored answer values (out of the 1-5 scale) are
// dropped entirely — returned as null so they're excluded from Ans % and the RAG Index,
// never considered for anything (a flaky judge occasionally returns e.g. 52 on a 1-5 scale).
function metricPct(run, key){
  const v = metricOf(run, key);
  if (v == null || isAnswerError(key, v)) return null;
  const c = COLMAP[key];
  return clamp(v / c.domain[1] * 100, 0, 100);
}
function retrievalPct(run){ return avg(RETRIEVAL.map(c => metricPct(run, c.key))); }
function answerPct(run){ return avg(ANSWER.map(c => metricPct(run, c.key))); }
function ragIndex(run){ const r = retrievalPct(run), a = answerPct(run); if (r == null && a == null) return null; return 0.5*(r ?? a) + 0.5*(a ?? r); }
const COMPFN = { retrievalPct, answerPct, ragIndex };
function valueFor(run, key){ return COMPMAP[key] ? COMPFN[key](run) : metricOf(run, key); }
// Answer scores live on a 1-5 scale; anything outside that range is a judge error,
// surfaced as "ERR" in the UI rather than a misleading number.
function isScoreKey(key){ const c = COLMAP[key]; return !!(c && c.domain && c.domain[1] === 5); }
function isAnswerError(key, v){ return v != null && isScoreKey(key) && (v > 5 || v < 1); }
function activeColumns(){ return state.view === "percent" ? [...COMPUTED, ...COLS] : COLS; }

const state = {
  runSet: "all",
  pipelines: new Set(DATA ? DATA.pipelines : []),
  embeddings: new Set(DATA ? DATA.embeddings : []),
  chats: new Set(DATA ? DATA.chats : []),
  sortKey: "overall",
  sortDir: -1,
  open: null,
  view: "metrics",
  cmpMetric: "ragIndex",
};

function metricOf(run, key) {
  const g = (RETRIEVAL.some(c => c.key === key)) ? run.retrieval : run.answer;
  const v = g.overall[key];
  return (v === undefined || v === null) ? null : v;
}
function shortEmb(m){ return m.replace(":latest","").replace(":l6-v2","-l6"); }
function fmt(v, dec){ return v == null ? "—" : v.toFixed(dec); }
// Faithful for in-range values; compact for out-of-scale judge anomalies so they don't
// blow out the column width (the raw number is preserved in the cell's title tooltip).
function fmtNative(v, dec){ if (v == null) return "—"; return Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(dec); }

function lerp(a,b,t){ return a + (b-a)*t; }
function heatColor(t){ // 0..1 -> red..amber..green, muted
  t = Math.max(0, Math.min(1, t));
  const stops = [[211,107,107],[216,178,74],[56,180,139]];
  let c;
  if (t < 0.5){ const u=t/0.5; c=[lerp(stops[0][0],stops[1][0],u),lerp(stops[0][1],stops[1][1],u),lerp(stops[0][2],stops[1][2],u)]; }
  else { const u=(t-0.5)/0.5; c=[lerp(stops[1][0],stops[2][0],u),lerp(stops[1][1],stops[2][1],u),lerp(stops[1][2],stops[2][2],u)]; }
  return `rgb(${c.map(x=>Math.round(x)).join(",")})`;
}

function filteredRuns(){
  return DATA.runs.filter(r =>
    (state.runSet === "all" || r.source === state.runSet) &&
    state.pipelines.has(r.pipeline) &&
    state.embeddings.has(r.embedding_model) &&
    state.chats.has(r.chat_model));
}
function sortedRuns(){
  const runs = filteredRuns();
  runs.sort((a,b) => {
    const av = valueFor(a, state.sortKey), bv = valueFor(b, state.sortKey);
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    // Errored answer values are not real scores — keep them at the bottom either way.
    const ae = isAnswerError(state.sortKey, av), be = isAnswerError(state.sortKey, bv);
    if (ae !== be) return ae ? 1 : -1;
    return (av - bv) * state.sortDir;
  });
  return runs;
}

function renderKpis(){
  const runs = filteredRuns();
  const el = document.getElementById("kpis");
  if (!runs.length){ el.innerHTML = ""; return; }
  function best(key){
    let top = null;
    for (const r of runs){
      const v = valueFor(r,key);
      if (v!=null && !isAnswerError(key,v) && (top==null || v>top.v)) top={v, r};
    }
    return top;
  }
  const bi = best("ragIndex"), bo = best("overall"), bm = best("mrr");
  const cards = [
    {label:"Runs in view", value:String(runs.length), meta:`${DATA.runs.length} total · ${DATA.pipelines.length} pipelines`, spark:""},
    {label:"Top RAG Index", value: bi? bi.v.toFixed(1)+"%":"—", meta: bi? `${bi.r.id} · ${shortEmb(bi.r.embedding_model)} · ${bi.r.chat_model}`:"", spark:"◆"},
    {label:"Best answer overall", value: bo? bo.v.toFixed(2):"—", meta: bo? `${bo.r.id} · ${shortEmb(bo.r.embedding_model)} · ${bo.r.chat_model}`:"", spark:"★"},
    {label:"Best retrieval MRR", value: bm? bm.v.toFixed(3):"—", meta: bm? `${bm.r.id} · ${shortEmb(bm.r.embedding_model)}`:"", spark:"★"},
  ];
  el.innerHTML = cards.map(c => `
    <div class="kpi">
      <span class="spark">${c.spark}</span>
      <div class="label">${c.label}</div>
      <div class="value">${c.value}</div>
      <div class="meta">${c.meta || "&nbsp;"}</div>
    </div>`).join("");
}

function chip(value, on, cls, onclick){
  const c = document.createElement("button");
  c.type = "button";
  c.className = "chip" + (on ? " on" : "") + (cls ? " " + cls : "");
  c.textContent = value;
  c.setAttribute("aria-pressed", on ? "true" : "false");
  c.onclick = onclick;
  return c;
}
function renderToolbar(){
  const tb = document.getElementById("toolbar");
  tb.innerHTML = "";

  // Run-set picker (single-select): "All" plus each discovered run set folder.
  if (DATA.runSets && DATA.runSets.length >= 1){
    const rsWrap = document.createElement("div");
    rsWrap.className = "filter-group";
    rsWrap.innerHTML = `<span class="gl">Run set</span>`;
    const rsChips = document.createElement("div");
    rsChips.className = "chips";
    const options = [{source:"all", count: DATA.runs.length}, ...DATA.runSets];
    for (const opt of options){
      const label = opt.source === "all" ? "All sets" : opt.source;
      rsChips.appendChild(chip(`${label} · ${opt.count}`, state.runSet === opt.source, "", () => {
        state.runSet = opt.source;
        render();
      }));
    }
    rsWrap.appendChild(rsChips);
    tb.appendChild(rsWrap);
  }

  const groups = [
    {gl:"Pipeline", values: DATA.pipelines, set: state.pipelines, pipe:true},
    {gl:"Embedding model", values: DATA.embeddings, set: state.embeddings},
    {gl:"Chat model", values: DATA.chats, set: state.chats},
  ];
  for (const g of groups){
    const wrap = document.createElement("div");
    wrap.className = "filter-group";
    wrap.innerHTML = `<span class="gl">${g.gl}</span>`;
    const chips = document.createElement("div");
    chips.className = "chips";
    for (const v of g.values){
      const cls = g.pipe ? `pipe-${v}` : "";
      chips.appendChild(chip(g.pipe ? v : (g.gl[0]==="E"? shortEmb(v): v), g.set.has(v), cls, () => {
        if (g.set.has(v)) g.set.delete(v); else g.set.add(v);
        if (g.set.size === 0) g.values.forEach(x => g.set.add(x)); // never empty
        render();
      }));
    }
    wrap.appendChild(chips);
    tb.appendChild(wrap);
  }
  const right = document.createElement("div");
  right.className = "right";
  const reset = document.createElement("button");
  reset.className = "reset";
  reset.textContent = "Reset filters";
  reset.onclick = () => {
    state.runSet = "all";
    state.pipelines = new Set(DATA.pipelines);
    state.embeddings = new Set(DATA.embeddings);
    state.chats = new Set(DATA.chats);
    render();
  };
  right.appendChild(reset);
  tb.appendChild(right);
}

function colExtent(runs, key){
  let lo=Infinity, hi=-Infinity;
  for (const r of runs){ const v=valueFor(r,key); if(v!=null){ lo=Math.min(lo,v); hi=Math.max(hi,v); } }
  if (lo===Infinity) return null;
  return [lo,hi];
}

function cellFor(run, col){
  const raw = valueFor(run, col.key);
  if (raw == null) return {bg:"transparent", color:"var(--faint)", text:"—"};
  if (isAnswerError(col.key, raw))
    return {err:true, text:"ERR", title:`Invalid answer score (expected 1–5): ${raw}`};
  let text, t;
  if (col.computed){ text = raw.toFixed(col.dec) + "%"; t = raw/100; }
  else {
    const pv = clamp(raw / col.domain[1] * 100, 0, 100);  // absolute, scale-based shade
    t = pv / 100;
    text = state.view === "percent" ? pv.toFixed(1) + "%" : fmtNative(raw, col.dec);  // raw stays faithful
  }
  const outOfScale = !col.computed && (raw < col.domain[0] || raw > col.domain[1]);
  return {bg: heatColor(t), color: "#0c0f14", text, title: outOfScale ? `Out of ${col.domain[0]}–${col.domain[1]} scale: ${raw}` : ""};
}

function renderTable(){
  const cols = activeColumns();
  const runs = sortedRuns();
  const table = document.getElementById("table");
  document.getElementById("count").textContent = `${runs.length} configuration${runs.length===1?"":"s"}`;
  if (!runs.length){ table.innerHTML = `<tbody><tr><td class="empty">No runs match these filters.</td></tr></tbody>`; return; }

  const best = {};
  for (const c of cols){
    let m = null;
    for (const r of runs){
      const v = valueFor(r, c.key);
      if (v != null && !isAnswerError(c.key, v) && (m == null || v > m)) m = v;
    }
    best[c.key] = m;
  }

  const head = `
    <thead><tr>
      <th class="lt" scope="col">Run</th>
      <th class="lt" scope="col">Config</th>
      ${cols.map((c,i) => {
        const sorted = state.sortKey===c.key;
        const ariaSort = sorted ? (state.sortDir<0?'descending':'ascending') : 'none';
        return `
        <th data-key="${c.key}" scope="col" tabindex="0" role="button"
            aria-sort="${ariaSort}" title="Sort by ${c.group} · ${c.label}"
            class="${sorted?'sorted':''}">
          <span class="grp">${(i===0||cols[i-1].group!==c.group)?c.group:'&nbsp;'}</span>
          ${c.label} <span class="arrow">${sorted?(state.sortDir<0?'▼':'▲'):''}</span>
        </th>`;
      }).join("")}
    </tr></thead>`;

  const rows = runs.map(r => {
    const cells = cols.map(c => {
      const cell = cellFor(r, c);
      const raw = valueFor(r, c.key);
      const isBest = !cell.err && raw != null && best[c.key] != null && Math.abs(raw - best[c.key]) < 1e-9;
      const titleAttr = cell.title ? ` title="${cell.title}"` : "";
      const cls = cell.err ? "err" : (isBest ? "best-col" : "");
      const styleAttr = cell.err ? "" : ` style="background:${cell.bg};color:${cell.color}"`;
      return `<td><span class="heat ${cls}"${styleAttr}${titleAttr}>${cell.text}</span></td>`;
    }).join("");
    const open = state.open === r.id;
    return `
      <tr class="run-row ${open?'open':''}" data-id="${r.id}" tabindex="0" role="button"
          aria-expanded="${open?'true':'false'}"
          aria-label="Run ${r.name}, ${r.pipeline}, ${shortEmb(r.embedding_model)}, ${r.chat_model}. Toggle details.">
        <td class="lt"><span class="runtag"><span class="dot ${r.pipeline}"></span>
          <span><span class="run-id">#${r.name}</span> <span class="run-src">${r.source}</span></span></span></td>
        <td class="lt">
          <span class="pill ${r.pipeline}">${r.pipeline}</span>
          <span class="pill">${shortEmb(r.embedding_model)}</span>
          <span class="pill">${r.chat_model}</span>
        </td>
        ${cells}
      </tr>
      ${open ? detailRow(r, cols.length) : ""}`;
  }).join("");

  table.innerHTML = head + `<tbody>${rows}</tbody>`;

  // Re-render rebuilds the table DOM; restore keyboard focus so the user keeps their place.
  const refocus = (selector) => { const el = table.querySelector(selector); if (el) el.focus(); };
  const sortBy = (k) => {
    if (state.sortKey === k) state.sortDir *= -1;
    else { state.sortKey = k; state.sortDir = -1; }
    renderTable();
    refocus(`thead th[data-key="${CSS.escape(k)}"]`);
  };
  table.querySelectorAll("thead th[data-key]").forEach(th => {
    th.onclick = () => sortBy(th.dataset.key);
    th.onkeydown = (e) => { if (e.key === "Enter" || e.key === " "){ e.preventDefault(); sortBy(th.dataset.key); } };
  });
  const toggleRow = (id) => {
    state.open = (state.open === id) ? null : id;
    renderTable();
    refocus(`tr.run-row[data-id="${CSS.escape(id)}"]`);
  };
  table.querySelectorAll("tr.run-row").forEach(tr => {
    tr.onclick = () => toggleRow(tr.dataset.id);
    tr.onkeydown = (e) => { if (e.key === "Enter" || e.key === " "){ e.preventDefault(); toggleRow(tr.dataset.id); } };
  });
}

function barBlock(title, buckets, metricDefs, orderHint){
  const keys = Object.keys(buckets);
  const order = orderHint.filter(k => keys.includes(k)).concat(keys.filter(k => !orderHint.includes(k)).sort());
  let html = `<h4>${title}</h4>`;
  for (const k of order){
    const b = buckets[k];
    html += `<div class="bk-row"><div class="lab">${k}${b.questions!=null?` · ${b.questions}q`:''}</div><div class="bars">`;
    for (const m of metricDefs){
      const v = b[m.key];
      const err = isAnswerError(m.key, v);
      const t = v==null ? 0 : (v - m.domain[0]) / (m.domain[1]-m.domain[0]);
      const fill = err
        ? `width:100%;background:var(--bad)`
        : `width:${Math.max(0,Math.min(100,t*100)).toFixed(1)}%`;
      const valText = v==null ? '—' : (err ? 'ERR' : v.toFixed(m.dec));
      const valTitle = err ? ` title="Invalid answer score (expected 1–5): ${v}"` : "";
      html += `<div class="bar-line">
        <span class="mlab">${m.label}</span>
        <span class="track"><span class="fill" style="${fill}"></span></span>
        <span class="mval${err?' err':''}"${valTitle}>${valText}</span></div>`;
    }
    html += `</div></div>`;
  }
  return html;
}

function detailRow(r, colCount){
  const diffOrder = ["easy","medium","hard"];
  const ret = barBlock("Retrieval · by difficulty", r.retrieval.byDifficulty, RETRIEVAL, diffOrder);
  const ans = barBlock("Answer · by difficulty", r.answer.byDifficulty, ANSWER.filter(c=>["accuracy","completeness","relevance","overall"].includes(c.key)), diffOrder);
  const retCat = Object.keys(r.retrieval.byCategory||{}).length
    ? barBlock("Retrieval · by category", r.retrieval.byCategory, [COLMAP.mrr, COLMAP.keyword_coverage], [])
    : "";
  const ansCat = Object.keys(r.answer.byCategory||{}).length
    ? barBlock("Answer · by category", r.answer.byCategory, [COLMAP.overall, COLMAP.answer_keyword_coverage], [])
    : "";
  const v2 = r.viz && r.viz.d2;
  const v3 = r.viz && r.viz.d3;
  const meta = `
    <div class="detail-meta">
      <span>k = <b>${r.k ?? '—'}</b></span>
      <span>questions = <b>${r.questions ?? '—'}</b></span>
      <span>judge = <b>${r.judge_model}</b></span>
      ${r.pipeline==='advanced' ? `<span>rerank = <b>${r.rerank_model}</b></span>` : ''}
      <span class="viz-links">
        <a class="${v2?'':'off'}" ${v2?`href="${v2}" target="_blank"`:''}>2D vectors ↗</a>
        <a class="${v3?'':'off'}" ${v3?`href="${v3}" target="_blank"`:''}>3D vectors ↗</a>
      </span>
    </div>`;
  return `<tr class="detail"><td colspan="${colCount+2}"><div class="detail-inner">
    <div>${ret}${retCat}</div>
    <div>${ans}${ansCat}</div>
    ${meta}
  </div></td></tr>`;
}

function renderCompare(){
  const sel = document.getElementById("cmpMetric");
  if (!sel.options.length){
    sel.innerHTML = [...COMPUTED, ...COLS].map(c => `<option value="${c.key}">${c.group} · ${c.label}</option>`).join("");
    sel.value = state.cmpMetric;
    sel.onchange = () => { state.cmpMetric = sel.value; renderCompare(); };
  }
  const c = COMPMAP[state.cmpMetric] || COLMAP[state.cmpMetric];
  const suffix = c.computed ? "%" : "";
  // Errored answer runs sort to the bottom and don't count toward the scale max.
  const runs = filteredRuns().filter(r => valueFor(r, c.key) != null).sort((a,b) => {
    const ae = isAnswerError(c.key, valueFor(a,c.key)), be = isAnswerError(c.key, valueFor(b,c.key));
    if (ae !== be) return ae ? 1 : -1;
    return valueFor(b,c.key) - valueFor(a,c.key);
  });
  const box = document.getElementById("compare");
  if (!runs.length){ box.innerHTML = `<div class="empty">No data.</div>`; return; }
  const valid = runs.map(r => valueFor(r,c.key)).filter(v => !isAnswerError(c.key, v));
  const max = valid.length ? Math.max(...valid) : 1;
  const ext = colExtent(runs.filter(r => !isAnswerError(c.key, valueFor(r,c.key))), c.key);
  box.innerHTML = runs.map(r => {
    const v = valueFor(r, c.key);
    if (isAnswerError(c.key, v)){
      return `<div class="cmp-row">
        <span class="cl"><span class="dot ${r.pipeline}" style="display:inline-block;margin-right:6px"></span>#${r.name} · ${shortEmb(r.embedding_model)} · ${r.chat_model}</span>
        <span class="ct"><span class="cf" style="width:100%;background:var(--bad);opacity:.5"></span></span>
        <span class="cv" style="color:var(--bad);font-weight:700" title="Invalid answer score (expected 1–5): ${v}">ERR</span>
      </div>`;
    }
    const t = c.computed ? v/100 : (ext && ext[1]>ext[0] ? (v-ext[0])/(ext[1]-ext[0]) : 1);
    const w = max>0 ? (v/max*100) : 0;
    return `<div class="cmp-row">
      <span class="cl"><span class="dot ${r.pipeline}" style="display:inline-block;margin-right:6px"></span>#${r.name} · ${shortEmb(r.embedding_model)} · ${r.chat_model}</span>
      <span class="ct"><span class="cf" style="width:${w.toFixed(1)}%;background:${heatColor(t)}"></span></span>
      <span class="cv">${v.toFixed(c.dec)}${suffix}</span>
    </div>`;
  }).join("");
}

function renderConstants(){
  const box = document.getElementById("constants");
  const sets = (DATA.runSets || []).filter(s => state.runSet === "all" || s.source === state.runSet);
  if (!sets.length){ box.innerHTML = `<div class="none">No run sets found.</div>`; return; }
  const ORDER = [["PREPROCESS_MODEL","Preprocess"],["REWRITE_MODEL","Rewrite"],["JUDGE_MODEL","Judge"],["RERANK_MODEL","Rerank"]];
  box.innerHTML = sets.map(s => {
    const c = s.constants;
    const kvs = c
      ? ORDER.filter(([k]) => c[k] != null).map(([k,lab]) =>
          `<span class="kv"><span class="k">${lab}</span><span class="v">${c[k]}</span></span>`).join('<span class="sep"></span>')
      : `<span class="none">constants not recorded for this set</span>`;
    return `<div class="const-set">
      <span class="set-name">${s.source} <span class="badge">${s.count} run${s.count===1?'':'s'}</span></span>
      <span class="sep"></span>${kvs}</div>`;
  }).join("");
}

function render(){
  renderToolbar();
  renderConstants();
  renderKpis();
  renderTable();
  renderCompare();
}

function boot(){
  document.getElementById("stamp").innerHTML =
    DATA ? `${DATA.runs.length} runs · generated ${DATA.generatedAt}` : "";
  document.getElementById("foot").innerHTML =
    "Cells are shaded by score on each metric's own scale (red = low → green = high). " +
    "Answer metrics are 1–5 (judge-scored); retrieval &amp; coverage are 0–1. " +
    "<b>RAG Index</b> = 50% retrieval-% + 50% answer-%, where each metric is taken as a percent of its scale. " +
    "Answer scores outside the 1–5 range are judge errors (shown as <span style='color:var(--bad);font-weight:700'>ERR</span>) " +
    "and are excluded from every aggregate — the RAG Index, percentages, best-of KPIs, and rankings. " +
    "Overall = question-weighted mean across difficulty buckets. " +
    "Regenerate with <code>uv run python -m app.build_benchmark</code>.";
  if (!DATA || !DATA.runs.length){
    document.getElementById("toolbar").innerHTML = `<div class="empty">No evaluation runs found. Run the evaluator, then rebuild this page.</div>`;
    return;
  }
  const toggle = document.getElementById("viewToggle");
  toggle.querySelectorAll("button").forEach(btn => {
    btn.onclick = () => {
      state.view = btn.dataset.view;
      toggle.querySelectorAll("button").forEach(b => b.classList.toggle("on", b === btn));
      // Lead with the headline metric for each view so the table is sorted usefully.
      state.sortKey = state.view === "percent" ? "ragIndex" : "overall";
      state.sortDir = -1;
      renderTable();
    };
  });
  render();
}
boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
