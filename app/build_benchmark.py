"""Build a self-contained ``rag-benchmark.html`` report from evaluation results.

Scans every ``<root>/results/<n>/`` folder (``run-*/results`` at the repo root, plus
``app/results``) for the ``config.json`` + ``evals.json`` a run produces, aggregates the
per-difficulty scores into overall metrics, and embeds everything into a single HTML
file. ``RESEARCH.md`` (when present) is rendered into the page as a "Research paper"
tab. The page needs no server and no network (data is inlined) — open it directly.

The page has three tabs:

* **Overview** — the study's story: best-per-run progression charts, an answer-quality
  heatmap by question category, and a diagram of the final pipeline.
* **Results explorer** — the full sortable/filterable leaderboard of every config.
* **Research paper** — ``RESEARCH.md`` rendered to HTML at build time.

Re-run after any new evaluation (or after editing ``RESEARCH.md``) to refresh::

    uv run python -m app.build_benchmark
    uv run python -m app.build_benchmark --output rag-benchmark.html
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import re
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

# Short human labels for the archived research runs (matches RESEARCH.md's
# "Main change" column). Unknown run sets fall back to their config summary.
RUN_LABELS = {
    "run-1": "Basic baseline",
    "run-2": "Rewrite + BGE rerank",
    "run-3": "Enriched docs (small)",
    "run-4": "Enriched docs (large)",
    "run-5": "No-enrichment control",
    "run-6": "Small-to-Big",
    "run-7": "Rollup documents",
    "run-8": "Jina reranker",
}


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


def _run_set_info(parent: Path) -> dict[str, Any]:
    """Shared models + description for a run set (from its top-level ``config.json``).

    Accepts either a ``constant_models`` or a ``metadata`` mapping for the fixed
    models, and a ``description`` object with ``summary`` / ``how_it_works``.
    """
    info: dict[str, Any] = {"constants": None, "summary": "", "how": ""}
    config_path = parent / "config.json"
    if not config_path.exists():
        return info
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return info
    if not isinstance(data, dict):
        return info
    constants = data.get("constant_models") or data.get("metadata")
    if isinstance(constants, dict):
        info["constants"] = {str(k): str(v) for k, v in constants.items()}
    description = data.get("description")
    if isinstance(description, dict):
        info["summary"] = str(description.get("summary", ""))
        info["how"] = str(description.get("how_it_works", ""))
    return info


def _run_set_label(source: str, summary: str) -> str:
    if source in RUN_LABELS:
        return RUN_LABELS[source]
    if summary:
        head = summary.split(":")[0].split(".")[0].strip()
        return head[:48] + ("…" if len(head) > 48 else "")
    return source


# Judge scores live on a 1-5 scale; anything outside is a judge error and must not
# poison the question-weighted mean (the page promises errors are excluded everywhere).
_SCORE_METRICS = {"accuracy", "completeness", "relevance", "overall"}


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
            if metric in _SCORE_METRICS and not (1.0 <= float(value) <= 5.0):
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
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        evals = json.loads(evals_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[build-benchmark] skipping {run_dir}: unreadable config/evals ({exc})")
        return None
    if not isinstance(config, dict) or not isinstance(evals, dict):
        print(f"[build-benchmark] skipping {run_dir}: config/evals is not a JSON object")
        return None

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
            info = _run_set_info(parent)
            run_sets.append(
                {
                    "source": source,
                    "count": set_count,
                    "constants": info["constants"],
                    "summary": info["summary"],
                    "how": info["how"],
                    "label": _run_set_label(source, info["summary"]),
                }
            )
    if skipped:
        print(
            f"[build-benchmark] skipped {len(skipped)} run(s) below "
            f"--min-questions={min_questions}: {', '.join(skipped)}"
        )
    return runs, run_sets


# --------------------------------------------------------------------------------------
# RESEARCH.md → HTML (the small markdown subset the paper actually uses)
# --------------------------------------------------------------------------------------


def _md_inline(text: str) -> str:
    text = html_mod.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    return text


def _md_to_html(md: str) -> str:
    """Render the markdown subset used by RESEARCH.md: #–#### headings, paragraphs,
    pipe tables (with an alignment row), flat ``-``/``*`` lists, and inline
    code / bold / links. Anything fancier will come through as a paragraph."""
    lines = md.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{_md_inline(heading.group(2))}</h{level}>")
            i += 1
            continue
        if stripped.startswith("|"):
            rows: list[list[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            body = [
                row
                for row in rows[1:]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in row)
            ]
            thead = "".join(f"<th>{_md_inline(c)}</th>" for c in rows[0])
            tbody = "".join(
                "<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in row) + "</tr>"
                for row in body
            )
            out.append(
                "<div class='tbl-scroll'><table><thead><tr>"
                + thead
                + "</tr></thead><tbody>"
                + tbody
                + "</tbody></table></div>"
            )
            continue
        if re.match(r"^[-*]\s+", stripped):
            items: list[str] = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(_md_inline(re.sub(r"^[-*]\s+", "", lines[i].strip())))
                i += 1
            out.append("<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")
            continue
        paragraph: list[str] = []
        while i < n:
            s = lines[i].strip()
            if not s or s.startswith(("|", "#")) or re.match(r"^[-*]\s+", s):
                break
            paragraph.append(s)
            i += 1
        out.append(f"<p>{_md_inline(' '.join(paragraph))}</p>")
    return "\n".join(out)


def _research_payload() -> dict[str, Any] | None:
    path = PROJECT_ROOT / "RESEARCH.md"
    if not path.exists():
        return None
    try:
        md = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return {"html": _md_to_html(md), "source": "RESEARCH.md"}


def build_payload(min_questions: int = 0) -> dict[str, Any]:
    runs, run_sets = collect(min_questions)
    return {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "runs": runs,
        "runSets": run_sets,
        "pipelines": sorted({run["pipeline"] for run in runs}),
        "embeddings": sorted({run["embedding_model"] for run in runs}),
        "chats": sorted({run["chat_model"] for run in runs}),
        "research": _research_payload(),
    }


def render_html(payload: dict[str, Any]) -> str:
    # "</" must be escaped inside a <script> block, or the embedded research HTML
    # would terminate the script tag mid-JSON.
    data_json = json.dumps(payload, indent=None, ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.replace("/*__DATA__*/null", data_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG benchmark report HTML.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "rag-benchmark.html"),
        help="Where to write the report (default: rag-benchmark.html at repo root).",
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


# The report template. Data is injected by replacing the `/*__DATA__*/null` token.
# Pure HTML/CSS/JS, no external assets beyond fonts — works offline from file://
# (fonts fall back to the system stacks).
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<title>RAGnosis · Benchmark report</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='3' fill='%232a78d6'/%3E%3C/svg%3E" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..500&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    /* Paper & ink (chart chrome tokens) */
    --page: #f6f6f3;
    --surface: #fcfcfb;
    --surface-2: #f1f1ed;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --muted: #6d6b66;   /* 4.9:1 on the page plane — #898781 fails AA at 11-12px */
    --hairline: #e1e0d9;
    --baseline: #c3c2b7;
    /* Data (validated palette: blue sequential ramp; blue/orange categorical pair) */
    --accent: #1c5cab;        /* interactive text / emphasis (blue 550, AA on paper) */
    --accent-soft: #e4eefb;
    --series: #2a78d6;        /* line + marks (blue 450) */
    --advanced: #2a78d6;
    --basic: #eb6834;
    --err: #d03b3b;           /* status: critical — reserved for judge errors */
    --good: #006300;          /* status: delta up (AA on paper) */
    --basic-ink: #a3441c;     /* orange text on paper */
    --basic-border: #f0c4ae;
    --accent-border: #b7d3f6;
    --basic-soft: #fdeee6;    /* orange chip wash */
    --page-glass: rgba(246,246,243,.92);
    --tip-bg: #0b0b0b;
    --tip-fg: #ffffff;
    --radius: 10px;
    --shadow: 0 1px 2px rgba(11,11,11,.05), 0 6px 20px rgba(11,11,11,.05);
    --serif: "Newsreader", Georgia, "Times New Roman", serif;
    --sans: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
    --focus: 0 0 0 2px var(--surface), 0 0 0 4px var(--accent);
    --t-fast: 130ms;
  }
  /* Dark theme — steps chosen for the dark surface (#1a1a19) and validated against it,
     not an automatic flip of the light values. Data ramps live in JS (PALETTE). */
  html[data-theme="dark"] {
    --page: #0d0d0d;
    --surface: #1a1a19;
    --surface-2: #242422;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #9a988f;         /* ~6:1 on the dark surface */
    --hairline: #2c2c2a;
    --baseline: #45443f;
    --accent: #6da7ec;        /* interactive text (blue 300, AA on dark surface) */
    --accent-soft: #172a44;
    --series: #3987e5;        /* line + marks (dark categorical blue) */
    --advanced: #3987e5;
    --basic: #e8794a;
    --err: #e66767;
    --good: #3ec46a;
    --basic-ink: #eb9066;
    --basic-border: #4a2a18;
    --accent-border: #24466f;
    --basic-soft: rgba(232,121,74,.20);
    --page-glass: rgba(13,13,13,.92);
    --tip-bg: #33322f;
    --tip-fg: #ffffff;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 6px 22px rgba(0,0,0,.55);
    color-scheme: dark;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    background: var(--page);
    color: var(--ink);
    font: 15px/1.55 var(--sans);
    -webkit-font-smoothing: antialiased;
    padding-bottom: 88px;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  code { font-family: var(--mono); font-size: .9em; background: var(--surface-2); border: 1px solid var(--hairline); border-radius: 5px; padding: 1px 5px; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 0 24px; }
  .num { font-family: var(--mono); font-feature-settings: "tnum" 1, "calt" 0; }

  :focus { outline: none; }
  :focus-visible { outline: none; box-shadow: var(--focus); border-radius: 6px; }
  @media (prefers-reduced-motion: reduce) {
    * { transition-duration: .01ms !important; animation-duration: .01ms !important; scroll-behavior: auto !important; }
  }

  /* ---------- header + tab nav ---------- */
  header.top {
    position: sticky; top: 0; z-index: 30;
    background: var(--page-glass);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--hairline);
  }
  .theme-toggle {
    margin-left: 16px; display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface); color: var(--ink-2); border: 1px solid var(--hairline);
    border-radius: 999px; padding: 5px 11px 5px 9px; font: 600 12px/1 var(--sans);
    cursor: pointer; transition: border-color var(--t-fast), color var(--t-fast);
  }
  .theme-toggle:hover { border-color: var(--baseline); color: var(--ink); }
  .theme-toggle svg { width: 14px; height: 14px; }
  .theme-toggle .moon { display: none; }
  html[data-theme="dark"] .theme-toggle .sun { display: none; }
  html[data-theme="dark"] .theme-toggle .moon { display: inline; }
  .masthead { display: flex; align-items: baseline; gap: 14px; padding: 18px 24px 0; max-width: 1180px; margin: 0 auto; }
  .wordmark { font-family: var(--serif); font-weight: 600; font-size: 22px; letter-spacing: .01em; }
  .wordmark .cross { color: var(--accent); }
  .masthead .sub { color: var(--ink-2); font-size: 13px; }
  .masthead .stamp { margin-left: auto; color: var(--muted); font-size: 12px; font-family: var(--mono); }
  .masthead .stamp + .theme-toggle { margin-left: 16px; }
  nav.tabs { display: flex; gap: 4px; padding: 8px 24px 0; max-width: 1180px; margin: 0 auto; overflow-x: auto; }
  nav.tabs button {
    appearance: none; border: 0; background: transparent; cursor: pointer;
    font: 600 13.5px/1 var(--sans); color: var(--ink-2);
    padding: 10px 14px 12px; border-bottom: 2px solid transparent;
    white-space: nowrap; transition: color var(--t-fast), border-color var(--t-fast);
  }
  nav.tabs button:hover { color: var(--ink); }
  nav.tabs button[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); }

  main section.panel { display: none; }
  main section.panel.active { display: block; }

  h2.sec { font-family: var(--serif); font-weight: 600; font-size: 26px; margin: 40px 0 6px; letter-spacing: .005em; }
  .sec-note { color: var(--ink-2); font-size: 13.5px; margin: 0 0 16px; max-width: 76ch; }
  .eyebrow { font: 600 11px/1 var(--sans); letter-spacing: .14em; text-transform: uppercase; color: var(--accent); margin: 36px 0 10px; }

  .card {
    background: var(--surface); border: 1px solid var(--hairline);
    border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden;
  }
  .card-head { display: flex; align-items: center; gap: 12px; padding: 13px 16px; border-bottom: 1px solid var(--hairline); flex-wrap: wrap; }
  .card-head h3 { margin: 0; font-size: 14px; font-weight: 650; }
  .card-head .hint { color: var(--muted); font-size: 12px; margin-left: auto; }

  /* ---------- overview: intro + stat tiles ---------- */
  .lede { max-width: 78ch; margin-top: 30px; }
  .lede h2 { font-family: var(--serif); font-weight: 500; font-size: clamp(26px, 4vw, 38px); line-height: 1.18; margin: 6px 0 14px; letter-spacing: .002em; }
  .lede h2 em { font-style: italic; color: var(--accent); }
  .lede p { color: var(--ink-2); font-size: 15.5px; max-width: 72ch; }
  .tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 26px 0 8px; }
  .tile { background: var(--surface); border: 1px solid var(--hairline); border-radius: var(--radius); box-shadow: var(--shadow); padding: 14px 16px 12px; }
  .tile .label { color: var(--ink-2); font-size: 12.5px; }
  .tile .value { font: 600 30px/1.15 var(--sans); margin-top: 6px; }
  .tile .value small { font-size: 15px; font-weight: 500; color: var(--muted); }
  .tile .meta { color: var(--muted); font-size: 11.5px; margin-top: 5px; font-family: var(--mono); }
  .tile .delta { font-size: 12px; font-weight: 600; color: var(--good); margin-left: 6px; }

  /* ---------- run strip (progression small multiples) ---------- */
  .strip { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .panel-viz { background: var(--surface); border: 1px solid var(--hairline); border-radius: var(--radius); box-shadow: var(--shadow); padding: 14px 14px 8px; position: relative; }
  .panel-viz h3 { margin: 0 0 2px; font-size: 13.5px; font-weight: 650; }
  .panel-viz .yunit { color: var(--muted); font-size: 11.5px; margin-bottom: 4px; }
  /* min-width keeps SVG text legible on phones: the chart scrolls instead of shrinking. */
  .panel-viz { overflow-x: auto; }
  .panel-viz svg { width: 100%; min-width: 540px; height: auto; display: block; }
  .viz-tip {
    position: absolute; pointer-events: none; z-index: 10; display: none;
    background: var(--tip-bg); color: var(--tip-fg); border-radius: 8px; padding: 8px 10px;
    font-size: 12px; line-height: 1.45; box-shadow: 0 6px 20px rgba(0,0,0,.35);
    max-width: 240px;
  }
  .viz-tip .v { font: 600 15px/1.2 var(--sans); }
  .viz-tip .k { color: rgba(255,255,255,.72); }
  .strip-caption { color: var(--muted); font-size: 12.5px; margin-top: 10px; max-width: 84ch; }
  .strip-caption b { color: var(--ink-2); }

  /* ---------- heatmap ---------- */
  .heatwrap { overflow-x: auto; padding: 6px 16px 14px; }
  table.heatmap { border-collapse: separate; border-spacing: 2px; min-width: 560px; }
  table.heatmap th { font: 600 11.5px/1.3 var(--sans); color: var(--ink-2); text-align: left; padding: 8px 10px 6px; }
  table.heatmap thead th { vertical-align: bottom; }
  table.heatmap thead th .sub { display: block; color: var(--muted); font-weight: 500; font-size: 10.5px; font-family: var(--mono); }
  table.heatmap td { border-radius: 6px; padding: 9px 12px; text-align: right; font-family: var(--mono); font-size: 12.5px; font-weight: 600; min-width: 96px; cursor: default; transition: filter var(--t-fast); }
  table.heatmap td:hover { filter: brightness(1.05); }
  table.heatmap th.rowlab { white-space: nowrap; padding-right: 14px; }
  table.heatmap th.rowlab .q { color: var(--muted); font-weight: 500; font-family: var(--mono); font-size: 10.5px; margin-left: 6px; }
  .scale-legend { display: flex; align-items: center; gap: 8px; padding: 0 16px 14px; color: var(--muted); font-size: 11.5px; }
  .scale-legend .ramp { height: 8px; width: 140px; border-radius: 4px; background: linear-gradient(90deg,
    #cde2fb 0 14.3%, #9ec5f4 14.3% 28.6%, #6da7ec 28.6% 42.9%, #3987e5 42.9% 57.2%,
    #256abf 57.2% 71.5%, #184f95 71.5% 85.8%, #0d366b 85.8% 100%); }
  html[data-theme="dark"] .scale-legend .ramp { background: linear-gradient(90deg,
    #12243d 0 14.3%, #173458 14.3% 28.6%, #1d4e8c 28.6% 42.9%, #2a6fc4 42.9% 57.2%,
    #3987e5 57.2% 71.5%, #63a2ea 71.5% 85.8%, #9ec5f4 85.8% 100%); }

  /* ---------- pipeline diagram ---------- */
  .flow { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; padding: 16px; }
  .step { background: var(--surface-2); border: 1px solid var(--hairline); border-radius: 9px; padding: 12px; position: relative; }
  .step .no { font: 600 10.5px/1 var(--mono); color: var(--accent); letter-spacing: .08em; }
  .step h4 { margin: 6px 0 5px; font-size: 13px; }
  .step p { margin: 0; color: var(--ink-2); font-size: 11.8px; line-height: 1.5; }
  .step:not(:last-child)::after {
    content: "→"; position: absolute; right: -13px; top: 42%; color: var(--baseline);
    font-size: 15px; z-index: 2;
  }
  .step .chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 7px; }
  .step .chips span { font: 500 10.5px/1.5 var(--mono); background: var(--surface); border: 1px solid var(--hairline); border-radius: 999px; padding: 1px 7px; color: var(--ink-2); }
  .step.hero { background: var(--accent-soft); border-color: var(--accent-border); }
  .flow-note { color: var(--muted); font-size: 12px; padding: 0 16px 14px; max-width: 90ch; }

  /* ---------- caveats ---------- */
  .caveats { border-left: 3px solid var(--baseline); background: var(--surface); border-radius: 0 var(--radius) var(--radius) 0; padding: 14px 18px; margin-top: 14px; box-shadow: var(--shadow); }
  .caveats h3 { margin: 0 0 6px; font-size: 13px; }
  .caveats ul { margin: 0; padding-left: 18px; color: var(--ink-2); font-size: 13px; }
  .caveats li { margin: 3px 0; }

  /* ---------- explorer: toolbar ---------- */
  .toolbar {
    display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: flex-end;
    background: var(--surface); border: 1px solid var(--hairline); border-radius: var(--radius);
    padding: 14px 16px; box-shadow: var(--shadow); margin-top: 18px;
  }
  .filter-group { display: flex; flex-direction: column; gap: 7px; }
  .filter-group > .gl { font-size: 10.5px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 600; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    border: 1px solid var(--hairline); background: var(--surface); color: var(--ink-2);
    padding: 5px 11px; border-radius: 999px; font-size: 12.5px; cursor: pointer;
    user-select: none; transition: border-color var(--t-fast), color var(--t-fast), background var(--t-fast);
    white-space: nowrap; font-family: inherit; line-height: 1.4;
  }
  .chip:hover { border-color: var(--baseline); color: var(--ink); }
  .chip.on { background: var(--accent-soft); border-color: var(--series); color: var(--ink); }
  .chip.on.pipe-basic { background: var(--basic-soft); border-color: var(--basic); }
  .toolbar .right { margin-left: auto; display: flex; gap: 16px; align-items: flex-end; }
  select, .reset {
    background: var(--surface); color: var(--ink); border: 1px solid var(--hairline);
    border-radius: 8px; padding: 7px 10px; font-size: 13px; cursor: pointer;
    font-family: inherit; transition: border-color var(--t-fast);
  }
  select:hover, .reset:hover { border-color: var(--baseline); }

  .count-pill { font-size: 12px; color: var(--ink-2); background: var(--surface-2); border: 1px solid var(--hairline); border-radius: 999px; padding: 3px 9px; }
  .seg { display: inline-flex; background: var(--surface-2); border: 1px solid var(--hairline); border-radius: 8px; padding: 2px; gap: 2px; }
  .seg button { background: transparent; border: 0; color: var(--ink-2); padding: 5px 12px; border-radius: 6px; font-size: 12.5px; cursor: pointer; font-weight: 600; font-family: inherit; }
  .seg button:hover { color: var(--ink); }
  .seg button.on { background: var(--surface); color: var(--ink); box-shadow: inset 0 0 0 1px var(--series); }

  /* ---------- leaderboard table ---------- */
  table.lb { width: 100%; border-collapse: collapse; }
  table.lb thead th {
    position: sticky; top: 0; background: var(--surface); z-index: 5;
    text-align: right; padding: 10px 12px; font-size: 10.5px; font-weight: 650;
    color: var(--ink-2); border-bottom: 1px solid var(--hairline); cursor: pointer;
    white-space: nowrap; text-transform: uppercase; letter-spacing: .05em;
    transition: color var(--t-fast), background var(--t-fast); user-select: none;
  }
  table.lb thead th[data-key]:hover { color: var(--ink); background: var(--surface-2); }
  table.lb thead th.lt { text-align: left; cursor: default; }
  table.lb thead th .grp { display: block; font-size: 9px; color: var(--muted); letter-spacing: .08em; }
  table.lb thead th.sorted { color: var(--accent); }
  table.lb thead th .arrow { opacity: .85; font-size: 9px; color: var(--accent); }
  table.lb tbody td { padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--surface-2); font-variant-numeric: tabular-nums; }
  table.lb tbody td.lt { text-align: left; }
  tr.run-row { cursor: pointer; transition: background var(--t-fast); }
  tr.run-row:hover { background: var(--accent-soft); }
  tr.run-row.open { background: var(--accent-soft); }
  tr.run-row:focus-visible { box-shadow: inset 0 0 0 2px var(--accent); }
  table.lb thead th:focus-visible { box-shadow: inset var(--focus); }
  .runtag { display: inline-flex; align-items: center; gap: 8px; white-space: nowrap; }
  /* Sticky Run column: keeps row identity visible while the metrics scroll. */
  table.lb thead th.lt:first-child,
  table.lb tbody tr.run-row > td.lt:first-child { position: sticky; left: 0; z-index: 6; background: var(--surface); }
  tr.run-row:hover > td.lt:first-child, tr.run-row.open > td.lt:first-child { background: var(--accent-soft); }
  .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
  .dot.basic { background: var(--basic); }
  .dot.advanced { background: var(--advanced); }
  .run-id { font-weight: 650; font-family: var(--mono); font-size: 12.5px; }
  .run-src { color: var(--muted); font-size: 11.5px; }
  .pill {
    display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 6px;
    background: var(--surface-2); border: 1px solid var(--hairline); color: var(--ink-2);
    font-family: var(--mono); margin: 1px 0;
  }
  .pill.basic { color: var(--basic-ink); border-color: var(--basic-border); }
  .pill.advanced { color: var(--accent); border-color: var(--accent-border); }
  .heat { border-radius: 5px; padding: 3px 8px; display: inline-block; min-width: 54px; font-weight: 600; font-family: var(--mono); font-size: 12px; transition: filter var(--t-fast); }
  .best-col { box-shadow: 0 0 0 1.5px var(--ink); }
  .heat.err { background: #fbeaea; color: var(--err); box-shadow: inset 0 0 0 1px var(--err); font-weight: 700; letter-spacing: .04em; cursor: help; }

  .detail td { background: var(--surface-2); padding: 0; }
  /* Sticky-left so the expanded breakdown reflows to the viewport instead of
     inheriting the table's scroll width on small screens. */
  .detail-inner { padding: 18px 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 26px; position: sticky; left: 0; max-width: calc(100vw - 64px); }
  .detail-inner h4 { margin: 0 0 12px; font-size: 11px; color: var(--ink-2); text-transform: uppercase; letter-spacing: .1em; }
  .bk-row { display: grid; grid-template-columns: 96px 1fr; align-items: center; gap: 10px; margin-bottom: 9px; }
  .bk-row .lab { font-size: 12px; color: var(--ink-2); text-transform: capitalize; }
  .bars { display: flex; flex-direction: column; gap: 5px; }
  .bar-line { display: grid; grid-template-columns: 64px 1fr 46px; align-items: center; gap: 8px; font-size: 11.5px; }
  .bar-line .mlab { color: var(--muted); text-align: right; }
  .track { height: 8px; background: var(--hairline); border-radius: 6px; overflow: hidden; }
  .fill { display: block; height: 100%; border-radius: 6px 4px 4px 6px; background: var(--series); }
  .bar-line .mval { color: var(--ink); text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono); }
  .bar-line .mval.err { color: var(--err); font-weight: 700; cursor: help; }
  .detail-meta { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: center; padding-top: 8px; border-top: 1px solid var(--hairline); margin-top: 4px; color: var(--ink-2); font-size: 12px; }
  .detail-meta b { color: var(--ink); font-weight: 600; font-family: var(--mono); font-size: 11.5px; }
  .viz-links { display: flex; gap: 8px; margin-left: auto; }
  .viz-links a { border: 1px solid var(--series); color: var(--accent); border-radius: 7px; padding: 4px 10px; font-size: 12px; font-weight: 600; }
  .viz-links a:hover { background: var(--accent-soft); text-decoration: none; }
  .viz-links a.off { border-color: var(--hairline); color: var(--muted); cursor: not-allowed; pointer-events: none; }

  /* ---------- compare ---------- */
  .compare { display: grid; grid-template-columns: 1fr; gap: 9px; padding: 16px; }
  .cmp-row { display: grid; grid-template-columns: 250px 1fr 60px; align-items: center; gap: 12px; }
  .cmp-row .cl { font-size: 12px; color: var(--ink-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--mono); }
  .cmp-row .ct { height: 16px; background: var(--surface-2); border-radius: 5px; overflow: hidden; }
  .cmp-row .cf { display: block; height: 100%; border-radius: 5px 4px 4px 5px; min-width: 2px; background: var(--series); }
  .cmp-row .cv { text-align: right; font-variant-numeric: tabular-nums; font-size: 12.5px; font-family: var(--mono); }

  /* ---------- constants ---------- */
  .constants { display: flex; flex-direction: column; gap: 10px; padding: 16px; }
  .const-set { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; padding: 11px 13px; border: 1px solid var(--hairline); border-radius: 9px; background: var(--surface-2); }
  .const-set .set-name { font-weight: 650; font-size: 13px; display: inline-flex; align-items: center; gap: 8px; }
  .const-set .set-name .badge { font-size: 11px; color: var(--ink-2); background: var(--surface); border: 1px solid var(--hairline); border-radius: 999px; padding: 2px 8px; font-weight: 500; }
  .const-set .set-sum { flex-basis: 100%; color: var(--muted); font-size: 12px; margin-top: 2px; }
  .const-set .sep { width: 1px; height: 18px; background: var(--hairline); margin: 0 4px; }
  .kv { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; }
  .kv .k { color: var(--muted); text-transform: uppercase; letter-spacing: .06em; font-size: 10px; font-weight: 600; }
  .kv .v { color: var(--ink); background: var(--surface); border: 1px solid var(--hairline); border-radius: 6px; padding: 2px 8px; font-family: var(--mono); font-size: 11.5px; }
  .const-set .none { color: var(--muted); font-size: 12px; font-style: italic; }
  .empty { padding: 40px; text-align: center; color: var(--muted); }
  .footnote { color: var(--muted); font-size: 12px; margin-top: 14px; line-height: 1.7; }

  /* ---------- research paper ---------- */
  article.paper {
    max-width: 760px; margin: 34px auto 0; background: var(--surface);
    border: 1px solid var(--hairline); border-radius: var(--radius); box-shadow: var(--shadow);
    padding: clamp(26px, 5vw, 56px);
    font-family: var(--serif); font-size: 17px; line-height: 1.68; color: var(--ink);
  }
  article.paper h1 { font-size: clamp(28px, 4vw, 36px); line-height: 1.2; font-weight: 600; margin: 0 0 18px; letter-spacing: .002em; }
  article.paper h2 { font-size: 23px; font-weight: 600; margin: 34px 0 10px; }
  article.paper h3 { font-size: 18.5px; font-weight: 600; margin: 26px 0 8px; }
  article.paper p { margin: 0 0 14px; }
  article.paper ul { margin: 0 0 14px; padding-left: 22px; }
  article.paper li { margin: 4px 0; }
  article.paper code { font-size: .82em; }
  /* Scroll shadows: a visible cue that columns continue off-screen (overlay
     scrollbars give none). background-attachment local/scroll fades the edge
     shadow out when that edge is fully scrolled into view. */
  .tbl-scroll, .heatwrap {
    background:
      linear-gradient(90deg, var(--surface) 30%, rgba(252,252,251,0)) 0 0,
      linear-gradient(270deg, var(--surface) 30%, rgba(252,252,251,0)) 100% 0,
      radial-gradient(farthest-side at 0 50%, rgba(11,11,11,.16), transparent) 0 0,
      radial-gradient(farthest-side at 100% 50%, rgba(11,11,11,.16), transparent) 100% 0;
    background-repeat: no-repeat;
    background-size: 44px 100%, 44px 100%, 12px 100%, 12px 100%;
    background-attachment: local, local, scroll, scroll;
  }
  .tbl-scroll::-webkit-scrollbar, .heatwrap::-webkit-scrollbar { height: 8px; }
  .tbl-scroll::-webkit-scrollbar-thumb, .heatwrap::-webkit-scrollbar-thumb { background: var(--baseline); border-radius: 4px; }
  article.paper .tbl-scroll { overflow-x: auto; margin: 18px 0; }
  article.paper table { border-collapse: collapse; width: 100%; font-family: var(--sans); font-size: 13px; }
  article.paper th { text-align: left; font-weight: 650; border-bottom: 2px solid var(--ink); padding: 7px 10px; white-space: nowrap; }
  article.paper td { border-bottom: 1px solid var(--hairline); padding: 7px 10px; font-variant-numeric: tabular-nums; }
  article.paper td:not(:first-child):not(:nth-child(2)):not(:nth-child(3)) { font-family: var(--mono); font-size: 12px; }
  .paper-meta { max-width: 760px; margin: 30px auto 0; color: var(--muted); font-size: 12.5px; display: flex; gap: 14px; align-items: baseline; padding: 0 4px; }

  @media (max-width: 960px) {
    .tiles { grid-template-columns: repeat(2, 1fr); }
    .strip { grid-template-columns: 1fr; }
    .flow { grid-template-columns: repeat(2, 1fr); }
    .step:not(:last-child)::after { content: ""; }
    .detail-inner { grid-template-columns: 1fr; }
    .cmp-row { grid-template-columns: 150px 1fr 54px; }
  }
  @media print {
    header.top, .toolbar, .footnote { display: none; }
    main section.panel { display: block !important; }
    body { background: #fff; }
    .card, article.paper, .tile, .panel-viz { box-shadow: none; }
  }
</style>
<script>
  /* Set the theme before first paint so there is no light-mode flash. */
  (function(){
    try {
      var t = localStorage.getItem("ragnosis-theme");
      if (t !== "light" && t !== "dark")
        t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      document.documentElement.dataset.theme = t;
    } catch (e) { document.documentElement.dataset.theme = "light"; }
  })();
</script>
</head>
<body>
<header class="top">
  <div class="masthead">
    <div class="wordmark">RAGnosis<span class="cross"> ·</span> benchmark report</div>
    <div class="sub">Retrieval experiments over a synthetic clinic database</div>
    <div class="stamp" id="stamp"></div>
    <button type="button" class="theme-toggle" id="themeToggle" aria-label="Switch color theme">
      <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.5M12 19v2.5M4.6 4.6l1.8 1.8M17.6 17.6l1.8 1.8M2.5 12H5M19 12h2.5M4.6 19.4l1.8-1.8M17.6 6.4l1.8-1.8"/></svg>
      <svg class="moon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a7 7 0 1 0 10.5 10.5z"/></svg>
      <span class="label">Dark</span>
    </button>
  </div>
  <nav class="tabs" role="tablist" aria-label="Report sections" id="tabs">
    <button role="tab" id="tab-overview" aria-controls="panel-overview" aria-selected="true" data-tab="overview">Overview</button>
    <button role="tab" id="tab-explorer" aria-controls="panel-explorer" aria-selected="false" data-tab="explorer">Results explorer</button>
    <button role="tab" id="tab-paper" aria-controls="panel-paper" aria-selected="false" data-tab="paper" hidden>Research paper</button>
  </nav>
</header>

<main class="wrap">
  <!-- ================= OVERVIEW ================= -->
  <section class="panel active" id="panel-overview" role="tabpanel" aria-labelledby="tab-overview">
    <div class="lede">
      <div class="eyebrow">Applied research · retrieval-augmented generation</div>
      <h2>Model choice helped. <em>Changing what the system retrieves</em> helped more.</h2>
      <p id="ledeText"></p>
    </div>

    <div class="tiles" id="tiles"></div>

    <div class="eyebrow">The experiment, run by run</div>
    <h2 class="sec" style="margin-top:0">Eight runs, two jumps</h2>
    <p class="sec-note">Each point is the best configuration of that run (highest answer overall).
      The two marked interventions changed <b>what was retrieved</b>, not which model ranked it —
      and they account for most of the gain.</p>
    <div class="strip" id="strip"></div>
    <p class="strip-caption" id="stripCaption"></p>

    <div class="eyebrow">Where the gains came from</div>
    <h2 class="sec" style="margin-top:0">Answer quality by question category</h2>
    <p class="sec-note">Judge scores on a 1–5 scale, per question category, at four milestones.
      Holistic (whole-corpus) questions did not move until precomputed rollup documents made their
      answers retrievable; numerical questions needed rollups to pass 4. Each category holds 3–5
      questions, so read swings as directional.</p>
    <div class="card">
      <div class="heatwrap" id="heatmap"></div>
      <div class="scale-legend"><span>1 (poor)</span><span class="ramp" aria-hidden="true"></span><span>5 (correct &amp; complete)</span></div>
    </div>

    <div class="eyebrow">The final architecture</div>
    <h2 class="sec" style="margin-top:0">How the final pipeline answers a question</h2>
    <p class="sec-note">Small documents are precise to <b>search</b>; large documents are complete to <b>answer from</b>.
      The pipeline searches one and answers from the other, and precomputed aggregates fill the gap
      neither can cover.</p>
    <div class="card">
      <div class="flow">
        <div class="step"><span class="no">01 · QUESTION</span><h4>A clinic-database question</h4><p>“Which doctor has the largest appointment load?” — a fact, a join, or an aggregate.</p></div>
        <div class="step"><span class="no">02 · REWRITE</span><h4>Query rewriting</h4><p>An LLM rewrites the question into a search query; retrieval runs with both versions and merges candidates.</p></div>
        <div class="step hero"><span class="no">03 · SEARCH</span><h4>Vector search over small children</h4><p>Only compact, identity-anchored documents are embedded:</p><div class="chips"><span>per-visit chunks</span><span>doctor / dept docs</span><span>rollup aggregates</span></div></div>
        <div class="step"><span class="no">04 · RERANK</span><h4>Rerank candidates</h4><p>A local reranker reorders the merged candidates (BGE cross-encoder or Jina listwise).</p></div>
        <div class="step hero"><span class="no">05 · EXPAND</span><h4>Small-to-Big expansion</h4><p>Winning children swap for their full parent patient record; rollups pass through verbatim with exact numbers.</p></div>
        <div class="step"><span class="no">06 · ANSWER</span><h4>Generate the answer</h4><p>The chat model answers from complete parent context instead of scattered chunks.</p></div>
      </div>
      <p class="flow-note">Rollup documents exist because aggregate answers (“how many”, “most”, “total”) appear in no single
        row — they are precomputed at indexing time as counts, rankings, and totals, so retrieval can find them as text.</p>
    </div>

    <div class="eyebrow">Reading the numbers</div>
    <div class="caveats" id="caveats"></div>
  </section>

  <!-- ================= EXPLORER ================= -->
  <section class="panel" id="panel-explorer" role="tabpanel" aria-labelledby="tab-explorer">
    <h2 class="sec">Every configuration, side by side</h2>
    <p class="sec-note">Filter, sort, and expand any run for its difficulty and category breakdown.
      Cells shade light→dark on each metric's own scale; the outlined cell is the best in view.</p>

    <div class="toolbar" id="toolbar"></div>

    <section style="margin-top:18px">
      <div class="card">
        <div class="card-head">
          <h3>Leaderboard</h3>
          <span class="count-pill" id="count"></span>
          <div class="seg" id="viewToggle">
            <button data-view="metrics" class="on">Metrics</button>
            <button data-view="percent">Percent + RAG Index</button>
          </div>
          <span class="hint">Click a row for its difficulty / category breakdown &amp; vector plots</span>
        </div>
        <div style="overflow:auto">
          <table class="lb" id="table"></table>
        </div>
      </div>
    </section>

    <section style="margin-top:18px">
      <div class="card">
        <div class="card-head">
          <h3>Compare on one metric</h3>
          <select id="cmpMetric" aria-label="Metric to compare"></select>
          <span class="hint">Sorted best → worst</span>
        </div>
        <div class="compare" id="compare"></div>
      </div>
    </section>

    <section style="margin-top:18px" id="constSection">
      <div class="card">
        <div class="card-head">
          <h3>Run sets &amp; fixed models</h3>
          <span class="hint">What each archived run changed, and the models it held constant</span>
        </div>
        <div class="constants" id="constants"></div>
      </div>
      <div class="footnote" id="foot"></div>
    </section>
  </section>

  <!-- ================= PAPER ================= -->
  <section class="panel" id="panel-paper" role="tabpanel" aria-labelledby="tab-paper">
    <article class="paper" id="paperBody"></article>
    <div class="paper-meta" id="paperMeta"></div>
  </section>
</main>

<script>
const DATA = /*__DATA__*/null;

/* ---------- metric definitions ---------- */
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
const COMPUTED = [
  {key:"retrievalPct", label:"Ret %", group:"Summary", dec:1, computed:true},
  {key:"answerPct", label:"Ans %", group:"Summary", dec:1, computed:true},
  {key:"ragIndex", label:"RAG Index", group:"Summary", dec:1, computed:true},
];
const COMPMAP = Object.fromEntries(COMPUTED.map(c => [c.key, c]));

/* ---------- helpers ---------- */
function esc(s){ return String(s ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch])); }
function avg(xs){ xs = xs.filter(x => x != null); return xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : null; }
function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
function metricOf(run, key) {
  const g = (RETRIEVAL.some(c => c.key === key)) ? run.retrieval : run.answer;
  const v = g.overall[key];
  return (v === undefined || v === null) ? null : v;
}
function isScoreKey(key){ const c = COLMAP[key]; return !!(c && c.domain && c.domain[1] === 5); }
function isAnswerError(key, v){ return v != null && isScoreKey(key) && (v > 5 || v < 1); }
function metricPct(run, key){
  const v = metricOf(run, key);
  if (v == null || isAnswerError(key, v)) return null;
  return clamp(v / COLMAP[key].domain[1] * 100, 0, 100);
}
function retrievalPct(run){ return avg(RETRIEVAL.map(c => metricPct(run, c.key))); }
function answerPct(run){ return avg(ANSWER.map(c => metricPct(run, c.key))); }
function ragIndex(run){ const r = retrievalPct(run), a = answerPct(run); if (r == null && a == null) return null; return 0.5*(r ?? a) + 0.5*(a ?? r); }
const COMPFN = { retrievalPct, answerPct, ragIndex };
function valueFor(run, key){ return COMPMAP[key] ? COMPFN[key](run) : metricOf(run, key); }
function activeColumns(){ return state.view === "percent" ? [...COMPUTED, ...COLS] : COLS; }
function shortEmb(m){ return String(m).replace(":latest","").replace(":l6-v2","-l6").replace("text-embedding-3-","te3-"); }
function shortRerank(m){ const s = String(m); return s.includes("/") ? s.split("/")[1] : s; }
function fmt(v, dec){ return v == null ? "—" : v.toFixed(dec); }
function fmtNative(v, dec){ if (v == null) return "—"; return Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(dec); }

/* Theme-dependent colors that JS draws directly (SVG chart + heat cells). Static
   chrome lives in CSS custom properties; these are the values CSS vars can't reach.
   Each ramp is QUANTIZED to 7 steps so every step's text color clears WCAG AA at
   12px — light goes light→dark (near-zero recedes to the paper), dark goes
   dark→light (near-zero recedes to the dark surface). Interpolated mid-blues would
   land in a band where neither ink clears 4.5:1, so continuous shading is unsafe. */
const PALETTE = {
  light: {
    grid: "#e1e0d9", axis: "#6d6b66", baseline: "#c3c2b7", jumpLabel: "#52514e",
    series: "#2a78d6", surface: "#fcfcfb", ink: "#0b0b0b",
    ramp: ["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"],
    darkStepFrom: 4,  // steps >= this are dark enough to need white text
  },
  dark: {
    grid: "#2c2c2a", axis: "#9a988f", baseline: "#45443f", jumpLabel: "#c3c2b7",
    series: "#3987e5", surface: "#1a1a19", ink: "#ffffff",
    ramp: ["#12243d","#173458","#1d4e8c","#2a6fc4","#3987e5","#63a2ea","#9ec5f4"],
    darkStepFrom: 0, whiteStepUpto: 3,  // steps 0..3 are dark (white text); 4..6 light (dark text)
  },
};
let THEME = "light";
function pal(){ return PALETTE[THEME]; }
function seqStep(t){ const r = pal().ramp; return Math.min(r.length - 1, Math.floor(clamp(t, 0, 1) * r.length)); }
function seqColor(t){ return pal().ramp[seqStep(t)]; }
function inkFor(t){
  const step = seqStep(t);
  return THEME === "dark"
    ? (step <= pal().whiteStepUpto ? "#ffffff" : "#0b0b0b")
    : (step >= pal().darkStepFrom ? "#ffffff" : "#0b0b0b");
}

/* ---------- story data (archived run-N sets only) ---------- */
function storyRuns(){
  const sets = (DATA.runSets || [])
    .filter(s => /^run-\d+$/.test(s.source))
    .sort((a,b) => parseInt(a.source.slice(4),10) - parseInt(b.source.slice(4),10));
  return sets.map(s => {
    let best = null, bestV = null;
    for (const r of DATA.runs){
      if (r.source !== s.source) continue;
      const v = metricOf(r, "overall");
      if (v == null || isAnswerError("overall", v)) continue;
      if (bestV == null || v > bestV){ best = r; bestV = v; }
    }
    return best ? { set: s, run: best } : null;
  }).filter(Boolean);
}
function topDeltaIdx(story, key, n){
  const deltas = story.map((s, i) => i === 0 ? null : ({ i, d: metricOf(s.run, key) - metricOf(story[i-1].run, key) }))
    .filter(x => x && x.d > 0).sort((a,b) => b.d - a.d);
  return deltas.slice(0, n).map(x => x.i).sort((a,b) => a-b);
}

/* ---------- overview: lede + tiles ---------- */
function renderLede(){
  const story = storyRuns();
  const lede = document.getElementById("ledeText");
  const archived = DATA.runs.filter(r => /^run-\d+$/.test(r.source)).length;
  const extra = DATA.runs.length - archived;
  const q = DATA.runs.length ? (DATA.runs[0].questions ?? "—") : "—";
  lede.textContent =
    `RAGnosis converts a relational clinic database (patients, doctors, appointments, records, ` +
    `prescriptions, billing) into retrievable context and measures what actually improves answers. ` +
    `${archived} configurations were evaluated on the same ${q}-question bank across ${story.length || "several"} ` +
    `archived runs — from a plain vector-search baseline to Small-to-Big retrieval with precomputed rollup documents` +
    (extra > 0 ? ` — plus ${extra} live result${extra === 1 ? "" : "s"} in the explorer.` : ".");
}
function bestRun(key){
  let top = null;
  for (const r of DATA.runs){
    const v = metricOf(r, key);
    if (v == null || isAnswerError(key, v)) continue;
    if (!top || v > top.v) top = { v, r };
  }
  return top;
}
function renderTiles(){
  const el = document.getElementById("tiles");
  const bm = bestRun("mrr"), bo = bestRun("overall");
  const story = storyRuns();
  const baseline = story.length ? story[0] : null;
  const validScore = v => (v != null && !isAnswerError("overall", v)) ? v : null;
  const hardBest = validScore(bo ? (bo.r.answer.byDifficulty.hard || {}).overall : null);
  const hardBase = validScore(baseline ? (baseline.run.answer.byDifficulty.hard || {}).overall : null);
  const tiles = [];
  if (bm) tiles.push({label:"Best retrieval MRR", value: bm.v.toFixed(3), meta: `${esc(bm.r.id)} · ${esc(shortEmb(bm.r.embedding_model))} · ${esc(shortRerank(bm.r.rerank_model))}`});
  if (bo) tiles.push({label:"Best answer overall", value: `${bo.v.toFixed(2)}<small> / 5</small>`, meta: `${esc(bo.r.id)} · ${esc(shortEmb(bo.r.embedding_model))} · ${esc(bo.r.chat_model)}`});
  if (hardBest != null) tiles.push({
    label:"Hard-question answer score",
    value: `${hardBest.toFixed(2)}<small> / 5</small>` + (hardBase != null ? `<span class="delta">▲ from ${hardBase.toFixed(2)}</span>` : ""),
    meta: hardBase != null ? "vs the best baseline configuration" : "hardest 10 questions"});
  tiles.push({label:"Study size", value:`${DATA.runs.length}<small> configs</small>`, meta:`${(DATA.runs[0]||{}).questions ?? "—"} questions · single pass each`});
  el.innerHTML = tiles.map(t => `
    <div class="tile">
      <div class="label">${t.label}</div>
      <div class="value">${t.value}</div>
      <div class="meta">${t.meta}</div>
    </div>`).join("");
}

/* ---------- overview: run strip (two small multiples; never dual-axis) ---------- */
function renderStrip(){
  const story = storyRuns();
  const strip = document.getElementById("strip");
  const caption = document.getElementById("stripCaption");
  strip.innerHTML = "";
  if (story.length < 2){ caption.textContent = ""; return; }
  const jumps = topDeltaIdx(story, "mrr", 2);
  // Jumps are remembered by run-set source so each panel can filter its own points
  // (a run set evaluated with --skip-retrieval has answer scores but no MRR).
  const jumpSrc = new Set(jumps.map(i => story[i].set.source));
  const mrrStory = story.filter(s => metricOf(s.run, "mrr") != null);
  const ansStory = story.filter(s => metricOf(s.run, "overall") != null);
  const left = linePanel(mrrStory, {key:"mrr", title:"Retrieval quality — MRR", unit:"Mean Reciprocal Rank · scale 0–1", domain:[0,1], ticks:[0,0.25,0.5,0.75,1], dec:3, jumpSrc, showJumpLabels:true});
  const right = linePanel(ansStory, {key:"overall", title:"Answer quality — overall", unit:"LLM-judge score · scale 1–5", domain:[1,5], ticks:[1,2,3,4,5], dec:2, jumpSrc, showJumpLabels:false});
  if (left) strip.appendChild(left);
  if (right) strip.appendChild(right);
  if (!left && !right){ caption.textContent = ""; return; }
  const jumpNames = jumps.map(i => `${esc(story[i].set.label)} (${esc(story[i].set.source)})`);
  caption.innerHTML = `Marked interventions — the two biggest retrieval jumps: <b>${jumpNames.join("</b> and <b>")}</b>. ` +
    `Hover or focus any point for that run's best configuration. Answer scores also reflect the generation model, ` +
    `which changed from run-3 onward; retrieval scores depend only on the embedding.`;
}
function linePanel(story, opt){
  if (story.length < 2) return null;
  const P = pal();
  const W = 620, H = 300, m = {t: 46, r: 18, b: 44, l: 46};
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = story.map((_, i) => m.l + (story.length === 1 ? iw/2 : i * iw / (story.length - 1)));
  const yOf = v => m.t + ih - ( (v - opt.domain[0]) / (opt.domain[1] - opt.domain[0]) ) * ih;
  const vals = story.map(s => metricOf(s.run, opt.key));

  let g = "";
  for (const tk of opt.ticks){
    const y = yOf(tk);
    g += `<line x1="${m.l}" y1="${y}" x2="${W-m.r}" y2="${y}" stroke="${P.grid}" stroke-width="1"/>` +
         `<text x="${m.l-8}" y="${y+3.5}" text-anchor="end" font-size="10.5" fill="${P.axis}" font-family="IBM Plex Mono, monospace">${tk}</text>`;
  }
  // x labels
  story.forEach((s, i) => {
    g += `<text x="${xs[i]}" y="${H-m.b+18}" text-anchor="middle" font-size="10.5" fill="${P.axis}" font-family="IBM Plex Mono, monospace">R${s.set.source.slice(4)}</text>`;
  });
  // intervention hairlines; labels staggered on two rows so adjacent jumps never collide
  let jumpRow = 0;
  const jumpIdx = [];
  story.forEach((s, j) => {
    if (!opt.jumpSrc.has(s.set.source)) return;
    jumpIdx.push(j);
    g += `<line x1="${xs[j]}" y1="${m.t-4}" x2="${xs[j]}" y2="${H-m.b}" stroke="${P.baseline}" stroke-width="1"/>`;
    if (opt.showJumpLabels){
      const lbl = `${s.set.label} ↓`;
      const half = lbl.length * 3.1;
      const cx = clamp(xs[j], m.l + half, W - m.r - half);
      g += `<text x="${cx}" y="${14 + jumpRow * 14}" text-anchor="middle" font-size="10.5" font-weight="600" fill="${P.jumpLabel}" font-family="system-ui, sans-serif">${esc(lbl)}</text>`;
      jumpRow++;
    }
  });
  // line
  const path = story.map((s, i) => `${i ? "L" : "M"}${xs[i].toFixed(1)},${yOf(vals[i]).toFixed(1)}`).join(" ");
  g += `<path d="${path}" fill="none" stroke="${P.series}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  // selective direct labels: first, last, and jump points
  const labelIdx = new Set([0, story.length - 1, ...jumpIdx]);
  story.forEach((s, i) => {
    const y = yOf(vals[i]);
    g += `<circle cx="${xs[i]}" cy="${y}" r="4.5" fill="${P.series}" stroke="${P.surface}" stroke-width="2"/>`;
    if (labelIdx.has(i)){
      const above = vals[i] >= (opt.domain[0]+opt.domain[1])/2;
      g += `<text x="${xs[i]}" y="${y + (above ? -10 : 18)}" text-anchor="middle" font-size="11" font-weight="600" fill="${P.ink}" font-family="IBM Plex Mono, monospace" paint-order="stroke" stroke="${P.surface}" stroke-width="3">${vals[i].toFixed(opt.dec)}</text>`;
    }
  });
  // crosshair
  g += `<line id="xh" x1="0" y1="${m.t}" x2="0" y2="${H-m.b}" stroke="${P.baseline}" stroke-width="1" visibility="hidden"/>`;

  const panel = document.createElement("div");
  panel.className = "panel-viz";
  panel.innerHTML = `<h3>${esc(opt.title)}</h3><div class="yunit">${esc(opt.unit)}</div>` +
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opt.title)} across runs">${g}</svg>` +
    `<div class="viz-tip" role="status"></div>`;

  const svg = panel.querySelector("svg");
  const tip = panel.querySelector(".viz-tip");
  const xh = svg.querySelector("#xh");
  function showTip(i, clientX, clientY){
    const s = story[i], v = vals[i];
    tip.replaceChildren();
    const val = document.createElement("div"); val.className = "v"; val.textContent = v.toFixed(opt.dec);
    const name = document.createElement("div"); name.textContent = `${s.set.source} — ${s.set.label}`;
    const cfg = document.createElement("div"); cfg.className = "k";
    cfg.textContent = `${shortEmb(s.run.embedding_model)} · ${s.run.chat_model} · ${shortRerank(s.run.rerank_model)}`;
    tip.append(val, name, cfg);
    tip.style.display = "block";
    const rect = panel.getBoundingClientRect();
    let lx = clientX - rect.left + 14, ly = clientY - rect.top - 10;
    if (lx + 240 > rect.width) lx = clientX - rect.left - 250;
    tip.style.left = Math.max(6, lx) + "px";
    tip.style.top = Math.max(6, ly) + "px";
    xh.setAttribute("x1", xs[i]); xh.setAttribute("x2", xs[i]);
    xh.setAttribute("visibility", "visible");
  }
  function hideTip(){ tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); }
  svg.addEventListener("pointermove", e => {
    const box = svg.getBoundingClientRect();
    const px = (e.clientX - box.left) * W / box.width;
    let bi = 0, bd = Infinity;
    xs.forEach((x, i) => { const d = Math.abs(x - px); if (d < bd){ bd = d; bi = i; } });
    showTip(bi, e.clientX, e.clientY);
  });
  svg.addEventListener("pointerleave", hideTip);
  // keyboard access: one focus stop per point
  story.forEach((s, i) => {
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", xs[i]); dot.setAttribute("cy", yOf(vals[i])); dot.setAttribute("r", 14);
    dot.setAttribute("fill", "transparent"); dot.setAttribute("tabindex", "0"); dot.setAttribute("role", "img");
    dot.setAttribute("aria-label", `${s.set.source}, ${s.set.label}: ${vals[i].toFixed(opt.dec)}`);
    dot.addEventListener("focus", () => {
      const box = svg.getBoundingClientRect();
      showTip(i, box.left + xs[i] * box.width / W, box.top + yOf(vals[i]) * box.height / H);
    });
    dot.addEventListener("blur", hideTip);
    svg.appendChild(dot);
  });
  return panel;
}

/* ---------- overview: category heatmap ---------- */
const CAT_ORDER = ["direct_fact","relationship","temporal","comparative","numerical","spanning","holistic"];
const CAT_LABEL = {direct_fact:"Direct fact", relationship:"Relationship", temporal:"Temporal", comparative:"Comparative", numerical:"Numerical", spanning:"Spanning", holistic:"Holistic"};
function milestoneRuns(){
  const story = storyRuns();
  if (!story.length) return [];
  const idx = new Set([0, ...topDeltaIdx(story, "mrr", 2), story.length - 1]);
  return [...idx].sort((a,b) => a-b).slice(0, 4).map(i => story[i]);
}
function renderHeatmap(){
  const el = document.getElementById("heatmap");
  const ms = milestoneRuns();
  if (!ms.length){ el.innerHTML = `<div class="empty">No archived run sets found.</div>`; return; }
  const cats = CAT_ORDER.filter(c => ms.some(m => (m.run.answer.byCategory || {})[c]));
  if (!cats.length){ el.innerHTML = `<div class="empty">No per-category results recorded in these runs.</div>`; return; }
  const head = ms.map(m =>
    `<th scope="col">${esc(m.set.label)}<span class="sub">${esc(m.set.source)} · ${esc(shortEmb(m.run.embedding_model))}</span></th>`).join("");
  const rows = cats.map(cat => {
    const anyQ = ms.map(m => (m.run.answer.byCategory || {})[cat]).find(b => b && b.questions != null);
    const cells = ms.map(m => {
      const b = (m.run.answer.byCategory || {})[cat];
      const v = b ? b.overall : null;
      if (v == null || isAnswerError("overall", v)) return `<td style="background:var(--surface-2);color:var(--muted)">—</td>`;
      const t = clamp((v - 1) / 4, 0, 1);
      return `<td style="background:${seqColor(t)};color:${inkFor(t)}" title="${esc(CAT_LABEL[cat])} · ${esc(m.set.source)}: ${v.toFixed(2)} / 5 over ${b.questions} question${b.questions===1?"":"s"}">${v.toFixed(2)}</td>`;
    }).join("");
    return `<tr><th scope="row" class="rowlab">${esc(CAT_LABEL[cat])}<span class="q">${anyQ ? anyQ.questions + "q" : ""}</span></th>${cells}</tr>`;
  }).join("");
  el.innerHTML = `<table class="heatmap"><thead><tr><th></th>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}

/* ---------- overview: caveats ---------- */
function renderCaveats(){
  const el = document.getElementById("caveats");
  const judges = [...new Set(DATA.runs.map(r => r.judge_model))].filter(j => j && j !== "—");
  const q = (DATA.runs[0] || {}).questions ?? "—";
  el.innerHTML = `<h3>How to read these numbers</h3><ul>
    <li>Each configuration ran <b>once</b> on ${esc(String(q))} questions — treat small gaps as directional, not significant.</li>
    <li>Answer scores come from a small local LLM judge (${judges.map(j => `<code>${esc(j)}</code>`).join(", ") || "see configs"}); absolute levels are calibration-dependent.</li>
    <li>Retrieval metrics depend only on the embedding model and compare cleanly across runs; answer scores also reflect the generation model, which changed between run-2 and run-3.</li>
    <li>Overview points use each run's <b>best</b> configuration (best-of-N selection), so they are upward-biased estimates.</li>
    <li>Full methodology, controls, and limitations: see the <a href="#paper" data-goto="paper">Research paper</a> tab.</li>
  </ul>`;
}

/* ---------- explorer state ---------- */
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
  tab: "overview",
};

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
    const ae = isAnswerError(state.sortKey, av), be = isAnswerError(state.sortKey, bv);
    if (ae !== be) return ae ? 1 : -1;
    return (av - bv) * state.sortDir;
  });
  return runs;
}

function chipBtn(value, on, cls, onclick){
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
  if (DATA.runSets && DATA.runSets.length >= 1){
    const rsWrap = document.createElement("div");
    rsWrap.className = "filter-group";
    rsWrap.innerHTML = `<span class="gl">Run set</span>`;
    const rsChips = document.createElement("div");
    rsChips.className = "chips";
    const options = [{source:"all", count: DATA.runs.length}, ...DATA.runSets];
    for (const opt of options){
      const label = opt.source === "all" ? "All sets" : opt.source;
      rsChips.appendChild(chipBtn(`${label} · ${opt.count}`, state.runSet === opt.source, "", () => {
        state.runSet = opt.source;
        renderExplorer();
      }));
    }
    rsWrap.appendChild(rsChips);
    tb.appendChild(rsWrap);
  }
  const groups = [
    {gl:"Pipeline", values: DATA.pipelines, set: state.pipelines, pipe:true},
    {gl:"Embedding model", values: DATA.embeddings, set: state.embeddings, emb:true},
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
      chips.appendChild(chipBtn(g.emb ? shortEmb(v) : v, g.set.has(v), cls, () => {
        if (g.set.has(v)) g.set.delete(v); else g.set.add(v);
        if (g.set.size === 0) g.values.forEach(x => g.set.add(x)); // never empty
        renderExplorer();
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
    renderExplorer();
  };
  right.appendChild(reset);
  tb.appendChild(right);
}

function cellFor(run, col){
  const raw = valueFor(run, col.key);
  if (raw == null) return {plain:true, text:"—"};
  if (isAnswerError(col.key, raw))
    return {err:true, text:"ERR", title:`Invalid answer score (expected 1–5): ${raw}`};
  let text, t;
  if (col.computed){ text = raw.toFixed(col.dec) + "%"; t = raw/100; }
  else {
    const pv = clamp(raw / col.domain[1] * 100, 0, 100);  // absolute, scale-based shade
    t = pv / 100;
    text = state.view === "percent" ? pv.toFixed(1) + "%" : fmtNative(raw, col.dec);
  }
  const outOfScale = !col.computed && (raw < col.domain[0] || raw > col.domain[1]);
  return {bg: seqColor(t), color: inkFor(t), text, title: outOfScale ? `Out of ${col.domain[0]}–${col.domain[1]} scale: ${raw}` : ""};
}

function renderTable(){
  const cols = activeColumns();
  const runs = sortedRuns();
  const table = document.getElementById("table");
  document.getElementById("count").textContent = `${runs.length} configuration${runs.length===1?"":"s"}`;
  if (!runs.length){ table.innerHTML = `<tbody><tr><td class="empty">No configurations match these filters.</td></tr></tbody>`; return; }

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
      const isBest = !cell.err && !cell.plain && raw != null && best[c.key] != null && Math.abs(raw - best[c.key]) < 1e-9;
      const titleAttr = cell.title ? ` title="${esc(cell.title)}"` : "";
      const cls = cell.err ? "err" : (isBest ? "best-col" : "");
      const styleAttr = (cell.err || cell.plain) ? "" : ` style="background:${cell.bg};color:${cell.color}"`;
      return `<td><span class="heat ${cls}"${styleAttr}${titleAttr}>${cell.text}</span></td>`;
    }).join("");
    const open = state.open === r.id;
    return `
      <tr class="run-row ${open?'open':''}" data-id="${esc(r.id)}" tabindex="0" role="button"
          aria-expanded="${open?'true':'false'}"
          aria-label="Run ${esc(r.name)}, ${esc(r.pipeline)}, ${esc(shortEmb(r.embedding_model))}, ${esc(r.chat_model)}. Toggle details.">
        <td class="lt"><span class="runtag"><span class="dot ${esc(r.pipeline)}"></span>
          <span><span class="run-id">#${esc(r.name)}</span> <span class="run-src">${esc(r.source)}</span></span></span></td>
        <td class="lt">
          <span class="pill ${esc(r.pipeline)}">${esc(r.pipeline)}</span>
          <span class="pill">${esc(shortEmb(r.embedding_model))}</span>
          <span class="pill">${esc(r.chat_model)}</span>
        </td>
        ${cells}
      </tr>
      ${open ? detailRow(r, cols.length) : ""}`;
  }).join("");

  table.innerHTML = head + `<tbody>${rows}</tbody>`;

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
  let html = `<h4>${esc(title)}</h4>`;
  for (const k of order){
    const b = buckets[k];
    html += `<div class="bk-row"><div class="lab">${esc(k)}${b.questions!=null?` · ${b.questions}q`:''}</div><div class="bars">`;
    for (const m of metricDefs){
      const v = b[m.key];
      const err = isAnswerError(m.key, v);
      const t = v==null ? 0 : (v - m.domain[0]) / (m.domain[1]-m.domain[0]);
      const fill = err
        ? `width:100%;background:var(--err)`
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
      <span>judge = <b>${esc(r.judge_model)}</b></span>
      ${r.pipeline==='advanced' ? `<span>rerank = <b>${esc(r.rerank_model)}</b></span>` : ''}
      <span class="viz-links">
        <a class="${v2?'':'off'}" ${v2?`href="${esc(v2)}" target="_blank"`:''}>2D vectors ↗</a>
        <a class="${v3?'':'off'}" ${v3?`href="${esc(v3)}" target="_blank"`:''}>3D vectors ↗</a>
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
  const runs = filteredRuns().filter(r => valueFor(r, c.key) != null).sort((a,b) => {
    const ae = isAnswerError(c.key, valueFor(a,c.key)), be = isAnswerError(c.key, valueFor(b,c.key));
    if (ae !== be) return ae ? 1 : -1;
    return valueFor(b,c.key) - valueFor(a,c.key);
  });
  const box = document.getElementById("compare");
  if (!runs.length){ box.innerHTML = `<div class="empty">No data.</div>`; return; }
  const valid = runs.map(r => valueFor(r,c.key)).filter(v => !isAnswerError(c.key, v));
  const max = valid.length ? Math.max(...valid) : 1;
  box.innerHTML = runs.map(r => {
    const v = valueFor(r, c.key);
    if (isAnswerError(c.key, v)){
      return `<div class="cmp-row">
        <span class="cl"><span class="dot ${esc(r.pipeline)}" style="display:inline-block;margin-right:6px"></span>#${esc(r.name)} · ${esc(shortEmb(r.embedding_model))} · ${esc(r.chat_model)}</span>
        <span class="ct"><span class="cf" style="width:100%;background:var(--err);opacity:.45"></span></span>
        <span class="cv" style="color:var(--err);font-weight:700" title="Invalid answer score (expected 1–5): ${esc(String(v))}">ERR</span>
      </div>`;
    }
    const w = max>0 ? (v/max*100) : 0;
    return `<div class="cmp-row">
      <span class="cl"><span class="dot ${esc(r.pipeline)}" style="display:inline-block;margin-right:6px"></span>#${esc(r.name)} · ${esc(shortEmb(r.embedding_model))} · ${esc(r.chat_model)}</span>
      <span class="ct"><span class="cf" style="width:${w.toFixed(1)}%"></span></span>
      <span class="cv">${v.toFixed(c.dec)}${suffix}</span>
    </div>`;
  }).join("");
}

function renderConstants(){
  const box = document.getElementById("constants");
  const sets = (DATA.runSets || []).filter(s => state.runSet === "all" || s.source === state.runSet);
  if (!sets.length){ box.innerHTML = `<div class="none">No run sets found.</div>`; return; }
  const ORDER = [
    ["PREPROCESS_MODEL","Preprocess"],["REWRITE_MODEL","Rewrite"],["JUDGE_MODEL","Judge"],
    ["RERANK_MODEL","Rerank"],["EMBEDDING_MODEL","Embedding"],
    ["EMBEDDING_MODEL_1","Embedding 1"],["EMBEDDING_MODEL_2","Embedding 2"],
  ];
  box.innerHTML = sets.map(s => {
    const c = s.constants;
    const kvs = c
      ? ORDER.filter(([k]) => c[k] != null).map(([k,lab]) =>
          `<span class="kv"><span class="k">${lab}</span><span class="v">${esc(c[k])}</span></span>`).join('<span class="sep"></span>')
      : `<span class="none">constants not recorded for this set</span>`;
    const sum = s.summary ? `<span class="set-sum">${esc(s.summary)}</span>` : "";
    return `<div class="const-set">
      <span class="set-name">${esc(s.source)} <span class="badge">${esc(s.label || "")}</span> <span class="badge">${s.count} run${s.count===1?'':'s'}</span></span>
      <span class="sep"></span>${kvs}${sum}</div>`;
  }).join("");
}

function renderExplorer(){
  renderToolbar();
  renderConstants();
  renderTable();
  renderCompare();
}

/* ---------- paper ---------- */
function renderPaper(){
  if (!DATA.research || !DATA.research.html) return;
  document.getElementById("tab-paper").hidden = false;
  document.getElementById("paperBody").innerHTML = DATA.research.html;
  document.getElementById("paperMeta").innerHTML =
    `<span>Rendered from <code>${esc(DATA.research.source)}</code></span>` +
    `<span>page generated ${esc(DATA.generatedAt)}</span>`;
}

/* ---------- tabs ---------- */
function selectTab(name){
  if (name === "paper" && document.getElementById("tab-paper").hidden) name = "overview";
  state.tab = name;
  document.querySelectorAll("nav.tabs button").forEach(b => {
    const on = b.dataset.tab === name;
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.querySelectorAll("main section.panel").forEach(p => {
    p.classList.toggle("active", p.id === `panel-${name}`);
  });
  if (history.replaceState) history.replaceState(null, "", `#${name}`);
}
function initTabs(){
  const tabs = [...document.querySelectorAll("nav.tabs button")];
  tabs.forEach(b => b.addEventListener("click", () => selectTab(b.dataset.tab)));
  document.getElementById("tabs").addEventListener("keydown", e => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const visible = tabs.filter(b => !b.hidden);
    const cur = visible.findIndex(b => b.dataset.tab === state.tab);
    const next = visible[(cur + (e.key === "ArrowRight" ? 1 : visible.length - 1)) % visible.length];
    next.focus(); selectTab(next.dataset.tab);
  });
  document.addEventListener("click", e => {
    const goto = e.target.closest("[data-goto]");
    if (goto){ e.preventDefault(); selectTab(goto.dataset.goto); window.scrollTo({top:0}); }
  });
  const applyHash = () => {
    const hash = (location.hash || "").replace("#", "");
    if (["overview","explorer","paper"].includes(hash) && hash !== state.tab) selectTab(hash);
  };
  window.addEventListener("hashchange", applyHash);
  applyHash();
}

/* ---------- theme ---------- */
function applyTheme(theme, persist){
  THEME = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = THEME;
  const btn = document.getElementById("themeToggle");
  if (btn){
    // The button offers the OTHER theme, so its label names the destination.
    btn.querySelector(".label").textContent = THEME === "dark" ? "Light" : "Dark";
    btn.setAttribute("aria-label", THEME === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }
  if (persist){ try { localStorage.setItem("ragnosis-theme", THEME); } catch (e) {} }
  // Re-render only the parts whose colors are drawn in JS (charts + heat cells);
  // all static chrome recolors via CSS custom properties automatically.
  if (DATA && DATA.runs.length){ renderStrip(); renderHeatmap(); renderTable(); }
}
function initTheme(){
  THEME = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  const btn = document.getElementById("themeToggle");
  if (btn){
    btn.querySelector(".label").textContent = THEME === "dark" ? "Light" : "Dark";
    btn.onclick = () => applyTheme(THEME === "dark" ? "light" : "dark", true);
  }
}

/* ---------- boot ---------- */
function boot(){
  initTheme();
  document.getElementById("stamp").textContent =
    DATA ? `${DATA.runs.length} configs · ${DATA.generatedAt}` : "";
  document.getElementById("foot").innerHTML =
    "Leaderboard cells shade light → dark by score on each metric's own scale (single-hue; darker is better). " +
    "Answer metrics are 1–5 (judge-scored); retrieval &amp; coverage are 0–1. " +
    "<b>RAG Index</b> = 50% retrieval-% + 50% answer-%, where each metric is taken as a percent of its scale. " +
    "Answer scores outside the 1–5 range are judge errors (shown as <span style='color:var(--err);font-weight:700'>ERR</span>) " +
    "and are excluded from every aggregate — the RAG Index, percentages, best-of tiles, and rankings. " +
    "Overall = question-weighted mean across difficulty buckets. " +
    "Regenerate with <code>uv run python -m app.build_benchmark</code>.";
  if (!DATA || !DATA.runs.length){
    document.getElementById("toolbar").innerHTML = `<div class="empty">No evaluation runs found. Run the evaluator, then rebuild this page.</div>`;
    document.getElementById("tiles").innerHTML = "";
    document.getElementById("ledeText").textContent = "No evaluation runs found yet — run the evaluator, then rebuild this page.";
    renderPaper();
    initTabs();
    return;
  }
  const toggle = document.getElementById("viewToggle");
  toggle.querySelectorAll("button").forEach(btn => {
    btn.onclick = () => {
      state.view = btn.dataset.view;
      toggle.querySelectorAll("button").forEach(b => b.classList.toggle("on", b === btn));
      state.sortKey = state.view === "percent" ? "ragIndex" : "overall";
      state.sortDir = -1;
      renderTable();
    };
  });
  renderLede();
  renderTiles();
  renderStrip();
  renderHeatmap();
  renderCaveats();
  renderExplorer();
  renderPaper();
  initTabs();
}
boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
