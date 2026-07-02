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
<meta name="theme-color" content="#2a78d6" />
<title>RAGnosis · Benchmark report</title>
<meta name="description" content="RAGnosis is an applied research benchmark for retrieval-augmented generation over a synthetic clinic database. It compares query rewriting, reranking, Small-to-Big retrieval, document enrichment, and aggregate rollups across 30 configurations with built-in retrieval and answer evals." />
<meta name="keywords" content="RAG, retrieval-augmented generation, RAG benchmark, reranking, query rewriting, Small-to-Big retrieval, vector search, LLM evaluation, MRR, NDCG" />
<meta name="author" content="Samarth M N" />
<meta name="robots" content="index, follow" />
<link rel="canonical" href="https://rag-nosis.vercel.app/" />
<!-- Open Graph / Facebook -->
<meta property="og:type" content="website" />
<meta property="og:site_name" content="RAGnosis" />
<meta property="og:title" content="RAGnosis · Benchmark report" />
<meta property="og:description" content="An applied research benchmark for retrieval-augmented generation over a synthetic clinic database — query rewriting, reranking, Small-to-Big retrieval, and enrichment compared across 30 configurations." />
<meta property="og:url" content="https://rag-nosis.vercel.app/" />
<meta property="og:image" content="https://rag-nosis.vercel.app/logo.png" />
<meta property="og:image:width" content="1400" />
<meta property="og:image:height" content="1400" />
<meta property="og:image:alt" content="RAGnosis logo" />
<!-- Twitter -->
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="RAGnosis · Benchmark report" />
<meta name="twitter:description" content="An applied research benchmark for retrieval-augmented generation over a synthetic clinic database — query rewriting, reranking, Small-to-Big retrieval, and enrichment compared across 30 configurations." />
<meta name="twitter:image" content="https://rag-nosis.vercel.app/logo.png" />
<meta name="twitter:image:alt" content="RAGnosis logo" />
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Dataset","name":"RAGnosis Benchmark","description":"An applied research benchmark for retrieval-augmented generation over a synthetic clinic database, comparing query rewriting, reranking, Small-to-Big retrieval, document enrichment, and aggregate rollups across 30 configurations with retrieval and answer evals.","url":"https://rag-nosis.vercel.app/","keywords":["retrieval-augmented generation","RAG benchmark","reranking","vector search","LLM evaluation"],"creator":{"@type":"Person","name":"Samarth M N"},"license":"https://github.com/samarthmn/RAGnosis","isAccessibleForFree":true}
</script>
<link rel="icon" type="image/png" href="data:image/png;base64,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" />
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
  .gh-link {
    margin-left: 16px; display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface); color: var(--ink-2); border: 1px solid var(--hairline);
    border-radius: 999px; padding: 5px 11px; font: 600 12px/1 var(--sans);
    text-decoration: none; transition: border-color var(--t-fast), color var(--t-fast);
  }
  .gh-link:hover { border-color: var(--baseline); color: var(--ink); }
  .gh-link svg { width: 14px; height: 14px; }
  .masthead .stamp + .gh-link { margin-left: 16px; }
  .gh-link + .theme-toggle { margin-left: 10px; }
  .theme-toggle svg { width: 14px; height: 14px; }
  .theme-toggle .moon { display: none; }
  html[data-theme="dark"] .theme-toggle .sun { display: none; }
  html[data-theme="dark"] .theme-toggle .moon { display: inline; }
  .masthead { display: flex; align-items: baseline; gap: 14px; padding: 18px 24px 0; max-width: 1180px; margin: 0 auto; }
  .brand-logo { width: 32px; height: 32px; border-radius: 8px; align-self: center; flex-shrink: 0; box-shadow: 0 1px 3px rgba(10,22,40,.28); }
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
    <img class="brand-logo" src="data:image/png;base64,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" alt="RAGnosis logo" width="32" height="32" />
    <div class="wordmark">RAGnosis<span class="cross"> ·</span> benchmark report</div>
    <div class="sub">Retrieval experiments over a synthetic clinic database</div>
    <div class="stamp" id="stamp"></div>
    <a class="gh-link" href="https://github.com/samarthmn/RAGnosis" target="_blank" rel="noopener" aria-label="View RAGnosis on GitHub">
      <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
      <span>GitHub</span>
    </a>
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
<!-- Vercel Web Analytics -->
<script>
  window.va = window.va || function () { (window.vaq = window.vaq || []).push(arguments); };
</script>
<script defer src="/_vercel/insights/script.js"></script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
