# RAGnosis

RAGnosis is a self-contained applied research project for studying how
retrieval-augmented generation behaves over a synthetic **clinic** database
(patients, doctors, departments, appointments, medical records, prescriptions and
billing). It was built to answer a practical question: which retrieval and
context-building techniques make a healthcare-style database easier for an LLM to
answer from?

The project starts from a plain vector-search baseline and evolves through query
rewriting, reranking, document enrichment, Small-to-Big retrieval, aggregate rollup
documents, and a final reranker ablation. The archived runs in `all-results/` record
each experiment with its config, metadata, visualizations, and evaluation scores. The
work was run mostly on local resources, so the results reflect a practical,
resource-constrained RAG study rather than an exhaustive search over every possible
model and setting.

Two pipelines are provided and can be compared head-to-head with built-in retrieval and
answer evals:

- **basic** — the baseline retrieval path used for controlled vector-search comparisons.
- **advanced** — query rewriting, dual retrieval, reranking, Small-to-Big parent
  expansion, optional document enrichment, and aggregate rollup documents.

Read [RESEARCH.md](RESEARCH.md) for the run-by-run research summary and the main
findings, or open [rag-benchmark.html](rag-benchmark.html) directly in a browser for
the interactive report — an overview of the experiment's progression with charts, a
sortable leaderboard of every configuration, and the research paper rendered inline
(regenerate it with `uv run python -m app.build_benchmark`). The pipeline code lives
under `app/`. The raw CSV dataset and the evaluation question bank live under `data/`.
Vector stores and eval results are generated under `app/`.

---

## Setup

Models are served by an OpenAI-compatible **Ollama** endpoint. The host is read from
`OLLAMA_HOST`, configured in `app/.env`:

```dotenv
OLLAMA_HOST=http://127.0.0.1:11434
```

A real shell environment variable overrides the file:

```bash
export OLLAMA_HOST=http://localhost:11434
```

OpenAI-hosted models (`gpt-*`, `o*`, `text-embedding-3-*`) route to the OpenAI API
using `OPENAI_API_KEY`. Run everything with `uv` from the repo root:

```bash
uv run python -m app.common.ingest basic
```

The raw CSV dataset is read only at chunking time. It is looked up under `data/dataset`
and can be pointed anywhere with `RAGNOSIS_DATASET_DIR`.

---

## Pipeline order

Each pipeline is built in three steps. Run them in order:

```
chunking  →  ingest  →  evaluator
(chunk)      (embed)     (score)
```

1. **chunking** — turns the dataset CSVs into retrievable child chunks
   (`app/<pipeline>/chunks.jsonl`) and answer-context parents
   (`app/<pipeline>/parents.jsonl`). Patient children are visit-level records with
   patient identity attached; parents hold complete patient records. Doctor,
   department, and aggregate rollup documents are also created.
2. **ingest** — embeds the stored chunks into a Chroma vector DB (`app/vector_db/<pipeline>`).
3. **evaluator** — runs retrieval + answer evals against the bundled question bank.

---

## Choosing models

Every command accepts per-run model overrides as `key=value` arguments.
**Precedence: CLI argument → `SELECTED_MODELS` default** (in `app/common/models.py`).

Both `key=value` and `--key=value` styles are accepted, and values may be quoted:

```bash
uv run python -m app.common.ingest basic embedding=all-minilm:l6-v2
uv run python -m app.common.ingest basic --embedding_model="all-minilm:l6-v2"
```

### Override keys

The embedding model, chat model, and advanced re-ranker are overridable from the CLI.
The preprocess, rewrite and judge models are constants (`PREPROCESS_MODEL`,
`REWRITE_MODEL`, `JUDGE_MODEL` in `app/common/models.py`).

| Key | Alias | Used by | Applies to |
| --- | --- | --- | --- |
| `embedding_model` | `embedding` | ingest | embedding the chunks |
| `chat_model` | `chat` | evaluator | generating answers |
| `rerank_model` | `rerank` | evaluator (advanced) | re-ranking retrieved chunks |

Unknown keys raise an error on ingest/chunking.

### Re-ranking (advanced)

The advanced pipeline re-ranks the merged retrieval results before expanding child
chunks to parent context. The re-ranker is selected with `RERANK_MODEL` or
`rerank=<repo-id>` and currently supports two local Hugging Face paths:

- **`jinaai/jina-reranker-v3`** (default) — a Qwen3-based listwise reranker that ranks
  all candidate chunks together via its `trust_remote_code` `.rerank()` API.
- **`BAAI/bge-reranker-v2-m3`** — a cross-encoder that scores each `(query, chunk)` pair
  directly via [FlagEmbedding](https://huggingface.co/BAAI/bge-reranker-v2-m3).

Weights download from the Hugging Face Hub on first use and cache under
`~/.cache/huggingface`. See `app/advanced/reranker.py` for model loading and device
selection.

```bash
uv run python -m app.evaluator advanced                                  # Jina reranker (default)
uv run python -m app.evaluator advanced rerank=BAAI/bge-reranker-v2-m3    # BGE cross-encoder
```

The resolved reranker is recorded in each run's `config.json` (`rerank_model`).

> An `embedding` change only takes effect at **ingest** time. Retrieval embeds queries
> with whatever model the vector DB was built with, so re-run `ingest` after changing
> the embedding model.

---

## Commands

### 1. Chunking

Both pipelines use deterministic recursive character splitting (no LLM). The same
module chunks either pipeline (passed as a positional argument); output goes to
`app/<pipeline>/chunks.jsonl`:

```bash
uv run python -m app.common.chunking basic
uv run python -m app.common.chunking advanced
```

Optional `chunk_size` / `chunk_overlap` overrides apply (defaults 1000 / 200).

**Optional — advanced preprocess.** Before chunking, the advanced pipeline can
enrich each document with an LLM-generated title + summary, written to
`app/advanced/enriched_documents.jsonl`:

```bash
uv run python -m app.advanced.preprocess
```

If that file exists, advanced `chunking` uses the enriched documents as parent
context; the visit-level child chunks are still built from the raw tables for precise
retrieval. It is a manual, standalone step (not part of multirun). Delete
`enriched_documents.jsonl` to go back to raw parent context.

### 2. Ingest

Embeds `app/<pipeline>/chunks.jsonl` into `app/vector_db/<pipeline>` (rebuilds the DB
each run). Pipeline is positional; accepts `embedding`:

```bash
uv run python -m app.common.ingest basic    embedding="all-minilm:l6-v2"
uv run python -m app.common.ingest advanced embedding="all-minilm:l6-v2"
```

### 3. Evaluator

Runs retrieval and answer evals for one pipeline. Mixes flags with model overrides.

```bash
# positional: basic | advanced
uv run python -m app.evaluator basic
uv run python -m app.evaluator advanced --limit 6 --k 10

# with a chat-model override
uv run python -m app.evaluator basic chat_model="deepseek-r1:1.5b"
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--limit N` | all | Evaluate at most N questions, stratified across difficulties. |
| `--k N` | 10 | Top-k documents retrieved per question. |
| `--skip-answers` | off | Retrieval metrics only; skips answer generation + judging. |

### 4. Multirun (full pipeline, batch)

Describe many runs in a JSONL file and execute them one by one. **Each run executes
the full pipeline — chunk → ingest → evaluate — rebuilding the vector DB** with that
run's models, so it's a self-contained, reproducible experiment. Each run saves to
its own `app/results/<n>/`.

```bash
uv run python -m app.multirun                 # uses app/multirun.jsonl
uv run python -m app.multirun path/to/runs.jsonl
```

Each line is one JSON object. Lines starting with `//` or `#` and blank lines are ignored.

```jsonl
{"name": "basic-default", "pipeline": "basic", "limit": 6, "k": 10}
{"name": "advanced-minilm", "pipeline": "advanced", "embedding_model": "all-minilm:l6-v2", "chat_model": "deepseek-r1:1.5b", "limit": 6}
{"name": "basic-eval-only", "pipeline": "basic", "stages": ["evaluate"], "limit": 6}
```

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `pipeline` | yes | — | `basic` or `advanced`. |
| `name` | no | pipeline | Label saved into `config.json` and printed. |
| `stages` | no | all three | Subset/order of `["chunk","ingest","evaluate"]`. |
| `embedding_model` | no | default | Embedding used at ingest + query time. |
| `chat_model` | no | default | Answer-generation model. |
| `chunk_size` / `chunk_overlap` | no | 1000 / 200 | Recursive chunking, in chars. |
| `rerank_model` | no | `BAAI/bge-reranker-v2-m3` | Advanced re-ranker: the bge cross-encoder, or an `LLM_MODELS` tag. Alias: `rerank`. |
| `limit` | no | all | Max questions, stratified across difficulties. |
| `k` | no | 10 | Top-k documents retrieved. |
| `include_answers` | no | true | Set `false` for retrieval-only runs. |

`embedding_model`, `chat_model` and (advanced) `rerank_model` vary per run;
preprocess/rewrite/judge models are constants. Use `"stages": ["evaluate"]` to re-score against an existing DB
without rebuilding, or `["chunk","ingest"]` to (re)build a DB without evaluating. A
failing run is reported and skipped so the rest of the batch still completes; a final
summary lists where each run was saved.

> **Cost:** the default flow re-chunks and re-embeds the whole dataset per run.
> Basic chunking is instant but embedding the chunks takes a couple of minutes.

Each run is saved to an auto-incremented folder under `app/results/<n>/`:

- `config.json` — pipeline, resolved models, `k`, `limit`, question count.
- `evals.json` — retrieval and answer metrics broken down by difficulty
  (`retrieval`, `answer`) and by question category (`retrieval_by_category`,
  `answer_by_category`); `NaN` is written as `null`.

---

## Available models

Local Ollama models are referenced by their Ollama tag; any tag pulled on the server works.
Named OpenAI models (`gpt-*`, `o*`) route to the OpenAI API using `OPENAI_API_KEY`.

- **Embeddings:** `all-minilm:l6-v2`, `bge-large:latest`, `qwen3-embedding:latest`, `text-embedding-3-small`, `text-embedding-3-large`
- **LLMs:** `deepseek-r1:1.5b`, `gpt-oss:20b`, `gemma4:e4b`, or any other tag on the server

Inspect the current defaults at any time:

```bash
uv run python -c "from app.common.models import selected_pipeline_models as s; print(s('basic')); print(s('advanced'))"
```

Defaults live in `SELECTED_MODELS` in `app/common/models.py`.

---

## Layout

```
.
├── pyproject.toml
├── data/
│   ├── dataset/             # raw clinic CSVs (read at chunking time)
│   └── eval-questions.json  # evaluation question bank
└── app/
    ├── .env                 # OLLAMA_HOST and other config
    ├── evaluator.py         # retrieval + answer evals (single run)
    ├── multirun.py          # batch runner over multirun.jsonl
    ├── multirun.jsonl       # batch run definitions
    ├── common/              # shared: models, chat, embeddings, rag, paths, documents, chunks, chunking, ingest
    ├── basic/               # basic implementation (retrieval + answer)
    ├── advanced/            # advanced preprocess + implementation (rewrite + rerank); reranker.py = BGE/Jina local rerankers
    ├── vector_db/           # Chroma stores (created by ingest)
    └── results/             # numbered eval runs (created by evaluator)
```
