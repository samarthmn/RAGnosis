# RAGnosis: An Applied Research Project on Healthcare RAG

## Abstract

RAGnosis is an applied research project on retrieval-augmented generation over a synthetic healthcare database. The goal was simple: start with a plain RAG baseline, measure where it fails, then keep changing the retrieval pipeline until the system could answer harder clinic-database questions with better context.

The final architecture is much stronger than the first baseline. The best baseline Basic run reached 0.406 retrieval MRR and 2.856 answer overall. The later Small-to-Big plus rollup-document pipeline spanned 0.753 to 0.792 retrieval MRR and 4.44 to 4.62 answer overall across its four embedding–reranker combinations, peaking at 4.622. The main lesson is that model choice helped, but data representation helped more. The biggest gains came from changing what the system retrieved: small visit-level chunks for precise matching, full parent records for answer context, and precomputed aggregate documents for questions whose answers do not exist in any single row.

## Why I built this

I started RAGnosis to actually apply the RAG techniques I had only been reading about — query rewriting, reranking, chunking strategies, Small-to-Big retrieval — on a dataset messy enough to make them matter. It was meant to be a learning exercise, nothing more: build a baseline, try a technique, see if it moved the needle.

Somewhere around the third or fourth run it turned into a game. Each run posted a number, the next run tried to beat it, and questions like "can I get the holistic answers off the floor?" became genuinely fun to chase. Optimising retrieval MRR and answer quality stopped feeling like homework and started feeling like a puzzle with a scoreboard — which is most of why there are eight runs here instead of two.

So this is less a paper with a hypothesis I set out to prove, and more a project that began as practice, became a game of moving numbers, and ended up as a small but honest study of what actually makes a healthcare-style database answerable. It was, in the end, just a really nice project to build.

## Research Question

The project asks a practical RAG question: how should a relational healthcare database be converted into retrievable context so that an LLM can answer factual, multi-table, and aggregate questions reliably?

The database is synthetic, not clinical data. It contains patients, doctors, departments, appointments, medical records, prescriptions, and billing. That makes it useful for RAG experiments because the answers often span several tables. A simple vector search can find a patient name or a doctor profile, but it struggles when the question asks for a status count, a total billed amount, or a visit summary that combines diagnosis, medication, doctor, follow-up, and payment details.

## Dataset and Evaluation

The dataset contains 1,200 patients, 120 doctors, 12 departments, 2,200 appointments, 700 medical records, 400 prescriptions, and 368 billing rows. The evaluation set has 30 questions: 10 easy, 10 medium, and 10 hard. The questions cover direct facts, relationships, temporal details, numerical answers, comparisons, spanning questions (whose answers join several tables for one entity), and holistic questions (aggregates over the whole corpus).

Retrieval was measured with MRR (Mean Reciprocal Rank), nDCG (normalized Discounted Cumulative Gain), and context keyword coverage. All three are keyword-substring proxies: each question carries expected keywords, each keyword is scored separately — a retrieved document counts as relevant to a keyword when it contains it as a substring — and the per-keyword scores are averaged per question, so these numbers are not directly comparable to gold-labeled IR benchmarks. nDCG tracked MRR closely (within about 0.04 in every run) and is omitted from the tables below; full values are in each run's `evals.json`. Answer quality was judged with accuracy, completeness, and relevance on a 1 to 5 scale, and the reported answer overall is the unweighted mean of those three judge scores. Deterministic answer keyword coverage — the fraction of expected keywords that appear in the generated answer — is tracked separately as a model-free second signal. All runs share a fixed retrieval configuration — top-k of 10 and recursive character chunking with chunk size 1,000 and overlap 200 — with chunks embedded into a Chroma vector store; the Advanced-pipeline configurations additionally share `gemma4:e4b` as the query-rewrite model (the Basic pipeline does no rewriting). This is not a clinical benchmark. It is a focused benchmark for whether the retrieval layer brings the right synthetic database evidence into the prompt.

One caveat matters for reading the cross-run numbers. The retrieval metrics (MRR, nDCG, context keyword coverage) depend only on the embedding model, so they are directly comparable across all runs. The answer-quality scores also depend on the generation model, which was not held constant across the whole progression. The best Run 1 and Run 2 configurations generate answers with the local `gemma4:e4b` model, while Runs 3 to 8 generate with `gpt-oss:20b`. So the answer-overall jump at the Run 2 to Run 3 boundary reflects an embedding change, document enrichment, and a generation-model change together, not retrieval alone. The cleanest embedding-controlled evidence for the representation thesis is the Run 5 to Run 7 retrieval chain, which holds `text-embedding-3-small` and `gpt-oss:20b` fixed while MRR rises from 0.471 to 0.614 to 0.779. Answer scoring uses one small local judge, `gemma4:e4b`, for every run; in Runs 1 and 2 that same model both generates and judges the best configurations, so those baseline answer scores are effectively self-graded.

## Compute and Budget Constraints

Most of the work was run locally, which made the experiments easier to reproduce but expensive in wall-clock time. Repeated chunking, embedding, reranking, answer generation, and LLM judging all had to be run across many configurations. Some steps were especially slow because local models, local vector rebuilds, and in-process rerankers were used instead of larger hosted infrastructure.

That constraint shaped the research. The experiment could compare the main retrieval ideas, but it could not exhaust every model, chunking setting, reranker, prompt variant, or evaluation size. With more compute and a larger API budget, the next version could run wider sweeps, use stronger hosted generation and judging models, test larger evaluation sets, and repeat runs to measure variance. The results here should be read as a strong local-first study, not as the ceiling for this approach.

## Experimental Progression

### Run 1: Basic RAG baseline

Run 1 used the simplest pipeline: build entity documents, split them with recursive character chunking, embed the chunks, retrieve the top-k similar chunks, and answer from that context. There was no query rewriting and no reranking. This established the baseline across local embedding and chat model combinations.

The baseline handled direct facts reasonably well, but it was weak on aggregate and multi-table questions. In the best Run 1 result, direct-fact answers scored 4.917 overall, while numerical answers scored 2.000 and holistic answers scored 1.889. That gap became the main target for the rest of the work.

### Run 2: Query rewriting and cross-encoder reranking

Run 2 added the Advanced pipeline: rewrite the question into a search query, retrieve with both the original and rewritten query, merge candidates, then rerank with `BAAI/bge-reranker-v2-m3` before answering. This improved retrieval and answer quality over the baseline, but it did not solve the core representation problem. If the needed answer was spread across chunks, or if the answer was an aggregate count, reranking could only reorder imperfect candidates.

### Runs 3 to 5: OpenAI embeddings and document enrichment

Runs 3 and 4 tested LLM-enriched documents with `text-embedding-3-small` and `text-embedding-3-large`. Each document received a generated title and summary (produced by `gpt-oss:20b` via `app/advanced/preprocess.py`) before chunking. Run 5 acted as the control: the same Advanced pipeline and OpenAI embeddings, but without enrichment.

The enrichment results were mixed. With `text-embedding-3-small`, the plain control was slightly better on answer overall than the enriched version. With `text-embedding-3-large`, enrichment helped more. Even then, the scores stayed in the low 3s. Enrichment gave the embedder more searchable text, but it did not change the unit of retrieval enough to fix hard spanning and aggregate questions.

### Run 6: Small-to-Big retrieval

Run 6 changed the retrieval unit. Instead of embedding only large patient documents or arbitrary character chunks, the system embedded small visit-level child documents. Each child kept the patient identity in its text and pointed back to a full parent patient document. At query time, the system searched the smaller child documents, reranked those precise matches, and expanded the winners back to full parent records before answering.

This was the first major jump. Retrieval MRR rose to 0.614, and answer overall rose to 4.044. Spanning questions improved sharply because the model received complete patient context after retrieval had found the right visit. Holistic aggregate questions did not improve — they dipped to 1.444, slightly below the best baseline, because no parent document contains a corpus-wide count. This also fixed an important failure mode in the earlier chunking: later chunks in a long patient record could lose the patient identity, making them hard to retrieve for name-based questions.

### Run 7: Rollup documents

Run 7 added precomputed rollup documents for aggregate questions. These documents stored counts, rankings, totals, and summaries that cannot be found in one raw visit record. Examples include patients by city, bills by payment status, doctors by appointment load, and patients ranked by total billed amount.

This was the second major jump. The best Run 7 result reached 0.779 retrieval MRR, 0.818 context keyword coverage, 4.622 answer overall, and 4.500 hard answer overall. Numerical answer quality rose from 3.333 in Run 6 to 4.667, and holistic answers rose from 1.444 to 5.000, while spanning answers were unchanged at 4.200 — rollups add aggregate facts rather than per-entity context. A question like "Which doctor has the largest appointment load?" is not a lookup problem. It is an aggregate computation. Once the aggregate exists as retrievable text, RAG has a fair chance to answer it.

### Run 8: Jina reranker ablation

Run 8 kept the Run 7 architecture and swapped the reranker from BGE to `jinaai/jina-reranker-v3`, testing what reranker quality adds after the data representation problem had mostly been addressed.

The result split by embedding. Holding the embedding fixed, Jina improved retrieval with `text-embedding-3-large` — MRR rose from 0.770 to 0.792 and context keyword coverage from 0.820 to 0.831, the highest in the archive — but regressed it with `text-embedding-3-small`, where MRR fell from 0.779 to 0.753. The best Run 8 answer overall was 4.522, slightly below the best Run 7 score but still far ahead of the early runs. The takeaway is that the listwise reranker can improve retrieval in the stronger embedding setup, but the architecture change from Runs 6 and 7 mattered more than the reranker swap itself.

## Results Summary

Rows below use the best answer-overall result from each run group — a best-of-N selection (N=9 for Runs 1 and 2, which swept three embeddings by three chat models; N=2 for Runs 3 to 8), so values are upward-biased relative to a single preregistered configuration.

| Run | Main change | Representative result | Retrieval MRR | Context keyword coverage | Answer overall | Hard answer overall |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Basic vector-search baseline | `all-results/run-1/results/8` | 0.406 | 0.547 | 2.856 | 2.200 |
| 2 | Query rewriting + BGE reranking | `all-results/run-2/results/5` | 0.425 | 0.562 | 3.056 | 2.867 |
| 3 | Enriched docs, `text-embedding-3-small` | `all-results/run-3/results/2` | 0.478 | 0.602 | 3.178 | 2.767 |
| 4 | Enriched docs, `text-embedding-3-large` | `all-results/run-4/results/2` | 0.488 | 0.607 | 3.300 | 2.567 |
| 5 | No-enrichment control with OpenAI embeddings | `all-results/run-5/results/1` | 0.471 | 0.572 | 3.200 | 2.900 |
| 6 | Small-to-Big retrieval | `all-results/run-6/results/1` | 0.614 | 0.671 | 4.044 | 3.433 |
| 7 | Rollup aggregate documents | `all-results/run-7/results/1` | 0.779 | 0.818 | 4.622 | 4.500 |
| 8 | Jina reranker on the rollup pipeline | `all-results/run-8/results/2` | 0.792 | 0.831 | 4.522 | 4.467 |

The category breakdown (three of the seven categories) shows where the gains came from; each category mean covers only 3 to 5 questions, so read the swings as directional.

| Experiment | Numerical answer | Holistic answer | Spanning answer |
| --- | ---: | ---: | ---: |
| Best baseline, Run 1 | 2.000 | 1.889 | 2.000 |
| Small-to-Big, Run 6 | 3.333 | 1.444 | 4.200 |
| Rollups, Run 7 | 4.667 | 5.000 | 4.200 |
| Jina final, Run 8 | 4.333 | 5.000 | 4.133 |

## Discussion

The experiments point to three findings.

First, retrieval quality depends heavily on document shape. Recursive chunking over large entity records is easy to build, but it can cut away the identity and relational context that healthcare questions depend on. Small-to-Big retrieval worked because it separated matching from answering: small child chunks improved precision, while parent documents preserved enough context for synthesis.

Second, aggregate questions need aggregate documents. RAG cannot reliably retrieve a count that was never written down. Rollup documents made totals and rankings explicit, which turned many hard questions from implicit computation into retrieval.

Third, reranking is useful after the corpus is shaped well. Query rewriting and BGE reranking improved the baseline, and Jina improved the strongest large-embedding retrieval run. But rerankers did not compensate for missing parent context or missing aggregates. The retrieval corpus had to be made answerable first.

## Limitations

The dataset is synthetic and should not be treated as a medical benchmark. The evaluation set is intentionally small at 30 questions, so the scores are best read as directional evidence. Each configuration was run once, with no repeats, error bars, or significance testing, so small cross-run gaps such as 4.622 versus 4.522 answer overall should be treated as directional rather than statistically distinguishable. The per-category breakdowns are computed over only 3 to 5 questions each (holistic has 3), so each one-point change in a single answer's judge score moves a category mean by 0.2 to 0.33 points — a fully flipped answer can move a small category by more than a point — and large category swings are illustrative rather than robust. Answer judging uses one small local LLM judge (`gemma4:e4b`), so the absolute 1 to 5 levels are calibration-dependent and best compared within this study rather than against external benchmarks; deterministic keyword coverage gives a second, model-free signal. The experiments were also bounded by local hardware and budget, so they favor careful comparisons over exhaustive search. Finally, the model choices carry different runtime and licensing tradeoffs; for example, `jinaai/jina-reranker-v3` is a stronger reranker candidate but has non-commercial licensing constraints.

## Conclusion

RAGnosis improved because the project moved from model tinkering to context engineering. The early system retrieved chunks from flattened healthcare records. The final system retrieves precise visit-level evidence, expands it to full patient context, and adds rollup documents for aggregate questions. With the embedding and generation model held fixed, that design raised retrieval MRR from 0.471 to 0.779 (Runs 5 to 7); end to end — embedding upgrade included — MRR rose from 0.406 to 0.792, and answer quality rose from the high 2s to the mid 4s.

For this dataset, a practical recipe emerged: rewrite and rerank queries, retrieve small visit-level children, answer from larger parents, and make aggregate facts explicit before retrieval. Stronger embeddings helped retrieval MRR, though the best answer overall came from the smaller `text-embedding-3-small` — one more sign that representation, not model size, did the heavy lifting.

## Sources

- Run descriptions and metadata: `all-results/run-1/config.json` through `all-results/run-8/config.json`
- Evaluation outputs: `all-results/run-*/results/*/evals.json`
- Dataset and question bank: `data/README.md`, `data/eval-questions.json`
- Final implementation: `app/common/chunking.py`, `app/common/rag.py`, `app/advanced/implementation.py`, `app/advanced/preprocess.py`, `app/advanced/reranker.py`, `app/evaluator.py`
- Git history reviewed: `b729889`, `86b0be5`, `2bd4272`, `29942d0`, `3c3e90d`, `aff6ece`, `751a857`, `0ef13cd`, `9b28795`, `787a066`
