"""Project a pipeline's Chroma vectors to 2D/3D with t-SNE and save interactive plots.

Reads the persisted vector store at ``app/vector_db/<pipeline>`` (no embedding model
needed — the vectors are already stored) and writes ``visualise_2d.html`` and
``visualise_3d.html`` into a target directory (typically a results run folder, next to
config.json/evals.json). Run standalone with::

    uv run python -m app.common.visualize basic app/results/1

Each plot keeps a single t-SNE layout but offers a dropdown to recolour the points by
several schemes (document type, type + visit count, type + chunk section, chunk size),
so the dominant patient cluster can be broken down along different attributes.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from app.common.embeddings import _reset_chroma_cache
from app.common.ingest import db_path_for

# Distinct, colour-blind-friendly palette cycled across category values.
_PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

Metadata = dict[str, Any]


def _load_vectors(
    pipeline: str, db_path: Path | None = None
) -> tuple[np.ndarray, list[Metadata], list[str]]:
    """Return (vectors, metadatas, documents) from a pipeline's persisted Chroma store."""
    import chromadb

    path = Path(db_path) if db_path is not None else db_path_for(pipeline)
    if not path.exists():
        raise FileNotFoundError(
            f"No vector store at {path}; ingest the {pipeline!r} pipeline first."
        )

    # Chroma caches an opened collection (with its embedding dimension) per path in
    # the shared system cache. Reset before reading so we never get a stale collection
    # from an earlier rebuild, and again after (below) so the client we open here can't
    # leak a stale dimension into the next run's ingest — the cause of
    # "Collection expecting embedding with dimension of N, got M".
    _reset_chroma_cache()
    client = chromadb.PersistentClient(path=str(path))
    try:
        collections = client.list_collections()
        if not collections:
            raise RuntimeError(f"No collections found in vector store at {path}")
        collection = client.get_collection(collections[0].name)
        data = collection.get(include=["embeddings", "metadatas", "documents"])
    finally:
        _reset_chroma_cache()

    embeddings = data.get("embeddings")
    # Chroma may return embeddings as a numpy array; avoid truthiness on arrays.
    vectors = np.asarray(
        embeddings if embeddings is not None else [], dtype=float
    )
    metadatas = [dict(meta or {}) for meta in (data.get("metadatas") or [])]
    documents = [str(doc or "") for doc in (data.get("documents") or [])]
    return vectors, metadatas, documents


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bucket(value: int, edges: tuple[tuple[int, str], ...], over: str) -> str:
    """Return the label of the first ``(threshold, label)`` whose threshold value covers it."""
    for threshold, label in edges:
        if value <= threshold:
            return label
    return over


def _scheme_document_type(meta: Metadata) -> str:
    return str(meta.get("doc_type", "unknown"))


def _scheme_type_visits(meta: Metadata) -> str:
    """Patients split by how many visits their record holds; others by doc type."""
    doc_type = str(meta.get("doc_type", "unknown"))
    if doc_type != "patient":
        return doc_type
    bucket = _bucket(
        _as_int(meta.get("visit_count")),
        ((0, "0 visits"), (2, "1-2 visits"), (5, "3-5 visits")),
        over="6+ visits",
    )
    return f"patient · {bucket}"


def _scheme_record_size(meta: Metadata) -> str:
    """Patients split by how many chunks their source record spans (record size); others by type."""
    doc_type = str(meta.get("doc_type", "unknown"))
    if doc_type != "patient":
        return doc_type
    bucket = _bucket(
        _as_int(meta.get("record_chunks"), 1),
        ((1, "1 chunk"), (2, "2 chunks")),
        over="3+ chunks",
    )
    return f"patient · {bucket}"


def _scheme_chunk_size(meta: Metadata) -> str:
    return _bucket(
        _as_int(meta.get("chunk_size")),
        ((499, "small (<500)"), (899, "medium (500-899)")),
        over="large (900+)",
    )


# Ordered so the first scheme is the default colouring shown on load.
_SCHEMES: tuple[tuple[str, Callable[[Metadata], str]], ...] = (
    ("Document type", _scheme_document_type),
    ("Type · visit count", _scheme_type_visits),
    ("Patient record size", _scheme_record_size),
    ("Chunk size", _scheme_chunk_size),
)


def _enrich(metadatas: list[Metadata]) -> None:
    """Add derived fields used by schemes (e.g. how many chunks each patient record spans)."""
    chunks_per_patient: dict[str, int] = {}
    for meta in metadatas:
        if meta.get("doc_type") == "patient" and meta.get("patient_id"):
            pid = str(meta["patient_id"])
            chunks_per_patient[pid] = chunks_per_patient.get(pid, 0) + 1
    for meta in metadatas:
        if meta.get("doc_type") == "patient" and meta.get("patient_id"):
            meta["record_chunks"] = chunks_per_patient[str(meta["patient_id"])]


def _hover_text(metadatas: list[Metadata], documents: list[str]) -> list[str]:
    lines = []
    for meta, doc in zip(metadatas, documents):
        parts = [f"Type: {meta.get('doc_type', 'unknown')}"]
        if meta.get("patient_id"):
            parts.append(f"Patient: {meta['patient_id']}")
        if meta.get("visit_count") is not None:
            parts.append(f"Visits: {meta['visit_count']}")
        parts.append(f"Chunk: {meta.get('chunk_id', '?')} ({meta.get('chunk_size', '?')} chars)")
        parts.append(f"Text: {doc[:100]}...")
        lines.append("<br>".join(str(part) for part in parts))
    return lines


def _scheme_traces(
    make_trace: Callable[..., Any],
    labels: list[str],
    *,
    visible: bool,
) -> list[Any]:
    """One trace per distinct label (stable colour + legend entry), all sharing a layout."""
    unique = sorted(set(labels))
    color_map = {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate(unique)}
    traces = []
    for label in unique:
        idxs = [i for i, value in enumerate(labels) if value == label]
        traces.append(
            make_trace(name=label, idxs=idxs, color=color_map[label], visible=visible)
        )
    return traces


def _build_figure(
    go: Any,
    coords: np.ndarray,
    metadatas: list[Metadata],
    hover: list[str],
    *,
    is_3d: bool,
    title: str,
) -> Any:
    """Assemble a figure with per-scheme traces and a dropdown to switch the colouring."""

    def make_trace(*, name: str, idxs: list[int], color: str, visible: bool) -> Any:
        text = [hover[i] for i in idxs]
        marker = dict(size=3 if is_3d else 5, color=color, opacity=0.8)
        common = dict(
            name=name, mode="markers", marker=marker, text=text,
            hoverinfo="text", visible=visible,
        )
        if is_3d:
            return go.Scatter3d(
                x=coords[idxs, 0], y=coords[idxs, 1], z=coords[idxs, 2], **common
            )
        return go.Scatter(x=coords[idxs, 0], y=coords[idxs, 1], **common)

    traces: list[Any] = []
    trace_scheme: list[int] = []
    for scheme_index, (_, classify) in enumerate(_SCHEMES):
        labels = [classify(meta) for meta in metadatas]
        scheme_traces = _scheme_traces(
            make_trace, labels, visible=(scheme_index == 0)
        )
        traces.extend(scheme_traces)
        trace_scheme.extend([scheme_index] * len(scheme_traces))

    buttons = []
    for scheme_index, (scheme_name, _) in enumerate(_SCHEMES):
        buttons.append(
            dict(
                label=scheme_name,
                method="update",
                args=[
                    {"visible": [s == scheme_index for s in trace_scheme]},
                    {"title": f"{title} — coloured by {scheme_name}"},
                ],
            )
        )

    fig = go.Figure(data=traces)
    layout = dict(
        title=f"{title} — coloured by {_SCHEMES[0][0]}",
        width=900 if is_3d else 820,
        height=700 if is_3d else 620,
        margin=dict(r=10, b=10, l=10, t=70),
        legend=dict(title="Category", itemsizing="constant"),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.12,
                yanchor="top",
                pad=dict(l=4, t=4),
            )
        ],
        annotations=[
            dict(
                text="Colour by:",
                showarrow=False,
                x=0.0,
                xref="paper",
                y=1.16,
                yref="paper",
                xanchor="left",
                font=dict(size=12),
            )
        ],
    )
    if is_3d:
        layout["scene"] = dict(xaxis_title="x", yaxis_title="y", zaxis_title="z")
    else:
        layout["xaxis_title"] = "x"
        layout["yaxis_title"] = "y"
    fig.update_layout(**layout)
    return fig


def visualize(
    pipeline: str,
    output_dir: Path,
    *,
    db_path: Path | None = None,
) -> list[Path]:
    """Write ``visualise_2d.html`` and ``visualise_3d.html`` for ``pipeline`` into ``output_dir``.

    Returns the paths written (empty list if the store has no vectors to plot).
    """
    import plotly.graph_objects as go

    vectors, metadatas, documents = _load_vectors(pipeline, db_path)
    if vectors.size == 0 or vectors.shape[0] < 2:
        print(
            f"[visualize] {pipeline}: need at least 2 vectors to plot "
            f"(found {vectors.shape[0] if vectors.size else 0}); skipping."
        )
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _enrich(metadatas)
    hover = _hover_text(metadatas, documents)
    written: list[Path] = []

    print(
        f"[visualize] {pipeline}: projecting {vectors.shape[0]} vectors "
        f"({vectors.shape[1]} dims) with t-SNE ..."
    )

    for n_components, filename in ((2, "visualise_2d.html"), (3, "visualise_3d.html")):
        coords = _tsne(vectors, n_components)
        fig = _build_figure(
            go,
            coords,
            metadatas,
            hover,
            is_3d=(n_components == 3),
            title=f"{n_components}D Vector Store Visualization ({pipeline})",
        )
        path = output_dir / filename
        fig.write_html(str(path), include_plotlyjs="cdn")
        written.append(path)

    scheme_names = ", ".join(name for name, _ in _SCHEMES)
    print(
        f"[visualize] {pipeline}: saved {', '.join(p.name for p in written)} "
        f"to {output_dir}/ (colour schemes: {scheme_names})"
    )
    return written


def _tsne(vectors: np.ndarray, n_components: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    n_samples = vectors.shape[0]
    # Perplexity must stay below the sample count; cap it for small stores.
    perplexity = max(1, min(30, n_samples - 1))
    tsne = TSNE(
        n_components=n_components,
        random_state=42,
        perplexity=perplexity,
        init="pca",
    )
    return tsne.fit_transform(vectors)


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2 or args[0] not in ("basic", "advanced"):
        raise SystemExit(
            "usage: python -m app.common.visualize <basic|advanced> <output_dir>"
        )
    visualize(args[0], Path(args[1]))


if __name__ == "__main__":
    main()
