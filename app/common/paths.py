from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]


def _resolve(*candidates: Path, default: Path) -> Path:
    """Return the first candidate that exists, else ``default``.

    Keeps the app self-contained: data bundled under ``app/data`` is preferred,
    with a fallback to the legacy parent-repo location so the original RAGnosis
    layout keeps working.
    """
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return default


# App-local data directory, independent of the parent repository.
DATA_DIR = _resolve(APP_ROOT / "data", APP_ROOT.parent / "data", default=APP_ROOT / "data")

_dataset_override = os.getenv("RAGNOSIS_DATASET_DIR")
DATASET_DIR = (
    Path(_dataset_override)
    if _dataset_override
    else _resolve(
        APP_ROOT / "data" / "dataset",
        APP_ROOT.parent / "data" / "dataset",
        default=DATA_DIR / "dataset",
    )
)
