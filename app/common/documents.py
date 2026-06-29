from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.common.paths import DATASET_DIR


def load_csv_tables(csv_dir: Path = DATASET_DIR) -> dict[str, pd.DataFrame]:
    """Load every ``*.csv`` under ``csv_dir`` as a string-typed DataFrame."""
    csv_paths = sorted(Path(csv_dir).glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {Path(csv_dir).resolve()}")
    return {
        path.stem: pd.read_csv(path, dtype=str, keep_default_na=False)
        for path in csv_paths
    }


def clean_value(value: Any) -> str:
    return "" if value is None else str(value).strip()
