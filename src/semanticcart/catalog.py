"""Load and normalize product catalogues for semantic recommendation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["product_id", "title", "description"]
OPTIONAL_TEXT_COLUMNS = ["category", "brand", "attributes"]


def load_catalog(path: str | Path) -> pd.DataFrame:
    """Load a CSV product catalogue and build embedding-ready text.

    Args:
        path: CSV file containing product_id, title, and description.
            Category, brand, and attributes are included when present.

    Returns:
        A deduplicated catalogue with normalized text fields and a
        catalog_text column.

    Raises:
        ValueError: If a required catalogue column is missing.
    """
    path = Path(path)
    df = pd.read_csv(path)

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Catalogue is missing required columns: {missing}")

    keep_columns = REQUIRED_COLUMNS + [
        col for col in OPTIONAL_TEXT_COLUMNS if col in df.columns
    ]

    df = df[keep_columns].copy()
    df["product_id"] = df["product_id"].astype(str).str.strip()

    df = df.dropna(subset=["product_id"])
    df = df.drop_duplicates(subset=["product_id"], keep="first")

    for col in keep_columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["catalog_text"] = df.apply(build_catalog_text, axis=1)

    return df.reset_index(drop=True)


def build_catalog_text(row: pd.Series) -> str:
    """Combine one normalized product row into labelled semantic text.

    Args:
        row: Product fields containing at least title and description.

    Returns:
        Newline-separated text suitable for an embedding model.
    """
    parts = [
        f"Title: {row.get('title', '')}",
        f"Description: {row.get('description', '')}",
    ]

    for col in OPTIONAL_TEXT_COLUMNS:
        value = row.get(col, "") 
        if value:
            parts.append(f"{col.title()}: {value}")

    return "\n".join(part for part in parts if part.strip())
