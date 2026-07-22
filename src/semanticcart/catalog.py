"""
This file loads, validates, cleans, deduplicates, 
and combines product catalogue fields into embedding-ready text.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["product_id", "title", "description"]
OPTIONAL_TEXT_COLUMNS = ["category", "brand", "attributes"]


def load_catalog(path: str | Path) -> pd.DataFrame:
    """Load and normalize a product catalogue for recommendation experiments."""
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
    """Create the text sent to embedding models."""
    parts = [
        f"Title: {row.get('title', '')}",
        f"Description: {row.get('description', '')}",
    ]

    for col in OPTIONAL_TEXT_COLUMNS:
        value = row.get(col, "") 
        if value:
            parts.append(f"{col.title()}: {value}")

    return "\n".join(part for part in parts if part.strip())