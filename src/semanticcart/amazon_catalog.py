"""Prepare Amazon product metadata for semantic recommendation models."""

import html
import math
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


TEXT_COLUMNS = [
    "main_category",
    "title",
    "features",
    "description",
    "price",
    "images",
    "store",
    "categories",
    "parent_asin",
]

OUTPUT_COLUMNS = [
    "item_id",
    "title",
    "main_category",
    "categories",
    "store",
    "description",
    "features",
    "price",
    "image_url",
    "catalog_text",
]

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
PRICE_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def clean_text(value: object) -> str:
    """Normalize one metadata value into plain single-spaced text."""

    if value is None:
        return ""

    text = html.unescape(str(value))
    text = HTML_TAG_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def join_text(values: object) -> str:
    """Normalize and concatenate a scalar or sequence of metadata values."""

    if not isinstance(values, (list, tuple)):
        return clean_text(values)

    cleaned_values = [
        clean_text(value)
        for value in values
        if value is not None
    ]
    return " ".join(filter(None, cleaned_values))


def parse_price(value: object) -> float | None:
    """Extract the first finite numeric price from a metadata value."""

    if isinstance(value, (int, float)):
        price = float(value)
        return price if math.isfinite(price) else None

    match = PRICE_PATTERN.search(clean_text(value))

    if match is None:
        return None

    return float(match.group().replace(",", ""))


def select_image_url(images: object) -> str:
    """Select the first available large, high-resolution, or thumbnail URL."""

    if not isinstance(images, dict):
        return ""

    for image_size in ("large", "hi_res", "thumb"):
        urls = images.get(image_size) or []

        if isinstance(urls, str):
            urls = [urls]

        for url in urls:
            cleaned_url = clean_text(url)
            if cleaned_url:
                return cleaned_url

    return ""


def build_product_record(row: dict) -> dict:
    """Convert one raw Amazon metadata row into a model-ready product record.

    Args:
        row: Raw metadata containing parent_asin and optional descriptive
            fields, price, and image variants.

    Returns:
        A normalized record with item_id, display fields, and catalog_text
        capped at 6,000 characters.
    """

    title = clean_text(row.get("title"))
    main_category = clean_text(row.get("main_category"))
    categories = join_text(row.get("categories"))
    store = clean_text(row.get("store"))
    features = join_text(row.get("features"))
    description = join_text(row.get("description"))

    sections = [
        ("Title", title),
        ("Main category", main_category),
        ("Categories", categories),
        ("Store", store),
        ("Features", features),
        ("Description", description),
    ]

    catalog_text = ". ".join(
        f"{label}: {value}"
        for label, value in sections
        if value
    )[:6000]

    return {
        "item_id": clean_text(row.get("parent_asin")),
        "title": title,
        "main_category": main_category,
        "categories": categories,
        "store": store,
        "description": description,
        "features": features,
        "price": parse_price(row.get("price")),
        "image_url": select_image_url(row.get("images")),
        "catalog_text": catalog_text,
    }


def load_amazon_catalog(
    path: str | Path,
    item_ids: Iterable[str] | None = None,
    batch_size: int = 8192,
) -> pd.DataFrame:
    """Load selected products from Amazon metadata without loading it all.

    Duplicate parent_asin records are resolved by retaining the record with
    the longest catalog_text.

    Args:
        path: Amazon metadata Parquet file.
        item_ids: Optional product IDs to retain; all products are loaded when
            omitted.
        batch_size: Number of Parquet rows processed at a time.

    Returns:
        A sorted, deduplicated catalogue using OUTPUT_COLUMNS.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    requested_ids = (
        {clean_text(item_id) for item_id in item_ids}
        if item_ids is not None
        else None
    )

    products: dict[str, dict] = {}
    parquet = pq.ParquetFile(path)

    for batch in parquet.iter_batches(
        columns=TEXT_COLUMNS,
        batch_size=batch_size,
    ):
        for row in batch.to_pylist():
            item_id = clean_text(row.get("parent_asin"))

            if not item_id:
                continue

            if requested_ids is not None and item_id not in requested_ids:
                continue

            product = build_product_record(row)
            existing = products.get(item_id)

            if (
                existing is None
                or len(product["catalog_text"])
                > len(existing["catalog_text"])
            ):
                products[item_id] = product

    if not products:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return (
        pd.DataFrame(products.values(), columns=OUTPUT_COLUMNS)
        .sort_values("item_id")
        .reset_index(drop=True)
    )
