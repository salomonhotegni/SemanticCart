from pathlib import Path

import pandas as pd

from semanticcart.amazon_catalog import load_amazon_catalog


RAW_METADATA_PATH = Path(
    "data/raw/amazon/meta_Video_Games.parquet"
)
INTERACTION_DIR = Path(
    "data/processed/amazon_video_games_5core"
)
CATALOG_PATH = INTERACTION_DIR / "catalog.parquet"
MISSING_ITEMS_PATH = (
    INTERACTION_DIR / "missing_metadata_items.parquet"
)
SPLIT_NAMES = ("train", "validation", "test")


def load_interaction_item_ids() -> pd.Index:
    item_frames = [
        pd.read_parquet(
            INTERACTION_DIR / f"{split}.parquet",
            columns=["item_id"],
        )
        for split in SPLIT_NAMES
    ]

    item_ids = (
        pd.concat(item_frames, ignore_index=True)["item_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return pd.Index(
        item_ids[item_ids.ne("")].unique(),
        name="item_id",
    )


def nonempty_rate(series: pd.Series) -> float:
    return float(
        series.fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .mean()
    )


def main() -> None:
    requested_item_ids = load_interaction_item_ids()

    catalog = load_amazon_catalog(
        path=RAW_METADATA_PATH,
        item_ids=requested_item_ids,
    )

    matched_item_ids = pd.Index(catalog["item_id"])
    missing_item_ids = requested_item_ids[
        ~requested_item_ids.isin(matched_item_ids)
    ]

    INTERACTION_DIR.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(CATALOG_PATH, index=False)

    pd.DataFrame(
        {"item_id": missing_item_ids}
    ).to_parquet(MISSING_ITEMS_PATH, index=False)

    metadata_coverage = (
        len(catalog) / len(requested_item_ids)
        if len(requested_item_ids)
        else 0.0
    )

    print(f"Requested items: {len(requested_item_ids):,}")
    print(f"Matched items:   {len(catalog):,}")
    print(f"Missing items:   {len(missing_item_ids):,}")
    print(f"Metadata coverage: {metadata_coverage:.2%}")
    print(
        "Text coverage:     "
        f"{nonempty_rate(catalog['catalog_text']):.2%}"
    )
    print(
        "Description coverage: "
        f"{nonempty_rate(catalog['description']):.2%}"
    )
    print(
        "Feature coverage:  "
        f"{nonempty_rate(catalog['features']):.2%}"
    )
    print(
        "Image coverage:    "
        f"{nonempty_rate(catalog['image_url']):.2%}"
    )
    print(
        "Price coverage:    "
        f"{catalog['price'].notna().mean():.2%}"
    )
    print(f"Catalogue:       {CATALOG_PATH}")


if __name__ == "__main__":
    main()