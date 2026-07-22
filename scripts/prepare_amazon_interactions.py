"""
This script prepares the Amazon Beauty dataset for use in recommendation systems.
It loads the raw interaction data, splits it into training, validation, and test sets,
and saves the processed data in Parquet format.
"""

from pathlib import Path

from semanticcart.interactions import (
    chronological_split,
    load_interactions,
)


RAW_PATH = Path("data/raw/amazon/All_Beauty.csv.gz")
OUTPUT_DIR = Path("data/processed/amazon_all_beauty")


def print_summary(name, interactions) -> None:
    print(
        f"{name:10} "
        f"interactions={len(interactions):,} "
        f"users={interactions['user_id'].nunique():,} "
        f"items={interactions['item_id'].nunique():,}"
    )


def main() -> None:
    interactions = load_interactions(RAW_PATH)
    splits = chronological_split(interactions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits.train.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    splits.validation.to_parquet(
        OUTPUT_DIR / "validation.parquet", index=False
    )
    splits.test.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

    print_summary("train", splits.train)
    print_summary("validation", splits.validation)
    print_summary("test", splits.test)


if __name__ == "__main__":
    main()