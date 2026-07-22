from pathlib import Path

from semanticcart.interactions import (
    leave_last_two_split,
    load_interactions,
)


RAW_PATH = Path("data/raw/amazon/Video_Games_5core.csv.gz")
OUTPUT_DIR = Path("data/processed/amazon_video_games_5core")


def print_summary(name, interactions) -> None:
    print(
        f"{name:10} "
        f"interactions={len(interactions):,} "
        f"users={interactions['user_id'].nunique():,} "
        f"items={interactions['item_id'].nunique():,}"
    )


def main() -> None:
    interactions = load_interactions(RAW_PATH)
    splits = leave_last_two_split(interactions)

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