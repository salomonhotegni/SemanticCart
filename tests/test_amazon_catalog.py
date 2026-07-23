import pyarrow as pa
import pyarrow.parquet as pq

from semanticcart.amazon_catalog import (
    build_product_record,
    clean_text,
    load_amazon_catalog,
    parse_price,
)


def make_row(
    item_id: str,
    title: str,
    description: list[str],
) -> dict:
    return {
        "main_category": "Video Games",
        "title": title,
        "features": ["Online multiplayer", "Controller support"],
        "description": description,
        "price": "$59.99",
        "images": {
            "large": [f"https://example.com/{item_id}.jpg"],
            "hi_res": [],
            "thumb": [],
        },
        "store": "Example Studio",
        "categories": ["Video Games", "PC", "Games"],
        "parent_asin": item_id,
    }


def test_text_and_price_cleaning():
    assert clean_text("<b>Hello</b>   world") == "Hello world"
    assert parse_price("$1,299.99") == 1299.99
    assert parse_price("None") is None


def test_product_record_builds_semantic_text():
    product = build_product_record(
        make_row(
            item_id="game-1",
            title="<b>Example Game</b>",
            description=["Explore a large world."],
        )
    )

    assert product["item_id"] == "game-1"
    assert product["title"] == "Example Game"
    assert product["price"] == 59.99
    assert product["image_url"].endswith("game-1.jpg")
    assert "Title: Example Game" in product["catalog_text"]
    assert "Features: Online multiplayer" in product["catalog_text"]
    assert "<b>" not in product["catalog_text"]


def test_loader_filters_and_keeps_richest_duplicate(tmp_path):
    metadata_path = tmp_path / "metadata.parquet"

    rows = [
        make_row("game-1", "Short title", []),
        make_row("game-2", "Other game", ["Other description"]),
        make_row(
            "game-1",
            "Detailed game title",
            [
                "A substantially longer product description.",
                "Includes campaign and multiplayer modes.",
            ],
        ),
    ]

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, metadata_path)

    catalog = load_amazon_catalog(
        metadata_path,
        item_ids=["game-1"],
        batch_size=1,
    )

    assert len(catalog) == 1
    assert catalog.iloc[0]["item_id"] == "game-1"
    assert catalog.iloc[0]["title"] == "Detailed game title"
    assert "campaign and multiplayer" in catalog.iloc[0]["catalog_text"]