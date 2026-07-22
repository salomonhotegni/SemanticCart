"""
Demo script for generating semantic product recommendations.
"""

from semanticcart.catalog import load_catalog
from semanticcart.embedding_cache import EmbeddingConfig, embed_catalog
from semanticcart.semantic import recommend_semantic


VIEWED_PRODUCT_IDS = [
    "P1938",
    "P1970",
    "P1044",
    "P1838",
    "P1048",
    "P1017",
    "P1310",
    "P1444",
]


def main() -> None:
    catalog = load_catalog("data/raw/home_depot_products.csv")

    embedded_catalog = embed_catalog(
        catalog,
        "data/processed/product_embeddings.parquet",
        EmbeddingConfig(),
    )

    recommendations = recommend_semantic(
        embedded_catalog,
        viewed_product_ids=VIEWED_PRODUCT_IDS,
        k=10,
        recency_decay=0.85,
    )

    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()