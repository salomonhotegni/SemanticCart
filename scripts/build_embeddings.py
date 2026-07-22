"""
It loads the product catalogue, generates or retrieves cached embeddings 
for every product, and reports the number and dimensionality of the resulting vectors.
"""

from pathlib import Path

from semanticcart.catalog import load_catalog
from semanticcart.embedding_cache import EmbeddingConfig, embed_catalog


def main() -> None:
    catalog = load_catalog("data/raw/home_depot_products.csv")

    embedded = embed_catalog(
        catalog=catalog,
        cache_path=Path("data/processed/product_embeddings.parquet"),
        config=EmbeddingConfig(),
    )

    dimensions = len(embedded["embedding"].iloc[0])
    print(f"Products embedded: {len(embedded):,}")
    print(f"Embedding dimensions: {dimensions}")


if __name__ == "__main__":
    main()