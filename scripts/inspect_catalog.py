"""
This file loads the product catalogue and prints its size, columns, 
and sample records to verify that preprocessing works correctly.
"""

from pathlib import Path

from semanticcart.catalog import load_catalog


CATALOG_PATH = Path("data/raw/home_depot_products.csv")


def main() -> None:
    catalog = load_catalog(CATALOG_PATH)

    print(f"Products: {len(catalog):,}")
    print(f"Columns: {list(catalog.columns)}")
    print()
    print(catalog[["product_id", "title", "catalog_text"]].head(3))


if __name__ == "__main__":
    main()