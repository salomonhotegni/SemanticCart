"""Build and query a FAISS index for dense product embeddings."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HnswConfig:
    """Configure approximate nearest-neighbour retrieval.

    Attributes:
        connections: Graph connections retained per indexed product.
        ef_construction: Candidate breadth used while building the graph.
        ef_search: Minimum candidate breadth used during retrieval.
    """

    connections: int = 32
    ef_construction: int = 200
    ef_search: int = 128


class DenseItemIndex:
    """Store normalized product vectors in a cosine-similarity HNSW index."""

    def __init__(
        self,
        item_ids: np.ndarray,
        item_vectors: np.ndarray,
        index,
        config: HnswConfig,
    ) -> None:
        self.item_ids = item_ids
        self.item_vectors = item_vectors
        self.index = index
        self.config = config

        self.item_to_index = {
            item_id: position
            for position, item_id in enumerate(item_ids)
        }

    @classmethod
    def from_catalog(
        cls,
        catalog: pd.DataFrame,
        config: HnswConfig | None = None,
    ) -> "DenseItemIndex":
        """Validate product embeddings and construct an HNSW index.

        Args:
            catalog: Products containing unique item_id and embedding values.
            config: Optional HNSW construction and search configuration.

        Returns:
            An index containing products sorted by item ID.

        Raises:
            ValueError: If columns, identifiers, vectors, or configuration
                values are invalid.
        """
        config = config or HnswConfig()
        missing = {"item_id", "embedding"} - set(catalog.columns)

        if missing:
            raise ValueError(
                f"Missing catalogue columns: {sorted(missing)}"
            )

        if min(
            config.connections,
            config.ef_construction,
            config.ef_search,
        ) <= 0:
            raise ValueError("HNSW configuration values must be positive.")

        items = catalog[["item_id", "embedding"]].copy()

        if items.empty:
            raise ValueError("Catalogue cannot be empty.")
        if items["item_id"].isna().any():
            raise ValueError("Item IDs cannot be missing.")

        items["item_id"] = items["item_id"].astype(str)

        if items["item_id"].eq("").any():
            raise ValueError("Item IDs cannot be empty.")
        if items["item_id"].duplicated().any():
            raise ValueError("Item IDs must be unique.")
        if any(vector is None for vector in items["embedding"]):
            raise ValueError("Product embeddings cannot be missing.")

        items = items.sort_values("item_id").reset_index(drop=True)

        try:
            vectors = np.vstack(
                items["embedding"].to_numpy()
            ).astype(np.float32)
        except ValueError as error:
            raise ValueError(
                "Product embeddings must share one dimension."
            ) from error

        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError("Product embeddings must be two-dimensional.")
        if not np.isfinite(vectors).all():
            raise ValueError("Product embeddings must contain finite values.")
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Product embeddings cannot be zero vectors.")

        vectors = np.ascontiguousarray(vectors)
        faiss.normalize_L2(vectors)

        index = faiss.IndexHNSWFlat(
            vectors.shape[1],
            config.connections,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = config.ef_construction
        index.hnsw.efSearch = config.ef_search
        index.add(vectors)

        return cls(
            item_ids=items["item_id"].to_numpy(),
            item_vectors=vectors,
            index=index,
            config=config,
        )

    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve product positions and cosine scores for query vectors."""
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        queries = np.asarray(query_vectors, dtype=np.float32)

        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if queries.ndim != 2:
            raise ValueError("Query vectors must be a matrix.")
        if queries.shape[1] != self.item_vectors.shape[1]:
            raise ValueError("Query and product dimensions do not match.")
        if not np.isfinite(queries).all():
            raise ValueError("Query vectors must contain finite values.")
        if np.any(np.linalg.norm(queries, axis=1) == 0):
            raise ValueError("Query vectors cannot be zero vectors.")

        queries = np.ascontiguousarray(queries.copy())
        faiss.normalize_L2(queries)

        result_count = min(k, len(self.item_ids))
        self.index.hnsw.efSearch = max(
            self.config.ef_search,
            result_count,
        )

        scores, indices = self.index.search(
            queries,
            result_count,
        )
        return scores, indices
    
    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index, vectors, item mapping, and configuration.

        Args:
            directory: Model artifact directory to create or update.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "item_index.faiss"),
        )

        np.save(
            directory / "item_vectors.npy",
            self.item_vectors,
            allow_pickle=False,
        )

        pd.DataFrame(
            {
                "item_index": np.arange(
                    len(self.item_ids),
                    dtype=np.int32,
                ),
                "item_id": self.item_ids,
            }
        ).to_parquet(
            directory / "items.parquet",
            index=False,
        )

        with (directory / "index_config.json").open(
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(asdict(self.config), output, indent=2)