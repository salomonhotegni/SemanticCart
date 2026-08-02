CREATE TABLE IF NOT EXISTS catalog_embeddings (
    model_version TEXT NOT NULL
        CHECK (length(btrim(model_version)) > 0),
    item_id TEXT NOT NULL
        CHECK (length(btrim(item_id)) > 0),
    embedding_model TEXT NOT NULL
        CHECK (length(btrim(embedding_model)) > 0),
    embedding_dimensions SMALLINT NOT NULL
        CHECK (embedding_dimensions = 512),
    embedding VECTOR(512) NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_version, item_id)
);

CREATE INDEX IF NOT EXISTS
    catalog_embeddings_model_version_idx
ON catalog_embeddings (model_version);

CREATE INDEX IF NOT EXISTS
    catalog_embeddings_embedding_hnsw_idx
ON catalog_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 32,
    ef_construction = 200
);