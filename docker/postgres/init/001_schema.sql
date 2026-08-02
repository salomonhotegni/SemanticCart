CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS interaction_events (
    event_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL
        CHECK (length(btrim(user_id)) > 0),
    item_id TEXT NOT NULL
        CHECK (length(btrim(item_id)) > 0),
    event_type TEXT NOT NULL
        CHECK (
            event_type IN (
                'view',
                'cart',
                'purchase'
            )
        ),
    occurred_at TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS
    interaction_events_user_recency_idx
ON interaction_events (
    user_id,
    occurred_at DESC,
    ingested_at DESC,
    event_id DESC
);

CREATE INDEX IF NOT EXISTS
    interaction_events_user_item_recency_idx
ON interaction_events (
    user_id,
    item_id,
    occurred_at DESC,
    ingested_at DESC,
    event_id DESC
);