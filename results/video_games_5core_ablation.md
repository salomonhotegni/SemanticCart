# Video Games 5-Core Validation Ablation

Protocol: leave-last-two chronological split, one validation target per user, 94,762 users, 25,527 training products, K=10.
Hybrid alpha was selected on validation; the test split remains reserved for final evaluation.

## Ranking Quality

| Model | Recall@10 | NDCG@10 | MRR@10 | Coverage | Notes |
|---|---:|---:|---:|---:|---|
| Popularity | 0.026783 | 0.013568 | 0.009663 | 0.000783 | Global Top-K |
| TF-IDF content | 0.030740 | 0.015560 | 0.011016 | 0.888158 | 50,000-feature sparse retrieval |
| OpenAI content | 0.033885 | 0.017503 | 0.012568 | 0.879422 | 512d embeddings with FAISS HNSW |
| Collaborative ALS | 0.081351 | 0.043505 | 0.032045 | 0.078662 | 64 latent factors |
| Hybrid | 0.081351 | 0.043854 | 0.032471 | 0.078662 | ALS Top-10 reranked; semantic weight 0.6 |

## Bulk Performance Context

| Model | Measured stage | Seconds | Users/second |
|---|---|---:|---:|
| Popularity | not recorded | - | - |
| TF-IDF content | candidate generation | 151.62 | 625 |
| OpenAI content | candidate generation | 11.54 | 8,212 |
| Collaborative ALS | candidate generation | 13.57 | 6,984 |
| Hybrid | reranking only | 1.56 | 60,870 |

Hybrid timing measures reranking after candidates already exist. It is not end-to-end serving latency.

## Findings

- OpenAI embeddings improve NDCG@10 by 12.5% over TF-IDF.
- Conservative hybrid reranking improves NDCG@10 by 0.8% over ALS while preserving Recall@10 and coverage.
- OpenAI content reaches broad catalogue coverage, while ALS remains the strongest source for warm-user recall.
- p50 and p95 online latency are intentionally omitted until the serving API is benchmarked.
