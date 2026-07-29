# Video Games 5-Core Final Test Results

Protocol: models were fitted on train plus validation (719,824 interactions and 25,600 products), then evaluated on 94,762 untouched chronological test events with K=10.
All configurations were fixed using validation results before test evaluation. No parameter was selected using test metrics.
See the complete [validation ablation](video_games_5core_ablation.md) for popularity, TF-IDF, OpenAI, ALS, and hybrid comparisons.

## Held-Out Ranking Quality

| Model | Recall@10 | NDCG@10 | MRR@10 | Coverage | Notes |
|---|---:|---:|---:|---:|---|
| Collaborative ALS | 0.071031 | 0.038802 | 0.029041 | 0.078828 | 64 latent factors |
| OpenAI content | 0.029484 | 0.015466 | 0.011238 | 0.848242 | 512d embeddings with FAISS HNSW |
| Hybrid | 0.071031 | 0.038933 | 0.029193 | 0.078828 | ALS Top-10 reranked; semantic weight 0.6 |

## Validation-to-Test Generalization

| Model | Validation Recall@10 | Test Recall@10 | Validation NDCG@10 | Test NDCG@10 | NDCG change |
|---|---:|---:|---:|---:|---:|
| Collaborative ALS | 0.081351 | 0.071031 | 0.043505 | 0.038802 | -10.8% |
| OpenAI content | 0.033885 | 0.029484 | 0.017503 | 0.015466 | -11.6% |
| Hybrid | 0.081351 | 0.071031 | 0.043854 | 0.038933 | -11.2% |

## Bulk Performance Context

| Model | Measured stage | Seconds | Users/second |
|---|---|---:|---:|
| Collaborative ALS | candidate generation | 10.77 | 8,801 |
| OpenAI content | candidate generation | 9.05 | 10,472 |
| Hybrid | reranking only | 1.72 | 54,994 |

Hybrid timing measures reranking after candidates already exist, not end-to-end serving latency.

## Findings

- The fixed hybrid improves test NDCG@10 by 0.336% and MRR@10 by 0.526% over ALS.
- Conservative reranking preserves ALS Recall@10 and catalogue coverage exactly.
- OpenAI semantic retrieval covers 10.8x as much of the fit catalogue as ALS.
- All three models score lower on the later test horizon, showing why chronological holdout evaluation matters.
- Online p50 and p95 latency remain unreported until the serving API is benchmarked.
