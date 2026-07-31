# Video Games 5-Core Final Test Results

Protocol: models were fitted on train plus validation (719,824 interactions and 25,600 products), then evaluated on 94,762 untouched chronological test events with K=10.
All configurations were fixed using validation results before test evaluation. No parameter was selected using test metrics.
See the complete [validation ablation](video_games_5core_ablation.md) and [diversity report](video_games_5core_diversity.md).

## Held-Out Ranking Quality

| Model | Recall@10 | NDCG@10 | MRR@10 | Coverage | Notes |
|---|---:|---:|---:|---:|---|
| Collaborative ALS | 0.071031 | 0.038802 | 0.029041 | 0.078828 | 64 latent factors |
| OpenAI content | 0.029484 | 0.015466 | 0.011238 | 0.848242 | 512d embeddings with FAISS HNSW |
| Long-term hybrid | 0.071031 | 0.038933 | 0.029193 | 0.078828 | ALS Top-10 reranked by long-term semantics; weight 0.6 |
| Returning-user hybrid | 0.071031 | 0.040135 | 0.030708 | 0.078828 | ALS Top-10 reranked by one-item session intent; weight 0.5 |
| Top-25 returning-user | 0.076001 | 0.042232 | 0.031964 | 0.099219 | Top-10 selected from 25 ALS candidates; session weight 0.5 |
| Diversity-aware reranker | 0.076128 | 0.042313 | 0.032030 | 0.098867 | MMR with semantic, category, and price redundancy |

## Validation-to-Test Generalization

| Model | Validation Recall@10 | Test Recall@10 | Validation NDCG@10 | Test NDCG@10 | NDCG change |
|---|---:|---:|---:|---:|---:|
| Collaborative ALS | 0.081351 | 0.071031 | 0.043505 | 0.038802 | -10.8% |
| OpenAI content | 0.033885 | 0.029484 | 0.017503 | 0.015466 | -11.6% |
| Long-term hybrid | 0.081351 | 0.071031 | 0.043854 | 0.038933 | -11.2% |
| Returning-user hybrid | 0.081351 | 0.071031 | 0.044963 | 0.040135 | -10.7% |
| Top-25 returning-user | 0.086765 | 0.076001 | 0.047146 | 0.042232 | -10.4% |
| Diversity-aware reranker | 0.086944 | 0.076128 | 0.047245 | 0.042313 | -10.4% |

## Bulk Performance Context

| Model | Measured stage | Seconds | Users/second |
|---|---|---:|---:|
| Collaborative ALS | candidate generation | 10.77 | 8,801 |
| OpenAI content | candidate generation | 9.05 | 10,472 |
| Long-term hybrid | Top-10 reranking only | 1.72 | 54,994 |
| Returning-user hybrid | Top-10 reranking only | 1.29 | 73,303 |
| Top-25 returning-user | Top-25 relevance reranking only | 4.03 | 23,537 |
| Diversity-aware reranker | MMR Top-10 selection only | 74.03 | 1,280 |

Timing rows measure different offline bulk stages and are not end-to-end serving latency.

## Findings

- The final diversity-aware model improves Recall@10 by 7.176%, NDCG@10 by 9.047%, and MRR@10 by 10.293% over ALS.
- Final catalogue coverage is 25.421% higher than ALS coverage.
- Expanding the returning-user candidate pool from 10 to 25 improves NDCG@10 by 5.225%.
- Frozen MMR adds another 0.191% NDCG@10 while improving semantic and category diversity.
- OpenAI semantic retrieval covers 10.8x as much of the fit catalogue as ALS.
- All six models score lower on the later test horizon, showing why chronological holdout evaluation matters.
- Online p50 and p95 latency remain unreported until the serving API is benchmarked.
