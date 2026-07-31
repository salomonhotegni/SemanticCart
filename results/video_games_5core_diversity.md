# Video Games 5-Core Diversity Reranking

Protocol: session blending was tuned on the full validation split. Sixteen MMR configurations were then compared on a deterministic 10,000-user validation cohort under a 99% NDCG-retention constraint. Test labels were loaded only after the frozen recommendations were written.

## Held-Out Test Results

| Model | Recall@10 | NDCG@10 | MRR@10 | Coverage | ILD | Category variety | Novelty | Price dispersion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-25 returning-user relevance | 0.076001 | 0.042232 | 0.031964 | 0.099219 | 0.479902 | 0.612474 | 11.262847 | 0.907143 |
| Diversity-aware reranker | 0.076128 | 0.042313 | 0.032030 | 0.098867 | 0.484085 | 0.639219 | 11.249758 | 0.918266 |

## Diversity Trade-Off

| Metric | Absolute change | Relative change |
|---|---:|---:|
| Recall@10 | +0.000127 | +0.167% |
| NDCG@10 | +0.000081 | +0.191% |
| MRR@10 | +0.000066 | +0.206% |
| Catalogue coverage | -0.000352 | -0.354% |
| Intra-list diversity | +0.004183 | +0.872% |
| Category variety | +0.026745 | +4.367% |
| Category coverage | +0.000000 | +0.000% |
| Popularity novelty | -0.013089 | -0.116% |
| Price dispersion | +0.011123 | +1.226% |

## Selected Configuration

| Setting | Value |
|---|---:|
| Session semantic weight | 0.500 |
| MMR relevance weight | 0.850 |
| Popularity novelty weight | 0.000 |
| Semantic redundancy weight | 0.700 |
| Category redundancy weight | 0.200 |
| Price redundancy weight | 0.100 |

## Candidate-Depth Contribution

The earlier returning-user hybrid used ten ALS candidates. The new relevance baseline selects ten products from a Top-25 pool using the same one-item session signal.

| Metric | Relative change from Top-10 |
|---|---:|
| Recall@10 | +6.997% |
| NDCG@10 | +5.225% |
| MRR@10 | +4.090% |
| Catalogue coverage | +25.867% |

## Bulk Performance Context

| Stage | Seconds | Users/second |
|---|---:|---:|
| Top-25 relevance ranking | 4.03 | 23,537 |
| MMR Top-10 selection | 74.03 | 1,280 |

These are offline bulk-stage measurements, not online p50 or p95 API latency.

## Findings

- Expanding the candidate pool from 10 to 25 improves test NDCG@10 by 5.225%.
- Frozen MMR adds another 0.191% NDCG@10 improvement while increasing intra-list diversity.
- Category variety improves by 4.367% and price dispersion improves by 1.226%.
- Catalogue coverage changes by -0.354%, a small trade-off.
- Validation selected a popularity-novelty weight of zero; the signal was evaluated but did not improve the constrained objective.
