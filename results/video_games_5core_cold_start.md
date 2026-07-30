# Video Games 5-Core Cold-Start Evaluation

Protocol: embeddings cover the complete 25,612-product catalogue, while behavioral fitting uses only train plus validation interactions. Test data was not used for tuning.
See the [final warm-user results](video_games_5core_final_test.md) for ALS, semantic, and hybrid comparisons.

## Real New-Item Cohort

These products have metadata and embeddings but no interactions in the final fitting data.

| Events | Users | Products | Hits@10 | Recall@10 | 95% CI | NDCG@10 | MRR@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 69 | 69 | 12 | 4 | 0.057971 | [0.022772, 0.139794] | 0.041394 | 0.036232 |

The semantic model surfaced 3 of 12 new products and produced 4 correct Top-10 matches.
ALS cannot represent these products because they have no fitted interaction factors.

## Simulated New Users

Each profile uses only the user's most recent N unique viewed products. User identity and older behavior are not model features.

| History items | Eligible users | Recall@10 | NDCG@10 | MRR@10 | Coverage | Users/second |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 94,762 | 0.032893 | 0.017797 | 0.013205 | 0.954240 | 14,251 |
| 3 | 94,762 | 0.032471 | 0.016913 | 0.012227 | 0.915313 | 14,162 |
| 5 | 64,311 | 0.029746 | 0.015720 | 0.011481 | 0.823325 | 13,985 |

## Controlled Session-Length Comparison

All session lengths below use the same 64,311-user cohort.

| History items | Recall@10 | NDCG@10 | MRR@10 | Coverage |
|---:|---:|---:|---:|---:|
| 1 | 0.032592 | 0.017578 | 0.013010 | 0.940536 |
| 3 | 0.032094 | 0.016662 | 0.012014 | 0.891106 |
| 5 | 0.029746 | 0.015720 | 0.011481 | 0.823325 |

## Interpretation and Limitations

- One-item session intent performs best under the current recency-weighted averaging strategy.
- This is a dataset-specific observation, not evidence that shorter histories are universally better.
- The real cold-item cohort contains only 69 events, so its wide confidence interval must accompany the headline recall.
- Simulated new users are established users whose histories were deliberately truncated; they are not organically new production users.
- Two complete runs produced identical recommendation artifact hashes using single-threaded HNSW construction.
