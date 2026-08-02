# SemanticCart Vector Retrieval

Model version: `semanticcart-c5ef1c664d715a32`  
Dataset: `video_games_5core`  
Workload: frozen returning-user dense profiles  
Protocol: 500 measured queries, 25 warmups, concurrency 1  

## Results

| Engine | ef_search | Top-10 overlap with FAISS | Top-1 agreement | Self Recall@1 | p50 (ms) | p95 (ms) | Throughput (queries/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FAISS HNSW | 128 | 1.0000 | 1.0000 | 0.9900 | 0.271 | 0.416 | 3448.9 |
| pgvector HNSW | 128 | 0.9996 | 1.0000 | 0.9800 | 2.585 | 4.121 | 359.4 |
| pgvector HNSW | 512 | 0.9996 | 1.0000 | 0.9800 | 5.734 | 8.183 | 175.9 |
| pgvector HNSW | 1000 | 0.9996 | 0.9940 | 0.9900 | 74.203 | 77.978 | 13.5 |

## Interpretation

- pgvector at `ef_search=128` retained 99.96% mean Top-10 overlap and full Top-1 agreement on user-profile queries.
- Its p50 and p95 were 9.6x and 9.9x FAISS latency, respectively.
- Increasing pgvector to `ef_search=512` did not improve measured profile overlap or systematic self-retrieval.
- `ef_search=1000` improved systematic self-retrieval to 99.00%, but raised p50 by 28.7x relative to pgvector at 128.
- FAISS remains the deployed in-process retrieval engine; pgvector is a validated durable alternative for deployments that prioritize centralized vector storage.

## Methodology

- Latency excludes model loading, pool startup, and warmup queries.
- Top-10 overlap uses FAISS HNSW as the reference, not an exact brute-force oracle.
- Self Recall@1 uses 100 systematic catalogue positions, including both catalogue endpoints.
