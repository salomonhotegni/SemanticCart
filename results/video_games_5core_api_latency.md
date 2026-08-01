# SemanticCart Online API Latency

Model version: `semanticcart-c5ef1c664d715a32`  
Dataset: `video_games_5core`  
Workload: returning-user hybrid Top-10 recommendations  
Transport: HTTP/1.1 loopback  
Server: one warm Uvicorn worker  

## Results

| Concurrency | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 500 | 46.314 | 49.783 | 54.173 | 21.49 |
| 8 | 500 | 757.304 | 813.658 | 842.629 | 10.53 |

## Interpretation

- Warm low-load latency is 46.314 ms p50 and 49.783 ms p95.
- At concurrency 8, p95 latency is 813.658 ms, or 16.3x the low-load p95.
- Concurrency-8 throughput is 0.49x low-load throughput, showing saturation in the single CPU-bound worker.
- Measurements include HTTP handling, JSON serialization, ALS retrieval, semantic scoring, MMR reranking, and response deserialization.
- Model startup and checksum verification are excluded.
- Loopback measurements exclude cross-host network latency and should not be presented as internet latency.
