| config | tasks | runs | recall@5 | recall@10 | resolve rate | regression rate | cost / resolved ($) | errors |
|---|---|---|---|---|---|---|---|---|
| bm25_only | 9 | 1 | 1.000 ± 0.000 | 1.000 ± 0.000 | - | - | - | 0 |
| vector_only | 9 | 1 | 0.667 ± 0.000 | 0.889 ± 0.000 | - | - | - | 0 |
| hybrid_rerank | 9 | 1 | 0.889 ± 0.000 | 0.889 ± 0.000 | - | - | - | 0 |
| navigator_full | 9 | 1 | 0.000 ± 0.000 | 0.000 ± 0.000 | - | - | - | 9 |

Values are mean ± std across repeats. Crashed runs count as zero recall and as unresolved.
