# LLM Zoomcamp 2026 — Homework 2: Vector Search

My solution for [Module 2: Vector Search](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/02-vector-search/homework.md).

We turn the course lesson pages into embeddings and search them by similarity.
Embeddings come from the lightweight ONNX `Embedder` (`Xenova/all-MiniLM-L6-v2`),
and the data is pulled from the repository at the pinned commit `8c1834d` (72
pages).

## Answers

| # | Question | Answer |
|---|----------|--------|
| Q1 | Embedding a query — `v[0]` | **-0.02** |
| Q2 | Cosine similarity | **0.37** (≈0.36) |
| Q3 | Highest-scoring chunk | **`02-vector-search/lessons/07-sqlitesearch-vector.md`** |
| Q4 | Vector search (minsearch) | **`04-evaluation/lessons/05-search-metrics.md`** |
| Q5 | In vector results, not text | **`02-vector-search/lessons/08-pgvector.md`** |
| Q6 | Hybrid search (RRF) | **`01-agentic-rag/lessons/13-function-calling.md`** |

## What the script does

- **Q1** — embeds the query with the ONNX `Embedder` and reads `v[0]`.
- **Q2** — embeds a page's content and takes the dot product with the query
  (vectors are normalized, so dot product = cosine similarity).
- **Q3** — chunks the pages (`size=2000`, `step=1000`), embeds every chunk with
  `encode_batch`, and scores them with `X.dot(v)`.
- **Q4** — vector search over the chunks with minsearch `VectorSearch`.
- **Q5** — compares the top-5 vector results against the top-5 text results
  (minsearch `Index`) for the same query.
- **Q6** — fuses vector and text results with Reciprocal Rank Fusion (`k=60`).

## How to run

This uses [uv](https://docs.astral.sh/uv/).

1. Install dependencies:

   ```bash
   uv add onnxruntime tokenizers numpy tqdm minsearch gitsource
   uv add --dev huggingface-hub
   ```

2. Download the ONNX model (`Xenova/all-MiniLM-L6-v2`):

   ```bash
   uv run python download.py
   ```

3. Run it:

   ```bash
   uv run python homework2.py
   ```

Output:

```
Q1: -0.02
Q2: 0.36
Q3: 02-vector-search/lessons/07-sqlitesearch-vector.md
Q4: 04-evaluation/lessons/05-search-metrics.md
Q5: ['02-vector-search/lessons/08-pgvector.md']
Q6: 01-agentic-rag/lessons/13-function-calling.md
```

`embedder.py` and `download.py` are the helper scripts from the course repo's
[`02-vector-search/embed/`](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/02-vector-search/embed)
directory.
