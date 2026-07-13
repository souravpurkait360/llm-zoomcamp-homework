# LLM Zoomcamp 2026 — Homework 4: Evaluation

My solution for [Module 4: Evaluation](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/04-evaluation/homework.md).

We generate a ground-truth dataset and use it to measure keyword, vector, and
hybrid search over the course lessons — so we can finally compare them on numbers
(Hit Rate and MRR) instead of intuition. Data is pulled from the repository at
the pinned commit `8c1834d` (72 pages, 295 chunks).

## Answers

| # | Question | Answer |
|---|----------|--------|
| Q1 | Avg input tokens (3 pages) | **1400** (≈1354) |
| Q2 | First text-search result | **`01-agentic-rag/lessons/03-rag.md`** |
| Q3 | First vector-search result | **`01-agentic-rag/lessons/01-intro.md`** |
| Q4 | Text search — Hit Rate | **0.76** |
| Q5 | Vector search — MRR | **0.55** |
| Q6 | Best `k` for hybrid (RRF) | **1** |

## What the script does

- **Q1** — generates 5 questions for each of the first 3 lesson pages with
  `gpt-5.4-mini` (structured output via `llm_structured`) and averages the input
  tokens from `response.usage`.
- **Q2/Q3** — takes the first ground-truth question and runs text search vs
  vector search (the question came from `01-intro.md`; vector finds it at the
  top, text does not — a nice illustration of why we measure across the whole
  set).
- **Q4/Q5** — evaluates each method over all 360 ground-truth questions with
  Hit Rate and MRR.
- **Q6** — sweeps the RRF constant `k` over `[1, 50, 100, 200]` for hybrid
  search and reports the `k` with the best MRR (ties → smallest `k`).

## How to run

This uses [uv](https://docs.astral.sh/uv/) and continues from Homework 2.

1. Install dependencies:

   ```bash
   uv add onnxruntime tokenizers numpy tqdm minsearch gitsource openai pydantic python-dotenv pandas
   uv add --dev huggingface-hub
   ```

2. Download the ONNX model:

   ```bash
   uv run python download.py
   ```

3. Set your OpenAI key (Q1 makes 3 LLM calls):

   ```bash
   echo 'OPENAI_API_KEY=sk-...' > .env
   ```

4. Run it:

   ```bash
   uv run python homework4.py
   ```

Output:

```
Q1: 1354
Q2: 01-agentic-rag/lessons/03-rag.md
Q3: 01-agentic-rag/lessons/01-intro.md
Q4: 0.76
Q5: 0.55
   k=1: mrr=0.6482
   k=50: mrr=0.6379
   k=100: mrr=0.6379
   k=200: mrr=0.6379
Q6: 1
```

`embedder.py`, `download.py`, `rag_helper.py`, and `evaluation_utils.py` are
helper scripts from the course repo. `ground-truth.csv` (360 questions) is the
pre-generated dataset provided with the homework.

> Q1 varies slightly between runs; it stays in the same order of magnitude, so
> the closest option is 1400.
