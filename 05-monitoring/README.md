# LLM Zoomcamp 2026 — Homework 5: Monitoring

My solution for [Module 5: Monitoring](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/05-monitoring/homework.md).

We instrument the course-lessons RAG (from homework 1) with
[OpenTelemetry](https://opentelemetry.io/): wrap `rag`, `search`, and `llm` in
spans, capture tokens and cost as span attributes, persist spans to SQLite with
a custom exporter, and query the trace data.

## Answers

| # | Question | Answer |
|---|----------|--------|
| Q1 | Spans per trace | **3** (`rag`, `search`, `llm`) |
| Q2 | Input tokens (llm span) | **7000** (≈7111) |
| Q3 | LLM call duration | **Over 2000ms** (≈2.5s) |
| Q4 | Span names in the table | **`rag`, `search`, and `llm`** |
| Q5 | Slowest span type | **`llm`** |
| Q6 | Input-token variation over 4 runs | **They're identical** (7111 every run) |

## What the script does

- Sets up a `TracerProvider` with a custom `SQLiteSpanExporter` **before**
  importing `starter`, so the tracer is ready first.
- Defines `RAGTraced(RAGBase)` that wraps `rag()`, `search()`, and `llm()` each
  in its own span, and records `input_tokens`, `output_tokens`, and `cost` as
  attributes on the `llm` span.
- Runs the query 4 times (Q6 needs 4 RAG calls), flushes the spans to
  `traces.db`, then queries SQLite for each answer:
  - **Q1** spans per trace, **Q2** first `llm` span input tokens,
  - **Q3** average `llm` duration, **Q4** distinct span names,
  - **Q5** total duration per span excluding `rag`, **Q6** input tokens per run.

## How to run

This uses [uv](https://docs.astral.sh/uv/).

1. Install dependencies:

   ```bash
   uv add gitsource minsearch openai python-dotenv opentelemetry-api opentelemetry-sdk
   ```

2. Set your OpenAI key:

   ```bash
   echo 'OPENAI_API_KEY=sk-...' > .env
   ```

3. Run it:

   ```bash
   uv run python homework5.py
   ```

Output:

```
Q1: 3
Q2: 7111
Q3 (llm ms): 2521.2
Q4 span names: ['llm', 'rag', 'search']
Q5 (slowest, total ns): [('llm', 10084691000), ('search', 3778000)]
Q6 input_tokens per run: [7111, 7111, 7111, 7111] -> spread 0.0%
```

`starter.py` and `rag_helper.py` are the starter package from the course repo.

> Q2 and Q3 vary between runs (token overhead, network latency); the first LLM
> call can be slower due to cold start. Pick the closest / most-common option.
