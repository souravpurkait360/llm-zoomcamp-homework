# LLM Zoomcamp 2026 — Homework 1: Agentic RAG

My solution for [Module 1: Agentic RAG](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/01-agentic-rag/homework.md).

We build a RAG system from scratch over the course lesson pages and then turn it
into an agent. The knowledge base is the course lessons themselves, pulled from
the repository at the pinned commit `8c1834d` so everyone works with the exact
same data.

## Answers

| # | Question | Answer |
|---|----------|--------|
| Q1 | How many lesson pages | **72** |
| Q2 | First search result | **`01-agentic-rag/lessons/14-agentic-loop.md`** |
| Q3 | Input (prompt) tokens | **7000** (≈7036) |
| Q4 | Number of chunks | **295** |
| Q5 | RAG with chunking | **3× fewer** (≈3.17×) |
| Q6 | Agent `search` calls | **4** |

## What the script does

- **Q1–Q2** — reads the lesson pages with `gitsource`, indexes them in `minsearch`
  (`content` as a text field, `filename` as a keyword field), and runs the query.
- **Q3** — builds a RAG over the index, calls `gpt-5.4-mini`, and reads the input
  tokens from `response.usage.input_tokens`.
- **Q4** — chunks the pages with a sliding window (`size=2000`, `step=1000`).
- **Q5** — points the same RAG at the chunk index and compares input tokens.
- **Q6** — gives the LLM a `search` tool (via `toyaikit`) and lets the agent
  decide when and what to search, counting the `search` calls.

## How to run

This uses [uv](https://docs.astral.sh/uv/) for dependency management.

1. Install the dependencies:

   ```bash
   uv add gitsource minsearch toyaikit openai python-dotenv
   ```

2. Set your OpenAI API key (Q3, Q5, and Q6 call the model):

   ```bash
   echo 'OPENAI_API_KEY=sk-...' > .env
   ```

3. Run it from the repository root:

   ```bash
   uv run python 01-agentic-rag/homework1.py
   ```

Output:

```
Q1: 72
Q2: 01-agentic-rag/lessons/14-agentic-loop.md
Q3: 7036
Q4: 295
Q5: 3.17 x fewer
Q6: 4
```

> Q3 and Q6 vary slightly between runs (message overhead and the agent deciding
> its own number of searches), so the numbers may differ a little — pick the
> closest option.
