import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch
from embedder import Embedder
from evaluation_utils import llm_structured

load_dotenv()


reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]


class Questions(BaseModel):
    questions: list[str]


data_gen_instructions = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()


client = OpenAI()

first_three = [
    "01-agentic-rag/lessons/01-intro.md",
    "01-agentic-rag/lessons/02-environment.md",
    "01-agentic-rag/lessons/03-rag.md",
]
by_name = {d["filename"]: d for d in documents}

input_tokens = []
for filename in first_three:
    doc = by_name[filename]
    user_prompt = json.dumps({"filename": doc["filename"], "content": doc["content"]})
    parsed, usage = llm_structured(client, data_gen_instructions, user_prompt, Questions)
    input_tokens.append(usage.input_tokens)

print("Q1:", round(sum(input_tokens) / len(input_tokens)))


ground_truth = pd.read_csv("ground-truth.csv").to_dict("records")


chunks = chunk_documents(documents, size=2000, step=1000)

text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)

embedder = Embedder()
X = embedder.encode_batch([c["content"] for c in chunks])
vector_index = VectorSearch(keyword_fields=["filename"])
vector_index.fit(X, chunks)


def text_search(query, num_results=5):
    return text_index.search(query, num_results=num_results)


def vector_search(query, num_results=5):
    q = embedder.encode(query)
    return vector_index.search(q, num_results=num_results)


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def hybrid_search(query, k=60):
    text_results = text_search(query, num_results=10)
    vector_results = vector_search(query, num_results=10)
    return rrf([text_results, vector_results], k=k)


q = ground_truth[0]["question"]
print("Q2:", text_search(q)[0]["filename"])
print("Q3:", vector_search(q)[0]["filename"])


def compute_relevance(search_function, record):
    results = search_function(record["question"])
    return [r["filename"] == record["filename"] for r in results]


def hit_rate(relevance_total):
    return sum(any(line) for line in relevance_total) / len(relevance_total)


def mrr(relevance_total):
    total = 0.0
    for line in relevance_total:
        for rank, hit in enumerate(line):
            if hit:
                total += 1 / (rank + 1)
                break
    return total / len(relevance_total)


def evaluate(search_function):
    relevance_total = [compute_relevance(search_function, rec) for rec in ground_truth]
    return {"hit_rate": hit_rate(relevance_total), "mrr": mrr(relevance_total)}


text_metrics = evaluate(text_search)
print("Q4:", round(text_metrics["hit_rate"], 2))

vector_metrics = evaluate(vector_search)
print("Q5:", round(vector_metrics["mrr"], 2))


best_k = None
best_mrr = -1.0
for k in [1, 50, 100, 200]:
    m = evaluate(lambda query, k=k: hybrid_search(query, k=k))
    if m["mrr"] > best_mrr:
        best_mrr = m["mrr"]
        best_k = k
    print(f"   k={k}: mrr={m['mrr']:.4f}")
print("Q6:", best_k)
