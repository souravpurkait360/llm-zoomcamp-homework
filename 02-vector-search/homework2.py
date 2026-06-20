import numpy as np
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import VectorSearch, Index
from embedder import Embedder


embedder = Embedder()


query = "How does approximate nearest neighbor search work?"
v = embedder.encode(query)
print("Q1:", round(float(v[0]), 2))


reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]


target = next(d for d in documents if d["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md")
page_vector = embedder.encode(target["content"])
print("Q2:", round(float(page_vector.dot(v)), 2))


chunks = chunk_documents(documents, size=2000, step=1000)
X = embedder.encode_batch([c["content"] for c in chunks])
scores = X.dot(v)
best = chunks[int(scores.argmax())]
print("Q3:", best["filename"])


vindex = VectorSearch(keyword_fields=["filename"])
vindex.fit(X, chunks)

q4_query = "What metric do we use to evaluate a search engine?"
q4_vector = embedder.encode(q4_query)
q4_results = vindex.search(q4_vector, num_results=5)
print("Q4:", q4_results[0]["filename"])


tindex = Index(text_fields=["content"], keyword_fields=["filename"])
tindex.fit(chunks)

q5_query = "How do I store vectors in PostgreSQL?"
q5_vec = embedder.encode(q5_query)
vec_results = vindex.search(q5_vec, num_results=5)
text_results = tindex.search(q5_query, num_results=5)

vec_files = {r["filename"] for r in vec_results}
text_files = {r["filename"] for r in text_results}
print("Q5:", sorted(vec_files - text_files))


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


q6_query = "How do I give the model access to tools?"
q6_vec = embedder.encode(q6_query)
vector_results = vindex.search(q6_vec, num_results=5)
text_results = tindex.search(q6_query, num_results=5)
results = rrf([vector_results, text_results])
print("Q6:", results[0]["filename"])
