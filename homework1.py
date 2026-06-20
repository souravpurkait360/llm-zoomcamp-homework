from dotenv import load_dotenv
load_dotenv()

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from openai import OpenAI


reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []
for file in files:
    doc = file.parse()
    documents.append(doc)

print("Q1:", len(documents))


query = "How does the agentic loop keep calling the model until it stops?"

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

results = index.search(query, num_results=5)
print("Q2:", results[0]["filename"])


INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


client = OpenAI()
MODEL = "gpt-5.4-mini"


def build_context(search_results):
    return "\n\n".join(d["content"] for d in search_results)


def build_prompt(query, search_results):
    context = build_context(search_results)
    return PROMPT_TEMPLATE.format(question=query, context=context)


def rag(query, search_index):
    search_results = search_index.search(query, num_results=5)
    prompt = build_prompt(query, search_results)

    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    )

    return response.output_text, response.usage


answer, usage = rag(query, index)
q3_tokens = usage.input_tokens
print("Q3:", q3_tokens)


chunks = chunk_documents(documents, size=2000, step=1000)
print("Q4:", len(chunks))


chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
chunk_index.fit(chunks)

answer, usage = rag(query, chunk_index)
q5_tokens = usage.input_tokens
print("Q5:", round(q3_tokens / q5_tokens, 2), "x fewer")


from toyaikit.tools import Tools
from toyaikit.llm import OpenAIClient
from toyaikit.chat.runners import OpenAIResponsesRunner, RunnerCallback


def search(query: str) -> list:
    """Search the course lessons for chunks relevant to the query.

    Args:
        query: The search query describing what to look for.
    Returns:
        A list of the most relevant lesson chunks.
    """
    return chunk_index.search(query, num_results=5)


tools = Tools()
tools.add_tool(search)

AGENT_INSTRUCTIONS = (
    "You're a course teaching assistant. Answer the student's question using the "
    "search tool. Make multiple searches with different keywords before answering."
)

agent_question = "How does the agentic loop work, and how is it different from plain RAG?"


class SearchCounter(RunnerCallback):
    def __init__(self):
        self.count = 0

    def on_function_call(self, function_call, result):
        if function_call.name == "search":
            self.count += 1

    def on_message(self, message):
        pass

    def on_reasoning(self, reasoning):
        pass

    def on_response(self, response):
        pass


runner = OpenAIResponsesRunner(
    tools=tools,
    developer_prompt=AGENT_INSTRUCTIONS,
    chat_interface=None,
    llm_client=OpenAIClient(model=MODEL, client=client),
)

counter = SearchCounter()
runner.loop(prompt=agent_question, callback=counter)
print("Q6:", counter.count)
