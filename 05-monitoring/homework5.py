import os
import sqlite3

from dotenv import load_dotenv

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
    SimpleSpanProcessor,
)

load_dotenv()

DB_PATH = "traces.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)


def calc_cost(input_tokens, output_tokens):
    return (input_tokens / 1_000_000) * 0.75 + (output_tokens / 1_000_000) * 4.50


class SQLiteSpanExporter(SpanExporter):
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self, timeout_millis=None):
        return True


provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter(DB_PATH)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-zoomcamp")

from starter import rag
from rag_helper import RAGBase


class RAGTraced(RAGBase):
    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search"):
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)
            usage = response.usage
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)
            span.set_attribute("cost", calc_cost(usage.input_tokens, usage.output_tokens))
            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag"):
            return super().rag(query)


traced = RAGTraced(index=rag.index, llm_client=rag.llm_client)

query = "How does the agentic loop keep calling the model until it stops?"

for _ in range(4):
    traced.rag(query)

provider.force_flush()


conn = sqlite3.connect(DB_PATH)

rows = conn.execute("SELECT name FROM spans").fetchall()
per_call_spans = len(rows) // 4
print("Q1:", per_call_spans)

first_llm = conn.execute(
    "SELECT input_tokens FROM spans WHERE name = 'llm' LIMIT 1"
).fetchone()
print("Q2:", first_llm[0])

durations = conn.execute("""
    SELECT name, AVG((end_time - start_time) / 1e6) AS ms
    FROM spans GROUP BY name
""").fetchall()
dur = {name: ms for name, ms in durations}
print("Q3 (llm ms):", round(dur.get("llm", 0), 1))

names = sorted({r[0] for r in rows})
print("Q4 span names:", names)

totals = conn.execute("""
    SELECT name, SUM(end_time - start_time) AS total
    FROM spans WHERE name != 'rag' GROUP BY name ORDER BY total DESC
""").fetchall()
print("Q5 (slowest, total ns):", totals)

tokens = [r[0] for r in conn.execute(
    "SELECT input_tokens FROM spans WHERE name = 'llm'"
).fetchall()]
lo, hi = min(tokens), max(tokens)
spread = (hi - lo) / lo * 100 if lo else 0
print("Q6 input_tokens per run:", tokens, f"-> spread {spread:.1f}%")

conn.close()
