"""GraphRAG multi-hop answer, a self-RAG run with its decision log, and the eval suite."""
from pathlib import Path

from agragr import Chunk, GraphRAG, KnowledgeGraph, SelfRAG, evaluate_suite, extract_triples

data = Path(__file__).resolve().parent.parent / "data"
texts = {p.stem: p.read_text() for p in sorted(data.glob("doc-*.md"))}

graph = KnowledgeGraph()
for text in texts.values():
    graph.add_triples(extract_triples(text))
print(f"Graph: {len(graph.entities)} entities, {graph.triple_count} triples from {len(texts)} documents")

rag = GraphRAG(graph, texts)
ans = rag.answer("What gripper does NovaBot use and how much does it cost?")
print("GraphRAG answer:", ans.answer)
print("Evidence chain:", ans.evidence)

self_rag = SelfRAG([Chunk(id=c, text=t) for c, t in texts.items()])
res = self_rag.answer("What is the battery life of NovaBot?")
print("Self-RAG answer:", res.answer)
print("Grounded:", res.grounded, "| iterations:", res.iterations)
for entry in res.log:
    print("  log:", entry.iteration, entry.action, entry.detail)

report = evaluate_suite(self_rag.answer, data)
print("Eval over gold set:", {k: round(v, 3) for k, v in report.means.items()})
