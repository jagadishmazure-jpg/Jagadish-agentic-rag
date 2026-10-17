# Agentic RAG: knowledge graphs, self-correction, and honest evaluation

Naive chunk-retrieve-and-generate RAG has two failure modes this repo
addresses head-on:

1. **It can't do multi-hop reasoning.** "What does the gripper on
   NovaBot cost?" needs two facts — *NovaBot uses TitanArm as its
   gripper*, *TitanArm costs $12,000* — and chunk similarity rarely
   surfaces both in the right relationship.
2. **It answers confidently from irrelevant context.** When retrieval
   misses, the generator still writes an answer. Nothing in the loop
   checks whether the answer is actually grounded.

`agragr` (agentic + graph RAG) implements the three patterns that fix
this, plus the evaluation to prove it — all deterministic, no network,
no model keys. The demo corpus is a fictional company (Acme Robotics);
swap in a real extractor, vector store, and LLM for production.

## Architecture

```
data/*.md ──extract_triples()──▶ KnowledgeGraph ──GraphRAG.answer()──▶ answer + cited chain
        │                              │
        │                         multi_hop BFS
        │
        └─▶ chunks ──SelfRAG.answer()──▶ retrieve → score → generate → critique
                       │                     ▲              │
                       │                     │         low relevance?
                       └──── corrective_retrieve (reformulate + aliases)

evaluate_suite() ──▶ faithfulness / answer_relevancy / context_precision / context_recall
```

## Quickstart

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

```python
from pathlib import Path
from agragr import extract_triples, KnowledgeGraph, GraphRAG, SelfRAG, Chunk, evaluate_suite

data = Path("data")
texts = {p.stem: p.read_text() for p in sorted(data.glob("doc-*.md"))}

# Build the knowledge graph
graph = KnowledgeGraph()
for text in texts.values():
    graph.add_triples(extract_triples(text))

# Multi-hop GraphRAG: answer cites the entity-relation chain
rag = GraphRAG(graph, texts)
ans = rag.answer("What gripper does NovaBot use and how much does it cost?")
print(ans.answer)     # $12,000 (NovaBot -> uses -> TitanArm -> costs -> $12,000).
print(ans.evidence)   # ['NovaBot --uses--> TitanArm', 'TitanArm --costs--> $12,000']

# Self-RAG: retrieve, reflect, reformulate when retrieval is weak
self_rag = SelfRAG([Chunk(id=c, text=t) for c, t in texts.items()])
res = self_rag.answer("What is the battery life of NovaBot?")
print(res.answer, res.grounded, res.iterations)
for entry in res.log:  # every decision is logged
    print(entry.iteration, entry.action, entry.detail)

# RAGAS-style evaluation over the gold Q&A set
report = evaluate_suite(self_rag.answer, data)
print(report.means)
```

## Example multi-hop walkthrough

Question: *"What gripper does NovaBot use and how much does it cost?"*

1. **Entity linking** finds `NovaBot` and `TitanArm` in the question.
2. **Intent detection** maps *gripper* → `uses`/`uses_as`, *cost* → `costs`.
3. **Traversal** explores 2-hop paths from both entities and scores
   chains by intent coverage: `NovaBot --uses--> TitanArm --costs--> $12,000`
   covers both intents and wins.
4. **Answer** leads with the final hop and cites the full chain, so the
   reasoning is auditable — not just the answer.

When the graph has no coverage (no linked entities, or no intent match),
the system falls back to keyword chunk retrieval rather than
hallucinating.

## The self-RAG loop

For questions better answered from prose than from triples:

1. `retrieve` ranks chunks by content-token overlap.
2. `score_relevance` measures question-term coverage.
3. Below threshold → `corrective_retrieve` reformulates the query
   (entity aliases like *gripper → TitanArm*, plus synonyms) and retries.
4. `generate` is extractive; `critique_grounding` rejects any answer
   sentence not token-supported by a retrieved chunk.
5. Bounded at 3 iterations; the full decision log ships with the result.

## Production path

- **Extractor**: replace the regex/gazetteer `extract_triples` with an
  NER + relation-extraction model (or an LLM with constrained decoding).
- **Retrieval**: replace keyword overlap with a vector store
  (embeddings + ANN search); keep the relevance threshold and the
  corrective loop — they are retrieval-agnostic.
- **Generation**: replace extractive generation with an LLM, keeping
  `critique_grounding` as the guardrail (or upgrading it to an NLI model).
- **Graph store**: persist the knowledge graph in Neo4j / Neptune for
  scale; the traversal and verbalization logic is unchanged.
- **Eval**: run `evaluate_suite` in CI over a growing gold set; the four
  metrics catch regressions in retrieval and grounding separately.

## Disclaimer

All documents describe the fictional Acme Robotics. This is a patterns
reference implementation, not legal, financial, or business advice.

## Knowledge

Distilled notes and architecture diagrams: [knowledge/](knowledge/README.md)
