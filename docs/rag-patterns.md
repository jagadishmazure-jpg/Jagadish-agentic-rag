# RAG patterns: when to use which

## Naive RAG

Retrieve top-k chunks by similarity, stuff them into a prompt, generate.
One pass, no reflection.

**Use when:** the answer lives in a single passage and retrieval quality
is high. **Fails when:** the question needs multiple facts in
relationship, or retrieval misses and the model answers anyway.

## Self-RAG

The pipeline reflects on its own retrieval: score whether the retrieved
chunks actually cover the question before generating. Implemented here
in `agragr/selfrag.py` (`score_relevance` + `critique_grounding`).

**Use when:** retrieval quality varies and you need the system to *know*
when it doesn't know. The grounding critique is the key property: every
answer sentence must be traceable to retrieved text.

## Corrective RAG

Extends self-RAG: when relevance is low, don't just abstain —
reformulate the query (keyword expansion, entity aliases, synonyms) and
retrieve again, bounded by a max iteration count. Implemented here in
`corrective_retrieve`; every reformulation is logged.

**Use when:** queries are terse, jargon-heavy, or use different
vocabulary than the corpus (*gripper* vs *TitanArm*). The failure mode
it fixes is vocabulary mismatch, not missing knowledge.

## GraphRAG

Build a knowledge graph from the corpus (entities + relations), then
answer by traversing multi-hop paths instead of ranking chunks.
Implemented here in `agragr/graph.py` + `agragr/graphrag.py`: entity
linking → intent detection → BFS traversal → chain scoring → verbalized
evidence with citations.

**Use when:** questions join facts across documents — *"What does the
gripper on NovaBot cost?"* needs `NovaBot --uses--> TitanArm` *and*
`TitanArm --costs--> $12,000`. Chunk similarity sees two unrelated
passages; the graph sees one connected answer.

**Trade-off:** graph quality is everything. A bad extractor produces a
bad graph, and traversal can't fix that. Keep the chunk fallback for
questions with no graph coverage.

## How they combine here

```
question ─┬─ entities found? ──yes──▶ GraphRAG (multi-hop, cited chains)
          │                              │ no coverage
          └─▶ SelfRAG loop (retrieve → reflect → correct → ground)
```

GraphRAG for relational questions, the self-RAG loop for prose
questions, and `evaluate_suite` (`agragr/eval.py`) measuring
faithfulness, answer relevancy, context precision, and context recall
over a gold set so regressions in either path show up separately.
