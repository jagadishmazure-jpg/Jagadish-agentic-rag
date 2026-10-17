# Agentic RAG: concepts

## Why "beyond basic RAG"

Basic RAG (chunk → embed → top-k → generate) fails in three predictable
ways: the answer spans multiple documents (multi-hop), the retrieval
pulls plausible-but-irrelevant context (low precision), and nobody can
say whether the system is actually good (no eval). This repo implements
the three production answers: **GraphRAG**, **self-RAG / corrective-RAG**,
and **RAGAS-style evals** — all deterministic and testable offline.

## 1. Knowledge-graph construction (`agragr/extract.py`, `agragr/graph.py`)

Raw text → entities + relations → a directed multigraph.

- Deterministic entity/relation extraction (pattern-based; no LLM needed
  for the demo corpus).
- Multi-hop traversal via BFS over the graph: questions that need
  "who supplied the part that caused the failure" are answered by
  walking edges, not by hoping two chunks co-occur in top-k.
- Interview line: *"The graph turns retrieval into navigation — each
  hop is an inspectable step, which is why a KG beats vector search for
  multi-hop questions."*

## 2. GraphRAG (`agragr/graphrag.py`)

Entity-link the question → score candidate evidence chains → generate
with chain citations.

- Chain scoring ranks competing evidence paths; the answer cites the
  chain it used.
- A real bug caught in verification: chain scoring preferred a longer
  *coincidental* path over the direct fact. Fixed — and it's a good
  interview story about why retrieval scoring needs negative tests.

## 3. Self-RAG / corrective-RAG (`agragr/selfrag.py`)

Retrieve → relevance-score → generate → grounding critique, with
corrective query reformulation when relevance is low.

- The loop: if retrieved context scores below the relevance threshold,
  reformulate the query and retrieve again (bounded retries).
- Another verification catch: the fallback was matching stop words
  against every document — fixed to content-bearing terms only.
- Interview line: *"Self-RAG is a control loop around retrieval — the
  relevance score is the feedback signal, and reformulation is the
  correction."*

## 4. RAGAS-style evals (`agragr/eval.py`)

Deterministic implementations of the four canonical metrics, run over a
gold Q&A set on the synthetic corpus:

- **Faithfulness** — is every claim in the answer grounded in context?
- **Answer relevancy** — does the answer address the question?
- **Context precision** — is the retrieved context mostly relevant?
- **Context recall** — did retrieval surface everything needed?

Eval is what turns "the demo worked once" into "the system is good."
This is the module you'd extend first in production (swap deterministic
scorers for LLM judges, keep the harness).

## Module map

| Module | Role |
|---|---|
| `agragr/extract.py` | entity/relation extraction |
| `agragr/graph.py` | directed multigraph + multi-hop BFS |
| `agragr/graphrag.py` | entity linking → chain scoring → cited answers |
| `agragr/selfrag.py` | retrieve → score → generate → critique loop |
| `agragr/eval.py` | faithfulness / relevancy / precision / recall |
