# Implementation guide: from offline build to Azure

> Status: **planned**. Nothing here has been deployed. The offline build in this
> repository is what exists today; this guide is how it would be taken to a
> real Azure environment.

## Azure service mapping

| Offline component | Azure service (planned) | Notes |
|---|---|---|
| Chunk retrieval | Azure AI Search hybrid (vector + keyword) with semantic ranker |  |
| Generation | Azure OpenAI chat model | Keep the grounding critique |
| Triple extraction | Azure OpenAI with structured outputs, or Azure AI Language |  |
| Graph store | Cosmos DB for Apache Gremlin, or Neo4j on Azure |  |
| Eval | Azure AI Foundry evaluations, with `evaluate_suite` kept in CI |  |

## Steps

1. Replace `retrieve` with Azure AI Search; keep the relevance threshold and corrective loop.
2. Replace extractive `generate` with an LLM and keep `critique_grounding` as a gate.
3. Extract triples with an LLM under a fixed schema; persist the graph.
4. Grow the gold set from real user questions and run eval on every change.

## Guardrails to keep when going live

- Grounding critique: answer sentences without token support in retrieved chunks are rejected.
- Bounded corrective loop (3 iterations), with the full decision log returned.
- GraphRAG answers cite the entity-relation chain they used.
- Fallback to chunk retrieval instead of guessing when the graph has no coverage.

## What to measure before calling it production-ready

- The four metrics on a gold set of 100+ real questions.
- Share of answers rejected by the grounding check.
- Multi-hop accuracy compared with plain top-k retrieval.
