# Interview points

Short, honest talking points for Jagadish-agentic-rag. Every number quoted is from the
real run shown in the README.

- A graph turns multi-hop retrieval into navigation: each hop is inspectable.
- A bug I caught: chain scoring preferred a longer coincidental path over the direct fact; negative tests fixed it.
- Self-RAG is a control loop: relevance is the signal, reformulation the correction.
- I am explicit that 7 gold questions and token-overlap metrics are regression checks, not proof of quality.

## Likely follow-up questions

- When is GraphRAG worth it? (Questions that join facts across documents; not for single-passage lookups.)
- How would you evaluate with an LLM judge? (Calibrate it against human labels first.)
- What breaks at scale? (Extraction quality and entity resolution.)
