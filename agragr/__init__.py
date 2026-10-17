"""Agentic RAG: knowledge-graph construction, multi-hop GraphRAG,
self-RAG/corrective-RAG loop, and RAGAS-style evaluation — all
deterministic, no network, no model keys."""

from .extract import extract_triples, link_entities, Triple
from .graph import KnowledgeGraph, PathStep
from .graphrag import GraphRAG, GraphAnswer
from .selfrag import SelfRAG, RagResult, Chunk, retrieve, score_relevance, critique_grounding
from .eval import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    evaluate_suite,
    EvalReport,
)

__all__ = [
    "extract_triples",
    "link_entities",
    "Triple",
    "KnowledgeGraph",
    "PathStep",
    "GraphRAG",
    "GraphAnswer",
    "SelfRAG",
    "RagResult",
    "Chunk",
    "retrieve",
    "score_relevance",
    "critique_grounding",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "evaluate_suite",
    "EvalReport",
]
