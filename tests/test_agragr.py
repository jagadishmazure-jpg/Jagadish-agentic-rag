"""Tests for the agentic RAG package (agragr).

All deterministic: no network, no model keys. The synthetic Acme Robotics
corpus in data/ is the shared fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agragr import (
    Chunk,
    GraphRAG,
    KnowledgeGraph,
    SelfRAG,
    answer_relevancy,
    context_precision,
    context_recall,
    critique_grounding,
    evaluate_suite,
    extract_triples,
    faithfulness,
    link_entities,
    retrieve,
    score_relevance,
)
from agragr.eval import load_gold
from agragr.selfrag import corrective_retrieve, generate

DATA = Path(__file__).resolve().parent.parent / "data"


def _doc_texts() -> dict[str, str]:
    return {p.stem: p.read_text() for p in sorted(DATA.glob("doc-*.md"))}


def _chunks() -> list[Chunk]:
    return [Chunk(id=cid, text=t) for cid, t in _doc_texts().items()]


@pytest.fixture(scope="module")
def graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    seen: set[tuple[str, str, str]] = set()
    for text in _doc_texts().values():
        for t in extract_triples(text):
            key = (t.subject, t.relation, t.obj)
            if key not in seen:
                seen.add(key)
                g.add_triples([t])
    return g


@pytest.fixture(scope="module")
def graphrag(graph: KnowledgeGraph) -> GraphRAG:
    return GraphRAG(graph, _doc_texts())


@pytest.fixture(scope="module")
def selfrag() -> SelfRAG:
    return SelfRAG(_chunks())


# ---------------------------------------------------------------- extraction

def test_extract_made_by():
    triples = extract_triples("NovaBot is made by Acme Robotics.")
    assert ("NovaBot", "made_by", "Acme Robotics") in {
        (t.subject, t.relation, t.obj) for t in triples
    }


def test_extract_founded_and_founded_in():
    triples = extract_triples("Maya Chen founded Acme Robotics in 2019.")
    keys = {(t.subject, t.relation, t.obj) for t in triples}
    assert ("Maya Chen", "founded", "Acme Robotics") in keys
    assert ("Acme Robotics", "founded_in", "2019") in keys


def test_extract_costs():
    triples = extract_triples("TitanArm costs $12,000.")
    keys = {(t.subject, t.relation, t.obj) for t in triples}
    assert ("TitanArm", "costs", "$12,000") in keys


def test_extract_uses_and_uses_as():
    triples = extract_triples("NovaBot uses the TitanArm as its gripper module.")
    keys = {(t.subject, t.relation, t.obj) for t in triples}
    assert ("NovaBot", "uses", "TitanArm") in keys
    assert ("NovaBot", "uses_as", "gripper") in keys


def test_extract_source_spans_present():
    text = _doc_texts()["doc-03-leadership"]
    triples = extract_triples(text)
    assert len(triples) > 0
    for t in triples:
        start, end = t.source_span
        assert end > start
        assert t.source_sentence
        assert t.source_sentence in text[start:end]


def test_extract_warranty():
    triples = extract_triples("Acme Robotics offers a 3-year warranty on NovaBot.")
    keys = {(t.subject, t.relation, t.obj) for t in triples}
    assert ("NovaBot", "warranty_years", "3") in keys


def test_link_entities_prefers_longest_match():
    assert "NovaBot team" in link_entities("Raj Patel leads the NovaBot team.")
    assert "NovaBot" in link_entities("NovaBot costs $45,000.")


# --------------------------------------------------------------------- graph

def test_neighbors(graph: KnowledgeGraph):
    rels = {r for r, _ in graph.neighbors("NovaBot")}
    assert "uses" in rels
    assert "costs" in rels
    assert "made_by" in rels


def test_neighbors_relation_filter(graph: KnowledgeGraph):
    objs = [o for _, o in graph.neighbors("TitanArm", relation="costs")]
    assert objs == ["$12,000"]


def test_multi_hop_two_hops(graph: KnowledgeGraph):
    paths = graph.multi_hop("NovaBot", max_hops=2)
    two_hop = [p for p in paths if len(p) == 2]
    assert any(
        p[0].relation == "uses"
        and p[0].obj == "TitanArm"
        and p[1].relation == "costs"
        and p[1].obj == "$12,000"
        for p in two_hop
    )


def test_multi_hop_no_path(graph: KnowledgeGraph):
    assert graph.multi_hop("Boston Dynamics", max_hops=2) == []


def test_subgraph_for(graph: KnowledgeGraph):
    sg = graph.subgraph_for(["NovaBot", "Raj Patel"], hops=1)
    assert set(sg) == {"NovaBot", "Raj Patel"}
    assert any(len(p) == 1 for p in sg["NovaBot"])


# ------------------------------------------------------------------ graphrag

def test_graphrag_multihop_answer(graphrag: GraphRAG):
    ans = graphrag.answer("What gripper does NovaBot use and how much does it cost?")
    assert ans.source == "graph"
    assert "TitanArm" in ans.answer
    assert "$12,000" in ans.answer


def test_graphrag_cites_chain(graphrag: GraphRAG):
    ans = graphrag.answer("What gripper does NovaBot use and how much does it cost?")
    assert len(ans.cited_chain) == 2
    assert ans.cited_chain[0].relation == "uses"
    assert ans.cited_chain[1].relation == "costs"
    assert len(ans.evidence) == 2
    assert all("--" in e for e in ans.evidence)


def test_graphrag_warranty(graphrag: GraphRAG):
    ans = graphrag.answer("What is the warranty on NovaBot?")
    assert ans.source == "graph"
    assert "3" in ans.answer


def test_graphrag_fallback_no_entities(graphrag: GraphRAG):
    ans = graphrag.answer("What is the capital of France?")
    assert ans.source == "chunk_fallback"
    assert "don't have information" in ans.answer


def test_graphrag_fallback_no_intent_match(graphrag: GraphRAG):
    ans = graphrag.answer("What color is NovaBot?")
    assert ans.source == "chunk_fallback"
    assert ans.evidence  # fell back to chunk retrieval, found NovaBot docs


# ------------------------------------------------------------------- selfrag

def test_selfrag_answers_directly(selfrag: SelfRAG):
    res = selfrag.answer("What is the battery life of NovaBot?")
    assert "18 hours" in res.answer
    assert res.iterations == 1
    assert res.grounded is True


def test_selfrag_irrelevant_triggers_reformulation(selfrag: SelfRAG):
    res = selfrag.answer("What is the speed of light?")
    actions = [e.action for e in res.log]
    assert "corrective_retrieve" in actions
    assert "reformulated" in res.log[2].detail


def test_selfrag_bounded_iterations(selfrag: SelfRAG):
    res = selfrag.answer("What is the speed of light?")
    assert res.iterations == selfrag.max_iterations == 3


def test_critique_rejects_unsupported():
    chunks = _chunks()
    grounded, unsupported = critique_grounding("Penguins enjoy ice cream.", chunks)
    assert grounded is False
    assert unsupported == ["Penguins enjoy ice cream."]


def test_critique_accepts_supported():
    chunks = _chunks()
    grounded, unsupported = critique_grounding("NovaBot costs $45,000.", chunks)
    assert grounded is True
    assert unsupported == []


def test_retrieve_ranks_relevant_first():
    top = retrieve("Who founded Acme Robotics?", _chunks(), k=2)
    assert top[0].id == "doc-03-leadership"


def test_corrective_retrieve_expands_query():
    expanded, _ = corrective_retrieve("What does TitanArm cost?", _chunks())
    # 'TitanArm' links to aliases 'robotic arm'/'gripper'; 'cost' -> 'price'
    assert "robotic arm" in expanded
    assert "price" in expanded


# ---------------------------------------------------------------------- eval

def test_faithfulness_perfect():
    assert faithfulness("NovaBot costs $45,000.", ["NovaBot costs $45,000."]) == 1.0


def test_faithfulness_drops_on_unsupported():
    assert faithfulness("Penguins enjoy ice cream.", ["NovaBot costs $45,000."]) == 0.0


def test_answer_relevancy():
    assert answer_relevancy("What costs $45,000?", "NovaBot costs $45,000.") == 1.0
    assert answer_relevancy("What costs $45,000?", "Penguins enjoy ice cream.") == 0.0


def test_context_precision_recall():
    assert context_precision(["a", "b"], ["a"]) == 0.5
    assert context_recall(["a", "c"], ["a", "b"]) == 0.5
    assert context_precision([], ["a"]) == 0.0
    assert context_recall(["a"], []) == 0.0


def test_evaluate_suite_end_to_end(selfrag: SelfRAG):
    report = evaluate_suite(selfrag.answer, DATA)
    assert len(report.per_case) == 7
    for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        assert key in report.means
        assert 0.0 <= report.means[key] <= 1.0
    assert report.means["context_recall"] > 0.0
    assert report.means["faithfulness"] > 0.5
