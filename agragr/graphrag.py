"""GraphRAG: answer questions by traversing the knowledge graph.

Pipeline: entity-link the question -> multi-hop traversal -> verbalize
paths as evidence -> compose an answer citing entity-relation chains.
Falls back to plain chunk retrieval when the graph has no coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .extract import link_entities
from .graph import KnowledgeGraph, PathStep


@dataclass
class GraphAnswer:
    answer: str
    evidence: list[str]        # verbalized chains
    cited_chain: list[PathStep]  # the chain the answer was built from
    source: str                # "graph" or "chunk_fallback"


# Question intent keywords -> relations that can satisfy them.
_INTENT_RELATIONS: dict[str, list[str]] = {
    "cost": ["costs"],
    "price": ["costs"],
    "warranty": ["warranty_years"],
    "who": ["leads_team", "has_role", "founded", "previously_worked_at"],
    "lead": ["leads_team"],
    "leads": ["leads_team"],
    "found": ["founded", "founded_in"],
    "founded": ["founded", "founded_in"],
    "headquarter": ["headquartered_in"],
    "where": ["headquartered_in", "located_in", "based_in"],
    "located": ["located_in"],
    "based": ["based_in"],
    "battery": ["battery_life"],
    "payload": ["payload"],
    "work": ["previously_worked_at", "joined"],
    "worked": ["previously_worked_at"],
    "join": ["joined", "joined_in"],
    "gripper": ["uses", "uses_as"],
    "use": ["uses", "uses_as"],
    "made": ["made_by"],
    "support": ["support_for", "support_hours"],
}


def _question_intents(question: str) -> list[str]:
    words = re.findall(r"[a-z]+", question.lower())
    intents: list[str] = []
    for w in words:
        if w in _INTENT_RELATIONS and w not in intents:
            intents.append(w)
    return intents


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "for", "to", "and", "or", "it", "its", "by", "with", "what",
    "who", "where", "when", "how", "does", "do", "did", "much", "many",
    "that", "this", "as",
}


def _content_tokens(text: str) -> set[str]:
    return _tokenize(text) - _STOP


class GraphRAG:
    def __init__(self, graph: KnowledgeGraph, chunks: dict[str, str]) -> None:
        self.graph = graph
        self.chunks = chunks  # chunk_id -> text (for fallback retrieval)

    def _find_answer_chain(
        self, entities: list[str], intents: list[str]
    ) -> list[PathStep] | None:
        """Best chain: covers the most question intents; ties broken by
        preferring a final hop that directly answers an intent, then the
        shortest path (a direct fact beats a coincidental detour)."""
        intent_rel_sets = {i: set(_INTENT_RELATIONS[i]) for i in intents}
        wanted = {r for rs in intent_rel_sets.values() for r in rs}
        best: list[PathStep] | None = None
        best_score: tuple[int, int, int] | None = None
        for ent in entities:
            for path in self.graph.multi_hop(ent, max_hops=2):
                rels = {s.relation for s in path}
                covered = sum(1 for i in intents if rels & intent_rel_sets[i])
                if covered == 0:
                    continue
                final_wanted = 1 if path[-1].relation in wanted else 0
                score = (covered, final_wanted, -len(path))
                if best_score is None or score > best_score:
                    best, best_score = path, score
        return best

    def _chunk_fallback(self, question: str) -> GraphAnswer:
        """Keyword chunk retrieval when the graph has no coverage."""
        qtok = _content_tokens(question)
        scored = []
        for cid, text in self.chunks.items():
            overlap = len(qtok & _content_tokens(text))
            scored.append((overlap, cid, text))
        scored.sort(reverse=True)
        top = [s for s in scored if s[0] > 0][:2]
        if not top:
            return GraphAnswer(
                answer="I don't have information to answer that question.",
                evidence=[],
                cited_chain=[],
                source="chunk_fallback",
            )
        evidence = [f"[{cid}] {text.strip()}" for _, cid, text in top]
        answer = " ".join(t.strip() for _, _, t in top)
        return GraphAnswer(answer, evidence, [], "chunk_fallback")

    def answer(self, question: str) -> GraphAnswer:
        entities = link_entities(question)
        intents = _question_intents(question)
        if not entities:
            return self._chunk_fallback(question)
        chain = self._find_answer_chain(entities, intents)
        if chain is None:
            return self._chunk_fallback(question)
        evidence = [s.verbalize() for s in chain]
        # Compose: lead with the answer-bearing final hop, cite full chain.
        final = chain[-1]
        chain_str = " -> ".join(
            [chain[0].subject]
            + [f"{s.relation} -> {s.obj}" for s in chain]
        )
        answer = (
            f"{final.obj} ({chain_str})."
        )
        return GraphAnswer(answer, evidence, chain, "graph")
