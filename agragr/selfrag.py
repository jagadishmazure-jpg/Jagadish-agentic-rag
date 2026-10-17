"""Self-RAG loop: retrieve, reflect, and correct.

Pipeline per iteration:
    retrieve -> score_relevance -> generate -> critique_grounding
If relevance is low, ``corrective_retrieve`` reformulates the query
(keyword expansion + entity aliases) and the loop regenerates.
Iterations are bounded; every decision is logged in a RetrievalLog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import ENTITY_ALIASES, link_entities


@dataclass
class Chunk:
    id: str
    text: str


@dataclass
class LogEntry:
    iteration: int
    action: str
    detail: str


@dataclass
class RagResult:
    answer: str
    chunks_used: list[str]
    iterations: int
    log: list[LogEntry]
    grounded: bool


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9$]+", text.lower()))


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "for", "to", "and", "or", "it", "its", "by", "with", "what",
    "who", "where", "when", "how", "does", "do", "did", "much", "many",
    "that", "this", "as",
}


def _content_tokens(text: str) -> set[str]:
    return _tokenize(text) - _STOP


def retrieve(query: str, chunks: list[Chunk], k: int = 2) -> list[Chunk]:
    """Keyword retrieval: rank chunks by content-token overlap with query."""
    qtok = _content_tokens(query)
    scored = sorted(
        chunks,
        key=lambda c: len(qtok & _content_tokens(c.text)),
        reverse=True,
    )
    return [c for c in scored[:k] if len(qtok & _content_tokens(c.text)) > 0]


def score_relevance(question: str, chunks: list[Chunk]) -> float:
    """Fraction of question content-tokens covered by retrieved chunks."""
    qtok = _content_tokens(question)
    if not qtok:
        return 0.0
    covered: set[str] = set()
    for c in chunks:
        covered |= qtok & _content_tokens(c.text)
    return len(covered) / len(qtok)


def generate(question: str, chunks: list[Chunk]) -> str:
    """Extractive generation: pick the highest-overlap sentences."""
    qtok = _content_tokens(question)
    sentences: list[tuple[int, str, str]] = []
    for c in chunks:
        for sent in re.split(r"(?<=[.!?])\s+", c.text.strip()):
            if sent.strip():
                overlap = len(qtok & _content_tokens(sent))
                sentences.append((overlap, c.id, sent.strip()))
    sentences.sort(key=lambda s: s[0], reverse=True)
    picked = [s for s in sentences if s[0] > 0][:2]
    return " ".join(s[2] for s in picked)


def critique_grounding(
    answer: str, chunks: list[Chunk], threshold: float = 0.5
) -> tuple[bool, list[str]]:
    """Every answer sentence must be token-supported by some chunk
    sentence: overlap / answer-sentence-tokens >= threshold."""
    chunk_sents: list[set[str]] = []
    for c in chunks:
        for sent in re.split(r"(?<=[.!?])\s+", c.text.strip()):
            if sent.strip():
                chunk_sents.append(_content_tokens(sent))
    unsupported: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", answer.strip()):
        stok = _content_tokens(sent)
        if not stok:
            continue
        best = max(
            (len(stok & cs) / len(stok) for cs in chunk_sents),
            default=0.0,
        )
        if best < threshold:
            unsupported.append(sent.strip())
    return (len(unsupported) == 0, unsupported)


def corrective_retrieve(
    question: str, chunks: list[Chunk], k: int = 2
) -> tuple[str, list[Chunk]]:
    """Query reformulation: expand with entity aliases + synonyms, retry."""
    expanded = question
    for ent in link_entities(question):
        for alias in ENTITY_ALIASES.get(ent, []):
            expanded += " " + alias
    synonyms = {"cost": "price", "price": "cost", "warranty": "guarantee"}
    for w, syn in synonyms.items():
        if w in question.lower():
            expanded += " " + syn
    return expanded, retrieve(expanded, chunks, k=k)


class SelfRAG:
    def __init__(
        self,
        chunks: list[Chunk],
        relevance_threshold: float = 0.4,
        max_iterations: int = 3,
    ) -> None:
        self.chunks = chunks
        self.relevance_threshold = relevance_threshold
        self.max_iterations = max_iterations

    def answer(self, question: str) -> RagResult:
        log: list[LogEntry] = []
        retrieved = retrieve(question, self.chunks)
        log.append(
            LogEntry(1, "retrieve",
                      f"query='{question}' -> {[c.id for c in retrieved]}")
        )
        iteration = 1
        while iteration <= self.max_iterations:
            relevance = score_relevance(question, retrieved)
            log.append(
                LogEntry(iteration, "score_relevance", f"{relevance:.2f}")
            )
            if relevance < self.relevance_threshold and iteration < self.max_iterations:
                expanded, retrieved = corrective_retrieve(question, self.chunks)
                log.append(
                    LogEntry(
                        iteration,
                        "corrective_retrieve",
                        f"reformulated='{expanded}' -> {[c.id for c in retrieved]}",
                    )
                )
                iteration += 1
                continue
            break
        answer = generate(question, retrieved)
        log.append(LogEntry(iteration, "generate", f"answer='{answer[:80]}'"))
        grounded, unsupported = critique_grounding(answer, retrieved)
        log.append(
            LogEntry(
                iteration,
                "critique_grounding",
                f"grounded={grounded}; unsupported={unsupported}",
            )
        )
        if not answer:
            answer = "I don't have relevant information to answer that question."
        return RagResult(
            answer=answer,
            chunks_used=[c.id for c in retrieved],
            iterations=iteration,
            log=log,
            grounded=grounded,
        )
