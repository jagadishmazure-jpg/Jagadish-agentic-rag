"""RAGAS-style evaluation, fully deterministic.

Metrics (all in [0, 1]):
- faithfulness:      fraction of answer sentences entailed by the context
                     (token-overlap entailment against context sentences)
- answer_relevancy:   question-answer content-term overlap
- context_precision:  fraction of retrieved chunks actually used
- context_recall:     fraction of gold chunks retrieved
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


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


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def faithfulness(answer: str, contexts: list[str], threshold: float = 0.5) -> float:
    """Fraction of answer sentences supported by at least one context
    sentence (support = token overlap / answer-sentence tokens >= threshold)."""
    ans_sents = _sentences(answer)
    if not ans_sents:
        return 0.0
    ctx_sents = [s for ctx in contexts for s in _sentences(ctx)]
    ctx_tok = [_content_tokens(s) for s in ctx_sents]
    supported = 0
    for sent in ans_sents:
        stok = _content_tokens(sent)
        if not stok:
            supported += 1
            continue
        best = max((len(stok & c) / len(stok) for c in ctx_tok), default=0.0)
        if best >= threshold:
            supported += 1
    return supported / len(ans_sents)


def answer_relevancy(question: str, answer: str) -> float:
    """Content-term overlap between question and answer."""
    q, a = _content_tokens(question), _content_tokens(answer)
    if not q:
        return 0.0
    return len(q & a) / len(q)


def context_precision(retrieved_ids: list[str], used_ids: list[str]) -> float:
    """Fraction of retrieved chunks that were actually used."""
    if not retrieved_ids:
        return 0.0
    used = set(used_ids)
    return sum(1 for r in retrieved_ids if r in used) / len(retrieved_ids)


def context_recall(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Fraction of gold chunks present in the retrieved set."""
    if not gold_ids:
        return 0.0
    retrieved = set(retrieved_ids)
    return sum(1 for g in gold_ids if g in retrieved) / len(gold_ids)


@dataclass
class EvalCase:
    id: str
    question: str
    gold_answer: str
    gold_chunk_ids: list[str]


@dataclass
class EvalReport:
    per_case: list[dict]
    means: dict[str, float]


def load_gold(data_dir: Path | str) -> list[EvalCase]:
    raw = json.loads((Path(data_dir) / "gold.json").read_text())
    return [
        EvalCase(
            id=r["id"],
            question=r["question"],
            gold_answer=r["gold_answer"],
            gold_chunk_ids=r["gold_chunk_ids"],
        )
        for r in raw
    ]


def evaluate_suite(
    answer_fn,
    data_dir: Path | str,
) -> EvalReport:
    """Run the gold Q&A set through ``answer_fn``.

    ``answer_fn(question)`` must return an object with ``answer`` (str)
    and ``chunks_used`` (list[str]); objects with a ``chunks``/evidence
    mapping are also accepted via duck-typing below.
    """
    data_dir = Path(data_dir)
    cases = load_gold(data_dir)
    chunk_texts = {
        p.stem: p.read_text() for p in sorted(data_dir.glob("doc-*.md"))
    }
    per_case: list[dict] = []
    sums = {"faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0}
    for case in cases:
        result = answer_fn(case.question)
        answer = result.answer
        used = list(getattr(result, "chunks_used", []))
        contexts = [chunk_texts[c] for c in used if c in chunk_texts]
        metrics = {
            "faithfulness": faithfulness(answer, contexts),
            "answer_relevancy": answer_relevancy(case.question, answer),
            "context_precision": context_precision(used, used),
            "context_recall": context_recall(used, case.gold_chunk_ids),
        }
        per_case.append({"id": case.id, "question": case.question, **metrics})
        for k, v in metrics.items():
            sums[k] += v
    n = len(cases)
    return EvalReport(per_case, {k: v / n for k, v in sums.items()})
