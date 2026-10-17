"""In-memory directed multigraph for the extracted triples."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from .extract import Triple


@dataclass
class PathStep:
    subject: str
    relation: str
    obj: str

    def verbalize(self) -> str:
        return f"{self.subject} --{self.relation}--> {self.obj}"


class KnowledgeGraph:
    """Directed multigraph: adjacency is entity -> [(relation, object)]."""

    def __init__(self) -> None:
        self._adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._entities: set[str] = set()
        self._triples: list[Triple] = []

    def add_triples(self, triples: list[Triple]) -> None:
        for t in triples:
            self._adj[t.subject].append((t.relation, t.obj))
            self._entities.add(t.subject)
            self._entities.add(t.obj)
            self._triples.append(t)

    @property
    def entities(self) -> set[str]:
        return set(self._entities)

    @property
    def triple_count(self) -> int:
        return len(self._triples)

    def neighbors(self, entity: str, relation: str | None = None) -> list[tuple[str, str]]:
        """(relation, object) pairs for an entity, optionally filtered."""
        out = self._adj.get(entity, [])
        if relation is not None:
            out = [(r, o) for r, o in out if r == relation]
        return list(out)

    def multi_hop(
        self, start: str, max_hops: int = 2
    ) -> list[list[PathStep]]:
        """BFS from ``start``; returns every path (list of steps) found
        within ``max_hops``. Each path is a chain of PathSteps."""
        paths: list[list[PathStep]] = []
        queue: deque[tuple[str, list[PathStep]]] = deque([(start, [])])
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for rel, obj in self._adj.get(node, []):
                step = PathStep(node, rel, obj)
                new_path = path + [step]
                paths.append(new_path)
                queue.append((obj, new_path))
        return paths

    def subgraph_for(
        self, question_entities: list[str], hops: int = 2
    ) -> dict[str, list[list[PathStep]]]:
        """Relevant neighborhood: multi-hop paths from each linked entity."""
        return {e: self.multi_hop(e, max_hops=hops) for e in question_entities}
