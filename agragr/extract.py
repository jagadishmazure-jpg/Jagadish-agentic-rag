"""Deterministic entity/relation extraction.

Regex + gazetteer based: no models, no network. ``extract_triples(text)``
returns (subject, relation, object) triples, each with the character span
of the source sentence it was extracted from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Known entities in the synthetic corpus. In production this is replaced by
# an NER model / entity linker; here the gazetteer keeps everything
# deterministic and testable.
GAZETTEER = [
    "Acme Robotics",
    "NovaBot",
    "TitanArm",
    "Maya Chen",
    "Raj Patel",
    "Boston Dynamics",
    "Austin, Texas",
    "NovaBot team",
    "NovaBot factory",
]

# Aliases used by corrective retrieval (selfrag) for query expansion.
ENTITY_ALIASES = {
    "NovaBot": ["warehouse robot"],
    "TitanArm": ["robotic arm", "gripper"],
    "Maya Chen": ["CEO", "founder"],
    "Raj Patel": ["CTO"],
    "Acme Robotics": ["the company"],
}


@dataclass
class Triple:
    subject: str
    relation: str
    obj: str
    source_span: tuple[int, int]  # char offsets of the source sentence
    source_sentence: str


# (compiled regex, relation template)
# Template groups: subj, obj, plus relation-specific extras.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"(?P<subj>[A-Z][\w ,\-']+?) is made by (?P<obj>[A-Z][\w ,\-']+?)[\.\,]"
        ),
        "made_by",
    ),
    (
        re.compile(
            r"(?P<subj>[A-Z][\w ,\-']+?) uses the (?P<obj>[A-Z][\w ,\-']+?) "
            r"as its (?P<role>[\w ]+?) module"
        ),
        "uses",
    ),
    (
        re.compile(
            r"(?P<subj>[A-Z][\w ,\-']+?) founded (?P<obj>[A-Z][\w ,\-']+?) "
            r"in (?P<year>\d{4})"
        ),
        "founded",
    ),
    (
        re.compile(
            r"(?P<subj>[A-Z][\w ,\-']+?) is the (?P<role>[A-Z][\w ]+?) "
            r"of (?P<obj>[A-Z][\w ,\-']+?)[\.\,]"
        ),
        "role",
    ),
    (
        re.compile(r"(?P<subj>[A-Z][\w ,\-']+?) leads the (?P<obj>[\w ]+?) team"),
        "leads_team",
    ),
    (
        re.compile(
            r"(?P<subj>[A-Z][\w ,\-']+?) offers a (?P<years>\d+)-year "
            r"warranty on (?P<obj>[A-Z][\w ,\-']+?)[\.\,]"
        ),
        "warranty",
    ),
    (
        re.compile(
            r"(?P<subj>[A-Z][\w ,\-']+?) is headquartered in "
            r"(?P<obj>[A-Z][\w ,\-']+?)[\.\,]"
        ),
        "headquartered_in",
    ),
    (
        re.compile(
            r"(?P<subj>The [A-Z][\w ,\-']+?) is located in "
            r"(?P<obj>[A-Z][\w ,\-']+?)[\.\,]"
        ),
        "located_in",
    ),
    (
        re.compile(
            r"(?P<subj>The [\w ]+?) is based in (?P<obj>[A-Z][\w ,\-']+?)[\.\,]"
        ),
        "based_in",
    ),
    (
        re.compile(r"(?P<subj>The [\w ]+?) includes (?P<obj>[\w ]+?)[\.\,]"),
        "includes",
    ),
    (
        re.compile(
            r"(?P<subj>[A-Z][\w ,\-']+?) has a battery life of "
            r"(?P<obj>[\w ]+?)[\.\,]"
        ),
        "battery_life",
    ),
    (
        re.compile(
            r"(?P<subj>[A-Z][\w ,\-']+?) can carry a payload of "
            r"(?P<obj>[\w ]+?)[\.\,]"
        ),
        "payload",
    ),
    (
        re.compile(
            r"(?P<subj>[A-Z][\w ,\-']+?) joined (?P<obj>[A-Z][\w ,\-']+?) "
            r"in (?P<year>\d{4})"
        ),
        "joined",
    ),
    (
        re.compile(
            r"(?P<subj>[A-Z][\w ,\-']+?) previously worked at "
            r"(?P<obj>[A-Z][\w ,\-']+?)[\.\,]"
        ),
        "previously_worked_at",
    ),
    (
        re.compile(r"(?P<subj>[A-Z][\w ,\-']+?) costs? \$(?P<amt>[\d,]+)"),
        "costs",
    ),
    (
        re.compile(
            r"(?P<subj>[A-Z][\w ,\-']+?) provides (?P<obj>[\w /]+?) support "
            r"for (?P<target>[A-Z][\w ,\-']+?)[\.\,]"
        ),
        "provides_support",
    ),
]


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split into sentences, keeping char offsets."""
    out: list[tuple[str, int, int]] = []
    for m in re.finditer(r"[^.!?]+[.!?]", text):
        sent = m.group(0).strip()
        if sent:
            out.append((sent, m.start(), m.end()))
    return out


def link_entities(text: str) -> list[str]:
    """Gazetteer entity linking: return known entities present in text,
    longest match first so 'NovaBot team' wins over 'NovaBot'."""
    found: list[str] = []
    lowered = text.lower()
    for ent in sorted(GAZETTEER, key=len, reverse=True):
        if ent.lower() in lowered:
            # avoid double counting a shorter entity inside a longer match
            if not any(ent.lower() in f.lower() and ent != f for f in found):
                found.append(ent)
    return found


def extract_triples(text: str) -> list[Triple]:
    """Extract (subject, relation, object) triples with source spans."""
    triples: list[Triple] = []
    for sent, start, end in _split_sentences(text):
        for pattern, relation in _PATTERNS:
            for m in pattern.finditer(sent):
                g = m.groupdict()
                subj = g.get("subj", "").strip()
                if relation == "uses":
                    triples.append(
                        Triple(subj, "uses", g["obj"].strip(), (start, end), sent)
                    )
                    triples.append(
                        Triple(
                            subj,
                            "uses_as",
                            g["role"].strip(),
                            (start, end),
                            sent,
                        )
                    )
                elif relation == "founded":
                    triples.append(
                        Triple(subj, "founded", g["obj"].strip(), (start, end), sent)
                    )
                    triples.append(
                        Triple(
                            g["obj"].strip(),
                            "founded_in",
                            g["year"],
                            (start, end),
                            sent,
                        )
                    )
                elif relation == "role":
                    triples.append(
                        Triple(subj, "has_role", g["role"].strip(), (start, end), sent)
                    )
                    triples.append(
                        Triple(
                            subj, "role_at", g["obj"].strip(), (start, end), sent
                        )
                    )
                elif relation == "warranty":
                    triples.append(
                        Triple(
                            g["obj"].strip(),
                            "warranty_years",
                            g["years"],
                            (start, end),
                            sent,
                        )
                    )
                elif relation == "costs":
                    triples.append(
                        Triple(subj, "costs", "$" + g["amt"], (start, end), sent)
                    )
                elif relation == "joined":
                    triples.append(
                        Triple(subj, "joined", g["obj"].strip(), (start, end), sent)
                    )
                    triples.append(
                        Triple(subj, "joined_in", g["year"], (start, end), sent)
                    )
                elif relation == "provides_support":
                    triples.append(
                        Triple(
                            subj,
                            "support_for",
                            g["target"].strip(),
                            (start, end),
                            sent,
                        )
                    )
                    triples.append(
                        Triple(
                            subj,
                            "support_hours",
                            g["obj"].strip(),
                            (start, end),
                            sent,
                        )
                    )
                else:
                    obj = (g.get("obj") or "").strip()
                    triples.append(
                        Triple(subj, relation, obj, (start, end), sent)
                    )
    # Deduplicate identical triples (same s/r/o).
    seen: set[tuple[str, str, str]] = set()
    unique: list[Triple] = []
    for t in triples:
        key = (t.subject, t.relation, t.obj)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique
