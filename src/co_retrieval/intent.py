"""Intent-conditioned retrieval query construction.

Raw left context is often a weak retrieval query because the code at the
cursor is incomplete.  The sketcher extracts cheap, deterministic hints from
the left context and appends them to the query so dense/BM25 retrieval can see
likely symbols, member-access owners, imports, and local type hints.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import List


_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
_MEMBER_ACCESS_RE = re.compile(
    r"(?P<owner>[_a-zA-Z][_a-zA-Z0-9]*)\s*\.\s*(?P<prefix>[_a-zA-Z][_a-zA-Z0-9]*)?$"
)
_ASSIGN_CALL_RE = re.compile(
    r"(?P<name>[_a-zA-Z][_a-zA-Z0-9]*)\s*=\s*(?P<class>[A-Z][_a-zA-Z0-9]*)\s*\("
)


@dataclass(frozen=True)
class IntentSketch:
    """Deterministic hints extracted from an incomplete left context."""

    prefix: str = ""
    member_owner: str = ""
    member_prefix: str = ""
    identifiers: List[str] = field(default_factory=list)
    class_hints: List[str] = field(default_factory=list)
    import_hints: List[str] = field(default_factory=list)
    query: str = ""


class IntentSketcher:
    """Build a retrieval query from left context plus static intent hints."""

    def __init__(self, max_tail_lines: int = 80, max_identifiers: int = 40) -> None:
        self.max_tail_lines = max_tail_lines
        self.max_identifiers = max_identifiers

    def build(self, left_context: str) -> IntentSketch:
        tail = self._tail(left_context)
        identifiers = self._identifiers(tail)
        prefix = self._last_identifier(left_context)
        member_owner, member_prefix = self._member_access(left_context)
        class_hints = self._class_hints(tail)
        import_hints = self._import_hints(tail)

        sketch_lines = ["### Intent sketch"]
        if prefix:
            sketch_lines.append(f"incomplete_prefix: {prefix}")
        if member_owner:
            sketch_lines.append(f"member_owner: {member_owner}")
        if member_prefix:
            sketch_lines.append(f"member_prefix: {member_prefix}")
        if class_hints:
            sketch_lines.append("class_hints: " + " ".join(class_hints[:12]))
        if import_hints:
            sketch_lines.append("imports: " + " ".join(import_hints[:12]))
        if identifiers:
            sketch_lines.append(
                "local_identifiers: "
                + " ".join(identifiers[: self.max_identifiers])
            )
        sketch_lines.append("### Left context tail")
        sketch_lines.append(tail)

        return IntentSketch(
            prefix=prefix,
            member_owner=member_owner,
            member_prefix=member_prefix,
            identifiers=identifiers,
            class_hints=class_hints,
            import_hints=import_hints,
            query="\n".join(sketch_lines).strip(),
        )

    def build_query(self, left_context: str) -> str:
        return self.build(left_context).query

    def _tail(self, text: str) -> str:
        lines = (text or "").splitlines()
        return "\n".join(lines[-self.max_tail_lines :])

    def _identifiers(self, text: str) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for match in _IDENTIFIER_RE.finditer(text or ""):
            token = match.group(0)
            if token in seen or keyword.iskeyword(token):
                continue
            seen.add(token)
            out.append(token)
        return out

    @staticmethod
    def _last_identifier(text: str) -> str:
        match = re.search(r"[_a-zA-Z][_a-zA-Z0-9]*$", (text or "").rstrip())
        return match.group(0) if match else ""

    @staticmethod
    def _member_access(text: str) -> tuple[str, str]:
        tail = (text or "").rstrip().splitlines()
        last_line = tail[-1] if tail else ""
        match = _MEMBER_ACCESS_RE.search(last_line)
        if not match:
            return "", ""
        return match.group("owner") or "", match.group("prefix") or ""

    def _class_hints(self, text: str) -> List[str]:
        hints: List[str] = []
        seen: set[str] = set()
        for match in _ASSIGN_CALL_RE.finditer(text or ""):
            for token in (match.group("name"), match.group("class")):
                if token not in seen:
                    seen.add(token)
                    hints.append(token)
        for token in self._identifiers(text):
            if token[:1].isupper() and token not in seen:
                seen.add(token)
                hints.append(token)
        return hints

    @staticmethod
    def _import_hints(text: str) -> List[str]:
        hints: List[str] = []
        seen: set[str] = set()
        for line in (text or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            for token in _IDENTIFIER_RE.findall(stripped):
                if token not in {"import", "from", "as"} and token not in seen:
                    seen.add(token)
                    hints.append(token)
        return hints
