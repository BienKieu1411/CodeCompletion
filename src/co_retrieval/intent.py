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


# ── Cost-Aware Query Enhancement ─────────────────────────────────────────────


def _score_entropy(scores: "List[float]") -> float:
    """Compute normalized score entropy in ``[0, 1]``.

    Parameters
    ----------
    scores : list of float
        Raw similarity scores from the retriever.

    Returns
    -------
    entropy : float
        Shannon entropy of softmax(scores).  Higher values indicate the
        retriever is uncertain about which chunk to prefer.
    """
    import math

    if not scores or len(scores) < 2:
        return 0.0

    # Softmax
    max_s = max(scores)
    exp_scores = [math.exp(s - max_s) for s in scores]
    total = sum(exp_scores)
    if total < 1e-12:
        return 0.0

    probs = [e / total for e in exp_scores]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return max(0.0, min(1.0, entropy / math.log(len(probs))))


@dataclass(frozen=True)
class QueryEnhancementResult:
    """Result of adaptive query enhancement."""

    query: str
    used_sampling: bool = False
    entropy: float = 0.0
    num_drafts_merged: int = 0
    query_changed: bool = False


class CostAwareQueryEnhancer:
    """Conditionally invoke LLM sampling based on retriever score entropy.

    Inherits AlignCoder's query enhancement idea (using LLM-generated
    completions to enrich the retrieval query) but only activates the
    expensive sampling path when the retriever is uncertain.

    Design
    ------
    * **Cheap path** (majority of samples): use static intent sketch.
      Triggered when retriever score entropy is below threshold.
    * **Expensive path** (minority): merge draft completions into query.
      Triggered when retriever is uncertain (high entropy / flat scores).

    The draft completions themselves must be provided by the caller
    (generated externally by a lightweight LLM or the main generator).
    """

    def __init__(
        self,
        sketcher: IntentSketcher,
        entropy_threshold: float = 0.8,
        max_draft_identifiers: int = 30,
    ) -> None:
        self.sketcher = sketcher
        self.entropy_threshold = entropy_threshold
        self.max_draft_identifiers = max_draft_identifiers

    def enhance(
        self,
        left_context: str,
        retriever_scores: "List[float] | None" = None,
        draft_completions: "List[str] | None" = None,
    ) -> QueryEnhancementResult:
        """Build an enhanced query, conditionally using draft completions.

        Parameters
        ----------
        left_context : str
            Code before cursor.
        retriever_scores : list of float, optional
            Similarity scores from the retriever for top-k chunks.
            Used to compute entropy and decide whether to use drafts.
            If None, always uses the cheap path.
        draft_completions : list of str, optional
            LLM-generated candidate completions (à la AlignCoder).
            Only merged when retriever entropy exceeds threshold.

        Returns
        -------
        QueryEnhancementResult
            Contains the final query string and metadata about which
            path was taken.
        """
        sketch = self.sketcher.build(left_context)

        # Compute retriever confidence
        entropy = 0.0
        if retriever_scores is not None:
            entropy = _score_entropy(retriever_scores)

        # Cheap path: retriever is confident OR no drafts available
        if entropy < self.entropy_threshold or not draft_completions:
            return QueryEnhancementResult(
                query=sketch.query,
                used_sampling=False,
                entropy=entropy,
            )

        # Expensive path: merge draft completions into query
        merged_query = self._merge(sketch, draft_completions)
        return QueryEnhancementResult(
            query=merged_query,
            used_sampling=True,
            entropy=entropy,
            num_drafts_merged=len(draft_completions),
            query_changed=merged_query != sketch.query,
        )

    def _merge(self, sketch: IntentSketch, drafts: List[str]) -> str:
        """Merge draft completions into the intent sketch query.

        Extracts novel identifiers from drafts and appends them to the
        sketch query, providing the retriever with tokens that may not
        appear in the left context (e.g. ``refund_payment`` from a draft).
        """
        # Collect identifiers already in the sketch
        existing = set(sketch.identifiers)
        existing.update(sketch.import_hints)
        existing.update(sketch.class_hints)
        if sketch.member_owner:
            existing.add(sketch.member_owner)
        if sketch.member_prefix:
            existing.add(sketch.member_prefix)

        # Extract novel identifiers from drafts
        novel: List[str] = []
        for draft in drafts:
            for token in _IDENTIFIER_RE.findall(draft or ""):
                if token not in existing and not keyword.iskeyword(token):
                    existing.add(token)
                    novel.append(token)

        if not novel:
            return sketch.query

        # Append draft hints to existing query
        draft_section = (
            "### Draft completion hints\n"
            + " ".join(novel[: self.max_draft_identifiers])
        )
        return sketch.query + "\n" + draft_section
