"""Context utility scoring for preference-optimized retrieval.

The central training signal is utility: how much a context lowers target NLL
relative to no retrieval.  Positive utility means retrieval helped for this
sample; negative utility means it added noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from co_retrieval.chunking import CodeChunk


@dataclass(frozen=True)
class ContextCandidate:
    """One candidate retrieval strategy for a sample."""

    name: str
    chunks: list[CodeChunk]
    is_stop: bool = False
    retrieval_query: str = ""


@dataclass(frozen=True)
class ContextScore:
    """NLL and utility for a candidate context."""

    name: str
    chunks: list[CodeChunk]
    is_stop: bool
    nll: float
    utility: float
    retrieval_query: str = ""


class ContextUtilityScorer:
    """Rank retrieval strategies by NLL improvement over no-retrieval."""

    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def score(
        self,
        left_context: str,
        target: str,
        candidates: Sequence[ContextCandidate],
        *,
        use_adapter: bool = True,
    ) -> list[ContextScore]:
        """Return candidates sorted by descending utility.

        Utility is defined as ``NLL(stop) - NLL(candidate)``.  Stop itself has
        utility 0.0 by definition.
        """
        stop_nll = self._nll(
            left_context,
            target,
            chunks=None,
            use_soft_prompt=False,
        )
        scores: list[ContextScore] = []
        for candidate in candidates:
            if candidate.is_stop:
                nll = stop_nll
                utility = 0.0
            else:
                nll = self._nll(
                    left_context,
                    target,
                    chunks=candidate.chunks,
                    use_soft_prompt=use_adapter,
                )
                utility = stop_nll - nll
            scores.append(
                ContextScore(
                    name=candidate.name,
                    chunks=list(candidate.chunks),
                    is_stop=candidate.is_stop,
                    nll=nll,
                    utility=utility,
                    retrieval_query=candidate.retrieval_query,
                )
            )
        scores.sort(key=lambda score: (score.utility, -score.nll), reverse=True)
        return scores

    def _nll(
        self,
        left_context: str,
        target: str,
        *,
        chunks: Sequence[CodeChunk] | None,
        use_soft_prompt: bool,
    ) -> float:
        value = self.generator.teacher_forcing_nll(
            left_context=left_context,
            target=target,
            retrieved_chunks=chunks,
            use_soft_prompt=use_soft_prompt,
        )
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
