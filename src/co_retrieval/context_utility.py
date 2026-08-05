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

    def __init__(self, generator: Any, micro_batch_size: int = 2) -> None:
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        self.generator = generator
        self.micro_batch_size = micro_batch_size

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
        # The stop baseline must use the same generator/adapter state as
        # retrieved candidates. Otherwise utility would conflate "retrieval
        # helped" with "the adapter helped", producing oracle-like gate labels.
        # SoftPromptLLM batches all strategy candidates on the frozen generator
        # when available; lightweight test doubles keep the scalar fallback.
        batch_scorer = getattr(self.generator, "teacher_forcing_nll_batch", None)
        if callable(batch_scorer):
            nlls = batch_scorer(
                left_context,
                target,
                [None if candidate.is_stop else candidate.chunks for candidate in candidates],
                use_soft_prompt=use_adapter,
                micro_batch_size=self.micro_batch_size,
            )
            stop_indices = [
                index for index, candidate in enumerate(candidates) if candidate.is_stop
            ]
            if stop_indices:
                stop_nll = float(nlls[stop_indices[0]])
            else:
                stop_nll = self._nll(
                    left_context,
                    target,
                    chunks=None,
                    use_soft_prompt=use_adapter,
                )
        else:
            stop_nll = self._nll(
                left_context,
                target,
                chunks=None,
                use_soft_prompt=use_adapter,
            )
        scores: list[ContextScore] = []
        for index, candidate in enumerate(candidates):
            if candidate.is_stop:
                nll = stop_nll
                utility = 0.0
            elif callable(batch_scorer):
                nll = float(nlls[index])
                utility = stop_nll - nll
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
