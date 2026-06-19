"""Completion quality metrics for Co-Retrieval.

This module provides the evaluation signals used throughout the pipeline:

* **Exact Match (EM)** — binary, 1.0 iff prediction == target after strip.
* **Edit Similarity** — ``SequenceMatcher.ratio()`` on stripped strings.
* **Identifier F1** — F1 score over Python-style identifiers.
* **Composite Quality** — weighted combination used for DPO pair ranking.

The composite score is the *single number* that determines which retrieval
strategy is "better" when constructing DPO preference pairs.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Set

from co_retrieval.chunking import CodeChunk

# ── Identifier extraction ─────────────────────────────────────────────────────

_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")


def identifiers(text: str) -> List[str]:
    """Return all Python-style identifiers found in *text*."""
    return _IDENTIFIER_RE.findall(text or "")


def identifier_set(text: str) -> Set[str]:
    """Return the *unique* identifiers in *text*."""
    return set(identifiers(text))


# ── Core metrics ──────────────────────────────────────────────────────────────


def exact_match(prediction: str, target: str) -> float:
    """1.0 if stripped prediction equals stripped target, else 0.0."""
    return 1.0 if (prediction or "").strip() == (target or "").strip() else 0.0


def edit_similarity(prediction: str, target: str) -> float:
    """Sequence-level edit similarity in [0, 1]."""
    pred = (prediction or "").strip()
    tgt = (target or "").strip()
    if pred == tgt:
        return 1.0
    if not tgt:
        return 0.0
    return SequenceMatcher(None, pred, tgt).ratio()


def identifier_precision_recall_f1(
    prediction: str,
    target: str,
) -> tuple[float, float, float]:
    """Identifier-level precision, recall, F1."""
    pred_ids = identifier_set(prediction)
    tgt_ids = identifier_set(target)
    if not tgt_ids:
        return (0.0, 0.0, 0.0)
    if not pred_ids:
        return (0.0, 0.0, 0.0)
    overlap = pred_ids & tgt_ids
    precision = len(overlap) / len(pred_ids)
    recall = len(overlap) / len(tgt_ids)
    if precision + recall == 0:
        return (0.0, 0.0, 0.0)
    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def identifier_f1(prediction: str, target: str) -> float:
    """Convenience: just the F1 value."""
    _, _, f1 = identifier_precision_recall_f1(prediction, target)
    return f1


# ── Composite quality ─────────────────────────────────────────────────────────

# Default weights for the composite score used in DPO pair ranking.
DEFAULT_EM_WEIGHT = 0.30
DEFAULT_EDIT_SIM_WEIGHT = 0.35
DEFAULT_ID_F1_WEIGHT = 0.35


def composite_quality(
    prediction: str,
    target: str,
    *,
    em_weight: float = DEFAULT_EM_WEIGHT,
    edit_sim_weight: float = DEFAULT_EDIT_SIM_WEIGHT,
    id_f1_weight: float = DEFAULT_ID_F1_WEIGHT,
) -> float:
    """Weighted composite quality score in [0, 1].

    This is the **single metric** that decides which retrieval strategy
    produced a "better" completion.  It is used to form DPO preference
    pairs and to supervise the adaptive gate.
    """
    em = exact_match(prediction, target)
    es = edit_similarity(prediction, target)
    f1 = identifier_f1(prediction, target)
    return em_weight * em + edit_sim_weight * es + id_f1_weight * f1


# ── Context-aware quality (backward compatible with proxy training.py) ────────

def completion_quality_with_context(
    prediction: str,
    target: str,
    context: Iterable[CodeChunk],
) -> float:
    """Composite quality **plus** a small bonus if retrieved context
    contained identifiers present in the target.

    This keeps backward compatibility with the proxy ``_completion_quality``
    function in ``training.py`` while delegating the core maths to the
    functions above.
    """
    base = composite_quality(prediction, target)
    tgt_ids = identifier_set(target)
    if not tgt_ids:
        return base

    context_symbols: set[str] = set()
    for chunk in context:
        context_symbols.update(chunk.defined_symbols)
        context_symbols.update(chunk.call_names)

    context_hit = 0.1 if context_symbols & tgt_ids else 0.0
    return min(1.0, base + context_hit)
