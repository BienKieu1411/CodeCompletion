"""Budgeted rendering of retrieved cross-file chunks.

Retrieval produces a ranked pool; this module performs the separate generator
prompt admission step.  It never mutates or deletes the pool.  A chunk is
either included in full or left in the pool for a later policy; it is never
silently truncated in the middle of source code.
"""

def _count(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def _render(candidate, language):
    marker = "# File: " if language == "python" else "// File: "
    path = candidate.get("path", "<unknown>")
    # model_text is cleaned input-only text. The raw ``text`` and offsets stay
    # in the candidate pool, so comments and source provenance are recoverable.
    body = candidate.get("model_text") or candidate.get("text") or ""
    return f"{marker}{path}\n{body}"


def select_cross_file_context(candidates, tokenizer, language, *,
                              budget_tokens=2344, max_snippets=10):
    """Select complete ranked snippets under an exact generator-token budget.

    The first pass gives distinct files a chance to contribute. A second pass
    fills unused capacity with additional high-ranked chunks. This is only a
    prompt view: ``candidates`` remains unchanged and all chunks remain
    available to a learned selector.
    """
    if budget_tokens <= 0 or max_snippets <= 0:
        raise ValueError("cross-file budget and snippet count must be positive")
    ranked = list(candidates or [])
    selected, selected_keys = [], set()
    selected_paths = set()
    used = 0
    skipped = 0

    def visit(require_new_path):
        nonlocal used, skipped
        for rank, candidate in enumerate(ranked):
            key = (candidate.get("path"), candidate.get("start"), candidate.get("end"),
                   candidate.get("source_sha256"))
            if key in selected_keys or (require_new_path and candidate.get("path") in selected_paths):
                continue
            text = _render(candidate, language)
            cost = _count(text, tokenizer)
            if cost + used > budget_tokens:
                skipped += 1
                continue
            selected_keys.add(key)
            selected_paths.add(candidate.get("path"))
            selected.append({
                "rank": rank,
                "path": candidate.get("path"),
                "start": candidate.get("start"),
                "end": candidate.get("end"),
                "source_sha256": candidate.get("source_sha256"),
                "text": text,
                "token_count": cost,
                "bm25_score": candidate.get("bm25_score"),
            })
            used += cost
            if len(selected) >= max_snippets:
                return

    visit(require_new_path=True)
    if len(selected) < max_snippets:
        visit(require_new_path=False)
    return {
        "items": selected,
        "token_count": used,
        "budget_tokens": budget_tokens,
        "max_snippets": max_snippets,
        "candidate_count": len(ranked),
        "selected_count": len(selected),
        "skipped_due_budget_or_duplicate": skipped,
    }
