"""Budgeted, AST-aware left-context packing for completion inputs."""

from collections import OrderedDict

import tree_sitter as ts
import tree_sitter_java
import tree_sitter_python

from src.data.code_input_cleanup import clean_code_input

SCOPE_TYPES = {
    "class_definition", "function_definition", "class_declaration",
    "method_declaration", "constructor_declaration", "interface_declaration",
}
IMPORT_TYPES = {
    "import_statement", "import_from_statement", "import_declaration",
    "package_declaration",
}


def _walk(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.named_children))


def _count(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def _suffix_within_budget(text, tokenizer, budget):
    """Return the longest suffix whose actual tokenizer length fits budget."""
    if budget <= 0 or not text:
        return ""
    if _count(text, tokenizer) <= budget:
        return text
    lo, hi, best = 0, len(text), len(text)
    while lo <= hi:
        middle = (lo + hi) // 2
        if _count(text[middle:], tokenizer) <= budget:
            best = middle
            hi = middle - 1
        else:
            lo = middle + 1
    # Tokenizers are almost monotonic on suffixes, but the final small scan
    # makes this safe around merges at a character boundary.
    while best > 0 and _count(text[best - 1:], tokenizer) <= budget:
        best -= 1
    return text[best:]


def _parse(source, language):
    grammar = tree_sitter_python if language == "python" else tree_sitter_java
    return ts.Parser(ts.Language(grammar.language())).parse(source.encode("utf-8"))


def _candidate_hints(source, cursor_byte, language):
    raw = source.encode("utf-8")
    tree = _parse(source, language)
    spans = OrderedDict()
    for node in _walk(tree.root_node):
        if node.start_byte >= cursor_byte:
            continue
        if node.type in IMPORT_TYPES:
            if node.end_byte > cursor_byte:
                continue
            spans[(node.start_byte, node.end_byte, node.type)] = raw[node.start_byte:node.end_byte].decode("utf-8")
        if node.type in SCOPE_TYPES:
            body = node.child_by_field_name("body")
            if body is not None and node.start_byte < body.start_byte <= cursor_byte:
                spans[(node.start_byte, body.start_byte, node.type)] = raw[node.start_byte:body.start_byte].decode("utf-8")

    # Keep only the latest few completed definitions as lightweight global
    # context; imports and active ancestors remain higher priority below.
    completed = []
    for node in tree.root_node.named_children:
        if node.type in SCOPE_TYPES and node.end_byte <= cursor_byte:
            body = node.child_by_field_name("body")
            if body is not None:
                completed.append((node.start_byte, body.start_byte, node.type))
    for start, end, kind in completed[-8:]:
        spans[(start, end, kind)] = raw[start:end].decode("utf-8")
    return list(spans.items())


def pack_left_context(source, cursor_byte, language, tokenizer, *,
                      max_tokens=2048, tail_tokens=1536, hint_tokens=512):
    """Pack only source before ``cursor_byte`` under an exact token budget.

    Short prefixes are preserved after input-only cleanup. Long prefixes retain
    the cursor-near suffix plus compact AST-derived imports/scope signatures.
    No bytes after the cursor can enter the returned text.
    """
    raw = source.encode("utf-8")
    if not 0 <= cursor_byte <= len(raw):
        raise ValueError("cursor_byte is outside the source")
    prefix_raw = raw[:cursor_byte].decode("utf-8")
    cleaned = clean_code_input(prefix_raw, language)["text"]
    if _count(cleaned, tokenizer) <= max_tokens:
        return {"text": cleaned, "tokens": _count(cleaned, tokenizer),
                "truncated": False, "hint_tokens": 0,
                "tail_tokens": _count(cleaned, tokenizer),
                "diagnostics": {"full_clean_prefix": 1,
                                "ast_hints_prefix_only": 1}}

    tail_budget = min(tail_tokens, max_tokens)
    tail = _suffix_within_budget(cleaned, tokenizer, tail_budget)
    marker = "\n# <AST_CONTEXT>\n" if language == "python" else "\n// <AST_CONTEXT>\n"
    available_hint = min(hint_tokens, max_tokens - _count(tail, tokenizer) - _count(marker, tokenizer))
    hint_parts = []
    hint_seen = set()
    # Parse the prefix only. Parsing the full file can let a parser recovery
    # decision depend on declarations after the cursor, which is a subtle form
    # of right-context leakage even when spans are later clipped.
    prefix_for_hints = prefix_raw
    prefix_cursor = len(prefix_for_hints.encode("utf-8"))
    for (start, end, kind), text in _candidate_hints(
            prefix_for_hints, prefix_cursor, language):
        if text.strip() in hint_seen:
            continue
        if start >= cursor_byte or end > cursor_byte:
            continue
        cleaned_hint = clean_code_input(text, language)["text"].strip()
        if not cleaned_hint or cleaned_hint in hint_seen:
            continue
        proposal = "\n".join(hint_parts + [cleaned_hint])
        if _count(proposal, tokenizer) > available_hint:
            continue
        hint_parts.append(cleaned_hint)
        hint_seen.add(cleaned_hint)
    hints = "\n".join(hint_parts)
    if hints:
        hints = hints + marker
    result = hints + tail
    if _count(result, tokenizer) > max_tokens:
        result = _suffix_within_budget(result, tokenizer, max_tokens)
    return {"text": result, "tokens": _count(result, tokenizer),
            "truncated": True, "hint_tokens": _count(hints, tokenizer),
            "tail_tokens": _count(tail, tokenizer),
            "diagnostics": {"ast_hints_added": int(bool(hints)),
                            "full_clean_prefix": 0,
                            "ast_hints_prefix_only": 1}}
