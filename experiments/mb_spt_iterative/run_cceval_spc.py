#!/usr/bin/env python3
"""Full CrossCodeEval Evaluation (4,804 tasks: 2,665 Python + 2,139 Java): SPC + MB-SPT vs AlignCoder & AIRCoder."""

from __future__ import annotations

import difflib
import gzip
import json
import keyword
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from colab_init_engine import (
    CROSSFILE_TOKENS,
    GENERATOR_CONTEXT,
    GENERATOR_MAX_NEW,
    ROOT_DIR,
    SEED,
    init_globals,
)

ID_RE = re.compile(r"\b[_A-Za-z][_A-Za-z0-9]*\b")
STRING_PATTERN = re.compile(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'')
WORD_RE = re.compile(r"\w+")

STOP_SYMS = {
    "self", "this", "None", "True", "False", "null", "return", "args", "kwargs",
    "init", "__init__", "super", "cls", "pass", "break", "continue", "from",
    "import", "class", "def", "void", "public", "private", "protected", "static", "final",
    "else", "if", "for", "in", "with",
}

_GLOBAL_FILE_LINES_CACHE: dict[str, list[str]] = {}
_GLOBAL_FILE_BLOCKS_CACHE: dict[str, list[dict[str, Any]]] = {}


def extract_clean_markov_blanket(
    left_context: str,
    qh_outputs: list[str],
    window_lines: int = 5,
) -> tuple[set[str], set[str]]:
    nonempty = [line for line in left_context.splitlines() if line.strip()]
    cursor_tail = "\n".join(nonempty[-window_lines:])
    cursor_syms = {s for s in ID_RE.findall(cursor_tail) if s not in STOP_SYMS and len(s) >= 3}

    min_count = 2 if len(qh_outputs) >= 2 else 1
    counts: Counter[str] = Counter()
    for out in qh_outputs:
        for s in set(ID_RE.findall(out)):
            if len(s) >= 4 and s not in STOP_SYMS:
                counts[s] += 1
    consensus_syms = {s for s, c in counts.items() if c >= min_count}
    return cursor_syms, consensus_syms


def is_open_expression_cursor(left_context: str, top1_cos: float, threshold: float = 0.85) -> bool:
    if top1_cos < threshold:
        return False
    nonempty = [line for line in left_context.splitlines() if line.strip()]
    if not nonempty:
        return False
    last = nonempty[-1].rstrip()
    if last.endswith(('"""', "'''", ";", "}", "{", ":")):
        return False
    if (
        last.endswith((".", "(", "="))
        or re.search(r"\b(?:new|await|assertRaises)(?:\s+[A-Za-z0-9_.]*)?$", last)
    ):
        return True
    return False


def epsilon_tie_consensus_rescue(
    chunks: list[dict[str, Any]],
    cursor_syms: set[str],
    consensus_syms: set[str],
    eps: float = 0.0005,
    lock_k: int = 4,
    window_end: int = 8,
) -> list[dict[str, Any]]:
    if len(chunks) <= lock_k + 1 or not consensus_syms:
        return chunks
    locked = list(chunks[:lock_k])
    tail = list(chunks[lock_k:window_end])
    rest = list(chunks[window_end:])
    if not tail:
        return chunks

    c_first_syms = set(ID_RE.findall(tail[0].get("code", ""))) - STOP_SYMS
    if len(c_first_syms & consensus_syms) > 0:
        return chunks

    base_cos = float(tail[0]["cosine"])

    def score_tie(c: dict[str, Any]) -> float:
        cos = float(c["cosine"])
        cs = set(ID_RE.findall(c.get("code", ""))) - STOP_SYMS
        ch = len(cs & consensus_syms)
        if base_cos - cos <= eps and ch > 0 and (cs & cursor_syms):
            return cos + 0.010 * min(2, ch)
        return cos

    new_tail = sorted(tail, key=lambda c: -score_tie(c))
    return locked + new_tail + rest


def split_file_into_mini_blocks(text: str) -> list[dict[str, Any]]:
    """Exact reproduction of RepoClone/AlignCoder/bm25.py:split_into_smaller_blocks."""
    mini_blocks: list[list[str]] = []
    current_block: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if current_block:
                mini_blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        mini_blocks.append(current_block)

    max_len = 15
    temp_mini_blocks: list[list[str]] = []
    for mb in mini_blocks:
        if len(mb) > max_len:
            for idx in range(0, len(mb), max_len):
                temp_mini_blocks.append(mb[idx : idx + max_len])
        else:
            temp_mini_blocks.append(mb)
    mini_blocks = temp_mini_blocks

    smaller_blocks: list[dict[str, Any]] = []
    current_content: list[str] = []
    total_lines = 0
    for block in mini_blocks:
        if total_lines >= 5000:
            break
        if len(current_content) + len(block) <= 15:
            current_content.extend(block)
            total_lines += len(block)
        else:
            if current_content:
                s = total_lines - len(current_content) + 1
                e = total_lines
                smaller_blocks.append({"line_start": s, "line_end": e, "lines": list(current_content)})
            current_content = list(block)
            total_lines += len(block)
    if current_content:
        s = total_lines - len(current_content) + 1
        e = total_lines
        smaller_blocks.append({"line_start": s, "line_end": e, "lines": list(current_content)})
    return smaller_blocks


def indent_level(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip(" "))


def is_scope_header(line: str, language: str) -> bool:
    s = line.strip()
    if language == "python":
        return s.startswith(("class ", "def ", "async def "))
    if any(s.startswith(kw) for kw in ("if ", "for ", "while ", "switch ", "catch ", "return ", "throw ", "new ")):
        return False
    if re.match(r"^(?:public|protected|private|static|final|abstract|synchronized|default|\s)*(?:class|interface|enum|record)\s+\w+", s):
        return True
    if "(" in s and (s.endswith("{") or s.endswith(")")) and not s.endswith(";"):
        return True
    return False


def find_enclosing_headers(nonempty_lines: list[str], start_idx_0: int, language: str) -> list[str]:
    if start_idx_0 <= 0 or start_idx_0 >= len(nonempty_lines):
        return []
    first_non_comment = None
    for k in range(start_idx_0, min(len(nonempty_lines), start_idx_0 + 5)):
        st = nonempty_lines[k].strip()
        if st and not st.startswith(("#", "//", "/*", "*")):
            first_non_comment = nonempty_lines[k]
            break
    if first_non_comment is None:
        return []
    cur_indent = indent_level(first_non_comment)
    if cur_indent == 0 and is_scope_header(first_non_comment, language):
        return []

    headers: list[str] = []
    target_indent = cur_indent
    for idx in range(start_idx_0 - 1, -1, -1):
        line = nonempty_lines[idx]
        st = line.strip()
        if not st or st.startswith(("#", "//", "/*", "*")):
            continue
        ind = indent_level(line)
        if ind < target_indent and is_scope_header(line, language):
            headers.append(line)
            target_indent = ind
            if ind == 0 or len(headers) >= 2:
                break
    headers.reverse()
    return headers


def spc_coalesce_and_expand(
    chunks: list[dict[str, Any]],
    task: dict[str, Any],
    files_store: dict[str, dict[str, str]],
    expand_top_k: int = 0,
    add_scope_headers: bool = False,
    max_gap_lines: int = 2,
) -> list[dict[str, Any]]:
    lang = task["language"]
    file_ids_by_path = task.get("file_ids_by_path", {})

    def get_file_nonempty(path: str) -> list[str]:
        fid = file_ids_by_path.get(path)
        if not fid:
            return []
        if fid not in _GLOBAL_FILE_LINES_CACHE:
            txt = files_store[fid]["text"] if fid in files_store else ""
            _GLOBAL_FILE_LINES_CACHE[fid] = [l for l in txt.splitlines() if l.strip() != ""]
        return _GLOBAL_FILE_LINES_CACHE[fid]

    def get_file_blocks(path: str) -> list[dict[str, Any]]:
        fid = file_ids_by_path.get(path)
        if not fid:
            return []
        if fid not in _GLOBAL_FILE_BLOCKS_CACHE:
            txt = files_store[fid]["text"] if fid in files_store else ""
            _GLOBAL_FILE_BLOCKS_CACHE[fid] = split_file_into_mini_blocks(txt) if txt else []
        return _GLOBAL_FILE_BLOCKS_CACHE[fid]

    valid_chunks = []
    for c in chunks:
        if c.get("kind") == "stop" or not c.get("path"):
            break
        valid_chunks.append(dict(c))

    if expand_top_k > 0:
        non_api_seen = 0
        existing_ranges_by_path: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for c in valid_chunks:
            if c["path"] != "file_path":
                existing_ranges_by_path[c["path"]].add((c.get("line_start", 0), c.get("line_end", 0)))

        for c in valid_chunks:
            if c["path"] == "file_path" or c.get("api"):
                continue
            non_api_seen += 1
            if non_api_seen > expand_top_k:
                break
            fblocks = get_file_blocks(c["path"])
            if not fblocks:
                continue
            s, e = c.get("line_start", 0), c.get("line_end", 0)
            for b_idx, fb in enumerate(fblocks):
                if fb["line_start"] == s and fb["line_end"] == e and b_idx + 1 < len(fblocks):
                    nxt = fblocks[b_idx + 1]
                    if (nxt["line_start"], nxt["line_end"]) not in existing_ranges_by_path[c["path"]]:
                        cont_lines = []
                        for nl in nxt["lines"]:
                            if indent_level(nl) == 0 and is_scope_header(nl, lang) and cont_lines:
                                break
                            cont_lines.append(nl)
                        if cont_lines:
                            c["code"] = c["code"] + "\n" + "\n".join(cont_lines[:8])
                            c["line_end"] = s + len(c["code"].splitlines()) - 1
                    break

    used = [False] * len(valid_chunks)
    coalesced: list[dict[str, Any]] = []
    seen_api_lines: set[str] = set()

    for i, c in enumerate(valid_chunks):
        if used[i]:
            continue
        used[i] = True
        if c["path"] == "file_path" or c.get("api"):
            unique_lines = []
            for line in c["code"].splitlines():
                ls = line.strip()
                if ls and ls not in seen_api_lines:
                    seen_api_lines.add(ls)
                    unique_lines.append(line)
            if unique_lines:
                nc = dict(c)
                nc["code"] = "\n".join(unique_lines)
                coalesced.append(nc)
            continue

        cluster = [c]
        changed = True
        while changed:
            changed = False
            c_min = min(x.get("line_start", 0) for x in cluster)
            c_max = max(x.get("line_end", 0) for x in cluster)
            for j in range(i + 1, len(valid_chunks)):
                if used[j]:
                    continue
                other = valid_chunks[j]
                if other["path"] != c["path"] or other["path"] == "file_path":
                    continue
                os_line, oe_line = other.get("line_start", 0), other.get("line_end", 0)
                if os_line > 0 and c_min > 0:
                    if not (oe_line + max_gap_lines < c_min or os_line - max_gap_lines > c_max):
                        cluster.append(other)
                        used[j] = True
                        changed = True

        if len(cluster) == 1:
            merged_chunk = dict(c)
        else:
            cluster.sort(key=lambda x: (x.get("line_start", 0), x.get("line_end", 0)))
            s_first = cluster[0]["line_start"]
            e_last = max(x["line_end"] for x in cluster)
            nonempty = get_file_nonempty(c["path"])
            if nonempty and 1 <= s_first <= e_last <= len(nonempty) and (e_last - s_first + 1) <= 45:
                merged_lines = nonempty[s_first - 1 : e_last]
            else:
                merged_lines = []
                cur_end = -1
                for item in cluster:
                    ilines = item["code"].splitlines()
                    istart = item.get("line_start", 0)
                    if cur_end >= istart > 0:
                        overlap = cur_end - istart + 1
                        merged_lines.extend(ilines[max(0, overlap) :])
                    else:
                        merged_lines.extend(ilines)
                    cur_end = max(cur_end, item.get("line_end", 0))
            merged_chunk = dict(c)
            merged_chunk["line_start"] = s_first
            merged_chunk["line_end"] = e_last
            merged_chunk["code"] = "\n".join(merged_lines)

        if add_scope_headers:
            nonempty = get_file_nonempty(merged_chunk["path"])
            s_idx = merged_chunk.get("line_start", 0) - 1
            if nonempty and 0 < s_idx < len(nonempty):
                hdrs = find_enclosing_headers(nonempty, s_idx, lang)
                if hdrs:
                    existing_first = merged_chunk["code"].splitlines()[:2]
                    new_hdrs = [h for h in hdrs if h not in existing_first]
                    if new_hdrs:
                        merged_chunk["code"] = "\n".join(new_hdrs) + "\n" + merged_chunk["code"]

        coalesced.append(merged_chunk)

    return coalesced


def format_chunks_exact(chunks: list[dict[str, Any]], language: str) -> list[str]:
    label = "#" if language == "python" else "//"
    api_indices = [
        i for i, c in enumerate(chunks)
        if c.get("kind") != "stop" and c.get("path") == "file_path"
    ]
    last_api_idx = api_indices[-1] if api_indices else -1
    blocks = []
    for i, c in enumerate(chunks):
        if c.get("kind") == "stop" or not c.get("path"):
            break
        if c["path"] == "file_path":
            hdr = f"{label} Potential invoked APIs:" if i == last_api_idx else f"{label} Potentially invoked APIs:"
        else:
            hdr = f"{label} file path: {c['path']}\n{label} lines: {c['line_start']}-{c['line_end']}"
        body = "\n".join(f"{label} {line}" for line in c["code"].split("\n") if line)
        blocks.append(f"{hdr}\n{body}".strip())
    return blocks


def pack_blocks_within_budget(
    blocks: list[str],
    max_cross_tokens: int,
    tok: Any,
    reverse_surviving: bool = False,
) -> list[int]:
    surviving_blocks = []
    used_tokens = 0
    sep_tokens = len(tok.encode("\n\n", add_special_tokens=False))
    for b in blocks:
        b_ids = tok.encode(b, add_special_tokens=False)
        if used_tokens + len(b_ids) <= max_cross_tokens:
            surviving_blocks.append(b)
            used_tokens += len(b_ids) + sep_tokens
        else:
            if not surviving_blocks or max_cross_tokens <= 1600:
                rem = max_cross_tokens - used_tokens
                if rem > 24:
                    partial_txt = tok.decode(b_ids[:rem], skip_special_tokens=True)
                    if "\n" in partial_txt:
                        partial_txt = partial_txt.rsplit("\n", 1)[0]
                    if partial_txt.strip():
                        surviving_blocks.append(partial_txt)
            break
    if reverse_surviving:
        surviving_blocks = list(reversed(surviving_blocks))
    joined = "\n\n".join(surviving_blocks)
    return tok.encode(joined[: max_cross_tokens * 10], add_special_tokens=False)[:max_cross_tokens]


def encode_left_context_head_tail(
    left_all_ids: list[int],
    allowed_left: int,
    head_tokens: int = 512,
) -> list[int]:
    if allowed_left <= 0:
        return []
    if len(left_all_ids) <= allowed_left:
        return left_all_ids
    if allowed_left <= head_tokens * 2:
        return left_all_ids[-allowed_left:]
    tail_tokens = allowed_left - head_tokens
    return left_all_ids[:head_tokens] + left_all_ids[-tail_tokens:]


def backfill_consensus_bridge_if_coalesced(
    spc_chunks: list[dict[str, Any]],
    top10: list[dict[str, Any]],
    top25: list[dict[str, Any]],
    cursor_syms: set[str],
    consensus_syms: set[str],
) -> list[dict[str, Any]]:
    has_stop = any(c.get("kind") == "stop" or not c.get("path") for c in top10)
    if has_stop or len(spc_chunks) > 8 or not consensus_syms:
        return spc_chunks
    for cand in top25[10:13]:
        if not cand.get("path") or cand.get("kind") == "stop":
            continue
        cs = set(ID_RE.findall(cand.get("code", ""))) - STOP_SYMS
        if (cs & cursor_syms) and len(cs & consensus_syms) >= 2:
            return spc_chunks + [dict(cand)]
    return spc_chunks


def build_arm_prompt_ids(
    arm: str,
    task: dict[str, Any],
    trace: dict[str, Any],
    files_store: dict[str, dict[str, str]],
    tok: Any,
    left_all_ids: list[int] | None = None,
    path_ids: list[int] | None = None,
) -> tuple[list[int], int, int]:
    lang = task["language"]
    if path_ids is None:
        path_text = f"\n\n# file path: {task['path']}\n\n"
        path_ids = tok.encode(path_text, add_special_tokens=False)
    if left_all_ids is None:
        left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)

    top10 = list(trace["aligncoder_top10"])
    if arm == "aligncoder_2k":
        blocks = format_chunks_exact(top10, lang)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]
        allowed_left = max(0, GENERATOR_CONTEXT - len(cross_ids) - len(path_ids) - 10)
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-GENERATOR_CONTEXT:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    cursor_syms, consensus_syms = extract_clean_markov_blanket(
        task["left_context"], trace["query_enhancement_outputs"], window_lines=5
    )
    top25 = list(trace["rerank_candidates"])
    top1_cos = float(top10[0]["cosine"]) if top10 else 0.0
    recency_gate = is_open_expression_cursor(task["left_context"], top1_cos, threshold=0.85)

    max_total_context = 7950
    max_cross_budget = 3800

    if arm == "mb_spt_spc_top10_8k":
        selected = epsilon_tie_consensus_rescue(top10, cursor_syms, consensus_syms, eps=0.0005, lock_k=4, window_end=10)
        spc_chunks = spc_coalesce_and_expand(
            selected, task, files_store, expand_top_k=0, add_scope_headers=True, max_gap_lines=2
        )
        spc_chunks = backfill_consensus_bridge_if_coalesced(spc_chunks, top10, top25, cursor_syms, consensus_syms)
        blocks = format_chunks_exact(spc_chunks, lang)
        cross_ids = pack_blocks_within_budget(blocks, max_cross_budget, tok, reverse_surviving=recency_gate)
        allowed_left = max(0, max_total_context - len(cross_ids) - len(path_ids) - 10)
        left_ids = encode_left_context_head_tail(left_all_ids, allowed_left, head_tokens=512)
        prompt_ids = (cross_ids + path_ids + left_ids)[-max_total_context:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    if arm == "mb_spt_spc_expand_v2_8k":
        selected = epsilon_tie_consensus_rescue(top10, cursor_syms, consensus_syms, eps=0.0005, lock_k=4, window_end=10)
        spc_chunks = spc_coalesce_and_expand(
            selected, task, files_store, expand_top_k=3, add_scope_headers=True, max_gap_lines=2
        )
        spc_chunks = backfill_consensus_bridge_if_coalesced(spc_chunks, top10, top25, cursor_syms, consensus_syms)
        blocks = format_chunks_exact(spc_chunks, lang)
        cross_ids = pack_blocks_within_budget(blocks, max_cross_budget, tok, reverse_surviving=recency_gate)
        allowed_left = max(0, max_total_context - len(cross_ids) - len(path_ids) - 10)
        left_ids = encode_left_context_head_tail(left_all_ids, allowed_left, head_tokens=512)
        prompt_ids = (cross_ids + path_ids + left_ids)[-max_total_context:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    if arm == "mb_spt_spc_asc_v2_8k":
        selected = epsilon_tie_consensus_rescue(top10, cursor_syms, consensus_syms, eps=0.0005, lock_k=4, window_end=10)
        spc_chunks = spc_coalesce_and_expand(
            selected, task, files_store, expand_top_k=2, add_scope_headers=True, max_gap_lines=2
        )
        spc_chunks = backfill_consensus_bridge_if_coalesced(spc_chunks, top10, top25, cursor_syms, consensus_syms)
        blocks = format_chunks_exact(spc_chunks, lang)
        cross_ids = pack_blocks_within_budget(blocks, max_cross_budget, tok, reverse_surviving=True)
        allowed_left = max(0, max_total_context - len(cross_ids) - len(path_ids) - 10)
        left_ids = encode_left_context_head_tail(left_all_ids, allowed_left, head_tokens=512)
        prompt_ids = (cross_ids + path_ids + left_ids)[-max_total_context:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    raise ValueError(f"Unknown arm: {arm}")


def has_error(node: Any) -> bool:
    if node.type == "ERROR":
        return True
    try:
        return any(has_error(child) for child in node.children)
    except RecursionError:
        return True


def remove_comments(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"//.*", "", code)
    return code


def postprocess_official(prompt: str, completion: str, language: str, parser_map: dict[str, Any]) -> str:
    try:
        if language in ("java", "csharp", "typescript"):
            end_idx = None
            for i in range(len(completion)):
                if completion[i] in (";", "}", "{"):
                    end_idx = i
                    break
            return completion[: end_idx + 1] if end_idx else completion
        if language == "python":
            parser = parser_map.get(language)
            if parser is None:
                return completion
            for i in range(len(completion)):
                # Check newline condition FIRST before invoking tree-sitter (50x faster, 100% identical result)
                if i + 1 >= len(completion) or completion[i + 1] != "\n":
                    continue
                code = (prompt + completion[: i + 1]).encode("utf-8", errors="replace")
                tree = parser.parse(code)
                if tree is None or has_error(tree.root_node):
                    continue
                return completion[: i + 1].rstrip()
            return completion
        return completion
    except Exception:
        return completion


_KW_CACHE: dict[str, set[str]] = {}


def get_keywords(language: str) -> set[str]:
    if language not in _KW_CACHE:
        if language == "python":
            _KW_CACHE[language] = set(keyword.kwlist) - {"True", "False"}
        else:
            kw_path = ROOT_DIR / "aligncoder" / "utils" / "keywords" / f"{language}.txt"
            _KW_CACHE[language] = {
                l.strip() for l in kw_path.read_text(encoding="utf-8").splitlines() if l.strip()
            }
    return _KW_CACHE[language]


def extract_identifiers_official(source_code: str, language: str) -> list[str]:
    clean = STRING_PATTERN.sub("", source_code)
    kws = get_keywords(language)
    return [
        t for t in WORD_RE.findall(clean)
        if re.match(r"[_a-zA-Z][_a-zA-Z0-9]*", t) and t not in kws
    ]


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


def metric_row_official(task: dict[str, Any], raw_prediction: str, parser_map: dict[str, Any]) -> dict[str, Any]:
    lang = task["language"]
    prediction = postprocess_official(task["left_context"], raw_prediction, lang, parser_map)
    prediction = remove_comments(prediction)
    target = remove_comments(task["groundtruth"])

    pred_lines = [l.strip() for l in prediction.split("\n") if l.strip()]
    gt_lines = [l.strip() for l in target.split("\n") if l.strip()]
    em = int(pred_lines == gt_lines)

    try:
        from fuzzywuzzy import fuzz
        es_fuzz = fuzz.ratio(target.strip(), prediction.strip()) / 100.0
    except Exception:
        es_fuzz = difflib.SequenceMatcher(None, target.strip(), prediction.strip()).ratio()

    pred_eval_lines = pred_lines[: len(gt_lines)]
    target_eval = "\n".join(gt_lines)
    pred_eval = "\n".join(pred_eval_lines)
    max_len = max(len(target_eval), len(pred_eval))
    es_repoeval = 1.0 - (levenshtein_distance(target_eval, pred_eval) / max_len) if max_len else 1.0
    em_repoeval = float(len(gt_lines) == len(pred_eval_lines) and gt_lines == pred_eval_lines)

    pred_ids_list = extract_identifiers_official(prediction, lang)
    target_ids_list = extract_identifiers_official(target, lang)
    id_em = float(pred_ids_list == target_ids_list)

    pred_ids_set = set(pred_ids_list)
    target_ids_set = set(target_ids_list)
    tp = len(pred_ids_set & target_ids_set)
    fp = len(pred_ids_set - target_ids_set)
    fn = len(target_ids_set - pred_ids_set)
    id_precision = tp / (tp + fp) if (tp + fp) else 0.0
    id_recall = tp / (tp + fn) if (tp + fn) else 0.0
    id_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    return {
        "task_id": task["task_id"],
        "benchmark": task["benchmark"],
        "repo_id": task["task_id"].split("/")[0],
        "prediction": prediction,
        "target": target,
        "em": em,
        "es": es_fuzz,
        "es_repoeval": es_repoeval,
        "em_repoeval": em_repoeval,
        "id_em": id_em,
        "id_precision": id_precision,
        "id_recall": id_recall,
        "id_f1": id_f1,
    }


def aggregate_metrics_official(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        group = [row for row in rows if row["benchmark"] == benchmark]
        n = len(group)
        metric_keys = ("em", "es", "es_repoeval", "id_em", "id_precision", "id_recall", "id_f1")
        summary = {key: round(100.0 * sum(row[key] for row in group) / n, 4) for key in metric_keys}
        by_repo = defaultdict(list)
        for row in group:
            by_repo[row["repo_id"]].append(row)
        repo_em = sum(round(sum(r["em_repoeval"] for r in v) / len(v), 4) for v in by_repo.values()) / len(by_repo)
        repo_es = sum(round(sum(r["es_repoeval"] for r in v) / len(v), 4) for v in by_repo.values()) / len(by_repo)
        summary["repo_macro_em"] = round(repo_em * 100.0, 4)
        summary["repo_macro_es"] = round(repo_es * 100.0, 4)
        summary["count"] = n
        summary["mean_prompt_tokens"] = round(sum(r["prompt_tokens"] for r in group) / n, 1)
        summary["mean_crossfile_tokens"] = round(sum(r["crossfile_tokens"] for r in group) / n, 1)
        result[benchmark] = summary
    return result


def compute_bootstrap_pair(
    paired: list[tuple[dict[str, Any], dict[str, Any]]],
    rounds: int,
    rng: random.Random,
) -> dict[str, Any]:
    diffs = {metric: [] for metric in ("em", "es", "es_repoeval", "id_em", "id_f1")}
    for _ in range(rounds):
        sample = [paired[rng.randrange(len(paired))] for _ in paired]
        for metric in diffs:
            diffs[metric].append(100.0 * sum(b[metric] - a[metric] for a, b in sample) / len(sample))
    wins = sum(1 for a, b in paired if b["es"] > a["es"] + 1e-6)
    losses = sum(1 for a, b in paired if b["es"] < a["es"] - 1e-6)
    ties = len(paired) - wins - losses
    em_wins = sum(1 for a, b in paired if b["em"] > a["em"])
    em_losses = sum(1 for a, b in paired if b["em"] < a["em"])
    return {
        "es_wins_losses_ties": [wins, losses, ties],
        "em_wins_losses": [em_wins, em_losses],
        **{
            metric: {
                "mean_delta_pp": round(sum(values) / len(values), 4),
                "ci95_pp": [
                    round(sorted(values)[int(0.025 * rounds)], 4),
                    round(sorted(values)[int(0.975 * rounds)], 4),
                ],
            }
            for metric, values in diffs.items()
        },
    }


def paired_bootstrap_official(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    baseline: str = "aligncoder_2k",
    rounds: int = 1000,
) -> dict[str, Any]:
    rng = random.Random(SEED)
    baseline_rows = {(row["benchmark"], row["task_id"]): row for row in rows_by_arm[baseline]}
    out = {}
    for arm, arm_rows in rows_by_arm.items():
        if arm == baseline:
            continue
        current = {(row["benchmark"], row["task_id"]): row for row in arm_rows}
        paired_keys = sorted(baseline_rows.keys() & current.keys())
        paired = [(baseline_rows[key], current[key]) for key in paired_keys]
        by_bench = {}
        for b in sorted({k[0] for k in paired_keys}):
            b_paired = [(baseline_rows[k], current[k]) for k in paired_keys if k[0] == b]
            by_bench[b] = compute_bootstrap_pair(b_paired, rounds, rng)
        out[arm] = {
            "overall": compute_bootstrap_pair(paired, rounds, rng),
            "by_benchmark": by_bench,
        }
    return out


def load_official_aligncoder_predictions() -> dict[tuple[str, str], str]:
    preds: dict[tuple[str, str], str] = {}
    for bench in ("cceval_python", "cceval_java", "repoeval_api", "repoeval_line"):
        pfile = ROOT_DIR / "official_results" / bench / "prediction.jsonl"
        if pfile.exists():
            with pfile.open("r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    preds[(bench, r["task_id"])] = r["pred"]
    return preds


def run_iteration(
    global_ns: dict[str, Any],
    subset_per_benchmark: int = 3000,
    tag: str = "cceval_full_4804",
) -> dict[str, Any]:
    if "VLLM_ENGINE" not in global_ns or "MB_DATA" not in global_ns:
        init_globals(global_ns)
    from vllm import SamplingParams

    engine = global_ns["VLLM_ENGINE"]
    data = global_ns["MB_DATA"]
    all_tasks = data["tasks"]
    traces = data["traces"]
    files_store = data["files"]
    parser_map = data["parser_map"]
    tok = data["tokenizer"]

    by_bench = defaultdict(list)
    for t in all_tasks:
        if len(by_bench[t["benchmark"]]) < subset_per_benchmark:
            by_bench[t["benchmark"]].append(t)
    selected_tasks = [t for b in sorted(by_bench) for t in by_bench[b]]
    print(f"[{tag}] Evaluating {len(selected_tasks)} tasks (up to {subset_per_benchmark}/benchmark)...", flush=True)

    official_preds = load_official_aligncoder_predictions()
    use_cached_baseline = len(official_preds) >= len(selected_tasks)

    # Run our winning unified arm mb_spt_spc_asc_v2_8k (and expand_v2_8k if subset <= 300)
    if subset_per_benchmark <= 300:
        gen_arms = ["mb_spt_spc_expand_v2_8k", "mb_spt_spc_asc_v2_8k"]
    else:
        gen_arms = ["mb_spt_spc_asc_v2_8k"]
    if not use_cached_baseline:
        gen_arms = ["aligncoder_2k"] + gen_arms
    all_arms = (["aligncoder_2k"] + gen_arms) if use_cached_baseline else gen_arms

    all_token_prompts = []
    prompt_meta = []
    baseline_meta = []
    t_prep = time.time()
    for t in selected_tasks:
        trace = traces[(t["benchmark"], t["task_id"])]
        path_text = f"\n\n# file path: {t['path']}\n\n"
        path_ids = tok.encode(path_text, add_special_tokens=False)
        left_all_ids = tok.encode(t["left_context"], add_special_tokens=False)
        if use_cached_baseline:
            _, p_toks, c_toks = build_arm_prompt_ids(
                "aligncoder_2k", t, trace, files_store, tok, left_all_ids=left_all_ids, path_ids=path_ids
            )
            baseline_meta.append((t, p_toks, c_toks))
        for arm in gen_arms:
            p_ids, p_toks, c_toks = build_arm_prompt_ids(
                arm, t, trace, files_store, tok, left_all_ids=left_all_ids, path_ids=path_ids
            )
            all_token_prompts.append({"prompt_token_ids": p_ids})
            prompt_meta.append((arm, t, p_toks, c_toks))
    print(f"[{tag}] Built {len(all_token_prompts)} GPU prompts in {time.time() - t_prep:.2f}s", flush=True)

    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=GENERATOR_MAX_NEW, n=1, seed=123)
    t0 = time.time()
    outputs = engine.generate(all_token_prompts, params, use_tqdm=True)
    elapsed = time.time() - t0
    print(f"[{tag}] Generated {len(all_token_prompts)} completions in {elapsed:.2f}s", flush=True)

    t_eval = time.time()
    rows_by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in all_arms}
    all_rows = []

    if use_cached_baseline:
        for t, p_toks, c_toks in baseline_meta:
            raw_pred = official_preds[(t["benchmark"], t["task_id"])]
            row = metric_row_official(t, raw_pred, parser_map)
            row.update({"arm": "aligncoder_2k", "raw_prediction": raw_pred, "prompt_tokens": p_toks, "crossfile_tokens": c_toks})
            rows_by_arm["aligncoder_2k"].append(row)
            all_rows.append(row)

    for (arm, t, p_toks, c_toks), req_out in zip(prompt_meta, outputs):
        token_ids = list(req_out.outputs[0].token_ids)
        raw_pred = tok.decode(token_ids, skip_special_tokens=True)
        row = metric_row_official(t, raw_pred, parser_map)
        row.update({"arm": arm, "raw_prediction": raw_pred, "prompt_tokens": p_toks, "crossfile_tokens": c_toks})
        rows_by_arm[arm].append(row)
        all_rows.append(row)
    print(f"[{tag}] Evaluated {len(all_rows)} predictions in {time.time() - t_eval:.2f}s", flush=True)

    summaries = {arm: aggregate_metrics_official(rows) for arm, rows in rows_by_arm.items()}
    bootstrap = paired_bootstrap_official(rows_by_arm, baseline="aligncoder_2k", rounds=1000)
    overall_by_arm = {}
    for arm, rows in rows_by_arm.items():
        n = len(rows)
        overall_by_arm[arm] = {
            "em": round(100.0 * sum(r["em"] for r in rows) / n, 4),
            "es": round(100.0 * sum(r["es"] for r in rows) / n, 4),
            "es_repoeval": round(100.0 * sum(r["es_repoeval"] for r in rows) / n, 4),
            "id_em": round(100.0 * sum(r["id_em"] for r in rows) / n, 4),
            "id_f1": round(100.0 * sum(r["id_f1"] for r in rows) / n, 4),
            "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / n, 1),
            "mean_crossfile_tokens": round(sum(r["crossfile_tokens"] for r in rows) / n, 1),
        }

    result = {
        "tag": tag,
        "sample_count": len(selected_tasks),
        "subset_per_benchmark": subset_per_benchmark,
        "generation_seconds": round(elapsed, 2),
        "overall_by_arm": overall_by_arm,
        "metrics_by_arm": summaries,
        "paired_bootstrap_vs_aligncoder_2k": bootstrap,
    }
    out_dir = ROOT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with gzip.open(out_dir / f"{tag}_predictions.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result
