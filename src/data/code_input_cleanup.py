"""Conservative input-only cleanup. Never apply to gold or overwrite source bytes.

Language is not a noise criterion. Functional comments, docstrings, literals,
indentation and the final cursor whitespace survive. Byte offsets refer to the
original source, never the cleaned representation.
"""
from collections import Counter
from functools import lru_cache
import re
import unicodedata

import tree_sitter as ts
import tree_sitter_python
import tree_sitter_java

VERSION = 'code_input_cleanup_v2'
COMMENT_TYPES = {'comment', 'line_comment', 'block_comment'}
STRING_TYPES = {'string', 'concatenated_string', 'string_literal', 'character_literal',
                'text_block', 'string_content', 'string_fragment', 'escape_sequence'}
PRAGMA = re.compile(r'(?:^#!|coding\s*[:=]|noqa|type:\s*ignore|pylint:|mypy:|'
                    r'fmt:|isort:|noinspection|CHECKSTYLE|@formatter|SPDX-License)', re.I)
LICENSE = re.compile(r'copyright|all rights reserved|licensed under|'
                     r'http[s]?://(?:www\.)?(?:apache\.org/licenses|opensource\.org/licenses)|'
                     r'this (?:file|program|software) is (?:free software|distributed)|'
                     r'without (?:any )?warranty|GNU (?:general public|lesser general)', re.I)


def punctuation_only(text):
    """Recognize punctuation/banner-only comments without assuming a language."""
    compact = ''.join(text.split())
    if not compact:
        return False
    return all(unicodedata.category(char)[0] in {'P', 'S'} for char in compact)


@lru_cache(maxsize=64)
def intervals(code, language):
    grammar = tree_sitter_python if language == 'python' else tree_sitter_java
    raw = code.encode('utf-8')
    tree = ts.Parser(ts.Language(grammar.language())).parse(raw)
    comments, protected = [], []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in COMMENT_TYPES:
            comments.append((node.start_byte, node.end_byte))
            continue
        if node.type in STRING_TYPES:
            protected.append((node.start_byte, node.end_byte))
            continue
        stack.extend(reversed(node.named_children))
    return raw, comments, protected, tree.root_node.has_error


@lru_cache(maxsize=256)
def _clean(code, language, max_blank_lines):
    if language not in {'python', 'java'}:
        raise ValueError('Unsupported source language')
    if max_blank_lines < 1:
        raise ValueError('At least one blank line must remain allowed')
    raw, comments, protected, parse_error = intervals(code, language)
    stats = Counter()
    stats['parse_error'] = int(parse_error)
    # Incomplete prefixes/chunk windows may not parse. Do not guess where strings
    # end: preserve the entire input in that case, including repeated newlines.
    if parse_error:
        # Completion prefixes commonly end with `obj.` or an open call. Clean a
        # parseable preceding prefix, never the uncertain cursor suffix.
        boundaries = [m.end() for m in re.finditer('\n', code)]
        for boundary in reversed(boundaries[-64:]):
            if boundary >= len(code):
                continue
            prefix = code[:boundary]
            if not intervals(prefix, language)[3]:
                text, prior_stats = _clean(prefix, language, max_blank_lines)
                stats.update(prior_stats)
                stats['preserved_uncertain_suffix'] += 1
                return text + code[boundary:], dict(stats)
        stats['preserved_uncertain_input'] += 1
        return code, dict(stats)
    edits = []
    last_comment, last_end = None, -1
    for start, end in sorted(comments):
        text = raw[start:end].decode('utf-8')
        body = text.strip().strip('/#* \t')
        reason = None
        if not PRAGMA.search(text):
            if LICENSE.search(text):
                reason = 'license_comments_removed'
            elif body and (re.fullmatch(r'[=\-_*#~/|+ .,;:!?()\[\]{}<>]+', body)
                           or punctuation_only(body)):
                reason = 'separator_comments_removed'
            elif not body:
                reason = 'empty_comments_removed'
            elif text == last_comment and not raw[last_end:start].strip():
                reason = 'adjacent_duplicate_comments_removed'
        if reason:
            # Spaces prevent accidental merging: Java int/*...*/value must stay two tokens.
            replacement = bytes(10 if b == 10 else 13 if b == 13 else 32 for b in raw[start:end])
            edits.append((start, end, replacement))
            stats[reason] += 1
        last_comment, last_end = text, end
    buffer = bytearray(raw)
    for start, end, replacement in edits:
        buffer[start:end] = replacement
    # Original byte positions are still valid because removals kept byte lengths.
    output, offset, blanks = [], 0, 0
    for line in bytes(buffer).splitlines(keepends=True):
        end = offset + len(line)
        intersects_literal = any(a < end and offset < b for a, b in protected)
        terminal_cursor = end == len(raw) and not line.endswith((b'\n', b'\r'))
        if intersects_literal:
            output.append(line)
            blanks = 0
        else:
            normalized = line.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
            blank = not normalized.strip()
            blanks = blanks + 1 if blank else 0
            if blank and blanks > max_blank_lines and not terminal_cursor:
                stats['blank_lines_removed'] += 1
            else:
                # Remove trailing spaces only on completed lines outside literals.
                if normalized.endswith(b'\n'):
                    normalized = normalized[:-1].rstrip(b' \t') + b'\n'
                output.append(normalized)
            stats['crlf_normalized_outside_literals'] += line.count(b'\r\n')
        offset = end
    result = b''.join(output).decode('utf-8')
    stats['bytes_saved'] = len(raw) - len(result.encode('utf-8'))
    return result, dict(stats)


def clean_code_input(code, language, max_blank_lines=2):
    text, stats = _clean(code, language, max_blank_lines)
    return {'text': text, 'diagnostics': dict(stats), 'version': VERSION}


def clean_chunk(chunk, source, language):
    """Clean with whole-file syntax context so split string windows stay literal."""
    raw, comments, protected, parse_error = intervals(source, language)
    start, end = chunk['start'], chunk['end']
    if raw[start:end].decode('utf-8') != chunk['text']:
        raise ValueError('Chunk offsets do not match the original source')
    # A window cutting through a string or comment cannot be parsed standalone.
    boundary_cut = any(a < start < b or a < end < b for a, b in protected + comments)
    if parse_error or boundary_cut:
        return {'text': chunk['text'], 'diagnostics': {'preserved_uncertain_input': 1}, 'version': VERSION}
    return clean_code_input(chunk['text'], language)
