"""
Context Composer for repository-level code completion.

Separates preserved local context from retrieved cross-file graph context,
with budget-aware deduplication.
"""

from __future__ import annotations

import re
from typing import List, Optional


class GraphContextComposer:
    def __init__(self, max_length: int = 2048):
        self.max_length = max_length

    @staticmethod
    def _deduplicate_blocks(text: str) -> str:
        lines = [ln for ln in (text or "").splitlines()]
        blocks: List[str] = []
        current: List[str] = []
        for ln in lines:
            if ln.strip().startswith("### File:") and current:
                blocks.append("\n".join(current).strip())
                current = [ln]
            else:
                current.append(ln)
        if current:
            blocks.append("\n".join(current).strip())

        seen = set()
        uniq: List[str] = []
        for block in blocks:
            norm = "\n".join([x.strip() for x in block.splitlines() if x.strip()])
            if not norm:
                continue
            if norm not in seen:
                seen.add(norm)
                uniq.append(block)
        return "\n\n".join(uniq)

    @staticmethod
    def _tail_lines(text: str, n: int) -> str:
        lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        return "\n".join(lines[-max(1, n):]).strip()

    @staticmethod
    def _compact_retrieved_block(block: str, max_lines: int = 18) -> str:
        lines = [ln for ln in (block or "").splitlines()]
        if not lines:
            return ""
        header = lines[0]
        body = lines[1:]
        if len(lines) <= max_lines:
            return block

        picked: List[str] = [header]
        signature_like = re.compile(r"^\s*(async\s+def|def|class|@|from\s+\S+\s+import|import\s+\S+)")

        for ln in body:
            if signature_like.match(ln):
                picked.append(ln)
            if len(picked) >= max_lines:
                break

        if len(picked) < max_lines:
            for ln in body:
                if ln not in picked:
                    picked.append(ln)
                if len(picked) >= max_lines:
                    break
        return "\n".join(picked)

    def _compact_retrieved_context(self, text: str, per_block_max_lines: int = 18) -> str:
        blocks = [b.strip() for b in (text or "").split("\n\n") if b.strip()]
        compacted = [self._compact_retrieved_block(b, max_lines=per_block_max_lines) for b in blocks]
        return "\n\n".join([x for x in compacted if x.strip()])

    def compose(
        self,
        retrieved_context: str,
        left_context: str,
        file_path: str,
        local_graph: Optional[object] = None,
        immediate_left_lines: int = 40,
        max_imports: int = 16,
    ) -> str:
        imports = []
        scope = ""
        local_vars = []
        if local_graph is not None:
            imports = list(getattr(local_graph, "imports", []) or [])
            scope = getattr(local_graph, "current_scope", "") or ""
            local_vars = list(getattr(local_graph, "local_variables", []) or [])

        immediate_left = self._tail_lines(left_context, immediate_left_lines)
        retrieved = self._deduplicate_blocks(retrieved_context)

        parts = [
            "# Retrieved cross-file graph context",
            retrieved if retrieved else "[None]",
            "",
            "# Current file context",
            f"[Path] {file_path}",
            "",
            "[Imports]",
            "\n".join(imports[:max_imports]) if imports else "[None]",
            "",
            "[Current scope]",
            scope if scope else "[Unknown]",
            "",
            "[Local variables]",
            ", ".join(local_vars[:20]) if local_vars else "[None]",
            "",
            "[Immediate left context]",
            immediate_left,
            "",
            "# Complete here",
        ]

        prompt = "\n".join(parts).strip()

        # Rough char-level budget safety
        max_chars = int(self.max_length * 4.0)
        if len(prompt) > max_chars:
            compact_retrieved = self._compact_retrieved_context(retrieved, per_block_max_lines=14)
            parts[1] = compact_retrieved if compact_retrieved else "[None]"
            prompt = "\n".join(parts).strip()

        if len(prompt) > max_chars:
            overflow = len(prompt) - max_chars
            trimmed_left = immediate_left[max(0, overflow):]
            parts[-3] = trimmed_left
            prompt = "\n".join(parts).strip()

        return prompt
