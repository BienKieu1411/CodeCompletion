"""AST-based repository chunking for Co-Retrieval.

The training sampler in ``co_retrieval.data.repository_dataset_loader`` already
creates code-completion cuts carefully. This module is the repository-side
candidate chunker from ``Novelty.md``: it chunks stable cross-file code into
entity-aligned evidence snippets with metadata useful for retrieval.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
_CALL_RE = re.compile(r"(?<!def\s)(?<!class\s)([_a-zA-Z][_a-zA-Z0-9]*)\s*\(")


@dataclass(frozen=True)
class CodeChunk:
    """One retrievable code chunk with entity-aware metadata."""

    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    text: str
    defined_symbols: List[str] = field(default_factory=list)
    used_symbols: List[str] = field(default_factory=list)
    call_names: List[str] = field(default_factory=list)
    parent_class: Optional[str] = None
    class_bases: List[str] = field(default_factory=list)
    method_names: List[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return f"{self.file_path}::L{self.start_line}-{self.end_line}::{self.chunk_type}"

    def retrieval_text(self) -> str:
        header = f"### File: {self.file_path} L{self.start_line}-{self.end_line} [{self.chunk_type}]"
        if self.parent_class:
            header += f" class={self.parent_class}"
        return f"{header}\n{self.text}".strip()


class RepositoryChunker:
    """Create entity-boundary chunks for Python repositories.

    Python is parsed with the stdlib AST because it provides reliable 1-based
    line metadata and works without optional parser wheels. Invalid or currently
    unsupported files fall back to small line windows marked as ``fallback``.
    """

    PYTHON_SUFFIXES = {".py"}
    DEFAULT_EXCLUDE_DIRS = {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        ".venv",
        "venv",
    }

    def __init__(self, max_chunk_lines: int = 120, fallback_lines: int = 40) -> None:
        if max_chunk_lines <= 0:
            raise ValueError("max_chunk_lines must be positive")
        if fallback_lines <= 0:
            raise ValueError("fallback_lines must be positive")
        self.max_chunk_lines = max_chunk_lines
        self.fallback_lines = fallback_lines

    def chunk_repository(self, root: str | Path, suffixes: Optional[Sequence[str]] = None) -> List[CodeChunk]:
        """Chunk all supported source files under ``root``."""
        root_path = Path(root)
        suffix_set = set(suffixes or self.PYTHON_SUFFIXES)
        chunks: List[CodeChunk] = []
        for path in sorted(root_path.rglob("*")):
            if not path.is_file() or path.suffix not in suffix_set:
                continue
            if any(part in self.DEFAULT_EXCLUDE_DIRS for part in path.parts):
                continue
            chunks.extend(self.chunk_file(path, repo_root=root_path))
        return chunks

    def chunk_file(self, file_path: str | Path, repo_root: str | Path | None = None) -> List[CodeChunk]:
        """Chunk a single source file."""
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        display_path = self._display_path(path, repo_root)

        if path.suffix not in self.PYTHON_SUFFIXES:
            return self._fallback_chunks(display_path, text)

        try:
            tree = ast.parse(text)
        except SyntaxError:
            return self._fallback_chunks(display_path, text)

        return self._python_chunks(display_path, text, tree)

    def chunk_source(self, file_path: str, source_code: str) -> List[CodeChunk]:
        """Chunk source text that has not been written to disk."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return self._fallback_chunks(file_path, source_code)
        return self._python_chunks(file_path, source_code, tree)

    @staticmethod
    def _display_path(path: Path, repo_root: str | Path | None) -> str:
        if repo_root is None:
            return path.as_posix()
        try:
            return path.relative_to(Path(repo_root)).as_posix()
        except ValueError:
            return path.as_posix()

    def _python_chunks(self, file_path: str, source_code: str, tree: ast.AST) -> List[CodeChunk]:
        lines = source_code.splitlines()
        chunks: List[CodeChunk] = []
        occupied_lines: set[int] = set()

        imports_and_globals = self._global_chunk(file_path, lines, tree)
        if imports_and_globals is not None:
            chunks.append(imports_and_globals)

        for node in getattr(tree, "body", []):
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._class_chunks(file_path, lines, node))
                occupied_lines.update(range(node.lineno, self._end_line(node) + 1))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.extend(self._function_chunks(file_path, lines, node, "function"))
                occupied_lines.update(range(node.lineno, self._end_line(node) + 1))

        leftovers = self._module_body_chunks(file_path, lines, occupied_lines)
        chunks.extend(leftovers)

        if not chunks:
            return self._fallback_chunks(file_path, source_code)
        return chunks

    def _global_chunk(self, file_path: str, lines: List[str], tree: ast.AST) -> Optional[CodeChunk]:
        global_nodes = []
        for node in getattr(tree, "body", []):
            is_global = isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom,
                    ast.Assign,
                    ast.AnnAssign,
                    ast.AugAssign,
                ),
            )
            if is_global:
                global_nodes.append(node)

        if not global_nodes:
            return None

        start = min(node.lineno for node in global_nodes)
        end = max(self._end_line(node) for node in global_nodes)
        text = self._slice(lines, start, end)
        defined = self._defined_symbols_from_nodes(global_nodes)
        return CodeChunk(
            file_path=file_path,
            start_line=start,
            end_line=end,
            chunk_type="global",
            text=text,
            defined_symbols=defined,
            used_symbols=self._identifiers(text),
            call_names=self._call_names(text),
        )

    def _class_chunks(self, file_path: str, lines: List[str], node: ast.ClassDef) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        bases = [self._name_from_expr(base) for base in node.bases]
        bases = [base for base in bases if base]
        methods = [
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        header_end = self._class_header_end(node)
        header_text = self._slice(lines, node.lineno, header_end)
        chunks.append(
            CodeChunk(
                file_path=file_path,
                start_line=node.lineno,
                end_line=header_end,
                chunk_type="class_header",
                text=header_text,
                defined_symbols=[node.name],
                used_symbols=self._identifiers(header_text),
                call_names=self._call_names(header_text),
                parent_class=node.name,
                class_bases=bases,
                method_names=methods,
            )
        )

        if not methods:
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=node.lineno,
                    end_line=self._end_line(node),
                    chunk_type="class_body",
                    text=self._slice(lines, node.lineno, self._end_line(node)),
                    defined_symbols=[node.name],
                    used_symbols=self._identifiers(self._slice(lines, node.lineno, self._end_line(node))),
                    call_names=self._call_names(self._slice(lines, node.lineno, self._end_line(node))),
                    parent_class=node.name,
                    class_bases=bases,
                    method_names=[],
                )
            )
            return chunks

        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            method_text = f"{header_text}\n{self._slice(lines, child.lineno, self._end_line(child))}"
            for chunk in self._split_text_chunk(
                file_path=file_path,
                start_line=child.lineno,
                text=method_text,
                chunk_type="method",
                defined_symbols=[child.name],
                parent_class=node.name,
                class_bases=bases,
                method_names=methods,
            ):
                chunks.append(chunk)
        return chunks

    def _function_chunks(
        self,
        file_path: str,
        lines: List[str],
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        chunk_type: str,
    ) -> List[CodeChunk]:
        return self._split_text_chunk(
            file_path=file_path,
            start_line=node.lineno,
            text=self._slice(lines, node.lineno, self._end_line(node)),
            chunk_type=chunk_type,
            defined_symbols=[node.name],
        )

    def _module_body_chunks(self, file_path: str, lines: List[str], occupied_lines: Iterable[int]) -> List[CodeChunk]:
        occupied = set(occupied_lines)
        blocks: List[CodeChunk] = []
        current: List[tuple[int, str]] = []

        def flush() -> None:
            nonlocal current
            if not current:
                return
            start = current[0][0]
            text_lines = [line for _, line in current]
            for offset in range(0, len(text_lines), self.fallback_lines):
                sub = text_lines[offset : offset + self.fallback_lines]
                sub_start = start + offset
                body = "\n".join(sub)
                if body.strip():
                    blocks.append(
                        CodeChunk(
                            file_path=file_path,
                            start_line=sub_start,
                            end_line=sub_start + len(sub) - 1,
                            chunk_type="global",
                            text=body,
                            defined_symbols=self._identifiers(body[: body.find("\n") if "\n" in body else len(body)]),
                            used_symbols=self._identifiers(body),
                            call_names=self._call_names(body),
                        )
                    )
            current = []

        for line_no, line in enumerate(lines, start=1):
            if line_no in occupied or not line.strip():
                flush()
                continue
            current.append((line_no, line))
        flush()
        return blocks

    def _split_text_chunk(
        self,
        file_path: str,
        start_line: int,
        text: str,
        chunk_type: str,
        defined_symbols: List[str],
        parent_class: Optional[str] = None,
        class_bases: Optional[List[str]] = None,
        method_names: Optional[List[str]] = None,
    ) -> List[CodeChunk]:
        lines = text.splitlines()
        if len(lines) <= self.max_chunk_lines:
            return [
                CodeChunk(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=start_line + len(lines) - 1,
                    chunk_type=chunk_type,
                    text=text,
                    defined_symbols=defined_symbols,
                    used_symbols=self._identifiers(text),
                    call_names=self._call_names(text),
                    parent_class=parent_class,
                    class_bases=class_bases or [],
                    method_names=method_names or [],
                )
            ]

        chunks: List[CodeChunk] = []
        for offset in range(0, len(lines), self.max_chunk_lines):
            sub = "\n".join(lines[offset : offset + self.max_chunk_lines])
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=start_line + offset,
                    end_line=start_line + offset + len(sub.splitlines()) - 1,
                    chunk_type=chunk_type,
                    text=sub,
                    defined_symbols=defined_symbols if offset == 0 else [],
                    used_symbols=self._identifiers(sub),
                    call_names=self._call_names(sub),
                    parent_class=parent_class,
                    class_bases=class_bases or [],
                    method_names=method_names or [],
                )
            )
        return chunks

    def _fallback_chunks(self, file_path: str, source_code: str) -> List[CodeChunk]:
        lines = source_code.splitlines() or [source_code]
        chunks: List[CodeChunk] = []
        for offset in range(0, len(lines), self.fallback_lines):
            sub_lines = lines[offset : offset + self.fallback_lines]
            text = "\n".join(sub_lines)
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=offset + 1,
                    end_line=offset + len(sub_lines),
                    chunk_type="fallback",
                    text=text,
                    defined_symbols=[],
                    used_symbols=self._identifiers(text),
                    call_names=self._call_names(text),
                )
            )
        return chunks

    @staticmethod
    def _slice(lines: List[str], start_line: int, end_line: int) -> str:
        return "\n".join(lines[start_line - 1 : end_line])

    @staticmethod
    def _end_line(node: ast.AST) -> int:
        return int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))

    @staticmethod
    def _class_header_end(node: ast.ClassDef) -> int:
        header_end = node.lineno
        if node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
                if isinstance(first.value.value, str):
                    header_end = int(getattr(first, "end_lineno", first.lineno))
        return header_end

    @staticmethod
    def _name_from_expr(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            left = RepositoryChunker._name_from_expr(node.value)
            return f"{left}.{node.attr}" if left else node.attr
        if isinstance(node, ast.Subscript):
            return RepositoryChunker._name_from_expr(node.value)
        return ""

    @staticmethod
    def _defined_symbols_from_nodes(nodes: Iterable[ast.AST]) -> List[str]:
        symbols: List[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name and name not in seen:
                seen.add(name)
                symbols.append(name)

        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    add((alias.asname or alias.name.split(".")[0]).strip())
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in RepositoryChunker._names_from_target(target):
                        add(name)
            elif isinstance(node, ast.AnnAssign):
                for name in RepositoryChunker._names_from_target(node.target):
                    add(name)
        return symbols

    @staticmethod
    def _names_from_target(target: ast.AST) -> List[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            out: List[str] = []
            for elt in target.elts:
                out.extend(RepositoryChunker._names_from_target(elt))
            return out
        return []

    @staticmethod
    def _identifiers(text: str) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for match in _IDENTIFIER_RE.finditer(text or ""):
            token = match.group(0)
            if token not in seen:
                seen.add(token)
                out.append(token)
        return out

    @staticmethod
    def _call_names(text: str) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for match in _CALL_RE.finditer(text or ""):
            token = match.group(1)
            if token not in {"if", "for", "while", "with", "return", "class", "def"} and token not in seen:
                seen.add(token)
                out.append(token)
        return out
