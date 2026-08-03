"""AST-based repository chunking for Co-Retrieval.

The training sampler in ``co_retrieval.data.repository_dataset_loader`` already
creates code-completion cuts carefully. This module is the repository-side
candidate chunker from ``Novelty.md``: it chunks stable cross-file code into
entity-aligned evidence snippets with metadata useful for retrieval.
"""

from __future__ import annotations

import ast
import importlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence


_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
_CALL_RE = re.compile(r"(?<!def\s)(?<!class\s)([_a-zA-Z][_a-zA-Z0-9]*)\s*\(")
logger = logging.getLogger(__name__)

_EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".rb": "ruby",
}
_TREE_SITTER_PARSERS: dict[str, Any] = {}
_TS_CLASS_TYPES = {
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
    "struct_specifier",
    "class_specifier",
    "type_declaration",
    "module",
}
_TS_FUNCTION_TYPES = {
    "function_declaration",
    "function_definition",
    "method_declaration",
    "constructor_declaration",
    "function_item",
    "method_definition",
}
_TS_FIELD_TYPES = {
    "field_declaration",
    "property_declaration",
    "lexical_declaration",
    "variable_declaration",
    "const_declaration",
    "var_declaration",
}
_TS_GLOBAL_TYPES = {
    "import_statement",
    "import_declaration",
    "package_declaration",
    "using_declaration",
    "include_declaration",
    "preproc_include",
}


def _detect_language(file_path: str | Path) -> str:
    return _EXTENSION_TO_LANGUAGE.get(Path(str(file_path)).suffix.lower(), "python")


def _load_tree_sitter_parser(language: str) -> Any:
    if language in _TREE_SITTER_PARSERS:
        return _TREE_SITTER_PARSERS[language]
    try:
        from tree_sitter_languages import get_parser

        parser = get_parser(language)
    except Exception as exc:
        logger.debug("tree-sitter-languages unavailable for %s: %s", language, exc)
        parser = None

    # Some server environments have the grammar wheels installed but an
    # incompatible tree-sitter-languages binary. Keep AST chunking alive by
    # falling back to the grammar module directly before using line windows.
    if parser is None:
        grammar_modules = {
            "python": "tree_sitter_python",
            "java": "tree_sitter_java",
            "javascript": "tree_sitter_javascript",
            "typescript": "tree_sitter_typescript",
            "go": "tree_sitter_go",
            "cpp": "tree_sitter_cpp",
            "c": "tree_sitter_c",
            "ruby": "tree_sitter_ruby",
        }
        module_name = grammar_modules.get(language)
        if module_name:
            try:
                grammar = importlib.import_module(module_name)
                import tree_sitter

                language_factory = getattr(grammar, "language")
                try:
                    parser = tree_sitter.Parser(
                        tree_sitter.Language(language_factory())
                    )
                except (TypeError, AttributeError):
                    language_obj = tree_sitter.Language(language_factory())
                    parser = tree_sitter.Parser()
                    parser.set_language(language_obj)
            except Exception as exc:
                logger.debug("No direct tree-sitter parser for %s: %s", language, exc)
    _TREE_SITTER_PARSERS[language] = parser
    return parser


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
    """Create entity-boundary chunks for repository source files.

    Python uses the stdlib AST for rich metadata. Other supported languages use
    tree-sitter entity chunks when parser wheels are available. Invalid or
    unsupported files fall back to small line windows marked as ``fallback``.
    """

    PYTHON_SUFFIXES = {".py"}
    SUPPORTED_SUFFIXES = set(_EXTENSION_TO_LANGUAGE)
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
        suffix_set = set(suffixes or self.SUPPORTED_SUFFIXES)
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

        return self.chunk_source(display_path, text)

    def chunk_source(self, file_path: str, source_code: str) -> List[CodeChunk]:
        """Chunk source text that has not been written to disk."""
        language = _detect_language(file_path)
        if language != "python":
            parser = _load_tree_sitter_parser(language)
            if parser is not None:
                try:
                    tree = parser.parse(bytes(source_code or "", "utf8"))
                    return self._tree_sitter_chunks(
                        file_path, source_code, language, tree
                    )
                except Exception as exc:
                    logger.debug("tree-sitter chunking failed for %s: %s", file_path, exc)
            return self._fallback_chunks(file_path, source_code)

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

        global_nodes = self._global_nodes(tree)
        imports_and_globals = self._global_chunk(file_path, lines, global_nodes)
        if imports_and_globals is not None:
            chunks.append(imports_and_globals)
            for node in global_nodes:
                occupied_lines.update(range(node.lineno, self._end_line(node) + 1))

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

    def _global_nodes(self, tree: ast.AST) -> List[ast.AST]:
        global_nodes: List[ast.AST] = []
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
        return global_nodes

    def _global_chunk(
        self,
        file_path: str,
        lines: List[str],
        global_nodes: Sequence[ast.AST],
    ) -> Optional[CodeChunk]:
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
            for chunk in self._split_text_chunk(
                file_path=file_path,
                start_line=child.lineno,
                text=self._slice(lines, child.lineno, self._end_line(child)),
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

    def _tree_sitter_chunks(
        self,
        file_path: str,
        source_code: str,
        language: str,
        tree: Any,
    ) -> List[CodeChunk]:
        lines = source_code.splitlines()
        root = getattr(tree, "root_node", None)
        if root is None:
            return self._fallback_chunks(file_path, source_code)

        chunks: List[CodeChunk] = []
        occupied_lines: set[int] = set()

        global_nodes = [
            child
            for child in self._ts_named_children(root)
            if child.type in _TS_GLOBAL_TYPES
        ]
        if global_nodes:
            start = min(self._ts_start_line(node) for node in global_nodes)
            end = max(self._ts_end_line(node) for node in global_nodes)
            text = self._slice(lines, start, end)
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    chunk_type="global",
                    text=text,
                    defined_symbols=self._identifiers(text),
                    used_symbols=self._identifiers(text),
                    call_names=self._call_names(text),
                )
            )
            self._mark_ts_lines(occupied_lines, global_nodes)

        for child in self._ts_named_children(root):
            if child.type in _TS_CLASS_TYPES:
                chunks.extend(self._ts_class_chunks(file_path, lines, child, language))
                self._mark_ts_lines(occupied_lines, [child])
            elif child.type in _TS_FUNCTION_TYPES:
                name = self._ts_node_name(child, lines)
                chunks.extend(
                    self._split_text_chunk(
                        file_path=file_path,
                        start_line=self._ts_start_line(child),
                        text=self._ts_node_text(lines, child),
                        chunk_type="function",
                        defined_symbols=[name] if name else [],
                    )
                )
                self._mark_ts_lines(occupied_lines, [child])

        chunks.extend(self._module_body_chunks(file_path, lines, occupied_lines))
        return chunks or self._fallback_chunks(file_path, source_code)

    def _ts_class_chunks(
        self,
        file_path: str,
        lines: List[str],
        node: Any,
        language: str,
    ) -> List[CodeChunk]:
        start = self._ts_start_line(node)
        end = self._ts_end_line(node)
        name = self._ts_node_name(node, lines) or "anonymous_class"
        body = self._ts_child_by_field_name(node, "body")
        body_children = self._ts_named_children(body) if body is not None else []

        member_nodes = [
            child
            for child in body_children
            if child.type in _TS_FUNCTION_TYPES or child.type in _TS_FIELD_TYPES
        ]
        method_names = [
            self._ts_node_name(child, lines)
            for child in member_nodes
            if child.type in _TS_FUNCTION_TYPES
        ]
        method_names = [method for method in method_names if method]

        header_end = start
        if body is not None:
            header_end = max(start, min(end, self._ts_start_line(body)))
        elif end > start:
            header_end = min(end, start + min(5, self.max_chunk_lines) - 1)
        header_text = self._slice(lines, start, header_end)
        bases = self._ts_bases_from_header(header_text, name, language)

        chunks: List[CodeChunk] = [
            CodeChunk(
                file_path=file_path,
                start_line=start,
                end_line=header_end,
                chunk_type="class_header",
                text=header_text,
                defined_symbols=[name],
                used_symbols=self._identifiers(header_text),
                call_names=self._call_names(header_text),
                parent_class=name,
                class_bases=bases,
                method_names=method_names,
            )
        ]

        for member in member_nodes:
            member_name = self._ts_node_name(member, lines)
            member_type = "method" if member.type in _TS_FUNCTION_TYPES else "field"
            chunks.extend(
                self._split_text_chunk(
                    file_path=file_path,
                    start_line=self._ts_start_line(member),
                    text=self._ts_node_text(lines, member),
                    chunk_type=member_type,
                    defined_symbols=[member_name] if member_name else [],
                    parent_class=name,
                    class_bases=bases,
                    method_names=method_names,
                )
            )

        if len(chunks) == 1 and end > header_end:
            body_text = self._slice(lines, start, end)
            chunks.extend(
                self._split_text_chunk(
                    file_path=file_path,
                    start_line=start,
                    text=body_text,
                    chunk_type="class_body",
                    defined_symbols=[name],
                    parent_class=name,
                    class_bases=bases,
                    method_names=method_names,
                )
            )
        return chunks

    @staticmethod
    def _ts_named_children(node: Any) -> List[Any]:
        return [
            child
            for child in getattr(node, "children", [])
            if getattr(child, "is_named", True)
        ]

    @staticmethod
    def _ts_child_by_field_name(node: Any, field_name: str) -> Any:
        try:
            return node.child_by_field_name(field_name)
        except (AttributeError, TypeError):
            return None

    @staticmethod
    def _ts_start_line(node: Any) -> int:
        return int(node.start_point[0]) + 1

    @staticmethod
    def _ts_end_line(node: Any) -> int:
        return int(node.end_point[0]) + 1

    @staticmethod
    def _ts_node_text(lines: List[str], node: Any) -> str:
        raw = getattr(node, "text", None)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            return raw
        return RepositoryChunker._slice(
            lines,
            RepositoryChunker._ts_start_line(node),
            RepositoryChunker._ts_end_line(node),
        )

    @staticmethod
    def _mark_ts_lines(occupied: set[int], nodes: Iterable[Any]) -> None:
        for node in nodes:
            occupied.update(
                range(
                    RepositoryChunker._ts_start_line(node),
                    RepositoryChunker._ts_end_line(node) + 1,
                )
            )

    def _ts_node_name(self, node: Any, lines: List[str]) -> str:
        name_node = self._ts_child_by_field_name(node, "name")
        if name_node is not None:
            text = self._ts_node_text(lines, name_node).strip()
            ids = self._identifiers(text)
            return ids[-1] if ids else text

        for child in self._ts_named_children(node):
            if child.type in {
                "identifier",
                "type_identifier",
                "property_identifier",
                "field_identifier",
            }:
                text = self._ts_node_text(lines, child).strip()
                ids = self._identifiers(text)
                return ids[-1] if ids else text
        return ""

    def _ts_bases_from_header(
        self,
        header_text: str,
        class_name: str,
        language: str,
    ) -> List[str]:
        keywords = {
            "class",
            "interface",
            "enum",
            "record",
            "struct",
            "extends",
            "implements",
            "public",
            "private",
            "protected",
            "abstract",
            "final",
            "static",
            "type",
            "module",
            language,
        }
        bases: List[str] = []
        for token in self._identifiers(header_text):
            if token == class_name or token in keywords:
                continue
            if token not in bases:
                bases.append(token)
        return bases

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
