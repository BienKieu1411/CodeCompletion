"""
AST-Aware Query Extractor
Trích xuất imports và class context thông qua phân tích cây cú pháp thay vì cắt dòng cứng.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import tree_sitter
    from tree_sitter import Parser
except Exception:
    tree_sitter = None  # type: ignore[assignment]
    Parser = Any  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class LocalGraph:
    """Đại diện cho đồ thị cục bộ (AST con) xung quanh con trỏ."""

    imports: List[str]
    parent_class: Optional[str]
    parent_function: Optional[str] = None
    local_code: str = ""
    file_path: str = "current_file.py"
    local_variables: List[str] = field(default_factory=list)
    cursor_line: int = 0

    @property
    def current_scope(self) -> Optional[str]:
        return self.parent_function or self.parent_class

    def to_prompt_string(self) -> str:
        """Chuyển đổi thông tin cấu trúc thành chuỗi dọn sạch cho Prompt."""
        res = f"[Path]\n{self.file_path}\n\n"
        if self.imports:
            res += "[Imports]\n" + "\n".join(self.imports) + "\n\n"
        if self.current_scope:
            res += "[Current scope]\n" + self.current_scope + "\n\n"
        if self.local_variables:
            res += "[Local variables]\n" + ", ".join(self.local_variables) + "\n\n"
        res += "[Immediate left context]\n" + self.local_code
        return res.strip()


class LeftContextAnchorExtractor:
    """Bộ trích xuất truy vấn dựa trên cây AST."""

    _LANG_ALIASES = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
    }

    def __init__(self, language: str = "python"):
        self.language = language
        self.parser = self._get_parser(language)

    _PARSER_CACHE: Dict[str, Optional[Parser]] = {}

    @classmethod
    def _normalize_language(cls, language: str) -> str:
        lang = (language or "python").strip().lower()
        return cls._LANG_ALIASES.get(lang, lang)

    @classmethod
    def _get_parser(cls, language: str) -> Optional[Parser]:
        """
        Tạo parser theo ngôn ngữ.

        Ưu tiên `tree_sitter_languages` (tiện + ổn định), fallback sang các
        module tree-sitter-<lang> nếu có.
        """
        language = cls._normalize_language(language)

        # Cache (Tree-sitter parser khá nặng)
        if language in cls._PARSER_CACHE:
            return cls._PARSER_CACHE[language]

        parser: Optional[Parser] = None

        # 1) Prefer tree_sitter_languages if installed
        try:
            import tree_sitter_languages  # type: ignore

            parser = tree_sitter_languages.get_parser(language)
            cls._PARSER_CACHE[language] = parser
            return parser
        except Exception:
            pass

        # 2) Fallback: individual language modules (best-effort)
        lang_modules = {
            "python": "tree_sitter_python",
            "java": "tree_sitter_java",
            "javascript": "tree_sitter_javascript",
            "typescript": "tree_sitter_typescript",
            "go": "tree_sitter_go",
            "cpp": "tree_sitter_cpp",
            "c": "tree_sitter_c",
            "ruby": "tree_sitter_ruby",
        }
        mod_name = lang_modules.get(language)
        if not mod_name:
            cls._PARSER_CACHE[language] = None
            return None

        try:
            import importlib

            mod = importlib.import_module(mod_name)
            # tree_sitter python bindings vary by version; support both styles.
            if tree_sitter is None:
                cls._PARSER_CACHE[language] = None
                return None
            ts_ver = tuple(int(x) for x in tree_sitter.__version__.split(".")[:2])
            if ts_ver >= (0, 21):
                lang_obj = tree_sitter.Language(mod.language())
                p = Parser()
                p.set_language(lang_obj)
                parser = p
                cls._PARSER_CACHE[language] = parser
                return parser

            so_path = f"/tmp/ts_{language}.so"
            tree_sitter.Language.build_library(so_path, [mod.__path__[0]])  # type: ignore[attr-defined]
            lang_obj = tree_sitter.Language(so_path, language)
            p = Parser()
            p.set_language(lang_obj)
            parser = p
            cls._PARSER_CACHE[language] = parser
            return parser
        except Exception:
            cls._PARSER_CACHE[language] = None
            return None

    @staticmethod
    def _coerce_cursor_idx(cursor_line: int, total_lines: int) -> int:
        """
        Quy ước nội bộ: cursor_idx là 0-based line index.

        Hỗ trợ input thường gặp trong codebase:
        - Co-Retrieval đang truyền: len(query.split("\\n")) (tức 1-based-ish / end-of-text)
        - IDE thường truyền 0-based hoặc 1-based tuỳ nơi
        """
        if total_lines <= 0:
            return 0

        if cursor_line <= 0:
            return 0

        # If cursor_line equals total_lines, likely "end of text" from len(splitlines)
        if cursor_line >= total_lines:
            return total_lines - 1

        # Heuristic: treat as 1-based line number in [1, total_lines-1]
        return max(0, min(total_lines - 1, cursor_line - 1))

    def extract_local_graph(
        self,
        source_code: str,
        cursor_line: int,
        context_lines: int = 15,
        file_path: str = "current_file.py",
    ) -> LocalGraph:
        """
        Trích xuất (Imports, Class cha) + local_code quanh con trỏ.
        """
        # splitlines() xử lý tốt hơn CRLF/trailing newline
        lines = source_code.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        cursor_idx = self._coerce_cursor_idx(cursor_line, len(lines))

        local_code = "\n".join(lines[max(0, cursor_idx - max(1, context_lines)) : cursor_idx + 1])

        imports: List[str] = []
        parent_class: Optional[str] = None
        parent_function: Optional[str] = None
        ast_block_range: Optional[Tuple[int, int]] = None

        if self.parser is not None:
            try:
                tree = self.parser.parse(bytes(source_code, "utf8"))
                imports = self._extract_imports(tree, cursor_idx)
                parent_class = self._extract_parent_class(tree, cursor_idx, source_code)
                parent_function = self._extract_parent_function(tree, cursor_idx, source_code)
                ast_block_range = self._extract_enclosing_block_range(tree, cursor_idx)
            except Exception:
                # Tree-sitter unavailable or parse failed; fallback to heuristic parsing.
                pass

        # Fallback: if AST didn't find imports/parent_class, do a cheap text scan.
        if not imports:
            imports = self._extract_imports_fallback(source_code, cursor_idx)
        if parent_class is None:
            parent_class = self._extract_parent_class_fallback(source_code, cursor_idx)
        if parent_function is None:
            parent_function = self._extract_parent_function_fallback(source_code, cursor_idx)

        # AST-aware local_code: ưu tiên block trong function/method nếu tìm được.
        if ast_block_range is not None:
            start, end = ast_block_range
            start = max(0, min(start, cursor_idx))
            end = max(start, min(end, cursor_idx))
            # Giới hạn kích thước để tránh prompt quá dài
            max_lines = max(30, context_lines * 4)
            if end - start + 1 > max_lines:
                start = end - max_lines + 1
            local_code = "\n".join(lines[start : end + 1])

        return LocalGraph(
            imports=imports,
            parent_class=parent_class,
            parent_function=parent_function,
            local_code=local_code,
            file_path=file_path,
            local_variables=self._extract_local_variables(local_code, parent_function),
            cursor_line=cursor_idx + 1,
        )

    def _extract_local_variables(self, local_code: str, parent_function: Optional[str]) -> List[str]:
        """
        Heuristic extraction of local variable state from current scope.
        Captures function parameters and assignment targets.
        """
        symbols: List[str] = []
        seen = set()

        def _add(name: str):
            if not name:
                return
            if name in {"self", "cls"}:
                return
            if name not in seen:
                seen.add(name)
                symbols.append(name)

        if parent_function:
            m = re.search(r"def\s+\w+\(([^)]*)\)", parent_function)
            if m:
                params = [p.strip() for p in m.group(1).split(",") if p.strip()]
                for p in params:
                    cleaned = p.split(":")[0].split("=")[0].strip()
                    _add(cleaned)

        for line in local_code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assign_match = re.match(r"([_a-zA-Z][_a-zA-Z0-9]*)\s*=", stripped)
            if assign_match:
                _add(assign_match.group(1))

            for_match = re.match(r"for\s+([_a-zA-Z][_a-zA-Z0-9]*)\s+in\s+", stripped)
            if for_match:
                _add(for_match.group(1))

        return symbols[:24]

    def _extract_imports(self, tree: tree_sitter.Tree, cursor_idx: int) -> List[str]:
        """Lấy các import (recursive, chỉ lấy những import trước con trỏ)."""
        imports: List[str] = []
        root_node = tree.root_node

        import_node_types_by_lang = {
            "python": {"import_statement", "import_from_statement"},
            "javascript": {"import_statement"},
            "typescript": {"import_statement"},
            "go": {"import_declaration"},
            "java": {"import_declaration"},
        }
        import_types = import_node_types_by_lang.get(
            self._normalize_language(self.language),
            {"import_statement", "import_from_statement"},
        )

        def traverse(node: tree_sitter.Node):
            if node.end_point and node.end_point[0] > cursor_idx:
                return
            if node.type in import_types and node.end_point and node.end_point[0] <= cursor_idx:
                try:
                    imports.append(node.text.decode("utf8"))
                except Exception:
                    pass
            for child in node.children:
                if child.start_point and child.start_point[0] <= cursor_idx:
                    traverse(child)

        traverse(root_node)

        # Dedup giữ thứ tự
        seen = set()
        deduped: List[str] = []
        for imp in imports:
            if imp not in seen:
                seen.add(imp)
                deduped.append(imp)
        return deduped

    def _extract_imports_fallback(self, source_code: str, cursor_idx: int) -> List[str]:
        """
        Fallback khi Tree-sitter không hoạt động: scan line-based, chỉ lấy các dòng
        import xuất hiện trước con trỏ.
        """
        lang = self._normalize_language(self.language)
        lines = source_code.split("\n")
        upper = min(len(lines), cursor_idx + 1)
        picked: List[str] = []

        if lang == "python":
            for ln in lines[:upper]:
                s = ln.strip()
                if s.startswith("import ") or (s.startswith("from ") and " import " in s):
                    picked.append(ln.rstrip())
        elif lang == "java":
            for ln in lines[:upper]:
                if ln.strip().startswith("import "):
                    picked.append(ln.rstrip())
        elif lang in ("javascript", "typescript"):
            for ln in lines[:upper]:
                if ln.strip().startswith("import "):
                    picked.append(ln.rstrip())

        # Dedup giữ thứ tự
        seen = set()
        out: List[str] = []
        for x in picked:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _extract_parent_class(self, tree: tree_sitter.Tree, cursor_idx: int, source_code: str) -> Optional[str]:
        """
        Tìm class_definition bao quanh cursor_idx và trả về header line.

        Hiện tại implement chắc chắn cho Python (đủ dùng cho Co-Retrieval pipeline).
        """
        if self._normalize_language(self.language) != "python":
            return None

        lines = source_code.split("\n")

        def contains_line(node: tree_sitter.Node) -> bool:
            return node.start_point[0] <= cursor_idx <= node.end_point[0]

        found: Optional[tree_sitter.Node] = None

        def traverse(node: tree_sitter.Node):
            nonlocal found
            # Deepest enclosing class: đi sâu và overwrite `found` nếu gặp class nhỏ hơn.
            if node.type == "class_definition" and contains_line(node):
                found = node
            for child in node.children:
                if child.start_point[0] <= cursor_idx <= child.end_point[0]:
                    traverse(child)

        try:
            traverse(tree.root_node)
            if not found:
                return None
            header_idx = found.start_point[0]
            if 0 <= header_idx < len(lines):
                return lines[header_idx].rstrip()
            return None
        except Exception:
            return None

    def _extract_parent_class_fallback(self, source_code: str, cursor_idx: int) -> Optional[str]:
        """
        Fallback: tìm class header gần nhất phía trên con trỏ (Python only).
        Đây là heuristic; dùng khi Tree-sitter không khả dụng.
        """
        if self._normalize_language(self.language) != "python":
            return None
        lines = source_code.split("\n")
        i = min(cursor_idx, len(lines) - 1)
        while i >= 0:
            s = lines[i].lstrip()
            if s.startswith("class ") and s.rstrip().endswith(":"):
                return lines[i].rstrip()
            i -= 1
        return None

    def _extract_parent_function(self, tree: tree_sitter.Tree, cursor_idx: int, source_code: str) -> Optional[str]:
        if self._normalize_language(self.language) != "python":
            return None
        lines = source_code.replace("\r\n", "\n").replace("\r", "\n").splitlines()

        def contains_line(node: tree_sitter.Node) -> bool:
            return node.start_point[0] <= cursor_idx <= node.end_point[0]

        found: Optional[tree_sitter.Node] = None

        def traverse(node: tree_sitter.Node):
            nonlocal found
            if node.type == "function_definition" and contains_line(node):
                found = node
            for child in node.children:
                if child.start_point[0] <= cursor_idx <= child.end_point[0]:
                    traverse(child)

        try:
            traverse(tree.root_node)
            if not found:
                return None
            header_idx = found.start_point[0]
            if 0 <= header_idx < len(lines):
                return lines[header_idx].rstrip()
            return None
        except Exception:
            return None

    def _extract_parent_function_fallback(self, source_code: str, cursor_idx: int) -> Optional[str]:
        if self._normalize_language(self.language) != "python":
            return None
        lines = source_code.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        i = min(cursor_idx, len(lines) - 1)
        while i >= 0:
            s = lines[i].lstrip()
            if s.startswith("def ") and s.rstrip().endswith(":"):
                return lines[i].rstrip()
            i -= 1
        return None

    def _extract_enclosing_block_range(self, tree: tree_sitter.Tree, cursor_idx: int) -> Optional[Tuple[int, int]]:
        """
        Trả về (start_line, end_line) cho block bao quanh con trỏ.
        Ưu tiên function_definition, fallback class_definition.
        """
        if self._normalize_language(self.language) != "python":
            return None

        def contains_line(node: tree_sitter.Node) -> bool:
            return node.start_point[0] <= cursor_idx <= node.end_point[0]

        best: Optional[tree_sitter.Node] = None

        def consider(node: tree_sitter.Node):
            nonlocal best
            if not contains_line(node):
                return
            if best is None:
                best = node
                return
            # choose smaller (deeper) span
            if (node.end_point[0] - node.start_point[0]) <= (best.end_point[0] - best.start_point[0]):
                best = node

        def traverse(node: tree_sitter.Node):
            if node.start_point[0] > cursor_idx:
                return
            if node.type in ("function_definition", "class_definition"):
                consider(node)
            for child in node.children:
                if child.start_point[0] <= cursor_idx <= child.end_point[0]:
                    traverse(child)

        try:
            traverse(tree.root_node)
            if best is None:
                return None
            return best.start_point[0], best.end_point[0]
        except Exception:
            return None
