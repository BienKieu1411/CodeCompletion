"""
Graph-Conditioned Retriever Policy (Production Implementation)
Build import dependency graph → BFS traversal → retrieve transitive dependencies.

Key improvements over v1:
- Real import graph (adjacency list) built from AST
- BFS for transitive dependencies (A→B→C)
- Extracts only relevant function/class from target file (not entire file)
- Graph distance-based ranking
"""

from __future__ import annotations

import re
import logging
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Import Parsing ────────────────────────────────────────────────────────────

def _extract_imported_modules(source_code: str, language: str = "python") -> List[str]:
    """
    Extract module names from import statements using AST.
    Falls back to regex if AST fails.

    Returns list of module names, e.g. ["os", "json", "utils.logger"]
    """
    modules: List[str] = []

    if language == "python":
        # Try AST first
        try:
            import tree_sitter_languages
            parser = tree_sitter_languages.get_parser("python")
            tree = parser.parse(bytes(source_code, "utf8"))

            for node in tree.root_node.children:
                if node.type == "import_statement":
                    # import os, sys
                    for child in node.children:
                        if child.type == "dotted_name":
                            modules.append(child.text.decode("utf8"))
                elif node.type == "import_from_statement":
                    # from utils.logger import Logger
                    for child in node.children:
                        if child.type == "dotted_name":
                            modules.append(child.text.decode("utf8"))
                            break  # Only take the module, not the imported names
        except Exception:
            pass

        # Fallback: regex
        if not modules:
            for line in source_code.split("\n"):
                line = line.strip()
                m = re.match(r'^from\s+([\w.]+)\s+import', line)
                if m:
                    modules.append(m.group(1))
                    continue
                m = re.match(r'^import\s+([\w.]+)', line)
                if m:
                    modules.append(m.group(1))

    elif language == "java":
        for line in source_code.split("\n"):
            m = re.match(r'^\s*import\s+([\w.]+)', line.strip())
            if m:
                modules.append(m.group(1))

    # Deduplicate while preserving order
    seen: Set[str] = set()
    deduped: List[str] = []
    for mod in modules:
        if mod not in seen:
            seen.add(mod)
            deduped.append(mod)
    return deduped


def _module_to_filename_candidates(module: str) -> List[str]:
    """
    Convert module name to possible filename matches.
    e.g. "utils.logger" → ["utils/logger.py", "utils.py", "logger.py", "utils/logger"]
    """
    parts = module.split(".")
    candidates = []
    # Full path
    candidates.append("/".join(parts) + ".py")
    candidates.append("/".join(parts))
    # Last part only
    candidates.append(parts[-1] + ".py")
    candidates.append(parts[-1])
    # Intermediate parts
    if len(parts) > 1:
        candidates.append(parts[0] + ".py")
        candidates.append(parts[0])
    return candidates


def _match_filename(module: str, available_files: List[str]) -> Optional[str]:
    """Find the best matching filename for an imported module."""
    candidates = _module_to_filename_candidates(module)
    for candidate in candidates:
        for filepath in available_files:
            # Normalize path separators
            norm_path = filepath.replace("\\", "/")
            if norm_path.endswith(candidate) or norm_path == candidate:
                return filepath
            # Also try matching just the basename
            basename = norm_path.split("/")[-1]
            cand_basename = candidate.split("/")[-1]
            if basename == cand_basename:
                return filepath
    return None


# ── Import Graph ──────────────────────────────────────────────────────────────

class ImportGraph:
    """
    Directed graph of import dependencies between files in a repository.

    Nodes: file paths
    Edges: file_a → file_b means file_a imports something from file_b
    """

    def __init__(self):
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # adj list
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)

    def add_edge(self, from_file: str, to_file: str):
        self.edges[from_file].add(to_file)
        self.reverse_edges[to_file].add(from_file)

    @classmethod
    def build_from_repo(cls, file_dict: Dict[str, str], language: str = "python") -> "ImportGraph":
        """Build import graph from all files in a repository."""
        graph = cls()
        available_files = list(file_dict.keys())

        for filepath, content in file_dict.items():
            imported_modules = _extract_imported_modules(content, language)
            for mod in imported_modules:
                target = _match_filename(mod, available_files)
                if target and target != filepath:
                    graph.add_edge(filepath, target)

        return graph

    def bfs_reachable(self, start: str, max_depth: int = 3) -> List[Tuple[str, int]]:
        """
        BFS from start node, return list of (node, depth) pairs.
        Sorted by depth (closest first).
        """
        visited: Set[str] = {start}
        queue: deque = deque([(start, 0)])
        result: List[Tuple[str, int]] = []

        while queue:
            node, depth = queue.popleft()
            if depth > 0:
                result.append((node, depth))
            if depth >= max_depth:
                continue
            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return sorted(result, key=lambda x: x[1])

    def get_importers(self, target: str) -> Set[str]:
        """Files that import target."""
        return self.reverse_edges.get(target, set())


# ── Relevant Content Extraction ───────────────────────────────────────────────

def _extract_relevant_symbols(
    source_content: str,
    imported_names: List[str],
    max_chars: int = 800,
) -> str:
    """
    Extract only the relevant function/class definitions from source.
    If imported_names is empty, return the first max_chars of the file.
    """
    if not imported_names:
        return source_content[:max_chars]

    try:
        import tree_sitter_languages
        parser = tree_sitter_languages.get_parser("python")
        tree = parser.parse(bytes(source_content, "utf8"))
        lines = source_content.split("\n")
        extracted_parts: List[str] = []

        for node in tree.root_node.children:
            if node.type in ("function_definition", "class_definition"):
                # Get the name of this definition
                name_node = None
                for child in node.children:
                    if child.type == "identifier":
                        name_node = child
                        break
                if name_node:
                    name = name_node.text.decode("utf8")
                    if name in imported_names or any(name in n for n in imported_names):
                        start = node.start_point[0]
                        end = node.end_point[0]
                        part = "\n".join(lines[start:end + 1])
                        extracted_parts.append(part)

        if extracted_parts:
            result = "\n\n".join(extracted_parts)
            return result[:max_chars]

    except Exception:
        pass

    return source_content[:max_chars]


# ── Graph Retriever ───────────────────────────────────────────────────────────

class GraphRetriever:
    """
    Graph-conditioned retriever that uses import dependency analysis
    to find structurally relevant files.

    Action space: select files via graph traversal paths.
    """

    def __init__(self, top_k_paths: int = 3, max_depth: int = 2, max_chars_per_file: int = 800):
        self.top_k_paths = top_k_paths
        self.max_depth = max_depth
        self.max_chars = max_chars_per_file

    def retrieve_paths(
        self,
        local_graph: Any,
        crossfile_dict: Dict[str, str],
        current_file: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve relevant code from other files based on structural analysis.

        Strategy:
        1. Build import graph from crossfile_dict + current file imports
        2. Find files reachable via import edges (direct + transitive)
        3. Extract only relevant symbols from those files
        4. Rank by graph distance (closer = higher priority)

        Args:
            local_graph: LocalGraph object from ASTQueryExtractor
            crossfile_dict: Dict[filename, content] of other files
            current_file: Path of the current file being completed

        Returns:
            List of formatted context strings
        """
        if not crossfile_dict:
            return []

        retrieved: List[str] = []
        available_files = list(crossfile_dict.keys())

        # Strategy 1: Direct import matching (fast path)
        imported_modules: List[str] = []
        if hasattr(local_graph, "imports") and local_graph.imports:
            for imp_line in local_graph.imports:
                modules = _extract_imported_modules(imp_line)
                imported_modules.extend(modules)

        # Match imports to files
        matched_files: List[Tuple[str, int]] = []  # (filename, priority)
        for mod in imported_modules:
            target = _match_filename(mod, available_files)
            if target:
                matched_files.append((target, 1))  # priority 1 = direct import

        # Strategy 2: Build full import graph for transitive deps
        if current_file:
            # Include current file in the graph analysis
            extended_dict = dict(crossfile_dict)
            if hasattr(local_graph, "local_code"):
                # Reconstruct imports + local code as proxy for current file
                current_content = "\n".join(local_graph.imports) if local_graph.imports else ""
                current_content += "\n" + (local_graph.local_code or "")
                extended_dict[current_file] = current_content

            graph = ImportGraph.build_from_repo(extended_dict)

            # BFS from current file
            if current_file in graph.edges:
                reachable = graph.bfs_reachable(current_file, max_depth=self.max_depth)
                for fname, depth in reachable:
                    if fname in crossfile_dict and fname not in [f for f, _ in matched_files]:
                        matched_files.append((fname, depth + 1))

        # Strategy 3: Parent class matching
        if hasattr(local_graph, "parent_class") and local_graph.parent_class:
            class_name = local_graph.parent_class.strip().rstrip(":")
            # Extract base class name: "class Foo(Bar):" → "Bar"
            m = re.search(r'\(([^)]+)\)', class_name)
            if m:
                base_classes = [c.strip() for c in m.group(1).split(",")]
                for base_class in base_classes:
                    base_class = base_class.split(".")[-1]  # Handle module.ClassName
                    for filename, content in crossfile_dict.items():
                        if f"class {base_class}" in content:
                            if filename not in [f for f, _ in matched_files]:
                                matched_files.append((filename, 2))

        # Sort by priority (lower = more relevant)
        matched_files.sort(key=lambda x: x[1])

        # Extract relevant content from matched files
        seen = set()
        for filename, priority in matched_files:
            if filename in seen:
                continue
            seen.add(filename)
            content = crossfile_dict.get(filename, "")
            if not content:
                continue

            # Try to extract only the relevant symbols
            imported_names = []
            for mod in imported_modules:
                parts = mod.split(".")
                imported_names.append(parts[-1])
            relevant = _extract_relevant_symbols(content, imported_names, self.max_chars)

            depth_label = f"depth={priority}" if priority > 1 else "direct"
            retrieved.append(
                f"### File: {filename} (Graph: {depth_label}) ###\n{relevant}"
            )

            if len(retrieved) >= self.top_k_paths:
                break

        return retrieved
