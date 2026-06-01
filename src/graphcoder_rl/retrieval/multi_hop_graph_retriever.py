"""
Graph-Conditioned Multi-hop Retriever

Implements a lightweight heterogeneous chunk-entity graph and
multi-hop traversal retrieval policy with quantized semantic states.
"""

from __future__ import annotations

import re
import ast
import logging
import os
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

from graphcoder_rl.retrieval.repository_graph_encoder import LightweightGraphEncoder, NodeFeature
from graphcoder_rl.rl.semantic_state_quantizer import SemanticStateQuantizer
from graphcoder_rl.cache.repository_graph_cache import RepositoryGraphCache
from graphcoder_rl.cache.ppl_entropy_cache import PPLEntropyCache

logger = logging.getLogger(__name__)


# ── Utilities ────────────────────────────────────────────────────────────────

_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
_CALL_RE = re.compile(r"([_a-zA-Z][_a-zA-Z0-9]*)\s*\(")
_CLASS_BASE_RE = re.compile(r"class\s+[_a-zA-Z][_a-zA-Z0-9]*\s*\(([^)]*)\)\s*:")
_CLASS_DEF_RE = re.compile(r"class\s+([_a-zA-Z][_a-zA-Z0-9]*)\s*(?:\([^)]*\))?\s*:")
_DEF_RE = re.compile(r"^\s*(?:async\s+def|def)\s+([_a-zA-Z][_a-zA-Z0-9]*)\s*\(", re.MULTILINE)
_CONTROL_RE = re.compile(r"^\s*(if|elif|else|for|while|try|except|with|match)\b", re.MULTILINE)


def _extract_identifiers(text: str) -> Set[str]:
    return {t for t in _IDENTIFIER_RE.findall(text or "") if len(t) > 2}


def _extract_call_names(text: str) -> Set[str]:
    return {m.group(1) for m in _CALL_RE.finditer(text or "") if len(m.group(1)) > 2}


def _extract_class_bases(text: str) -> Set[str]:
    bases: Set[str] = set()
    for m in _CLASS_BASE_RE.finditer(text or ""):
        raw = m.group(1)
        for item in raw.split(","):
            sym = item.strip().split(".")[-1]
            if sym and len(sym) > 2 and sym.isidentifier():
                bases.add(sym)
    return bases


def _extract_defined_classes(text: str) -> Set[str]:
    return {m.group(1) for m in _CLASS_DEF_RE.finditer(text or "") if len(m.group(1)) > 2}


def _extract_defined_methods(text: str) -> Set[str]:
    return {m.group(1) for m in _DEF_RE.finditer(text or "") if len(m.group(1)) > 2}


def _has_control_flow_header(text: str) -> bool:
    return _CONTROL_RE.search(text or "") is not None


def _extract_python_class_profiles(source_code: str) -> Dict[str, Dict[str, Any]]:
    """
    AST-aware class profiles:
      class_name -> {"bases": set[str], "methods": set[str], "lineno": int, "end_lineno": int}
    Line numbers are 1-based in AST.
    """
    profiles: Dict[str, Dict[str, Any]] = {}
    try:
        tree = ast.parse(source_code or "")
    except Exception:
        return profiles

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        class_name = node.name
        bases: Set[str] = set()
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.add(b.id)
            elif isinstance(b, ast.Attribute):
                bases.add(b.attr)
        methods: Set[str] = set()
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.add(child.name)
        lineno = int(getattr(node, "lineno", 1))
        end_lineno = int(getattr(node, "end_lineno", lineno))
        profiles[class_name] = {
            "bases": bases,
            "methods": methods,
            "lineno": lineno,
            "end_lineno": end_lineno,
        }
    return profiles


def _extract_python_control_flow_lines(source_code: str) -> Set[int]:
    """
    Return 1-based line numbers where control-flow statements start.
    """
    lines: Set[int] = set()
    try:
        tree = ast.parse(source_code or "")
    except Exception:
        return lines

    control_types = (
        ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith
    )
    if hasattr(ast, "Match"):
        control_types = control_types + (ast.Match,)  # type: ignore[operator]

    for node in ast.walk(tree):
        if isinstance(node, control_types):
            ln = int(getattr(node, "lineno", 0) or 0)
            if ln > 0:
                lines.add(ln)
    return lines


def _semantic_boundary_score(prev_line: str, line: str, next_line: str) -> float:
    """
    Heuristic proxy for semantic/PPL boundary.
    High score near blank-lines, scope boundaries, and topic shifts.
    """
    s_prev = prev_line.strip()
    s_cur = line.strip()
    s_next = next_line.strip()

    score = 0.0
    if not s_cur:
        score += 0.8
    if s_cur.startswith(("return ", "raise ", "yield ", "break", "continue")):
        score += 0.55
    if s_cur.startswith(("if ", "for ", "while ", "try", "except", "with ", "elif ", "else")):
        score += 0.45
    if s_cur.startswith(("def ", "class ", "@")):
        score += 0.65
    if s_prev.endswith((")", "]", "}", "\"", "'")) and s_next.startswith(("if ", "for ", "while ", "return ")):
        score += 0.35

    ids_prev = _extract_identifiers(s_prev)
    ids_next = _extract_identifiers(s_next)
    if ids_prev or ids_next:
        overlap = len(ids_prev & ids_next) / max(1, len(ids_prev | ids_next))
        score += (1.0 - overlap) * 0.35
    return score


def _split_long_chunk_semantic(
    filename: str,
    start_line: int,
    lines: List[str],
    max_lines: int = 80,
    min_lines: int = 20,
    boundary_scores: Optional[List[float]] = None,
) -> List[Tuple[str, str, int, int, Set[str], Set[str], Set[str]]]:
    """
    PPL-guided proxy split:
    - keep short entities intact
    - split long entities at high semantic-boundary points
    """
    n = len(lines)
    if n <= max_lines:
        text = "\n".join(lines)
        defs = _extract_identifiers(lines[0] if lines else "")
        used = _extract_identifiers(text)
        calls = _extract_call_names(text)
        chunk_id = f"{filename}::L{start_line}-{start_line + max(0, n - 1)}"
        return [(chunk_id, text, start_line, start_line + max(0, n - 1), defs, used, calls)]

    boundaries = []
    for i in range(min_lines, n - min_lines):
        if boundary_scores is not None and i < len(boundary_scores):
            score = float(boundary_scores[i])
        else:
            prev_line = lines[i - 1]
            cur_line = lines[i]
            next_line = lines[i + 1]
            score = _semantic_boundary_score(prev_line, cur_line, next_line)
        boundaries.append((score, i))
    boundaries.sort(reverse=True, key=lambda x: x[0])

    # Greedy picks of top boundaries while respecting min segment length.
    chosen = []
    used_idx: Set[int] = set()
    for _, idx in boundaries:
        if idx in used_idx:
            continue
        # respect spacing
        if any(abs(idx - c) < min_lines for c in chosen):
            continue
        chosen.append(idx)
        used_idx.add(idx)
        # stop when segments likely under max_lines
        est_segments = len(chosen) + 1
        if n / est_segments <= max_lines:
            break
    chosen = sorted(chosen)

    split_points = [0] + chosen + [n]
    out = []
    for s_i, e_i in zip(split_points[:-1], split_points[1:]):
        block = lines[s_i:e_i]
        if len(block) < min_lines and out:
            # merge tiny tail into previous block
            prev = out.pop()
            merged_lines = prev[1].split("\n") + block
            text = "\n".join(merged_lines)
            s_line = prev[2]
            e_line = start_line + e_i - 1
            defs = prev[4]
            used = _extract_identifiers(text)
            calls = _extract_call_names(text)
            chunk_id = f"{filename}::L{s_line}-{e_line}"
            out.append((chunk_id, text, s_line, e_line, defs, used, calls))
            continue

        text = "\n".join(block)
        s_line = start_line + s_i
        e_line = start_line + e_i - 1
        defs = _extract_identifiers(block[0] if block else "")
        used = _extract_identifiers(text)
        calls = _extract_call_names(text)
        chunk_id = f"{filename}::L{s_line}-{e_line}"
        out.append((chunk_id, text, s_line, e_line, defs, used, calls))

    return out


def _extract_imported_modules(source_code: str, language: str = "python") -> List[str]:
    modules: List[str] = []
    if language == "python":
        try:
            import tree_sitter_languages

            parser = tree_sitter_languages.get_parser("python")
            tree = parser.parse(bytes(source_code, "utf8"))
            for node in tree.root_node.children:
                if node.type == "import_statement":
                    for child in node.children:
                        if child.type == "dotted_name":
                            modules.append(child.text.decode("utf8"))
                elif node.type == "import_from_statement":
                    for child in node.children:
                        if child.type == "dotted_name":
                            modules.append(child.text.decode("utf8"))
                            break
        except Exception:
            pass

        if not modules:
            for line in source_code.split("\n"):
                line = line.strip()
                m = re.match(r"^from\s+([\w.]+)\s+import", line)
                if m:
                    modules.append(m.group(1))
                    continue
                m = re.match(r"^import\s+([\w.]+)", line)
                if m:
                    modules.append(m.group(1))

    seen: Set[str] = set()
    out: List[str] = []
    for x in modules:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _module_to_filename_candidates(module: str) -> List[str]:
    parts = module.split(".")
    candidates = ["/".join(parts) + ".py", "/".join(parts), parts[-1] + ".py", parts[-1]]
    if len(parts) > 1:
        candidates.extend([parts[0] + ".py", parts[0]])
    return candidates


def _match_filename(module: str, available_files: List[str]) -> Optional[str]:
    candidates = _module_to_filename_candidates(module)
    for candidate in candidates:
        for filepath in available_files:
            norm_path = filepath.replace("\\", "/")
            if norm_path.endswith(candidate) or norm_path == candidate:
                return filepath
            if norm_path.split("/")[-1] == candidate.split("/")[-1]:
                return filepath
    return None


def _candidate_paths_from_chunk_ids(candidate_chunk_ids: Optional[List[str]]) -> Set[str]:
    """
    Convert retriever chunk ids (e.g., 'foo/bar.py::L10-20') into file paths.
    """
    out: Set[str] = set()
    if not candidate_chunk_ids:
        return out
    for cid in candidate_chunk_ids:
        if not isinstance(cid, str):
            continue
        file_path = cid.split("::", 1)[0].strip()
        if file_path:
            out.add(file_path)
    return out


def _extract_python_chunks(
    filename: str,
    content: str,
    ppl_entropy_cache: Optional[PPLEntropyCache] = None,
) -> List[Tuple[str, str, int, int, Set[str], Set[str], Set[str]]]:
    """
    Returns chunk tuples:
    (chunk_id, chunk_text, start_line, end_line, defined_symbols, used_symbols, call_names)
    """
    chunks: List[Tuple[str, str, int, int, Set[str], Set[str], Set[str]]] = []

    try:
        import tree_sitter_languages

        parser = tree_sitter_languages.get_parser("python")
        tree = parser.parse(bytes(content, "utf8"))
        lines = content.split("\n")

        def _node_text(node) -> str:
            return "\n".join(lines[node.start_point[0] : node.end_point[0] + 1])

        for node in tree.root_node.children:
            if node.type in ("function_definition", "class_definition", "decorated_definition"):
                s, e = node.start_point[0], node.end_point[0]
                entity_lines = lines[s:e + 1]
                boundary_scores = None
                if ppl_entropy_cache is not None and entity_lines:
                    key_text = f"{filename}:{s}:{e}\n" + "\n".join(entity_lines)
                    boundary_scores = ppl_entropy_cache.get_or_compute(
                        key_text=key_text,
                        lines=entity_lines,
                        scorer=_semantic_boundary_score,
                    )
                chunks.extend(
                    _split_long_chunk_semantic(
                        filename=filename,
                        start_line=s,
                        lines=entity_lines,
                        max_lines=80,
                        min_lines=20,
                        boundary_scores=boundary_scores,
                    )
                )

        if not chunks:
            content_lines = lines[:120]
            chunk = "\n".join(content_lines)
            defs = _extract_identifiers(chunk)
            calls = _extract_call_names(chunk)
            chunks.append((f"{filename}::all", chunk, 0, max(0, len(content_lines) - 1), defs, defs, calls))

    except Exception:
        lines = content.split("\n")
        step = 30
        for i in range(0, len(lines), step):
            block = "\n".join(lines[i : i + step])
            defs = _extract_identifiers(block)
            calls = _extract_call_names(block)
            chunks.append((
                f"{filename}::L{i}-{min(i + step - 1, len(lines) - 1)}",
                block,
                i,
                min(i + step - 1, len(lines) - 1),
                defs,
                defs,
                calls,
            ))

    return chunks


# ── Graph Data Structures ────────────────────────────────────────────────────

@dataclass
class GraphNode:
    node_id: str
    node_type: str
    file_path: str
    text: str
    semantic_state: str = "generic"
    semantic_state_id: int = 0
    defined_symbols: Set[str] = field(default_factory=set)
    used_symbols: Set[str] = field(default_factory=set)
    calls: Set[str] = field(default_factory=set)
    start_line: int = 0
    end_line: int = 0


class HeteroRepoGraph:
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, src: str, dst: str, edge_type: str) -> None:
        if src in self.nodes and dst in self.nodes:
            self.adj[src].append((dst, edge_type))

    def neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        return self.adj.get(node_id, [])

    def has_edge(self, src: str, dst: str, edge_type: str) -> bool:
        return any(nbr == dst and et == edge_type for nbr, et in self.adj.get(src, []))


# ── Retriever ────────────────────────────────────────────────────────────────

class MultiHopGraphRetriever:
    """
    Multi-hop traversal retriever on a heterogeneous repository graph.

    State abstraction uses graph-aware embeddings + quantized semantic state ids.
    """

    def __init__(
        self,
        top_k_paths: int = 3,
        max_depth: int = 2,
        max_chars_per_file: int = 800,
        max_branch: int = 6,
        encoder_hidden_dim: int = 16,
        encoder_layers: int = 2,
        quantizer_codes: int = 16,
        use_graph_cache: bool = True,
        graph_cache_dir: str = "cache/graph",
        use_ppl_entropy_cache: bool = True,
        ppl_cache_dir: str = "cache/ppl",
        enable_left_context_anchors: bool = True,
        enable_quantization: bool = True,
        enable_multi_hop: bool = True,
        enable_structural_edges: bool = True,
        enable_control_dependency: bool = True,
        enable_override_edges: bool = True,
    ):
        self.top_k_paths = top_k_paths
        self.max_depth = max_depth
        self.max_chars = max_chars_per_file
        self.max_branch = max_branch
        self.encoder = LightweightGraphEncoder(hidden_dim=encoder_hidden_dim, n_layers=encoder_layers)
        self.quantizer = SemanticStateQuantizer(dim=encoder_hidden_dim, n_codes=quantizer_codes)
        self._policy_feat_dim = 8
        self.graph_cache = RepositoryGraphCache(graph_cache_dir) if use_graph_cache else None
        self.ppl_entropy_cache = PPLEntropyCache(ppl_cache_dir) if use_ppl_entropy_cache else None
        self.enable_left_context_anchors = enable_left_context_anchors
        self.enable_quantization = enable_quantization
        self.enable_multi_hop = enable_multi_hop
        self.enable_structural_edges = enable_structural_edges
        self.enable_control_dependency = enable_control_dependency
        self.enable_override_edges = enable_override_edges

    def _build_file_artifact(self, file_path: str, content: str) -> Dict[str, Any]:
        imports = _extract_imported_modules(content)
        chunks = _extract_python_chunks(
            file_path,
            content,
            ppl_entropy_cache=self.ppl_entropy_cache,
        )
        class_profiles = _extract_python_class_profiles(content)
        control_flow_lines = _extract_python_control_flow_lines(content)
        return {
            "imports": imports,
            "chunks": chunks,
            "class_profiles": class_profiles,
            "control_flow_lines": sorted(control_flow_lines),
        }

    def _build_graph(
        self,
        crossfile_dict: Dict[str, str],
        coarse_candidate_files: Optional[Set[str]] = None,
    ) -> HeteroRepoGraph:
        graph = HeteroRepoGraph()
        available_files = list(crossfile_dict.keys())

        if coarse_candidate_files:
            file_pool = {f for f in available_files if f in coarse_candidate_files}
            if not file_pool:
                file_pool = set(available_files)
        else:
            file_pool = set(available_files)

        file_imports: Dict[str, List[str]] = {}
        symbol_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        file_to_chunks: Dict[str, List[str]] = defaultdict(list)
        chunk_methods: Dict[str, Set[str]] = {}
        chunk_class_bases: Dict[str, Set[str]] = {}
        chunk_defined_classes: Dict[str, Set[str]] = {}
        file_control_flow_lines: Dict[str, Set[int]] = {}
        class_name_registry: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        class_profile_registry: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        repo_files = {fp: crossfile_dict.get(fp, "") for fp in sorted(file_pool)}
        artifacts: Dict[str, Any] = {}
        changed_files: List[str] = []
        if self.graph_cache is not None:
            repo_id = self.graph_cache.repo_id_from_files(repo_files)
            artifacts, changed_files = self.graph_cache.get_or_update_artifacts(
                repo_id=repo_id,
                file_dict=repo_files,
                artifact_builder=self._build_file_artifact,
            )
        else:
            for fp, text in repo_files.items():
                artifacts[fp] = self._build_file_artifact(fp, text)

        for file_path in sorted(file_pool):
            file_id = f"file::{file_path}"
            graph.add_node(GraphNode(
                node_id=file_id,
                node_type="file",
                file_path=file_path,
                text="",
            ))
            artifact = artifacts.get(file_path, {})
            file_imports[file_path] = list(artifact.get("imports", []))
            class_profiles: Dict[str, Dict[str, Any]] = dict(artifact.get("class_profiles", {}) or {})
            control_lines = set(int(x) for x in (artifact.get("control_flow_lines", []) or []))
            file_control_flow_lines[file_path] = control_lines
            for cls_name, prof in class_profiles.items():
                class_profile_registry[cls_name].append({
                    "class_name": cls_name,
                    "bases": set(prof.get("bases", set()) or set()),
                    "methods": set(prof.get("methods", set()) or set()),
                    "file_path": file_path,
                })

            chunks = list(artifact.get("chunks", []))
            prev_chunk_id: Optional[str] = None
            for cid, ctext, s, e, defs, used, calls in chunks:
                chunk_node = GraphNode(
                    node_id=f"chunk::{cid}",
                    node_type="chunk",
                    file_path=file_path,
                    text=ctext,
                    defined_symbols=defs,
                    used_symbols=used,
                    calls=calls,
                    start_line=s,
                    end_line=e,
                )
                graph.add_node(chunk_node)
                graph.add_edge(file_id, chunk_node.node_id, "contains")
                graph.add_edge(chunk_node.node_id, file_id, "inside")
                file_to_chunks[file_path].append(chunk_node.node_id)

                if self.enable_structural_edges and prev_chunk_id is not None:
                    graph.add_edge(prev_chunk_id, chunk_node.node_id, "adjacent_chunk")
                    graph.add_edge(chunk_node.node_id, prev_chunk_id, "adjacent_chunk")
                prev_chunk_id = chunk_node.node_id

                for sym in defs:
                    symbol_to_chunks[sym].add(chunk_node.node_id)
                    symbol_id = f"symbol::{sym}"
                    if symbol_id not in graph.nodes:
                        graph.add_node(GraphNode(
                            node_id=symbol_id,
                            node_type="symbol",
                            file_path=file_path,
                            text=sym,
                        ))
                    if not graph.has_edge(chunk_node.node_id, symbol_id, "defines"):
                        graph.add_edge(chunk_node.node_id, symbol_id, "defines")

                for sym in used:
                    symbol_id = f"symbol::{sym}"
                    if symbol_id not in graph.nodes:
                        graph.add_node(GraphNode(
                            node_id=symbol_id,
                            node_type="symbol",
                            file_path=file_path,
                            text=sym,
                        ))
                    if not graph.has_edge(chunk_node.node_id, symbol_id, "mentions"):
                        graph.add_edge(chunk_node.node_id, symbol_id, "mentions")

                class_bases = set(_extract_class_bases(ctext))
                methods = set(_extract_defined_methods(ctext))
                defined_classes = set(_extract_defined_classes(ctext))

                # AST-aware enrichment: map chunk range to class profiles.
                # Chunk lines are 0-based, AST lines are 1-based.
                s1, e1 = s + 1, e + 1
                for cls_name, prof in class_profiles.items():
                    c_start = int(prof.get("lineno", 1))
                    c_end = int(prof.get("end_lineno", c_start))
                    if not (e1 < c_start or s1 > c_end):
                        defined_classes.add(cls_name)
                        class_bases.update(set(prof.get("bases", set()) or set()))
                        methods.update(set(prof.get("methods", set()) or set()))
                        class_name_registry[cls_name].append(
                            {
                                "chunk_id": chunk_node.node_id,
                                "methods": set(prof.get("methods", set()) or set()),
                                "file_path": file_path,
                            }
                        )

                chunk_class_bases[chunk_node.node_id] = class_bases
                chunk_methods[chunk_node.node_id] = methods
                chunk_defined_classes[chunk_node.node_id] = defined_classes
                if self.enable_structural_edges:
                    for base in class_bases:
                        base_id = f"symbol::{base}"
                        if base_id not in graph.nodes:
                            graph.add_node(GraphNode(
                                node_id=base_id,
                                node_type="symbol",
                                file_path=file_path,
                                text=base,
                            ))
                        if not graph.has_edge(chunk_node.node_id, base_id, "inherits"):
                            graph.add_edge(chunk_node.node_id, base_id, "inherits")
                        if not graph.has_edge(chunk_node.node_id, base_id, "uses_type"):
                            graph.add_edge(chunk_node.node_id, base_id, "uses_type")

                for cls_name in defined_classes:
                    cls_id = f"symbol::{cls_name}"
                    if cls_id not in graph.nodes:
                        graph.add_node(GraphNode(
                            node_id=cls_id,
                            node_type="symbol",
                            file_path=file_path,
                            text=cls_name,
                        ))
                    if not graph.has_edge(chunk_node.node_id, cls_id, "defines"):
                        graph.add_edge(chunk_node.node_id, cls_id, "defines")

        for from_file, mods in file_imports.items():
            from_id = f"file::{from_file}"
            if from_id not in graph.nodes:
                continue
            for mod in mods:
                to_file = _match_filename(mod, available_files)
                if self.enable_structural_edges and to_file and to_file != from_file and to_file in file_pool:
                    graph.add_edge(from_id, f"file::{to_file}", "imports")

        for node_id, node in list(graph.nodes.items()):
            if node.node_type != "chunk":
                continue
            if self.enable_structural_edges:
                for called in node.calls:
                    for tgt in symbol_to_chunks.get(called, set()):
                        if tgt != node_id:
                            graph.add_edge(node_id, tgt, "calls")

        # Intra-file lightweight data-dependency edges:
        # producer chunk (defines symbol) -> consumer chunk (uses symbol).
        if self.enable_structural_edges:
            for file_path, chunk_ids in file_to_chunks.items():
                for src_id in chunk_ids:
                    src_node = graph.nodes.get(src_id)
                    if src_node is None:
                        continue
                    if not src_node.defined_symbols:
                        continue
                    for dst_id in chunk_ids:
                        if dst_id == src_id:
                            continue
                        dst_node = graph.nodes.get(dst_id)
                        if dst_node is None:
                            continue
                        if src_node.defined_symbols & dst_node.used_symbols:
                            if not graph.has_edge(src_id, dst_id, "data_dependency"):
                                graph.add_edge(src_id, dst_id, "data_dependency")

        # Intra-file control-dependency edges:
        # control-flow chunk -> next adjacent chunk in source order.
        if self.enable_structural_edges and self.enable_control_dependency:
            for file_path, chunk_ids in file_to_chunks.items():
                control_lines = file_control_flow_lines.get(file_path, set())
                sorted_chunks = sorted(
                    chunk_ids,
                    key=lambda nid: (
                        graph.nodes[nid].start_line,
                        graph.nodes[nid].end_line,
                    ),
                )
                for i in range(len(sorted_chunks) - 1):
                    src_id = sorted_chunks[i]
                    dst_id = sorted_chunks[i + 1]
                    src_node = graph.nodes.get(src_id)
                    if src_node is None:
                        continue
                    src_has_control = _has_control_flow_header(src_node.text)
                    if control_lines:
                        s1 = int(src_node.start_line) + 1
                        e1 = int(src_node.end_line) + 1
                        src_has_control = src_has_control or any(s1 <= ln <= e1 for ln in control_lines)
                    if src_has_control:
                        if not graph.has_edge(src_id, dst_id, "control_dependency"):
                            graph.add_edge(src_id, dst_id, "control_dependency")

        # Cross-chunk overrides edges:
        # derived-class chunk that defines method m and inherits base B
        # -> base-class chunk that also defines method m.
        if self.enable_structural_edges and self.enable_override_edges:
            for chunk_id, bases in chunk_class_bases.items():
                cur_methods = chunk_methods.get(chunk_id, set())
                if not bases or not cur_methods:
                    continue
                for base_name in bases:
                    base_entries = class_name_registry.get(base_name, [])
                    for base_entry in base_entries:
                        base_chunk_id = str(base_entry.get("chunk_id", ""))
                        if not base_chunk_id or base_chunk_id == chunk_id:
                            continue
                        base_methods = set(base_entry.get("methods", set()) or set())
                        if cur_methods & base_methods:
                            if not graph.has_edge(chunk_id, base_chunk_id, "overrides"):
                                graph.add_edge(chunk_id, base_chunk_id, "overrides")

        # Class symbol-level overrides edges (works even if classes co-locate in one chunk).
        if self.enable_structural_edges and self.enable_override_edges:
            for derived_name, entries in class_profile_registry.items():
                for derived_entry in entries:
                    derived_methods = set(derived_entry.get("methods", set()) or set())
                    if not derived_methods:
                        continue
                    for base_name in set(derived_entry.get("bases", set()) or set()):
                        for base_entry in class_profile_registry.get(base_name, []):
                            base_methods = set(base_entry.get("methods", set()) or set())
                            if not (derived_methods & base_methods):
                                continue
                            derived_sym = f"symbol::{derived_name}"
                            base_sym = f"symbol::{base_name}"
                            if derived_sym in graph.nodes and base_sym in graph.nodes:
                                if not graph.has_edge(derived_sym, base_sym, "overrides"):
                                    graph.add_edge(derived_sym, base_sym, "overrides")

        # Graph-aware encoding + vector quantization
        node_features: Dict[str, NodeFeature] = {}
        for node_id, node in graph.nodes.items():
            edge_counts: Dict[str, int] = defaultdict(int)
            for _, e_type in graph.neighbors(node_id):
                edge_counts[e_type] += 1
            node_features[node_id] = NodeFeature(
                node_type=node.node_type,
                text=node.text,
                edge_counts=dict(edge_counts),
                defined_symbols=len(node.defined_symbols),
                used_symbols=len(node.used_symbols),
                call_count=len(node.calls),
            )

        embeddings = self.encoder.encode(node_features=node_features, adjacency=graph.adj)
        if self.enable_quantization:
            q_states = self.quantizer.quantize_batch(embeddings)
            for node_id, q in q_states.items():
                if node_id in graph.nodes:
                    graph.nodes[node_id].semantic_state = q.state_label
                    graph.nodes[node_id].semantic_state_id = q.state_id

        if changed_files:
            logger.debug(
                "Graph cache incremental update: rebuilt %d/%d files",
                len(changed_files),
                len(file_pool),
            )
        return graph

    def _inject_left_context_anchors(
        self,
        graph: HeteroRepoGraph,
        local_graph: Any,
        crossfile_dict: Dict[str, str],
        query_symbols: Set[str],
    ) -> None:
        """
        Add left-context anchor states as explicit graph nodes:
        CursorNode, CurrentFileNode, CurrentFunctionNode, CurrentClassNode,
        ImportBlockNode, ImmediateLeftContextNode, LocalVariableStateNode.
        """
        cursor_id = "anchor::cursor"
        graph.add_node(GraphNode(node_id=cursor_id, node_type="anchor", file_path="", text="cursor"))

        current_file = getattr(local_graph, "file_path", "") or ""
        if current_file:
            cf_id = f"anchor::current_file::{current_file}"
            graph.add_node(GraphNode(node_id=cf_id, node_type="anchor", file_path=current_file, text=current_file))
            graph.add_edge(cursor_id, cf_id, "inside")

        parent_fn = getattr(local_graph, "parent_function", "") or ""
        if parent_fn:
            fn_id = "anchor::current_function"
            graph.add_node(GraphNode(node_id=fn_id, node_type="anchor", file_path=current_file, text=parent_fn))
            graph.add_edge(cursor_id, fn_id, "inside")

        parent_cls = getattr(local_graph, "parent_class", "") or ""
        if parent_cls:
            cls_id = "anchor::current_class"
            graph.add_node(GraphNode(node_id=cls_id, node_type="anchor", file_path=current_file, text=parent_cls))
            if parent_fn:
                graph.add_edge("anchor::current_function", cls_id, "inside")
            else:
                graph.add_edge(cursor_id, cls_id, "inside")

        imports = list(getattr(local_graph, "imports", []) or [])
        if imports:
            imp_id = "anchor::import_block"
            graph.add_node(GraphNode(node_id=imp_id, node_type="anchor", file_path=current_file, text="\n".join(imports)))
            graph.add_edge(cursor_id, imp_id, "contains")
            available_files = list(crossfile_dict.keys())
            for imp_line in imports:
                for mod in _extract_imported_modules(imp_line):
                    tgt = _match_filename(mod, available_files)
                    if tgt:
                        file_node_id = f"file::{tgt}"
                        if file_node_id in graph.nodes:
                            graph.add_edge(imp_id, file_node_id, "imports")

        local_vars = list(getattr(local_graph, "local_variables", []) or [])
        if local_vars:
            lv_id = "anchor::local_var_state"
            graph.add_node(GraphNode(node_id=lv_id, node_type="anchor", file_path=current_file, text=" ".join(local_vars)))
            graph.add_edge(cursor_id, lv_id, "contains")
            lv_symbols = set(local_vars)
            for node_id, node in list(graph.nodes.items()):
                if node.node_type != "chunk":
                    continue
                if lv_symbols & (node.defined_symbols | node.used_symbols):
                    graph.add_edge(lv_id, node_id, "mentions")
                if query_symbols & node.defined_symbols:
                    graph.add_edge(cursor_id, node_id, "mentions")

        immediate_left = getattr(local_graph, "local_code", "") or ""
        if immediate_left.strip():
            ilc_id = "anchor::immediate_left_context"
            graph.add_node(GraphNode(
                node_id=ilc_id,
                node_type="anchor",
                file_path=current_file,
                text=immediate_left,
            ))
            graph.add_edge(cursor_id, ilc_id, "contains")
            ilc_symbols = _extract_identifiers(immediate_left)
            for node_id, node in list(graph.nodes.items()):
                if node.node_type != "chunk":
                    continue
                if ilc_symbols & (node.defined_symbols | node.used_symbols):
                    graph.add_edge(ilc_id, node_id, "mentions")

        cursor_line = int(getattr(local_graph, "cursor_line", 0) or 0)
        cursor_line_id = f"anchor::cursor_line::{cursor_line}"
        graph.add_node(GraphNode(
            node_id=cursor_line_id,
            node_type="anchor",
            file_path=current_file,
            text=str(cursor_line),
        ))
        graph.add_edge(cursor_id, cursor_line_id, "inside")

    def _select_anchor_nodes(
        self,
        graph: HeteroRepoGraph,
        local_graph: Any,
        crossfile_dict: Dict[str, str],
        query_symbols: Set[str],
    ) -> List[str]:
        anchors: List[str] = []
        if "anchor::cursor" in graph.nodes:
            anchors.append("anchor::cursor")
        if "anchor::immediate_left_context" in graph.nodes:
            anchors.append("anchor::immediate_left_context")
        available_files = list(crossfile_dict.keys())

        imports = list(getattr(local_graph, "imports", []) or [])
        for imp_line in imports:
            for mod in _extract_imported_modules(imp_line):
                tgt = _match_filename(mod, available_files)
                if tgt:
                    anchors.append(f"file::{tgt}")

        for node_id, node in graph.nodes.items():
            if node.node_type != "chunk":
                continue
            if query_symbols & node.defined_symbols:
                anchors.append(node_id)

        seen = set()
        out: List[str] = []
        for a in anchors:
            if a in graph.nodes and a not in seen:
                seen.add(a)
                out.append(a)

        if not out:
            candidates = [nid for nid, n in graph.nodes.items() if n.node_type == "chunk"]
            if candidates:
                out.append(candidates[0])

        return out[: max(1, self.top_k_paths)]

    def _edge_bonus(self, edge_type: str) -> float:
        if edge_type == "calls":
            return 1.0
        if edge_type == "imports":
            return 0.8
        if edge_type == "data_dependency":
            return 0.75
        if edge_type == "control_dependency":
            return 0.7
        if edge_type == "overrides":
            return 0.72
        if edge_type == "inherits":
            return 0.72
        if edge_type == "uses_type":
            return 0.68
        if edge_type == "defines":
            return 0.55
        if edge_type == "mentions":
            return 0.5
        if edge_type == "contains":
            return 0.2
        if edge_type == "adjacent_chunk":
            return 0.15
        return 0.05

    def _score_neighbor(
        self,
        node: GraphNode,
        edge_type: str,
        query_symbols: Set[str],
        visited_states: Set[int],
    ) -> Tuple[float, float, float, float]:
        overlap = len(query_symbols & (node.defined_symbols | node.used_symbols))
        overlap_score = min(1.0, overlap / 4.0)
        edge_score = self._edge_bonus(edge_type)

        state_bonus = 0.15 if node.semantic_state_id not in visited_states else -0.05

        token_cost = max(1.0, len(node.text.split())) / 200.0
        irrelevant_penalty = 0.25 if overlap == 0 and edge_score < 0.8 else 0.0

        final_score = overlap_score + edge_score + state_bonus - 0.10 * token_cost - irrelevant_penalty
        return final_score, token_cost, irrelevant_penalty, overlap_score

    def retrieve_paths(
        self,
        local_graph: Any,
        crossfile_dict: Dict[str, str],
        current_file: Optional[str] = None,
        left_context: str = "",
        return_metadata: bool = False,
        coarse_candidate_chunks: Optional[List[str]] = None,
        policy_model: Optional[Any] = None,
        policy_device: str = "cpu",
    ):
        if not crossfile_dict:
            empty_meta = {
                "path_relevance": 0.0,
                "token_cost": 0.0,
                "redundancy_penalty": 0.0,
                "irrelevant_node_penalty": 0.0,
                "semantic_state_ids": [],
                "retrieval_path": [],
                "policy_action_features": [],
                "policy_selected_indices": [],
                "policy_feat_dim": self._policy_feat_dim,
            }
            return ([], empty_meta) if return_metadata else []

        coarse_files = _candidate_paths_from_chunk_ids(coarse_candidate_chunks)
        graph = self._build_graph(crossfile_dict, coarse_candidate_files=coarse_files)

        query_text = "\n".join([
            left_context or "",
            getattr(local_graph, "parent_class", "") or "",
            getattr(local_graph, "parent_function", "") or "",
            "\n".join(getattr(local_graph, "imports", []) or []),
            getattr(local_graph, "local_code", "") or "",
        ])
        query_symbols = _extract_identifiers(query_text)
        if self.enable_left_context_anchors:
            self._inject_left_context_anchors(graph, local_graph, crossfile_dict, query_symbols)
            anchors = self._select_anchor_nodes(graph, local_graph, crossfile_dict, query_symbols)
        else:
            anchors = []
            for node_id, node in graph.nodes.items():
                if node.node_type != "chunk":
                    continue
                if query_symbols & (node.defined_symbols | node.used_symbols):
                    anchors.append(node_id)
            if not anchors:
                anchors = [nid for nid, n in graph.nodes.items() if n.node_type == "chunk"][:1]

        selected_nodes: List[str] = []
        selected_paths: List[List[str]] = []
        selected_edges: List[List[str]] = []
        visited_states: Set[int] = set()
        scored_candidates: List[Tuple[float, str, List[str], List[str], float, float, float]] = []
        effective_max_depth = self.max_depth if self.enable_multi_hop else 0

        for anchor in anchors:
            queue = deque([(anchor, 0, [anchor], [])])
            visited = {anchor}

            while queue:
                node_id, depth, path_nodes, path_edges = queue.popleft()
                node = graph.nodes[node_id]

                if node.node_type == "chunk" and node.text.strip():
                    score, token_cost, ir_penalty, overlap_score = self._score_neighbor(
                        node, path_edges[-1] if path_edges else "contains", query_symbols, visited_states
                    )
                    scored_candidates.append((
                        score,
                        node_id,
                        path_nodes[:],
                        path_edges[:],
                        token_cost,
                        ir_penalty,
                        overlap_score,
                    ))

                if depth >= effective_max_depth:
                    continue

                nbrs = graph.neighbors(node_id)
                ranked: List[Tuple[float, str, str]] = []
                for nbr, e_type in nbrs:
                    if nbr in visited:
                        continue
                    nnode = graph.nodes[nbr]
                    score, _, _, _ = self._score_neighbor(nnode, e_type, query_symbols, visited_states)
                    ranked.append((score, nbr, e_type))

                ranked.sort(key=lambda x: x[0], reverse=True)
                for score, nbr, e_type in ranked[: self.max_branch]:
                    if score < -0.2:
                        continue
                    visited.add(nbr)
                    queue.append((nbr, depth + 1, path_nodes + [nbr], path_edges + [e_type]))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Build graph action candidate features for policy learning.
        action_features: List[List[float]] = []
        action_nodes: List[Tuple[float, str, List[str], List[str], float, float, float]] = []
        seen_action_nodes: Set[str] = set()
        for cand in scored_candidates:
            score, node_id, path_nodes, path_edges, token_cost, ir_penalty, overlap_score = cand
            if node_id in seen_action_nodes:
                continue
            seen_action_nodes.add(node_id)
            node = graph.nodes[node_id]
            if node.node_type != "chunk" or not node.text.strip():
                continue
            edge_bonus = self._edge_bonus(path_edges[-1] if path_edges else "contains")
            path_len_norm = min(1.0, len(path_edges) / max(1.0, float(effective_max_depth + 1)))
            state_norm = min(1.0, node.semantic_state_id / max(1.0, float(self.quantizer.n_codes - 1)))
            feat = [
                float(overlap_score),
                float(edge_bonus),
                float(min(1.0, token_cost)),
                float(min(1.0, ir_penalty)),
                float(path_len_norm),
                float(state_norm),
                float(max(-1.0, min(1.0, score))),
                0.0,  # not STOP
            ]
            action_features.append(feat)
            action_nodes.append(cand)

        # Explicit move actions by edge-type (for policy training trace fidelity).
        move_edge_types = sorted({e for _, _, _, edges, _, _, _ in action_nodes for e in edges})
        edge_action_idx_map: Dict[str, int] = {}
        for edge_type in move_edge_types:
            edge_bonus = self._edge_bonus(edge_type)
            move_feat = [
                0.0,                      # overlap_score
                float(edge_bonus),        # edge bonus
                0.0,                      # token cost
                0.0,                      # irrelevant penalty
                0.5,                      # path length proxy
                0.0,                      # state norm
                float(max(-1.0, min(1.0, edge_bonus))),
                0.0,                      # not STOP
            ]
            edge_action_idx_map[edge_type] = len(action_features)
            action_features.append(move_feat)

        # STOP action: explicit policy action requested by novelty.
        stop_feature = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        stop_idx = len(action_features)
        action_features.append(stop_feature)

        retrieved: List[str] = []
        total_token_cost = 0.0
        irrelevant_count = 0.0
        overlap_scores: List[float] = []
        semantic_state_ids: List[int] = []
        selected_action_indices: List[int] = []
        policy_action_trace: List[Dict[str, Any]] = []
        stop_selected = False

        # Learned-policy mode: let graph policy rank/stop actions.
        if policy_model is not None and action_nodes and torch is not None:
            with torch.no_grad():
                feat_t = torch.tensor(action_features, dtype=torch.float32, device=policy_device)
                logits = policy_model(feat_t)
                decision_pool = list(range(len(action_nodes))) + [stop_idx]
                decision_logits = logits[decision_pool]
                decision_logprobs = torch.log_softmax(decision_logits, dim=-1)
                ranked_pos = torch.argsort(decision_logprobs, descending=True).tolist()
                ranked_idx = [decision_pool[p] for p in ranked_pos]
                decision_lp_map = {
                    decision_pool[i]: float(decision_logprobs[i].item())
                    for i in range(len(decision_pool))
                }
                stop_lp = float(decision_logprobs[len(decision_pool) - 1].item())

            for action_idx in ranked_idx:
                if action_idx == stop_idx:
                    stop_selected = True
                    selected_action_indices.append(stop_idx)
                    policy_action_trace.append({"action": "stop", "reason": "policy_selected_stop"})
                    break
                if len(retrieved) >= self.top_k_paths:
                    selected_action_indices.append(stop_idx)
                    stop_selected = True
                    policy_action_trace.append({"action": "stop", "reason": "top_k_limit"})
                    break
                # If stop action is already preferred over this action, stop.
                if stop_lp >= decision_lp_map.get(action_idx, float("-inf")):
                    selected_action_indices.append(stop_idx)
                    stop_selected = True
                    policy_action_trace.append({"action": "stop", "reason": "stop_logprob_higher"})
                    break

                score, node_id, path_nodes, path_edges, token_cost, ir_penalty, overlap_score = action_nodes[action_idx]
                if node_id in selected_nodes:
                    continue
                node = graph.nodes[node_id]
                text = node.text[: self.max_chars]
                if not text.strip():
                    continue

                for step_edge in path_edges:
                    move_idx = edge_action_idx_map.get(step_edge)
                    if move_idx is not None:
                        selected_action_indices.append(move_idx)
                    policy_action_trace.append({"action": "move_to_neighbor", "edge_type": step_edge})
                policy_action_trace.append({"action": "select_node_as_context", "node_id": node_id})
                selected_action_indices.append(action_idx)
                selected_nodes.append(node_id)
                selected_paths.append(path_nodes)
                selected_edges.append(path_edges)
                visited_states.add(node.semantic_state_id)
                total_token_cost += token_cost
                irrelevant_count += ir_penalty
                overlap_scores.append(overlap_score)
                semantic_state_ids.append(node.semantic_state_id)

                path_str = " -> ".join(path_edges) if path_edges else "anchor"
                retrieved.append(
                    f"### File: {node.file_path} (state={node.semantic_state}, score={score:.3f}, path={path_str}) ###\n{text}"
                )
        else:
            stop_score = 0.12
            for action_idx, cand in enumerate(action_nodes):
                score, node_id, path_nodes, path_edges, token_cost, ir_penalty, overlap_score = cand
                if score < stop_score:
                    stop_selected = True
                    selected_action_indices.append(stop_idx)
                    policy_action_trace.append({"action": "stop", "reason": "score_below_threshold"})
                    break
                if node_id in selected_nodes:
                    continue
                node = graph.nodes[node_id]
                text = node.text[: self.max_chars]
                if not text.strip():
                    continue

                for step_edge in path_edges:
                    move_idx = edge_action_idx_map.get(step_edge)
                    if move_idx is not None:
                        selected_action_indices.append(move_idx)
                    policy_action_trace.append({"action": "move_to_neighbor", "edge_type": step_edge})
                policy_action_trace.append({"action": "select_node_as_context", "node_id": node_id})
                selected_action_indices.append(action_idx)
                selected_nodes.append(node_id)
                selected_paths.append(path_nodes)
                selected_edges.append(path_edges)
                visited_states.add(node.semantic_state_id)
                total_token_cost += token_cost
                irrelevant_count += ir_penalty
                overlap_scores.append(overlap_score)
                semantic_state_ids.append(node.semantic_state_id)

                path_str = " -> ".join(path_edges) if path_edges else "anchor"
                retrieved.append(
                    f"### File: {node.file_path} (state={node.semantic_state}, score={score:.3f}, path={path_str}) ###\n{text}"
                )

                if len(retrieved) >= self.top_k_paths:
                    selected_action_indices.append(stop_idx)
                    stop_selected = True
                    policy_action_trace.append({"action": "stop", "reason": "top_k_limit"})
                    break

        if not selected_action_indices:
            selected_action_indices = [stop_idx]
            stop_selected = True
            policy_action_trace.append({"action": "stop", "reason": "no_selected_actions"})
        elif not stop_selected:
            selected_action_indices.append(stop_idx)
            stop_selected = True
            policy_action_trace.append({"action": "stop", "reason": "auto_finalize"})

        all_symbols: List[str] = []
        for nid in selected_nodes:
            n = graph.nodes[nid]
            all_symbols.extend(list(n.defined_symbols))
        if all_symbols:
            uniq_ratio = len(set(all_symbols)) / max(1, len(all_symbols))
            redundancy_penalty = max(0.0, 1.0 - uniq_ratio)
        else:
            redundancy_penalty = 0.0

        call_or_import_steps = sum(
            1 for edges in selected_edges for e in edges if e in {"calls", "imports"}
        )
        total_steps = sum(len(edges) for edges in selected_edges)
        path_relevance = (
            (call_or_import_steps / max(1, total_steps))
            + (sum(overlap_scores) / max(1, len(overlap_scores)))
        ) / 2.0

        meta = {
            "path_relevance": float(path_relevance),
            "token_cost": float(total_token_cost),
            "redundancy_penalty": float(redundancy_penalty),
            "irrelevant_node_penalty": float(irrelevant_count),
            "semantic_state_ids": semantic_state_ids,
            "retrieval_path": selected_paths,
            "coarse_candidate_files": sorted(list(coarse_files)) if coarse_files else [],
            "policy_action_features": action_features,
            "policy_selected_indices": selected_action_indices,
            "policy_feat_dim": self._policy_feat_dim,
            "policy_action_trace": policy_action_trace,
            "stop_selected": stop_selected,
        }

        if return_metadata:
            return retrieved, meta
        return retrieved
