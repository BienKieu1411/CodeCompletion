"""
Graph-Guided Iterative Retrieval (GGIR)

Thay thế Forward Generation của AlignCoder.

╔══════════════════════════════════════════════════════════════════════╗
║  Vấn đề AlignCoder (Forward Generation):                           ║
║  - Generate → append → re-retrieve → generate (4 rounds)           ║
║  - Chậm: 4× inference cost                                         ║
║  - Lỗi tích lũy: generated code sai → query sai → retrieve sai     ║
║                                                                     ║
║  Giải pháp GraphFRL (GGIR):                                        ║
║  - Round 1: Normal retrieval → top-k snippets                      ║
║  - Analyze: Parse imports/calls trong snippets → discover new deps  ║
║  - Round 2: Retrieve from discovered deps (graph-guided)            ║
║  - Merge: Combine + deduplicate                                     ║
║  - 2 rounds thay 4, dùng graph thay generated code                  ║
╚══════════════════════════════════════════════════════════════════════╝

Định lý 2 (Graph-Guided Error Bound):
    Cho retrieval error ε_r và generation error ε_g:
    - AlignCoder Forward Gen: ε_total = 1 - (1-ε_r)^T × (1-ε_g)^T  (T rounds)
      → Tăng nhanh khi T lớn do ε_g tích lũy
    - GGIR: ε_total = 1 - (1-ε_r)^2
      → Chỉ phụ thuộc retrieval error, không phụ thuộc generation error
      → Luôn nhỏ hơn khi ε_g > 0
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _extract_api_calls(code: str) -> List[str]:
    """
    Extract function/method calls from code.
    e.g. 'self.processor.transform(data)' → ['processor', 'transform']
    """
    # Match function calls: word followed by (
    calls = re.findall(r'(\w+)\s*\(', code)
    # Match attribute accesses: word.word
    attrs = re.findall(r'(\w+)\.(\w+)', code)
    result = list(calls)
    for obj, attr in attrs:
        result.extend([obj, attr])
    # Filter noise
    noise = {'self', 'cls', 'print', 'len', 'range', 'str', 'int', 'float',
             'list', 'dict', 'set', 'type', 'super', 'isinstance', 'True',
             'False', 'None', 'if', 'for', 'while', 'return', 'import'}
    return [r for r in result if r not in noise and len(r) > 2]


def _extract_imports_from_snippet(code: str) -> List[str]:
    """Extract import targets from code snippet."""
    modules = []
    for line in code.split('\n'):
        line = line.strip()
        m = re.match(r'^from\s+([\w.]+)\s+import\s+(.*)', line)
        if m:
            modules.append(m.group(1))
            # Also extract specific imported names
            names = [n.strip().split(' as ')[0] for n in m.group(2).split(',')]
            modules.extend(names)
            continue
        m = re.match(r'^import\s+([\w.]+)', line)
        if m:
            modules.append(m.group(1))
    return [m for m in modules if m and len(m) > 1]


class GraphGuidedIterativeRetriever:
    """
    2-round retrieval that uses graph analysis instead of generation.

    Round 1: Standard retrieval (BM25 + Dense)
    Analyze: Parse imports and API calls in retrieved snippets
    Round 2: Retrieve from newly discovered dependencies
    Merge:   Combine and deduplicate

    Comparison with AlignCoder Forward Generation:
    ┌──────────────────────┬───────────────────────────────────────┐
    │ AlignCoder           │ GraphFRL (GGIR)                       │
    ├──────────────────────┼───────────────────────────────────────┤
    │ 4 rounds             │ 2 rounds                              │
    │ Needs LLM generate   │ No generation needed                  │
    │ Error compounds       │ Error bounded by retrieval only       │
    │ O(4×(R+G)) cost      │ O(2×R) cost (no generation)           │
    └──────────────────────┴───────────────────────────────────────┘
    """

    def __init__(self, max_round2_files: int = 3):
        self.max_round2_files = max_round2_files

    def retrieve_iterative(
        self,
        round1_snippets: List[str],
        crossfile_dict: Dict[str, str],
        dense_retriever=None,
        graph_retriever=None,
        local_graph=None,
        query: str = "",
        top_k: int = 2,
    ) -> List[str]:
        """
        Graph-Guided Iterative Retrieval.

        Args:
            round1_snippets: Snippets from first retrieval round
            crossfile_dict: All crossfile contents
            dense_retriever: UniXCoderRetriever instance
            graph_retriever: GraphRetriever instance
            local_graph: LocalGraph from AST extractor
            query: Original query (left context)
            top_k: Number of additional snippets to retrieve

        Returns:
            Merged list of snippets (round1 + round2, deduplicated)
        """
        if not round1_snippets:
            return []

        # ── Step 1: Analyze Round 1 snippets ──
        # Extract identifiers, imports, and API calls from retrieved code
        discovered_modules: Set[str] = set()
        discovered_apis: Set[str] = set()

        for snippet in round1_snippets:
            # Extract imports from retrieved code
            imports = _extract_imports_from_snippet(snippet)
            discovered_modules.update(imports)

            # Extract API calls
            apis = _extract_api_calls(snippet)
            discovered_apis.update(apis)

        logger.debug(
            f"GGIR analysis: {len(discovered_modules)} modules, "
            f"{len(discovered_apis)} API calls discovered"
        )

        # ── Step 2: Find NEW files that weren't in Round 1 ──
        # Files already represented in Round 1 snippets
        round1_files: Set[str] = set()
        for snippet in round1_snippets:
            # Parse filename from snippet header
            m = re.search(r'### File: ([^\s(]+)', snippet)
            if m:
                fname = m.group(1)
                # Extract base filename without line range
                base = fname.split('::')[0]
                round1_files.add(base)

        # Find files matching discovered dependencies
        new_files: List[Tuple[str, float]] = []  # (filename, relevance_score)

        for filename, content in crossfile_dict.items():
            if filename in round1_files:
                continue

            score = 0.0

            # Check if this file is imported by any discovered module
            for mod in discovered_modules:
                mod_parts = mod.split('.')
                basename = filename.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                if basename in mod_parts or mod_parts[-1] == basename:
                    score += 2.0  # High priority: import match
                    break

            # Check if this file contains any discovered API names
            if score == 0.0:
                content_lower = content.lower()
                api_matches = sum(1 for api in discovered_apis
                                  if f"def {api}" in content or f"class {api}" in content)
                if api_matches > 0:
                    score += api_matches * 1.0

            if score > 0:
                new_files.append((filename, score))

        # Sort by relevance score (highest first)
        new_files.sort(key=lambda x: x[1], reverse=True)
        new_files = new_files[:self.max_round2_files]

        if not new_files:
            logger.debug("GGIR Round 2: no new dependencies found")
            return round1_snippets

        # ── Step 3: Round 2 Retrieval ──
        round2_snippets: List[str] = []

        if dense_retriever is not None:
            # Create a sub-dict with only new files
            new_dict = {f: crossfile_dict[f] for f, _ in new_files if f in crossfile_dict}
            if new_dict:
                # Enhanced query: original + discovered APIs
                enhanced_query = query
                if discovered_apis:
                    api_suffix = " ".join(list(discovered_apis)[:10])
                    enhanced_query = query + " " + api_suffix

                r2_snippets, _ = dense_retriever.retrieve_top_k(
                    enhanced_query, new_dict, top_k=top_k,
                    repo_key="ggir_round2",
                )
                round2_snippets.extend(r2_snippets)

        elif graph_retriever is not None and local_graph is not None:
            # Fallback: use graph retriever on new files
            new_dict = {f: crossfile_dict[f] for f, _ in new_files if f in crossfile_dict}
            r2_snippets = graph_retriever.retrieve_paths(
                local_graph=local_graph,
                crossfile_dict=new_dict,
            )
            round2_snippets.extend(r2_snippets)

        if round2_snippets:
            logger.debug(f"GGIR Round 2: retrieved {len(round2_snippets)} additional snippets")

        # ── Step 4: Merge and deduplicate ──
        merged = list(round1_snippets)

        # Simple dedup by filename
        existing_ids = set()
        for s in merged:
            m = re.search(r'### File: ([^\s(]+)', s)
            if m:
                existing_ids.add(m.group(1))

        for s in round2_snippets:
            m = re.search(r'### File: ([^\s(]+)', s)
            if m and m.group(1) not in existing_ids:
                merged.append(s)
                existing_ids.add(m.group(1))

        return merged
