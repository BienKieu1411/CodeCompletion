"""
Test script for Tree-based CSSI with Inheritance Resolution and Multiple Embedding Modes.
"""

import ast
import json
import gzip
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class MethodNode:
    name: str
    signature: str
    docstring: Optional[str] = None
    is_inherited: bool = False
    source_class: str = ""


@dataclass
class ClassNode:
    name: str
    file_path: str
    language: str
    bases: List[str] = field(default_factory=list)
    methods: Dict[str, MethodNode] = field(default_factory=dict)
    fields: Dict[str, str] = field(default_factory=dict)
    parent_classes: List["ClassNode"] = field(default_factory=list)


class TypedContextTree:
    """
    Typed Context Tree (T_CSSI) representing classes, methods, and inheritance hierarchy.
    """
    def __init__(self):
        self.classes: Dict[str, ClassNode] = {}

    def parse_python_ast(self, file_path: str, source_code: str):
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    cname = node.name
                    bases = []
                    for b in node.bases:
                        if isinstance(b, ast.Name):
                            bases.append(b.id)
                        elif isinstance(b, ast.Attribute):
                            bases.append(ast.unparse(b))

                    cnode = ClassNode(name=cname, file_path=file_path, language="python", bases=bases)
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            mname = item.name
                            if mname.startswith("__") and mname != "__init__":
                                continue
                            sig_parts = []
                            for a in item.args.args:
                                ann = f": {ast.unparse(a.annotation)}" if a.annotation else ""
                                sig_parts.append(f"{a.arg}{ann}")
                            ret = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                            sig = f"def {mname}({', '.join(sig_parts)}){ret}"
                            doc = ast.get_docstring(item)
                            first_doc = doc.splitlines()[0].strip() if doc else None

                            cnode.methods[mname] = MethodNode(
                                name=mname,
                                signature=sig,
                                docstring=first_doc,
                                is_inherited=False,
                                source_class=cname
                            )
                    self.classes[cname] = cnode
        except Exception:
            self._parse_python_regex(file_path, source_code)

    def _parse_python_regex(self, file_path: str, source_code: str):
        lines = source_code.splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^class\s+([A-Za-z0-9_]+)(?:\(([^)]+)\))?:", line.strip())
            if m:
                cname = m.group(1)
                bases_raw = m.group(2) or ""
                bases = [b.strip() for b in bases_raw.split(",") if b.strip()]
                cnode = ClassNode(name=cname, file_path=file_path, language="python", bases=bases)
                for k in range(i + 1, min(len(lines), i + 60)):
                    l = lines[k].strip()
                    if l.startswith("class "):
                        break
                    if l.startswith("def "):
                        mname = re.match(r"def\s+([A-Za-z0-9_]+)", l)
                        if mname and not (mname.group(1).startswith("__") and mname.group(1) != "__init__"):
                            sig = l.split(":")[0]
                            cnode.methods[mname.group(1)] = MethodNode(
                                name=mname.group(1),
                                signature=sig,
                                is_inherited=False,
                                source_class=cname
                            )
                self.classes[cname] = cnode

    def parse_java_regex(self, file_path: str, source_code: str):
        for m in re.finditer(r"class\s+([A-Za-z0-9_]+)(?:\s+extends\s+([A-Za-z0-9_]+))?(?:\s+implements\s+([A-Za-z0-9_,\s]+))?", source_code):
            cname = m.group(1)
            bases = []
            if m.group(2): bases.append(m.group(2).strip())
            if m.group(3): bases.extend([b.strip() for b in m.group(3).split(",") if b.strip()])
            cnode = ClassNode(name=cname, file_path=file_path, language="java", bases=bases)
            # Find methods
            idx = m.end()
            chunk = source_code[idx:idx + 4000]
            for mm in re.finditer(r"(?:public|protected|private|\s)\s+([A-Za-z0-9_<>[\]]+)\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*(?:throws\s+[\w,\s]+)?\s*[{;]", chunk):
                ret_type = mm.group(1)
                mname = mm.group(2)
                params = mm.group(3)
                if mname not in ("if", "for", "while", "switch", "catch"):
                    sig = f"{ret_type} {mname}({params});"
                    cnode.methods[mname] = MethodNode(
                        name=mname,
                        signature=sig,
                        is_inherited=False,
                        source_class=cname
                    )
            self.classes[cname] = cnode

    def resolve_inheritance(self, max_depth: int = 3):
        """
        Traverses class hierarchy: pulls inherited methods into descendant nodes.
        """
        for cname, cnode in list(self.classes.items()):
            visited = set()
            curr_bases = list(cnode.bases)
            depth = 0
            while curr_bases and depth < max_depth:
                depth += 1
                next_bases = []
                for base_name in curr_bases:
                    if base_name in visited or base_name in ("object", "type", "Exception"):
                        continue
                    visited.add(base_name)
                    if base_name in self.classes:
                        parent = self.classes[base_name]
                        cnode.parent_classes.append(parent)
                        # Add inherited methods
                        for pm_name, pm_node in parent.methods.items():
                            if pm_name not in cnode.methods:
                                cnode.methods[pm_name] = MethodNode(
                                    name=pm_name,
                                    signature=pm_node.signature,
                                    docstring=pm_node.docstring,
                                    is_inherited=True,
                                    source_class=parent.name
                                )
                        next_bases.extend(parent.bases)
                curr_bases = next_bases

    def linearize(
        self,
        target_class_names: List[str],
        mode: str = "decorated_stub",
        max_methods: int = 10,
    ) -> str:
        """
        Linearizes the tree into text using one of the embedding strategies.
        Modes:
        - 'decorated_stub': Native class stub with [Inherited from Base] annotations.
        - 'chained_stub': Sequence of BaseClass -> ChildClass code stubs.
        - 'ascii_tree': Hierarchical ASCII tree representation.
        - 'schema': Compact JSON-like schema representation.
        """
        matched = [self.classes[c] for c in target_class_names if c in self.classes]
        if not matched:
            return ""

        blocks = []
        for cnode in matched:
            comment = "#" if cnode.language == "python" else "//"

            if mode == "decorated_stub":
                # Strategy: Embed directly inside class scope with explicit origin tags
                local_methods = [m for m in cnode.methods.values() if not m.is_inherited]
                inherited_methods = [m for m in cnode.methods.values() if m.is_inherited]

                base_str = f"({', '.join(cnode.bases)})" if cnode.bases else ""
                lines = [f"{comment} [CSSI Scope Tree: {cnode.name}{base_str} in {cnode.file_path}]"]
                lines.append(f"class {cnode.name}{base_str}:")

                if inherited_methods:
                    lines.append(f"    {comment} --- Inherited from {', '.join(cnode.bases)} ---")
                    for m in inherited_methods[:max_methods // 2]:
                        stub_sig = m.signature + ": ..." if cnode.language == "python" else m.signature
                        lines.append(f"    {stub_sig}")
                        if m.docstring:
                            lines.append(f'        """{m.docstring}"""')

                lines.append(f"    {comment} --- Declared in {cnode.name} ---")
                for m in local_methods[:max_methods]:
                    stub_sig = m.signature + ": ..." if cnode.language == "python" else m.signature
                    lines.append(f"    {stub_sig}")
                    if m.docstring:
                        lines.append(f'        """{m.docstring}"""')

                blocks.append("\n".join(lines))

            elif mode == "chained_stub":
                # Strategy: Base class stub followed by Child class stub
                chain = list(reversed(cnode.parent_classes)) + [cnode]
                chain_lines = []
                for node in chain:
                    b_str = f"({', '.join(node.bases)})" if node.bases else ""
                    chain_lines.append(f"{comment} [Interface Stub: {node.file_path}]\nclass {node.name}{b_str}:")
                    for m in list(node.methods.values())[:max_methods]:
                        if not m.is_inherited:
                            stub_sig = m.signature + ": ..." if node.language == "python" else m.signature
                            chain_lines.append(f"    {stub_sig}")
                            if m.docstring:
                                chain_lines.append(f'        """{m.docstring}"""')
                blocks.append("\n".join(chain_lines))

            elif mode == "ascii_tree":
                # Strategy: Explicit visual tree hierarchy
                base_str = f" (inherits: {', '.join(cnode.bases)})" if cnode.bases else ""
                lines = [f"{comment} [CSSI Context Tree]"]
                lines.append(f"{comment} └── Class: {cnode.name}{base_str}")
                for m in list(cnode.methods.values())[:max_methods]:
                    prefix = f"inherited from {m.source_class}" if m.is_inherited else "declared"
                    lines.append(f"{comment}     ├── [{prefix}] {m.signature}")
                blocks.append("\n".join(lines))

        return "\n\n".join(blocks) + "\n\n"


if __name__ == "__main__":
    # Test on a realistic inheritance case
    source_base = '''
class BaseService:
    def connect(self, timeout: int = 10) -> bool:
        """Establish network connection."""
        return True
    def ping(self) -> bool:
        """Health check endpoint."""
        return True
'''
    source_child = '''
class MessengerClient(BaseService):
    def __init__(self, token: str):
        self.token = token
    def get_delivery(self, data: dict) -> dict:
        """Extract delivery status."""
        return {}
'''
    tree = TypedContextTree()
    tree.parse_python_ast("services/base.py", source_base)
    tree.parse_python_ast("services/messenger.py", source_child)
    tree.resolve_inheritance()

    print("=== MODE 1: DECORATED STUB EMBEDDING ===")
    print(tree.linearize(["MessengerClient"], mode="decorated_stub"))

    print("=== MODE 2: CHAINED STUB EMBEDDING ===")
    print(tree.linearize(["MessengerClient"], mode="chained_stub"))

    print("=== MODE 3: ASCII TREE EMBEDDING ===")
    print(tree.linearize(["MessengerClient"], mode="ascii_tree"))
