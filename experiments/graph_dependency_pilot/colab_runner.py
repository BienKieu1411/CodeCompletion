#!/usr/bin/env python3
"""Run the 2x2 AlignCoder-vs-prefix-dependency pilot on a Colab A100."""

from __future__ import annotations

import difflib
import gzip
import hashlib
import json
import keyword
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path("/content/graph_dependency_pilot")
ARCHIVE = Path("/content/graph_dependency_pilot_bundle.tar.gz")
OUT_DIR = DATA_DIR / "output"
ALIGN_RETRIEVER = "AlignCoder/AlignRetriever"
ALIGN_REVISION = "a179c0fdbc5a56d83ae142064ef1ea773a841753"
GENERATOR = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
BM25_TOPK = 100
RETRIEVER_TOPK = 10
CROSSFILE_TOKENS = 1536
GENERATOR_CONTEXT = 2048
GENERATOR_MAX_NEW = 64
GRAPH_WEIGHT = 0.12
SEED = 20261024

PY_DECL = re.compile(r"(?m)^\s*(?:async\s+)?(?:class|def)\s+([A-Za-z_]\w*)")
JAVA_DECL = re.compile(r"\b(?:class|interface|enum|record)\s+([A-Za-z_$][\w$]*)")
IDENTIFIER = re.compile(r"[_A-Za-z][_A-Za-z0-9]*")


def ensure_dependencies() -> None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        version("vllm")
    except PackageNotFoundError:
        # Use vLLM's CUDA/PyTorch backend resolver instead of blindly replacing
        # the Colab torch build with a potentially incompatible pip wheel.
        print("Installing vLLM with the CUDA/PyTorch backend matched to Colab...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv_path = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
        subprocess.check_call([uv_path, "pip", "install", "--system", "vllm",
                               "--torch-backend=auto"])
    # The Colab base image may retain a torchaudio wheel built against its
    # original PyTorch CUDA version after vLLM's resolver upgrades PyTorch.
    # This pilot is code/text-only; remove torchaudio only when that mismatch
    # is confirmed, so Transformers won't import an incompatible optional wheel.
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        pass
    except RuntimeError as exc:
        if "different CUDA versions" not in str(exc):
            raise
        print("Removing incompatible optional torchaudio wheel (text-only pilot)...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])
    missing = []
    for module, package in (("torch", "torch"), ("transformers", "transformers"),
                            ("rank_bm25", "rank_bm25"), ("fuzzywuzzy", "fuzzywuzzy")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    try:
        if version("tree-sitter") != "0.20.4":
            missing.append("tree-sitter==0.20.4")
    except PackageNotFoundError:
        missing.append("tree-sitter==0.20.4")
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def load_bundle() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            archive.extractall(DATA_DIR, filter="data")
    task_path = DATA_DIR / "tasks.jsonl.gz"
    pool_path = DATA_DIR / "pools.jsonl.gz"
    file_path = DATA_DIR / "files.jsonl.gz"
    if not task_path.exists() or not pool_path.exists() or not file_path.exists():
        raise FileNotFoundError(f"Expected tasks, pools, and files in {DATA_DIR}; upload the pilot bundle first")
    with gzip.open(task_path, "rt", encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle]
    with gzip.open(file_path, "rt", encoding="utf-8") as handle:
        files = {row["file_id"]: {"path": row["path"], "text": row["text"]}
                 for row in map(json.loads, handle)}
    with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
        pools = {row["pool_id"]: [files[file_id] for file_id in row["file_ids"]]
                 for row in map(json.loads, handle)}
    with open(DATA_DIR / "selection.json", encoding="utf-8") as handle:
        selection = json.load(handle)
    return tasks, pools, selection


class Chunk:
    __slots__ = ("path", "description", "code", "kind", "api", "key", "start", "end", "language", "source_paths")

    def __init__(self, path: str, description: str, code: str, kind: str,
                 api: bool = False, start: int = 0, end: int = 0,
                 language: str | None = None, source_paths: list[str] | None = None):
        self.path, self.description, self.code = path, description, code
        self.kind, self.api, self.start, self.end = kind, api, start, end
        self.language = language or ("java" if path.endswith(".java") else "python")
        self.source_paths = source_paths or []
        self.key = hashlib.sha1(f"{path}\0{start}\0{end}\0{kind}\0{code}".encode()).hexdigest()

    def __str__(self) -> str:
        label = "#" if self.language == "python" else "//"
        header = "\n".join(f"{label} {line}" for line in self.description.strip().splitlines() if line)
        body = "\n".join(f"{label} {line}" for line in self.code.split("\n") if line)
        return f"{header}\n{body}".strip()

    @property
    def kind_language(self) -> str:
        return self.language


def aligncoder_chunks(path: str, source: str) -> list[Chunk]:
    """AlignCoder's default blank-line mini-blocking, max 15 nonempty lines."""
    all_lines = source.splitlines()
    nonempty = [(line_no, ordinal, line) for ordinal, (line_no, line) in enumerate(
        (pair for pair in enumerate(all_lines) if pair[1].strip()))]
    nonempty = nonempty[:5000]
    mini: list[list[tuple[int, int, str]]] = []
    current: list[tuple[int, int, str]] = []
    previous_line = -2
    for pair in nonempty:
        line_no, _, line = pair
        if line_no > previous_line + 1 and current:
            mini.append(current)
            current = []
        current.append(pair)
        previous_line = line_no
    if current:
        mini.append(current)
    split_mini: list[list[tuple[int, int, str]]] = []
    for block in mini:
        split_mini.extend(block[i:i + 15] for i in range(0, len(block), 15))
    merged: list[list[tuple[int, str]]] = []
    current = []
    for block in split_mini:
        if current and len(current) + len(block) > 15:
            merged.append(current)
            current = []
        current.extend(block)
    if current:
        merged.append(current)
    output = []
    for block in merged:
        # AlignCoder's default description counts nonempty lines, not source lines.
        start, end = block[0][1] + 1, block[-1][1] + 1
        text = "\n".join(line for _, _, line in block)
        desc = f"file path: {path}\nlines: {start}-{end}"
        output.append(Chunk(path, desc, text, "mini_block", start=start, end=end))
    return output


def api_chunks(task: dict[str, Any], files: list[dict[str, Any]]) -> list[Chunk]:
    """Use the cloned AlignCoder GET_API and its <=15-line API-block format."""
    import sys
    from types import SimpleNamespace
    aligncoder_dir = str(DATA_DIR / "aligncoder")
    if aligncoder_dir not in sys.path:
        sys.path.insert(0, aligncoder_dir)
    from api import GET_API

    related_files = [SimpleNamespace(file_path=item["path"], code_content=item["text"])
                     for item in files if item.get("path") != task["path"]]
    helper = GET_API([], task["language"])
    descriptions = []
    if task["language"] == "python":
        definitions: dict[str, list[Any]] = defaultdict(list)
        for file in related_files:
            for name in PY_DECL.findall(file.code_content):
                definitions[name].append(file)
        for entity in helper._extract_python_imports(task["left_context"]):
            for file in definitions.get(entity["name"], []):
                found = helper._find_python_definition(entity["name"], [file])
                if found:
                    descriptions.append(found["content"] if found["type"] == "class" else found["signature"])
                    break
    else:
        imported = helper._extract_java_imports(task["left_context"])
        processed = set()
        for entity in imported:
            name = entity["name"]
            if name in processed or name == "*":
                continue
            processed.add(name)
            found = helper._find_java_definition(name, related_files)
            if found:
                descriptions.append(found["content"] if found["type"] == "class" else found["signature"])
        for entity in imported:
            if entity["name"] != "*":
                continue
            package = entity["package"]
            for name in helper._find_classes_in_package(package, related_files):
                if name in processed:
                    continue
                processed.add(name)
                found = helper._find_java_definition(name, related_files)
                if found:
                    descriptions.append(found["content"] if found["type"] == "class" else found["signature"])
    api_text = "\n\n".join(descriptions)
    if not api_text:
        return []
    mini: list[list[str]] = []
    current: list[str] = []
    for line in api_text.splitlines():
        if not line.strip():
            if current:
                mini.append(current)
                current = []
        else:
            current.append(line)
    if current:
        mini.append(current)
    split_mini = [block[i:i + 15] for block in mini for i in range(0, len(block), 15)]
    merged: list[list[str]] = []
    current = []
    for block in split_mini:
        if current and len(current) + len(block) > 15:
            merged.append(current)
            current = []
        current.extend(block)
    if current:
        merged.append(current)
    chunks = []
    for index, block in enumerate(merged):
        code = "\n".join(block)
        source_paths = [edge["path"] for edge in task["dependency_edges"]
                        if edge.get("symbol") and re.search(rf"\b{re.escape(edge['symbol'])}\b", code)]
        description = "Potentially invoked APIs:\n" if index < len(merged) - 1 else "Potential invoked APIs:\n"
        chunks.append(Chunk("file_path", description, code,
                            "mini_block", api=True, language=task["language"],
                            source_paths=list(dict.fromkeys(source_paths))))
    return chunks


def last_nonempty_lines(text: str, count: int = 20) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-count:]) if count > 0 else "\n".join(lines)


def sort_indices(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def chunk_token_ids(text: str, tokenizer: Any, max_length: int, is_query: bool) -> list[int]:
    tokens = tokenizer.tokenize(text)
    if is_query:
        tokens = tokens[-(max_length - 4):]
    else:
        tokens = tokens[:max_length - 4]
    tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + tokens + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = ids[:max_length]
    return ids + [tokenizer.pad_token_id] * (max_length - len(ids))


class AlignEmbedder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        print(f"Loading retriever {ALIGN_RETRIEVER}@{ALIGN_REVISION}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(ALIGN_RETRIEVER, revision=ALIGN_REVISION, use_fast=False)
        self.model = AutoModel.from_pretrained(ALIGN_RETRIEVER, revision=ALIGN_REVISION).cuda().eval()
        self.cache: dict[str, Any] = {}

    @staticmethod
    def candidate_text(chunk: Chunk) -> str:
        return str(chunk)

    def embed(self, texts: list[str], *, query: bool, batch_size: int = 32):
        import torch
        max_length = 256 if query else 512
        all_vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                part = texts[start:start + batch_size]
                ids = torch.tensor([chunk_token_ids(text, self.tokenizer, max_length, query) for text in part], device="cuda")
                mask = ids.ne(self.tokenizer.pad_token_id)
                hidden = self.model(ids, attention_mask=mask).last_hidden_state
                vectors = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
                all_vectors.append(vectors.float().cpu())
        return torch.cat(all_vectors, dim=0) if all_vectors else torch.empty((0, 768))

    def candidate_vectors(self, chunks: list[Chunk]):
        missing = [chunk for chunk in chunks if chunk.key not in self.cache]
        if missing:
            vectors = self.embed([self.candidate_text(chunk) for chunk in missing], query=False)
            for chunk, vector in zip(missing, vectors):
                self.cache[chunk.key] = vector
        return self.torch.stack([self.cache[chunk.key] for chunk in chunks])

    def rank(self, query: str, chunks: list[Chunk]) -> list[float]:
        qvec = self.embed([query], query=True)[0]
        cvec = self.candidate_vectors(chunks)
        return (cvec @ qvec).tolist()


def matching_dependency_edges(chunk: Chunk, edges: list[dict[str, str]]) -> list[dict[str, str]]:
    matches = []
    for edge in edges:
        matching_paths = chunk.source_paths if chunk.api else [chunk.path]
        if edge["path"] not in matching_paths:
            continue
        symbol = edge.get("symbol", "")
        if symbol and symbol != "*" and re.search(rf"\b{re.escape(symbol)}\b", chunk.code):
            matches.append(edge)
        elif edge.get("relation") == "imports_module":
            matches.append(edge)
    return matches


def candidate_graph_score(chunk: Chunk, edges: list[dict[str, str]]) -> float:
    hits = {edge.get("import", edge.get("symbol", ""))
            for edge in matching_dependency_edges(chunk, edges)}
    total = max(1, len({edge.get("import", edge.get("symbol", "")) for edge in edges}))
    return min(1.0, len(hits) / total)


def compact_graph_evidence(chunk: Chunk, edges: list[dict[str, str]], language: str) -> str:
    label = "#" if language == "python" else "//"
    matches = [f"{edge['relation']}: {edge['import']} -> {edge['path']}::{edge['symbol']}"
               for edge in matching_dependency_edges(chunk, edges) if edge.get("symbol")]
    matches = list(dict.fromkeys(matches))[:2]
    return "\n".join(f"{label} DEPENDENCY {item}" for item in matches)


def build_prompt(task: dict[str, Any], chunks: list[Chunk], generator_tokenizer: Any,
                 with_graph_evidence: bool) -> tuple[str, int, int]:
    language = task["language"]
    evidence = task["dependency_edges"]
    blocks = []
    for chunk in chunks:
        # AlignCoder's generator stops consuming returned retrievals at the
        # explicit no-cross-file-context sentinel.
        if chunk.kind == "stop" or not chunk.path:
            break
        note = compact_graph_evidence(chunk, evidence, language) if with_graph_evidence else ""
        block = str(chunk)
        blocks.append(f"{note}\n{block}" if note else block)
    joined = "\n\n".join(blocks)
    cross_ids = generator_tokenizer.encode(joined[:CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]
    path_text = f"\n\n# file path: {task['path']}\n\n"
    path_ids = generator_tokenizer.encode(path_text, add_special_tokens=False)
    allowed_left = max(0, GENERATOR_CONTEXT - len(cross_ids) - len(path_ids) - 10)
    left_ids = generator_tokenizer.encode(task["left_context"], add_special_tokens=False)[-allowed_left:] if allowed_left else []
    prompt_ids = (cross_ids + path_ids + left_ids)[-GENERATOR_CONTEXT:]
    prompt = generator_tokenizer.decode(prompt_ids, skip_special_tokens=True)
    return prompt, len(prompt_ids), len(cross_ids)


def load_generator():
    from vllm import LLM
    print(f"Loading vLLM generator {GENERATOR}@{GENERATOR_REVISION}", flush=True)
    engine = LLM(
        model=GENERATOR,
        tokenizer=GENERATOR,
        revision=GENERATOR_REVISION,
        tokenizer_revision=GENERATOR_REVISION,
        dtype="float16",
        max_model_len=GENERATOR_CONTEXT + GENERATOR_MAX_NEW,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.78,
        seed=SEED,
        disable_log_stats=True,
    )
    return engine


def generate(engine: Any, prompts: list[str], *, samples: int,
             temperature: float, top_p: float) -> list[list[str]]:
    from vllm import SamplingParams
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=GENERATOR_MAX_NEW,
        n=samples,
        seed=123,
    )
    started = time.time()
    print(f"vLLM generation starting: {len(prompts)} prompts × {samples} completion(s)", flush=True)
    requests = engine.generate(prompts, params, use_tqdm=True)
    results = [[completion.text for completion in request.outputs] for request in requests]
    print(f"vLLM generation finished in {time.time() - started:.1f}s", flush=True)
    return results


def load_parsers():
    from tree_sitter import Language, Parser
    parser_map = {}
    for lang in ("python", "java"):
        path = DATA_DIR / "parsers" / f"{lang}-lang-parser.so"
        try:
            language = Language(str(path), lang)
            parser = Parser()
            parser.set_language(language)
            parser_map[lang] = parser
        except Exception as error:
            print(f"WARNING parser unavailable for {lang}: {error}", flush=True)
            parser_map[lang] = None
    return parser_map


def has_error(node: Any) -> bool:
    if node.type == "ERROR":
        return True
    return any(has_error(child) for child in node.children)


def remove_comments(code: str) -> str:
    # Preserve the original AlignCoder metric's comment filtering behavior.
    return re.sub(r"#.*|//.*", "", code)


def postprocess(prompt: str, completion: str, language: str, parser_map: dict[str, Any]) -> str:
    # Match AlignCoder's postprocess_code_lines behavior (including its
    # first-statement policy); for Python only newline boundaries can be a
    # return point, so parsing those boundaries is equivalent to its scan.
    try:
        if language in ("java", "csharp", "typescript"):
            for index, char in enumerate(completion):
                if char in (";", "}", "{"):
                    return completion[:index + 1] if index else completion
            return completion
        if language == "python":
            parser = parser_map.get(language)
            if parser is None:
                return completion
            for index, char in enumerate(completion):
                if char != "\n":
                    continue
                parsed = parser.parse((prompt + completion[:index]).encode("utf-8", errors="replace"))
                if parsed is not None and not has_error(parsed.root_node):
                    return completion[:index].rstrip()
            return completion
        return completion
    except Exception:
        return completion


def identifiers(code: str, language: str) -> list[str]:
    clean = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'', "", code)
    names = re.findall(r"\w+", clean)
    if language == "python":
        ignored = set(keyword.kwlist) - {"True", "False"}
    else:
        keyword_path = DATA_DIR / "aligncoder" / "utils" / "keywords" / "java.txt"
        ignored = {line.strip() for line in keyword_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return [name for name in names if re.match(r"[_a-zA-Z][_a-zA-Z0-9]*", name) and name not in ignored]


def metric_row(task: dict[str, Any], prediction: str, parser_map: dict[str, Any]) -> dict[str, Any]:
    pred = remove_comments(postprocess(task["left_context"], prediction, task["language"], parser_map)).strip()
    target = remove_comments(task["groundtruth"]).strip()
    pred_lines = [line.strip() for line in pred.splitlines() if line.strip()]
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    em = int(pred_lines == target_lines)
    # AlignCoder uses fuzzywuzzy.fuzz.ratio; use it if present, with stdlib fallback.
    try:
        from fuzzywuzzy import fuzz
        es = fuzz.ratio(target, pred) / 100.0
    except Exception:
        es = difflib.SequenceMatcher(None, target, pred).ratio()
    pred_ids, target_ids = set(identifiers(pred, task["language"])), set(identifiers(target, task["language"]))
    tp = len(pred_ids & target_ids)
    fp, fn = len(pred_ids - target_ids), len(target_ids - pred_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    repo_em, repo_es = None, None
    if task["benchmark"].startswith("repoeval"):
        pred_eval_lines = pred_lines[:len(target_lines)]
        target_eval = "\n".join(target_lines)
        pred_eval = "\n".join(pred_eval_lines)
        distance = levenshtein_distance(target_eval, pred_eval)
        repo_es = 1.0 - distance / max(len(target_eval), len(pred_eval)) if max(len(target_eval), len(pred_eval)) else 1.0
        repo_em = float(pred_eval_lines == target_lines)
    return {"task_id": task["task_id"], "benchmark": task["benchmark"], "repo_id": task["task_id"].split("/")[0],
            "prediction": pred, "target": target, "em": em, "es": es,
            "repo_em": repo_em, "repo_es": repo_es,
            "id_em": float(pred_ids == target_ids), "id_precision": precision,
            "id_recall": recall, "id_f1": f1}


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        group = [row for row in rows if row["benchmark"] == benchmark]
        metric_keys = ("em", "es") if benchmark.startswith("repoeval") else (
            "em", "es", "id_em", "id_precision", "id_recall", "id_f1")
        summary = {key: 100 * sum(row[key] for row in group) / len(group) for key in metric_keys}
        if benchmark.startswith("repoeval"):
            by_repo = defaultdict(list)
            for row in group:
                by_repo[row["repo_id"]].append(row)
            summary["repo_macro_em"] = 100 * sum(sum(r["repo_em"] for r in v) / len(v) for v in by_repo.values()) / len(by_repo)
            summary["repo_macro_es"] = 100 * sum(sum(r["repo_es"] for r in v) / len(v) for v in by_repo.values()) / len(by_repo)
        result[benchmark] = {k: round(v, 4) for k, v in summary.items()}
    return result


def paired_bootstrap(rows_by_arm: dict[str, list[dict[str, Any]]], baseline: str = "aligncoder_code", rounds: int = 2000) -> dict[str, Any]:
    import random
    rng = random.Random(SEED)
    # RepoEval line/API can reuse task_id values for different completion
    # positions. Pair by benchmark as well, or one benchmark silently
    # overwrites the other in these maps.
    baseline_rows = {(row["benchmark"], row["task_id"]): row for row in rows_by_arm[baseline]}
    out = {}
    for arm, arm_rows in rows_by_arm.items():
        if arm == baseline:
            continue
        current = {(row["benchmark"], row["task_id"]): row for row in arm_rows}
        paired_keys = sorted(baseline_rows.keys() & current.keys())
        paired = [(baseline_rows[key], current[key]) for key in paired_keys]
        diffs = {metric: [] for metric in ("em", "es", "id_f1")}
        for _ in range(rounds):
            sample = [paired[rng.randrange(len(paired))] for _ in paired]
            for metric in diffs:
                diffs[metric].append(100 * sum(b[metric] - a[metric] for a, b in sample) / len(sample))
        out[arm] = {metric: {"mean_delta_pp": round(sum(values) / len(values), 4),
                            "ci95_pp": [round(sorted(values)[int(0.025 * rounds)], 4),
                                        round(sorted(values)[int(0.975 * rounds)], 4)]}
                    for metric, values in diffs.items()}
    return out


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    ensure_dependencies()
    import torch
    from rank_bm25 import BM25Okapi

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Colab GPU: {gpu_name}, {gpu_total_gib:.1f} GiB; CUDA={torch.version.cuda}; torch={torch.__version__}", flush=True)
    if "A100" not in gpu_name:
        raise RuntimeError(f"Requested A100 runtime but received {gpu_name}")
    if not 35 <= gpu_total_gib <= 45:
        raise RuntimeError(f"Requested 40-GiB A100 but received {gpu_total_gib:.1f} GiB")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks, pools, selection = load_bundle()
    print(f"Loaded {len(tasks)} selected tasks; {len(pools)} unique repository pools", flush=True)
    if len(tasks) < 200 or len(tasks) > 300:
        raise ValueError(f"Pilot size outside requested range: {len(tasks)}")

    # Build the exact AlignCoder-default mini-block corpus once per unique repo pool.
    corpora: dict[str, list[Chunk]] = {}
    file_chunk_cache: dict[str, list[Chunk]] = {}
    for pool_id, files in pools.items():
        chunks = []
        for file in files:
            if not file.get("path") or not file.get("text"):
                continue
            file_key = hashlib.sha1((file["path"] + "\0" + file["text"]).encode()).hexdigest()
            if file_key not in file_chunk_cache:
                file_chunk_cache[file_key] = aligncoder_chunks(file["path"], file["text"])
            chunks.extend(file_chunk_cache[file_key])
        corpora[pool_id] = chunks
    corpus_chunks = sum(len(value) for value in corpora.values())
    unique_chunk_count = sum(len(value) for value in file_chunk_cache.values())
    print(f"Chunked {len(pools)} pools into {corpus_chunks:,} pool references / {unique_chunk_count:,} unique AlignCoder mini-blocks", flush=True)

    # Query enhancement is one shared stage, using the same AlignCoder BM25/API
    # candidates for every arm. Graph extraction never sees the generated text.
    from transformers import AutoTokenizer
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR, revision=GENERATOR_REVISION, use_fast=True)
    if gen_tokenizer.pad_token_id is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.padding_side = "left"
    parser_map = load_parsers()
    task_state = []
    qh_prompts = []
    for task in tasks:
        chunks = corpora[task["pool_id"]]
        api = api_chunks(task, pools[task["pool_id"]])
        query = last_nonempty_lines(task["left_context"], 20)
        mixed = chunks + api
        qh_scores = BM25Okapi([chunk.code.lower().split() for chunk in mixed]).get_scores(query.lower().split())
        # `query_with_api` builds one BM25 over all chunks and APIs, includes at
        # most two API blocks, and returns top-100 mini-blocks.
        qh_ranked = []
        api_count = 0
        for ix in sort_indices([float(x) for x in qh_scores]):
            chunk = mixed[ix]
            if chunk.api:
                if api_count >= 2:
                    continue
                api_count += 1
            qh_ranked.append(chunk)
            if len(qh_ranked) >= BM25_TOPK:
                break
        qh_prompts.append(build_prompt(task, qh_ranked, gen_tokenizer, False)[0])
        task_state.append({"task": task, "chunks": chunks, "api": api, "initial_query": query,
                           "qh_pool": qh_ranked})
    print(f"Built {len(qh_prompts)} fixed query-enhancement prompts", flush=True)

    generator = load_generator()
    qh_outputs = generate(generator, qh_prompts, samples=8, temperature=0.8, top_p=0.95)
    for state, outputs in zip(task_state, qh_outputs):
        cleaned = []
        for output in outputs:
            text = remove_comments(postprocess(state["task"]["left_context"], output,
                                               state["task"]["language"], parser_map)).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        state["qh_outputs"] = cleaned
        state["enhanced_query"] = state["initial_query"] + "\n\n" + "\n\n".join(
            f"# candidate answer {i + 1}:\n{text}" for i, text in enumerate(cleaned))
    print("Shared AlignCoder query-enhancement stage complete", flush=True)

    # Frozen UniXcoder/AlignRetriever scoring; graph reranking only adds an
    # import-to-definition edge bonus over the exact same BM25 candidate pool.
    embedder = AlignEmbedder()
    arms = {name: [] for name in ("aligncoder_code", "graph_rerank_code",
                                  "aligncoder_dep_prompt", "graph_rerank_dep_prompt")}
    retrieval_traces = []
    prompt_rows = []
    all_completion_prompts = []
    completion_prompt_map = []
    retrieval_started = time.time()
    for index, state in enumerate(task_state, 1):
        task, chunks, api = state["task"], state["chunks"], state["api"]
        query = state["enhanced_query"]
        raw_scores = BM25Okapi([chunk.code.lower().split() for chunk in chunks]).get_scores(query.lower().split())
        raw_top = [chunks[ix] for ix in sort_indices([float(x) for x in raw_scores])[:BM25_TOPK]]
        raw_score_by_key = {chunk.key: float(score) for chunk, score in zip(chunks, raw_scores)}
        stop_chunk = Chunk("", "Don't need cross file context for completion", "", "stop",
                           language=task["language"])
        candidates = raw_top + api + [stop_chunk]
        cosine_scores = embedder.rank(query, candidates)
        graph_scores = [candidate_graph_score(chunk, task["dependency_edges"]) for chunk in candidates]
        base_order = sorted(range(len(candidates)), key=lambda i: (-cosine_scores[i], i))[:RETRIEVER_TOPK]
        graph_order = sorted(range(len(candidates)), key=lambda i: (-(cosine_scores[i] + GRAPH_WEIGHT * graph_scores[i]), i))[:RETRIEVER_TOPK]
        base_selected = [candidates[i] for i in base_order]
        graph_selected = [candidates[i] for i in graph_order]
        chosen = {"aligncoder_code": base_selected, "aligncoder_dep_prompt": base_selected,
                  "graph_rerank_code": graph_selected, "graph_rerank_dep_prompt": graph_selected}
        for arm, selected_chunks in chosen.items():
            use_graph = arm.endswith("dep_prompt")
            prompt, prompt_tokens, cross_tokens = build_prompt(task, selected_chunks, gen_tokenizer, use_graph)
            prompt_index = len(all_completion_prompts)
            all_completion_prompts.append(prompt)
            completion_prompt_map.append((arm, task, prompt_tokens, cross_tokens, selected_chunks))
            prompt_rows.append({"task_id": task["task_id"], "benchmark": task["benchmark"], "arm": arm,
                                "prompt": prompt, "prompt_tokens": prompt_tokens,
                                "crossfile_tokens": cross_tokens})
        unique_dep_symbols = {edge["import"] for edge in task["dependency_edges"]}
        covered_base = {edge["import"] for chunk in base_selected
                        for edge in matching_dependency_edges(chunk, task["dependency_edges"])}
        covered_graph = {edge["import"] for chunk in graph_selected
                         for edge in matching_dependency_edges(chunk, task["dependency_edges"])}
        dep_recall_base = len(covered_base) / max(1, len(unique_dep_symbols))
        dep_recall_graph = len(covered_graph) / max(1, len(unique_dep_symbols))
        retrieval_traces.append({
            "task_id": task["task_id"], "benchmark": task["benchmark"], "language": task["language"],
            "query_enhancement_outputs": state["qh_outputs"], "dependency_edges": task["dependency_edges"],
            "dependency_symbol_count": len(unique_dep_symbols), "candidate_count": len(candidates),
            "bm25_top100": [{"path": c.path, "line_start": c.start, "line_end": c.end,
                             "kind": c.kind, "api": c.api, "bm25_score": raw_score_by_key.get(c.key),
                             "code": c.code} for c in raw_top],
            "aligncoder_top10": [{"path": candidates[i].path, "line_start": candidates[i].start,
                                  "line_end": candidates[i].end, "cosine": cosine_scores[i],
                                  "graph_score": graph_scores[i], "kind": candidates[i].kind,
                                  "source_paths": candidates[i].source_paths,
                                  "code": candidates[i].code} for i in base_order],
            "graph_top10": [{"path": candidates[i].path, "line_start": candidates[i].start,
                             "line_end": candidates[i].end, "cosine": cosine_scores[i],
                             "graph_score": graph_scores[i], "combined_score": cosine_scores[i] + GRAPH_WEIGHT * graph_scores[i],
                             "kind": candidates[i].kind,
                             "source_paths": candidates[i].source_paths,
                             "code": candidates[i].code} for i in graph_order],
            "rerank_candidates": [{"pool_rank": rank + 1, "path": candidate.path,
                                   "line_start": candidate.start, "line_end": candidate.end,
                                   "kind": candidate.kind, "api": candidate.api,
                                   "bm25_score": raw_score_by_key.get(candidate.key),
                                   "cosine": cosine_scores[candidate_index],
                                   "graph_score": graph_scores[candidate_index],
                                   "combined_score": cosine_scores[candidate_index] + GRAPH_WEIGHT * graph_scores[candidate_index],
                                   "aligncoder_selected": candidate_index in base_order,
                                   "graph_selected": candidate_index in graph_order,
                                   "source_paths": candidate.source_paths,
                                   "code": candidate.code}
                                  for rank, candidate_index in enumerate(sort_indices(cosine_scores))
                                  for candidate in [candidates[candidate_index]]],
            "dep_recall_at10_aligncoder": dep_recall_base,
            "dep_recall_at10_graph": dep_recall_graph,
        })
        if index % 16 == 0 or index == len(task_state):
            print(f"Reranked {index}/{len(task_state)} tasks", flush=True)
    print(f"Retriever stage finished in {time.time() - retrieval_started:.1f}s; cached {len(embedder.cache):,} candidate embeddings", flush=True)
    del embedder
    torch.cuda.empty_cache()

    # Final generator: same fixed model, greedy decoding, exact same selected
    # top-10 per retrieval arm; dependency annotations replace code tokens
    # within the existing 1,536-token cross-file ceiling.
    final_outputs = generate(generator, all_completion_prompts, samples=1,
                             temperature=0.0, top_p=1.0)
    prediction_rows = []
    rows_by_arm = {name: [] for name in arms}
    for (arm, task, prompt_tokens, cross_tokens, selected_chunks), output_list in zip(completion_prompt_map, final_outputs):
        raw_prediction = output_list[0]
        row = metric_row(task, raw_prediction, parser_map)
        row.update({"arm": arm, "raw_prediction": raw_prediction,
                    "prompt_tokens": prompt_tokens, "crossfile_tokens": cross_tokens})
        prediction_rows.append(row)
        rows_by_arm[arm].append(row)
    del generator
    torch.cuda.empty_cache()

    summaries = {arm: aggregate_metrics(rows) for arm, rows in rows_by_arm.items()}
    bootstrap = paired_bootstrap(rows_by_arm)
    retrieval_summary = {}
    for benchmark in sorted({row["benchmark"] for row in retrieval_traces}):
        group = [row for row in retrieval_traces if row["benchmark"] == benchmark]
        retrieval_summary[benchmark] = {
            "mean_dep_recall_at10_aligncoder": round(100 * sum(x["dep_recall_at10_aligncoder"] for x in group) / len(group), 4),
            "mean_dep_recall_at10_graph": round(100 * sum(x["dep_recall_at10_graph"] for x in group) / len(group), 4),
            "dependency_hit_rate_aligncoder": round(100 * sum(x["dep_recall_at10_aligncoder"] > 0 for x in group) / len(group), 4),
            "dependency_hit_rate_graph": round(100 * sum(x["dep_recall_at10_graph"] > 0 for x in group) / len(group), 4),
        }
    metrics = {"sample_count": len(tasks), "metrics_by_arm": summaries,
               "paired_bootstrap_vs_aligncoder_code": bootstrap,
               "retrieval_dependency_metrics": retrieval_summary,
               "prompt_token_means": {arm: round(sum(r["prompt_tokens"] for r in prediction_rows if r["arm"] == arm) /
                                                  max(1, sum(r["arm"] == arm for r in prediction_rows)), 2)
                                      for arm in arms},
               "selection": selection}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    write_jsonl_gz(OUT_DIR / "predictions.jsonl.gz", prediction_rows)
    write_jsonl_gz(OUT_DIR / "retrieval_trace.jsonl.gz", retrieval_traces)
    write_jsonl_gz(OUT_DIR / "prompts.jsonl.gz", prompt_rows)
    from importlib.metadata import version as package_version
    config = {"retriever": ALIGN_RETRIEVER, "retriever_revision": ALIGN_REVISION,
              "generator": GENERATOR, "generator_revision": GENERATOR_REVISION,
              "chunking": "AlignCoder default mini-block (blank-line grouping, <=15 nonempty lines)",
              "bm25_topk": BM25_TOPK, "retriever_topk": RETRIEVER_TOPK,
              "generator_crossfile_tokens": CROSSFILE_TOKENS, "generator_context_tokens": GENERATOR_CONTEXT,
              "generator_max_new_tokens": GENERATOR_MAX_NEW, "query_enhancement": "AlignCoder-style shared 8 samples, T=.8, top_p=.95",
              "graph": "prefix-only import-to-definition edges; score=cosine+0.12*edge-coverage",
              "arms": list(arms), "seed": SEED, "device": gpu_name,
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "vllm_version": package_version("vllm"),
              "vllm_gpu_memory_utilization": 0.78, "vllm_max_num_seqs": 64,
              "vllm_max_num_batched_tokens": 8192}
    (OUT_DIR / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    archive_path = OUT_DIR / "graph_dependency_pilot_results.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in ("metrics.json", "predictions.jsonl.gz", "retrieval_trace.jsonl.gz",
                     "prompts.jsonl.gz", "run_config.json"):
            archive.add(OUT_DIR / name, arcname=name)
    print(json.dumps({"status": "complete", "results": str(archive_path),
                      "metrics": summaries, "retrieval": retrieval_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
