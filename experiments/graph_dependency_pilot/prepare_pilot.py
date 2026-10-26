#!/usr/bin/env python3
"""Select leakage-safe, high-confidence cross-file dependency tasks.

Selection reads only each benchmark row's left_context, target path, and
repository candidate files. Ground truth/right context are copied only after
eligibility is decided, for evaluation on the remote runner.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import tarfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_SETS = {
    "cceval_python": ["cceval/python/test.parquet"],
    "cceval_java": ["cceval/java/test.parquet"],
    "repoeval_line": [
        "repoeval/line_level/test_0.parquet",
        "repoeval/line_level/test_1.parquet",
    ],
    "repoeval_api": [
        "repoeval/api_level/test_0.parquet",
        "repoeval/api_level/test_1.parquet",
    ],
}

PY_FROM_IMPORT = re.compile(
    r"(?m)^\s*from\s+([.\w]+)\s+import\s+(\([^)]*\)|[^\n]+)"
)
PY_IMPORT = re.compile(r"(?m)^\s*import\s+([^\n#]+)")
PY_DECL = re.compile(r"(?m)^\s*(?:async\s+)?(?:class|def)\s+([A-Za-z_]\w*)")
JAVA_IMPORT = re.compile(r"(?m)^\s*import\s+(?:static\s+)?([\w.$*]+)\s*;")
JAVA_DECL = re.compile(r"\b(?:class|interface|enum|record)\s+([A-Za-z_$][\w$]*)")
JAVA_METHOD = re.compile(r"(?m)^\s*(?:public|protected|private|static|final|abstract|synchronized|native|default|\s)+[\w$<>?,.\[\] ]+\s+([A-Za-z_$][\w$]*)\s*\(")
JAVA_PACKAGE = re.compile(r"(?m)^\s*package\s+([\w.]+)\s*;")


def _split_names(text: str) -> list[str]:
    text = text.strip().strip("()")
    names = []
    for item in text.replace("\n", " ").split(","):
        item = item.strip()
        if not item or item == "*":
            continue
        names.append(item.split(" as ", 1)[0].strip())
    return names


def imports_from_prefix(prefix: str, language: str) -> list[dict[str, str]]:
    """Extract only explicit import declarations visible before the cursor."""
    found: list[dict[str, str]] = []
    if language == "python":
        for match in PY_FROM_IMPORT.finditer(prefix):
            module = match.group(1).strip(".")
            for name in _split_names(match.group(2)):
                found.append({"module": module, "name": name, "kind": "from_import"})
        for match in PY_IMPORT.finditer(prefix):
            for item in match.group(1).split(","):
                item = item.strip()
                if not item:
                    continue
                module = item.split(" as ", 1)[0].strip()
                found.append({"module": module, "name": module.split(".")[0], "kind": "import"})
    else:
        for match in JAVA_IMPORT.finditer(prefix):
            full_name = match.group(1)
            if full_name.endswith(".*"):
                found.append({"module": full_name[:-2], "name": "*", "kind": "package_import"})
            else:
                module, _, name = full_name.rpartition(".")
                found.append({"module": module, "name": name or full_name, "kind": "class_import"})
    # Stable deduplication keeps edge counts meaningful.
    unique = {}
    for item in found:
        key = (item["module"], item["name"], item["kind"])
        unique[key] = item
    return list(unique.values())


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./").lower()


def _module_path_match(path: str, module: str, language: str) -> bool:
    """Match an import module to a candidate path without using target text."""
    if not module:
        return False
    path_norm = _norm_path(path)
    parts = [part.lower() for part in module.split(".") if part]
    if not parts:
        return False
    path_parts = path_norm.split("/")
    if language == "python":
        if path_parts[-1].endswith(".py"):
            path_parts[-1] = path_parts[-1][:-3]
        if path_parts[-1] == "__init__":
            path_parts.pop()
    else:
        if path_parts[-1].endswith(".java"):
            path_parts[-1] = path_parts[-1][:-5]
    # Match a module file/package exactly (allowing source roots such as
    # src/, lib/, or main/), not every descendant of a broad package name.
    return len(path_parts) >= len(parts) and path_parts[-len(parts):] == parts


def dependency_edges(prefix: str, target_path: str, files: list[dict[str, Any]], language: str) -> tuple[list[dict[str, str]], int]:
    imports = imports_from_prefix(prefix, language)
    target_norm = _norm_path(target_path)
    edges: list[dict[str, str]] = []
    resolved_entities: set[tuple[str, str]] = set()
    for candidate in files:
        path = str(candidate.get("path") or "")
        code = str(candidate.get("text") or "")
        if not path or _norm_path(path) == target_norm:
            continue
        if language == "python":
            declarations = set(PY_DECL.findall(code))
            for imp in imports:
                name = imp["name"]
                path_hit = _module_path_match(path, imp["module"], language)
                symbol_hit = bool(name and name != "*" and name in declarations)
                module_import = imp["kind"] == "import" and path_hit
                submodule_import = (imp["kind"] == "from_import" and
                                    _module_path_match(path, f"{imp['module']}.{name}", language))
                reexport_hit = (path.endswith("/__init__.py") and name and
                                bool(re.search(rf"\b{re.escape(name)}\b", code)))
                if not (symbol_hit or module_import or submodule_import or reexport_hit):
                    continue
                relation = ("imports_definition" if symbol_hit else
                            "imports_reexport" if reexport_hit else "imports_module")
                edges.append({"import": f"{imp['module']}.{name}".strip("."), "path": path, "relation": relation, "symbol": name})
                resolved_entities.add((imp["module"], name))
        else:
            declarations = set(JAVA_DECL.findall(code)) | set(JAVA_METHOD.findall(code))
            package_match = JAVA_PACKAGE.search(code)
            package = package_match.group(1) if package_match else ""
            for imp in imports:
                name = imp["name"]
                path_hit = _module_path_match(path, imp["module"], language)
                symbol_hit = name != "*" and name in declarations
                package_hit = bool(package and package == imp["module"] and symbol_hit)
                exact_file_hit = bool(name != "*" and path.lower().endswith("/" + name.lower() + ".java") and path_hit)
                if package_hit or (symbol_hit and (exact_file_hit or not imp["module"])):
                    edges.append({"import": f"{imp['module']}.{name}".strip("."), "path": path,
                                  "relation": "package_class" if package_hit else "imports_definition", "symbol": name})
                    resolved_entities.add((imp["module"], name))
                elif name == "*" and package == imp["module"]:
                    for symbol in sorted(declarations):
                        edges.append({"import": f"{imp['module']}.*", "path": path,
                                      "relation": "package_member", "symbol": symbol})
                    if declarations:
                        resolved_entities.add((imp["module"], "*"))
    # Multiple files can re-export the same symbol; count one source dependency.
    dedup = {}
    for edge in edges:
        dedup[(edge["import"], edge["path"], edge["relation"], edge["symbol"])] = edge
    return list(dedup.values()), len(resolved_entities)


def _pool_id(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(_file_id(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def _file_id(item: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(item.get("path") or "").encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(str(item.get("text") or "").encode("utf-8", errors="replace"))
    return digest.hexdigest()[:24]


def _iter_rows(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4, columns=[
            "task_id", "path", "left_context", "right_context", "groundtruth", "crossfile_context"]):
        yield from batch.to_pylist()


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare(data_root: Path, output_dir: Path, per_benchmark: int, seed: int,
            max_scan: int, min_pool_files: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    pools: dict[str, dict[str, Any]] = {}
    file_records: dict[str, dict[str, str]] = {}
    report: dict[str, Any] = {"seed": seed, "requested_per_benchmark": per_benchmark,
                              "max_scan_per_benchmark": max_scan, "benchmarks": {}}

    for benchmark, relpaths in DEFAULT_SETS.items():
        eligible: list[dict[str, Any]] = []
        scanned = 0
        for relpath in relpaths:
            parquet_path = data_root / relpath
            for row in _iter_rows(parquet_path):
                scanned += 1
                if scanned > max_scan:
                    break
                pool_files = row.get("crossfile_context") or []
                if len(pool_files) < min_pool_files:
                    continue
                language = "java" if benchmark == "cceval_java" else "python"
                left = str(row.get("left_context") or "")
                edges, resolved = dependency_edges(left, str(row.get("path") or ""), pool_files, language)
                if resolved < 1 or not edges:
                    continue
                file_ids = [_file_id(item) for item in pool_files]
                for file_id, item in zip(file_ids, pool_files):
                    file_records.setdefault(file_id, {"file_id": file_id,
                                                       "path": str(item.get("path") or ""),
                                                       "text": str(item.get("text") or "")})
                pool_id = _pool_id(pool_files)
                pools.setdefault(pool_id, {"pool_id": pool_id, "file_ids": file_ids})
                eligible.append({
                    "task_id": str(row["task_id"]), "benchmark": benchmark,
                    "language": language, "path": str(row.get("path") or ""),
                    "left_context": left, "right_context": str(row.get("right_context") or ""),
                    "groundtruth": str(row.get("groundtruth") or ""),
                    "pool_id": pool_id, "dependency_edges": edges,
                    "resolved_dependency_count": resolved,
                    "candidate_file_count": len(pool_files),
                })
            if scanned >= max_scan:
                break
            # Avoid an order-biased subset: continue until there is a buffer
            # of eligible rows, but cap the scan for a quick pilot.
            if len(eligible) >= max(per_benchmark * 2, per_benchmark + 40):
                break

        if len(eligible) < per_benchmark:
            raise RuntimeError(
                f"{benchmark}: only {len(eligible)} high-confidence dependency tasks "
                f"among {min(scanned, max_scan)} scanned; need {per_benchmark}. "
                "Increase --max-scan or lower --min-pool-files after inspecting the report."
            )
        # Uniform sample after filtering, so benchmark order cannot select the
        # evaluation examples. Store the exact decision evidence for audit.
        chosen = rng.sample(eligible, per_benchmark)
        selected.extend(chosen)
        report["benchmarks"][benchmark] = {
            "scanned": min(scanned, max_scan), "eligible": len(eligible),
            "selected": len(chosen), "mean_resolved_dependencies": round(
                sum(x["resolved_dependency_count"] for x in chosen) / len(chosen), 3),
            "mean_candidate_files": round(sum(x["candidate_file_count"] for x in chosen) / len(chosen), 3),
            "task_ids": [x["task_id"] for x in chosen],
        }
        print(f"{benchmark}: scanned={min(scanned, max_scan)} eligible={len(eligible)} selected={len(chosen)}", flush=True)

    rng.shuffle(selected)
    selected_pool_ids = {task["pool_id"] for task in selected}
    pools = {pool_id: pool for pool_id, pool in pools.items() if pool_id in selected_pool_ids}
    selected_file_ids = {file_id for pool in pools.values() for file_id in pool["file_ids"]}
    file_records = {file_id: item for file_id, item in file_records.items() if file_id in selected_file_ids}
    _write_jsonl_gz(output_dir / "tasks.jsonl.gz", selected)
    _write_jsonl_gz(output_dir / "pools.jsonl.gz", list(pools.values()))
    _write_jsonl_gz(output_dir / "files.jsonl.gz", list(file_records.values()))
    report["total_tasks"] = len(selected)
    report["unique_pools"] = len(pools)
    report["unique_source_files"] = len(file_records)
    report["selection_contract"] = (
        "Eligibility uses only left_context, target path, and crossfile repository sources. "
        "Groundtruth/right_context are carried solely for final scoring."
    )
    (output_dir / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    archive_path = output_dir / "pilot_bundle.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in ("tasks.jsonl.gz", "pools.jsonl.gz", "files.jsonl.gz", "selection.json"):
            archive.add(output_dir / name, arcname=name)
        project_root = Path(__file__).resolve().parents[2]
        for language in ("python", "java"):
            parser = project_root / "RepoClone/AlignCoder/utils/build" / f"{language}-lang-parser.so"
            archive.add(parser, arcname=f"parsers/{language}-lang-parser.so")
        archive.add(project_root / "RepoClone/AlignCoder/api.py", arcname="aligncoder/api.py")
        archive.add(project_root / "RepoClone/AlignCoder/utils/keywords/java.txt",
                    arcname="aligncoder/utils/keywords/java.txt")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("dataset/data4aligncoder"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-benchmark", type=int, default=64)
    parser.add_argument("--max-scan", type=int, default=400)
    parser.add_argument("--min-pool-files", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    report = prepare(args.data_root, args.output_dir, args.per_benchmark,
                     args.seed, args.max_scan, args.min_pool_files)
    print(json.dumps({k: v for k, v in report.items() if k != "benchmarks"}, indent=2))
    print(f"Bundle: {args.output_dir / 'pilot_bundle.tar.gz'}")


if __name__ == "__main__":
    main()
