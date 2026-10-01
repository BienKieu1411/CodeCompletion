"""Repository-level AST pool and per-epoch stochastic task sampling.

The expensive operations (parsing, AST chunking, dependency evidence) happen
once while building the pool.  A training epoch then samples 2,000 distinct
repositories, chooses an eligible target file/span per repository, and mines a
fresh BM25 candidate slate from the visible prefix.  No right context or gold
snippet is used in retrieval input.
"""

from collections import defaultdict
import hashlib
import random

from src.data.ast_training_data import (DataConfig, SourceFile, dependency_evidence,
                                         encoder_text, mine_candidates)
from src.data.code_input_cleanup import clean_chunk
from src.data.cross_file_budget import select_cross_file_context
from src.data.left_context import pack_left_context

REPO_POOL_SCHEMA = "ast_repo_pool_v3"
TASK_SCHEMA = "ast_completion_task_v3"


def _target_cap(cuts, limit):
    """Keep a balanced, deterministic subset without turning one file into a label flood."""
    buckets = defaultdict(list)
    for cut in cuts:
        buckets[cut["kind"]].append(cut)
    for values in buckets.values():
        values.sort(key=lambda item: (item["start"], item["end"], item["kind"]))
    chosen = []
    kinds = ("member_suffix", "line", "api_statement")
    while len(chosen) < limit and any(buckets.values()):
        for kind in kinds:
            if buckets[kind] and len(chosen) < limit:
                chosen.append(buckets[kind].pop(0))
    return chosen


def _file_record(file):
    return {"path": file.path, "code": file.code, "source_sha256": file.digest}


def build_repo_pool_record(repo, generator, retriever, config=DataConfig()):
    """Build one self-contained repository record or an explicit rejection."""
    parsed = {
        path: SourceFile(path, code, repo["language"], generator, retriever, config)
        for path, code in repo["files"].items()
    }
    parsed = {path: file for path, file in parsed.items()
              if not file.tree.root_node.has_error}
    if len(parsed) < config.min_files:
        return {"status": "too_few_parseable_files", "payload": None}

    # The first file is the repository's anchor, matching the established
    # AlignCoder/RLCoder target-file convention. Targets are sampled only from
    # the remaining files, so the model cannot memorize the anchor as target.
    ordered_paths = list(parsed)
    root_path = ordered_paths[0]
    symbols = [symbol for file in parsed.values() for symbol in file.symbols()]
    all_chunks = []
    for file in parsed.values():
        for chunk in file.chunks():
            model = clean_chunk(chunk, file.code, repo["language"])
            model_text = model["text"]
            # Cleanup is input-only; keep raw offsets/text and the exact gold
            # source untouched. Fall back to the raw AST span if cleanup ever
            # makes tokenization exceed the established generator cap.
            if len(generator.encode(model_text, add_special_tokens=False)) > config.chunk_tokens:
                model_text = chunk["text"]
                model = {"text": model_text, "diagnostics": {"raw_fallback_token_cap": 1},
                         "version": model.get("version")}
            enriched = {**chunk, "model_text": model_text,
                        "cleanup_diagnostics": model["diagnostics"],
                        "retrieval_text": encoder_text(
                            retriever, config.retriever_tokens,
                            "\n".join(h["text"] for h in chunk["scope_headers"]), model_text)}
            all_chunks.append(enriched)
    chunk_files = {chunk["path"] for chunk in all_chunks}
    targets = []
    target_file_counts = {}

    for path in ordered_paths[1:]:
        file = parsed[path]
        if sum(bool(line.strip()) for line in file.code.splitlines()) < config.min_file_code_lines:
            continue
        cuts = []
        for cut in file.target_cuts():
            evidence = dependency_evidence(file, cut, parsed, symbols)
            cuts.append({**cut, "path": path, "dependent": bool(evidence),
                         "dependency_evidence": evidence})
        cuts = _target_cap(cuts, config.max_target_spans_per_file)
        target_file_counts[path] = len(cuts)
        targets.extend(cuts)

    # A target is useful only when at least two other files can contribute
    # candidates. This prevents single-file or parse-fragment repositories from
    # entering the training pool merely because they contain a valid AST cut.
    usable_targets = [target for target in targets
                      if len({chunk["path"] for chunk in all_chunks
                              if chunk["path"] != target["path"]}) >= 2]
    usable_targets = _target_cap(usable_targets, config.max_target_spans_per_repo)
    if not usable_targets:
        return {"status": "no_usable_target_or_cross_file_pool", "payload": None}
    if len(chunk_files) < config.min_files:
        return {"status": "too_few_chunked_files", "payload": None}

    uid = repo["uid"]
    payload = {
        "schema": REPO_POOL_SCHEMA,
        "repo_uid": uid,
        "language": repo["language"],
        "repo_id": repo["repo_id"],
        "split": repo["split"],
        "root_path": root_path,
        "files": [_file_record(parsed[path]) for path in ordered_paths],
        "chunks": all_chunks,
        "target_pool": usable_targets,
        "quality": repo.get("quality", {}),
        "stats": {
            "parseable_files": len(parsed),
            "chunked_files": len(chunk_files),
            "chunk_count": len(all_chunks),
            "target_count": len(usable_targets),
            "dependent_target_count": sum(t["dependent"] for t in usable_targets),
            "target_files": len({t["path"] for t in usable_targets}),
        },
    }
    return {"status": "kept", "payload": payload}


def _choose_target(payload, rng, config):
    targets = payload["target_pool"]
    dependent = [target for target in targets if target["dependent"]]
    controls = [target for target in targets if not target["dependent"]]
    if dependent and (not controls or rng.random() < config.dependent_target_probability):
        pool = dependent
    else:
        pool = controls or dependent
    # First sample a file, then a cut. This avoids large files monopolising the
    # epoch simply because they have many AST nodes.
    by_file = defaultdict(list)
    for target in pool:
        by_file[target["path"]].append(target)
    file_path = rng.choice(sorted(by_file))
    candidates = by_file[file_path]
    kind = rng.choice(sorted({target["kind"] for target in candidates}))
    return rng.choice([target for target in candidates if target["kind"] == kind])


def sample_repo_task(payload, generator, retriever, rng, config=DataConfig()):
    """Sample one target and produce a fresh task row from a repository pool."""
    files = {item["path"]: item for item in payload["files"]}
    for _ in range(config.max_target_attempts):
        target = _choose_target(payload, rng, config)
        file = files[target["path"]]
        context = pack_left_context(
            file["code"], target["start"], payload["language"], generator,
            max_tokens=config.max_left_context_tokens,
            tail_tokens=config.left_tail_tokens,
            hint_tokens=config.left_hint_tokens)
        left = context["text"]
        target_code = file["code"].encode("utf-8")[target["start"]:target["end"]].decode("utf-8")
        candidates = mine_candidates(left, target["path"], payload["chunks"], retriever, config)
        if len(candidates) >= 2 and len({candidate["path"] for candidate in candidates}) >= 2:
            cross_file = select_cross_file_context(
                candidates, generator, payload["language"],
                budget_tokens=config.cross_file_budget_tokens,
                max_snippets=config.max_cross_file_snippets)
            task_id = hashlib.sha256(
                f"{payload['repo_uid']}\n{target['path']}\n{target['start']}\n{target['end']}\n{rng.random()}".encode()
            ).hexdigest()[:24]
            return {
                "schema": TASK_SCHEMA,
                "task_id": task_id,
                "repo_uid": payload["repo_uid"],
                "language": payload["language"],
                "repo_id": payload["repo_id"],
                "file_path": target["path"],
                "left_context": left,
                "left_context_metadata": context,
                "target_code": target_code,
                "target_kind": target["kind"],
                "target_start": target["start"],
                "target_end": target["end"],
                "target_token_count": len(generator.encode(target_code, add_special_tokens=False)),
                "source_sha256": file["source_sha256"],
                "root_path": payload["root_path"],
                "candidates": candidates,
                "cross_file_context": cross_file["items"],
                "cross_file_metadata": {key: value for key, value in cross_file.items()
                                         if key != "items"},
                "label_metadata": {
                    "dependent_target": target["dependent"],
                    "dependency_evidence": target["dependency_evidence"],
                    "utility_labels": None,
                },
            }
    return None


def sample_repo_epoch(repo_payloads, generator, retriever, epoch, config=DataConfig(), seed=123):
    """Select up to 2,000 unique repositories and one stochastic task/repo."""
    train = [payload for payload in repo_payloads if payload["split"] == "train"]
    rng = random.Random(seed + epoch * 1_000_003)
    selected = train if len(train) <= config.repos_per_epoch else rng.sample(train, config.repos_per_epoch)
    rows = []
    for payload in selected:
        task = sample_repo_task(payload, generator, retriever, rng, config)
        if task is not None:
            rows.append(task)
    return rows
