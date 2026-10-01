"""Quality filtering and leakage-aware repository indexing."""

from collections import Counter, defaultdict
import hashlib
from pathlib import Path

from src.data.audit_completion_data import source_hash
from src.data.raw_groups import groups
from src.data.ast_training_data import (DataConfig, is_test_source, key,
                                        valid_source)

SOURCE_SHA256 = {
    "python": "c7ead86805619020ca881be05c15896bce6cc75df6496c54dc7c0014511bef61",
    "java": "1abc3546f19d83c2461a3683a6c06e1d2258a1d544700b6507b0c8b6d8bb59f7",
}


def digest_file(path):
    import hashlib
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def clean_sources(group, language, config):
    files, hashes = {}, set()
    for path, code in group:
        if not path or not code or not valid_source(path, code, language, config):
            continue
        path = key(path)
        digest = source_hash(code)
        if path in files or digest in hashes:
            continue
        files[path] = code
        hashes.add(digest)
    return files


def code_lines(code):
    return sum(bool(line.strip()) and not line.lstrip().startswith(
        ("#", "//", "/*", "*", "*/")) for line in code.splitlines())


def substantive(code, config):
    return code_lines(code) >= config.min_file_code_lines


def _assign_splits(repos):
    parent = list(range(len(repos)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, repo in enumerate(repos):
        for digest in repo["split_hashes"]:
            if digest in seen:
                parent[root(index)] = root(seen[digest])
            else:
                seen[digest] = index
    components = defaultdict(list)
    for index in range(len(repos)):
        components[root(index)].append(index)
    conflicts = 0
    for members in components.values():
        component_id = min(repos[index]["uid"] for index in members)
        # Stable 15% validation split at repository-component level.
        split = "valid" if int(component_id[:8], 16) / 2**32 < 0.15 else "train"
        for index in members:
            repos[index]["split"] = split
            repos[index]["split_group"] = component_id
    return conflicts


def index_repositories(root, excluded, config):
    root = Path(root)
    repos, stats = [], Counter()
    for language in ("python", "java"):
        source = root / "github_repos" / language / "train.parquet"
        if digest_file(source) != SOURCE_SHA256[language]:
            raise ValueError(f"Unrecognized source revision: {source}")
        for repo_id, group in enumerate(groups(source)):
            stats[f"{language}:raw_repositories"] += 1
            files = clean_sources(group, language, config)
            if len(files) < config.min_files:
                stats[f"{language}:fewer_than_{config.min_files}_unique_code_files"] += 1
                continue
            if len(files) > config.max_files:
                stats[f"{language}:more_than_{config.max_files}_unique_code_files"] += 1
                continue
            substantive_count = sum(substantive(code, config) for code in files.values())
            if substantive_count < config.min_substantive_files:
                stats[f"{language}:too_few_substantive_files"] += 1
                continue
            production_count = sum(
                not is_test_source(path) and substantive(code, config)
                for path, code in files.items())
            if production_count < config.min_production_files:
                stats[f"{language}:too_few_production_files"] += 1
                continue
            total_lines = sum(code_lines(code) for code in files.values())
            if total_lines < config.min_repo_code_lines:
                stats[f"{language}:too_few_repo_code_lines"] += 1
                continue
            if total_lines > config.max_repo_code_lines:
                stats[f"{language}:too_many_repo_code_lines"] += 1
                continue
            all_hashes = {source_hash(code) for _, code in group if code}
            if all_hashes.intersection(excluded):
                stats[f"{language}:benchmark_exact_overlap"] += 1
                continue
            # Include every substantive file in component assignment. Hashing
            # only files >=200 characters allowed small shared files to cross
            # the train/validation boundary.
            split_hashes = sorted({source_hash(code) for code in files.values()
                                   if substantive(code, config)})
            uid = hashlib.sha256(
                (language + "\n" + "\n".join(sorted(all_hashes))).encode()
            ).hexdigest()
            repos.append({"language": language, "repo_id": repo_id, "files": files,
                          "uid": uid, "split_hashes": split_hashes,
                          "quality": {"file_count": len(files),
                                      "substantive_file_count": substantive_count,
                                      "production_file_count": production_count,
                                      "test_file_count": sum(is_test_source(path)
                                                              for path in files),
                                      "repo_code_lines": total_lines}})
        print("Index", language, dict(stats), flush=True)
    stats["shared_file_split_conflicts"] = _assign_splits(repos)
    return [repo for repo in repos if repo["split"] != "exclude_split_conflict"], stats
