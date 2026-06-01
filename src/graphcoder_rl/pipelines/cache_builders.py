from __future__ import annotations

import hashlib
import logging
from typing import Dict, List

from graphcoder_rl.data.repository_dataset_loader import DatasetLoader
from graphcoder_rl.retrieval.multi_hop_graph_retriever import MultiHopGraphRetriever

logger = logging.getLogger(__name__)


def _repo_key(file_dict: Dict[str, str]) -> str:
    basis = "\n".join(sorted(file_dict.keys()))
    return hashlib.sha256(basis.encode("utf8")).hexdigest()[:16]


def _collect_repo_crossfiles(dataset_loader: DatasetLoader, max_repos: int = 0) -> List[Dict[str, str]]:
    repos = dataset_loader.load_github_repos()
    repo_dicts: List[Dict[str, str]] = []
    for repo_files in repos:
        cross = {}
        for item in repo_files:
            path = item.get("path", "")
            content = item.get("content", "")
            if path:
                cross[path] = content
        if cross:
            repo_dicts.append(cross)
        if max_repos > 0 and len(repo_dicts) >= max_repos:
            break
    return repo_dicts


def build_graph_cache(dataset_path: str, graph_cache_dir: str, ppl_cache_dir: str, max_repos: int = 0) -> dict:
    dataset_loader = DatasetLoader(dataset_path=dataset_path, fixed_train=False)
    retriever = MultiHopGraphRetriever(
        use_graph_cache=True,
        graph_cache_dir=graph_cache_dir,
        use_ppl_entropy_cache=True,
        ppl_cache_dir=ppl_cache_dir,
    )
    repo_dicts = _collect_repo_crossfiles(dataset_loader, max_repos=max_repos)
    n_files = 0
    for repo in repo_dicts:
        retriever._build_graph(repo)
        n_files += len(repo)
        logger.info("Cached graph for repo %s (%d files)", _repo_key(repo), len(repo))
    return {"repos": len(repo_dicts), "files": n_files, "graph_cache_dir": graph_cache_dir}


def build_ppl_cache(dataset_path: str, ppl_cache_dir: str, max_repos: int = 0) -> dict:
    dataset_loader = DatasetLoader(dataset_path=dataset_path, fixed_train=False)
    retriever = MultiHopGraphRetriever(
        use_graph_cache=False,
        use_ppl_entropy_cache=True,
        ppl_cache_dir=ppl_cache_dir,
    )
    repo_dicts = _collect_repo_crossfiles(dataset_loader, max_repos=max_repos)
    n_files = 0
    for repo in repo_dicts:
        for path, content in repo.items():
            retriever._build_file_artifact(path, content)
            n_files += 1
        logger.info("Cached ppl-entropy for repo %s (%d files)", _repo_key(repo), len(repo))
    return {"repos": len(repo_dicts), "files": n_files, "ppl_cache_dir": ppl_cache_dir}
