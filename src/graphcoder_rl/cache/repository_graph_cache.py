"""
Repository graph cache with incremental artifact reuse.

Caches per-file parse artifacts keyed by content hash.
Changed files are re-parsed; unchanged files are reused.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Callable, Dict, List, Tuple


class RepositoryGraphCache:
    def __init__(self, cache_dir: str = "cache/graph"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf8")).hexdigest()

    @staticmethod
    def repo_id_from_files(file_dict: Dict[str, str]) -> str:
        basis = "\n".join(sorted(file_dict.keys()))
        return hashlib.sha256(basis.encode("utf8")).hexdigest()[:16]

    def _cache_path(self, repo_id: str) -> str:
        return os.path.join(self.cache_dir, f"{repo_id}.pkl")

    def load(self, repo_id: str) -> Dict[str, Any]:
        path = self._cache_path(repo_id)
        if not os.path.exists(path):
            return {"file_hashes": {}, "artifacts": {}}
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                return {"file_hashes": {}, "artifacts": {}}
            return {
                "file_hashes": data.get("file_hashes", {}),
                "artifacts": data.get("artifacts", {}),
            }
        except Exception:
            return {"file_hashes": {}, "artifacts": {}}

    def save(self, repo_id: str, file_hashes: Dict[str, str], artifacts: Dict[str, Any]) -> None:
        path = self._cache_path(repo_id)
        payload = {
            "file_hashes": file_hashes,
            "artifacts": artifacts,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def get_or_update_artifacts(
        self,
        repo_id: str,
        file_dict: Dict[str, str],
        artifact_builder: Callable[[str, str], Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Returns:
            artifacts: filepath -> artifact
            changed_files: files that were rebuilt this round
        """
        cache = self.load(repo_id)
        old_hashes: Dict[str, str] = cache.get("file_hashes", {})
        old_artifacts: Dict[str, Any] = cache.get("artifacts", {})

        new_hashes: Dict[str, str] = {}
        artifacts: Dict[str, Any] = {}
        changed_files: List[str] = []

        for file_path, content in file_dict.items():
            h = self._hash_text(content)
            new_hashes[file_path] = h
            if old_hashes.get(file_path) == h and file_path in old_artifacts:
                artifacts[file_path] = old_artifacts[file_path]
            else:
                artifacts[file_path] = artifact_builder(file_path, content)
                changed_files.append(file_path)

        # Drop removed files automatically by writing only new files.
        self.save(repo_id, new_hashes, artifacts)
        return artifacts, changed_files
