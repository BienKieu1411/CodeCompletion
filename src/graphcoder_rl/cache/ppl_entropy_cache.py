"""
Offline PPL/entropy proxy cache for semantic boundary scoring.

Stores per-entity boundary scores so chunking does not re-score unchanged entities.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, List


class PPLEntropyCache:
    def __init__(self, cache_dir: str = "cache/ppl"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf8")).hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get_or_compute(
        self,
        key_text: str,
        lines: List[str],
        scorer: Callable[[str, str, str], float],
    ) -> List[float]:
        key = self._hash_text(key_text)
        path = self._path(key)

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                scores = payload.get("scores", [])
                if isinstance(scores, list) and len(scores) == len(lines):
                    return [float(x) for x in scores]
            except Exception:
                pass

        scores: List[float] = []
        for i in range(len(lines)):
            prev_line = lines[i - 1] if i > 0 else ""
            cur_line = lines[i]
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            scores.append(float(scorer(prev_line, cur_line, next_line)))

        with open(path, "w", encoding="utf-8") as f:
            json.dump({"scores": scores}, f)

        return scores
