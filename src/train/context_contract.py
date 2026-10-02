"""One deterministic input renderer for CUR annotations and selection."""
from dataclasses import asdict, dataclass
import hashlib
import json

from src.data.left_context import pack_left_context

VERSION = "cur_prefix_v1"


@dataclass(frozen=True)
class ContextConfig:
    prompt_tokens: int = 3072
    cross_file_tokens: int = 2344
    max_snippets: int = 10
    output_tokens: int = 192

    def __post_init__(self):
        if min(asdict(self).values()) <= 0 or self.prompt_tokens <= self.cross_file_tokens + 64:
            raise ValueError("Positive budgets and >64 tokens for prefix/wrappers required")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class ContextRenderer:
    """Gold-free renderer: fixed prefix across every intervention of a query.

    Complete snippets are admitted by actual joined token cost. Infeasible
    sets raise ValueError; no silent truncation or budget-driven set mutation.
    """
    def __init__(self, task, tokenizer, config=ContextConfig()):
        self.tokenizer, self.config = tokenizer, config
        self.candidates = task["candidates"]
        self.language = task["language"]
        marker = "#" if self.language == "python" else "//"
        self.header = f"\n\n{marker} File: {task['file_path']}\n"
        self.pieces = []
        for candidate in self.candidates:
            if candidate["path"] == task["file_path"]:
                raise ValueError("Current-file candidate would expose hidden target context")
            body = candidate.get("model_text", candidate["text"])
            self.pieces.append(f"{marker} File: {candidate['path']}\n{body}")
        self.order = sorted(range(len(self.candidates)), key=lambda i: (
            self.candidates[i]["path"], self.candidates[i]["start"],
            self.candidates[i]["end"], self.candidates[i].get("source_sha256", ""), i))
        remaining = config.prompt_tokens - config.cross_file_tokens - self.count(self.header) - 32
        if remaining < 1:
            raise ValueError("Current-file header exceeds reserved prefix space")
        prefix = task["left_context"]
        packed = pack_left_context(prefix, len(prefix.encode()), self.language, tokenizer,
                                   max_tokens=remaining, tail_tokens=max(1, remaining * 3 // 4),
                                   hint_tokens=max(1, remaining // 4))
        self.left_context = packed["text"]
        self.prefix_metadata = packed
        self._canonical_cache = {(): ()}
        self._cross_cache = {(): ("", 0)}
        self._prompt_count_cache = {}
        self._feasible_cache = {}
        self.render(())

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_many(self, texts):
        """Batch-count exact tokenizer lengths, with a small-tokenizer fallback."""
        texts = list(texts)
        if not texts:
            return []
        if callable(self.tokenizer):
            try:
                encoded = self.tokenizer(
                    texts, add_special_tokens=False, padding=False, truncation=False,
                    return_attention_mask=False)
                return [len(row) for row in encoded["input_ids"]]
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        return [self.count(text) for text in texts]

    def canonical(self, indices):
        values = tuple(indices)
        if len(values) != len(set(values)) or any(i < 0 or i >= len(self.candidates) for i in values):
            raise ValueError("Invalid/duplicate candidate indices")
        cache_key = tuple(sorted(values))
        cached = self._canonical_cache.get(cache_key)
        if cached is None:
            selected = set(cache_key)
            cached = tuple(i for i in self.order if i in selected)
            self._canonical_cache[cache_key] = cached
        return cached

    def _ensure_cross_counts(self, keys):
        missing = [key for key in dict.fromkeys(keys) if key not in self._cross_cache]
        if not missing:
            return
        texts = ["\n\n".join(self.pieces[i] for i in key) for key in missing]
        costs = self.count_many(texts)
        self._cross_cache.update(zip(missing, zip(texts, costs)))

    def _ensure_prompt_counts(self, keys):
        keys = list(dict.fromkeys(keys))
        self._ensure_cross_counts(keys)
        missing = [key for key in keys if key not in self._prompt_count_cache]
        if not missing:
            return
        prompts = [self._cross_cache[key][0] + self.header + self.left_context for key in missing]
        self._prompt_count_cache.update(zip(missing, self.count_many(prompts)))

    def cross_file(self, indices):
        key = self.canonical(indices)
        self._ensure_cross_counts([key])
        return self._cross_cache[key]

    def cost_many(self, index_sets):
        keys = [self.canonical(indices) for indices in index_sets]
        self._ensure_cross_counts(keys)
        return [self._cross_cache[key][1] for key in keys]

    def render(self, indices):
        key = self.canonical(indices)
        self._ensure_cross_counts([key])
        text, cost = self._cross_cache[key]
        if len(key) > self.config.max_snippets or cost > self.config.cross_file_tokens:
            raise ValueError("Set exceeds cross-file budget or snippet limit")
        self._ensure_prompt_counts([key])
        prompt = text + self.header + self.left_context
        if self._prompt_count_cache[key] > self.config.prompt_tokens:
            raise ValueError("Rendered prompt exceeds generator input budget")
        return prompt, cost

    def feasible_many(self, index_sets):
        keys = [self.canonical(indices) for indices in index_sets]
        self._ensure_cross_counts(keys)
        prompt_keys = [key for key in dict.fromkeys(keys)
                       if len(key) <= self.config.max_snippets
                       and self._cross_cache[key][1] <= self.config.cross_file_tokens]
        self._ensure_prompt_counts(prompt_keys)
        result = []
        for key in keys:
            if key in self._feasible_cache:
                result.append(self._feasible_cache[key])
                continue
            feasible = (len(key) <= self.config.max_snippets
                        and self._cross_cache[key][1] <= self.config.cross_file_tokens
                        and self._prompt_count_cache[key] <= self.config.prompt_tokens)
            self._feasible_cache[key] = feasible
            result.append(feasible)
        return result

    def feasible(self, indices):
        return self.feasible_many([indices])[0]

    def contract(self):
        return {"version": VERSION, **asdict(self.config)}
