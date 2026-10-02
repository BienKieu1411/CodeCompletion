"""Target-only teacher-forced utility from the pinned local vLLM server.

Continuation IDs are encoded independently once and appended to each prompt's
IDs. This defines p(target_ids | prompt_ids), avoiding context-dependent BPE
boundary changes. It is not joint-string tokenization. No EOS is scored.
"""
import json
import math

from src.train.build_utility_bundles import CompletionOracle


class LikelihoodOracle(CompletionOracle):
    def score(self, prompts, targets, tokenizer, max_model_len=3264):
        if len(prompts) != len(targets):
            raise ValueError("One target per prompt required")
        encoded, boundaries, keys = [], [], []
        unique, target_cache = {}, {}
        for prompt, target in zip(prompts, targets):
            if target not in target_cache:
                target_cache[target] = tokenizer.encode(target, add_special_tokens=False)
            target_ids = target_cache[target]
            prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not prefix_ids or not target_ids:
                raise ValueError("Teacher forcing requires nonempty prefix and target tokens")
            ids = prefix_ids + target_ids
            if len(ids) + 1 > max_model_len:
                raise ValueError("Teacher-forcing sequence exceeds server context; never truncate target")
            key = (tuple(ids), len(prefix_ids))
            if key not in unique:
                unique[key] = len(encoded)
                encoded.append(ids)
                boundaries.append(len(prefix_ids))
            keys.append(unique[key])
        scores = []
        for start in range(0, len(encoded), self.batch_size):
            if self.check_continue is not None:
                self.check_continue()
            batch = encoded[start:start+self.batch_size]
            request = {"model": self.contract["model"], "prompt": batch,
                       "max_tokens": 1, "temperature": 0., "seed": 123,
                       "prompt_logprobs": 0, "add_special_tokens": False, "n": 1}
            response = self.session.post(self.base_url + "/v1/completions", json=request,
                                         timeout=(10, self.request_timeout))
            if not response.ok:
                raise RuntimeError(f"Likelihood HTTP {response.status_code}: {response.text[:600]}")
            choices = response.json().get("choices", [])
            by_index = {c["index"]: c for c in choices}
            if len(choices) != len(batch) or set(by_index) != set(range(len(batch))):
                raise ValueError("Likelihood response has missing/duplicate indices")
            for i, ids in enumerate(batch):
                positions = by_index[i].get("prompt_logprobs")
                if not isinstance(positions, list) or len(positions) != len(ids):
                    raise ValueError("Missing or misaligned prompt_logprobs; cannot score target")
                target_scores = []
                for pos in range(boundaries[start+i], len(ids)):
                    choices_at_token = positions[pos]
                    item = (choices_at_token.get(str(ids[pos]), choices_at_token.get(ids[pos]))
                            if isinstance(choices_at_token, dict) else None)
                    if not isinstance(item, dict) or "logprob" not in item:
                        raise ValueError("Actual target token is missing from prompt_logprobs")
                    value = float(item["logprob"])
                    if not math.isfinite(value) or value > 1e-5:
                        raise FloatingPointError("Invalid target log probability")
                    target_scores.append(value)
                scores.append(math.fsum(target_scores) / len(target_scores))
            print(json.dumps({"phase": "likelihood_scored", "completed": len(scores),
                              "contexts": len(encoded)}), flush=True)
        return [scores[index] for index in keys]
