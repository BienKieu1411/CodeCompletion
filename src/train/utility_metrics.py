"""Versioned training metrics; original labels never change.

Line/suffix utility: first nonempty generated line, fuzzywuzzy ratio / 100.
API utility: RepoEval line normalization + normalized Levenshtein distance.
These are explicit training adapters, not a universal official CCEval metric.
"""
from fuzzywuzzy import fuzz

VERSION = "cur_line_fuzz_api_repoeval_v1"


def levenshtein(a, b):
    if len(a) > len(b):
        a, b = b, a
    row = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        next_row = [i]
        for j, ca in enumerate(a, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j-1] + (ca != cb)))
        row = next_row
    return row[-1]


def completion_score(prediction, target, kind):
    if kind not in {"line", "member_suffix", "api_statement"}:
        raise ValueError(f"Unknown completion target kind: {kind}")
    lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    gold = [line.strip() for line in target.splitlines() if line.strip()]
    if not gold:
        raise ValueError("Empty completion target")
    if kind == "api_statement":
        prediction, target = "\n".join(lines[:len(gold)]), "\n".join(gold)
        es = 1 - levenshtein(prediction, target) / max(len(prediction), len(target), 1)
    else:
        if len(gold) != 1:
            raise ValueError("Line/suffix target must contain exactly one nonempty line")
        prediction, target = (lines[0] if lines else ""), target.strip()
        es = fuzz.ratio(prediction, target) / 100.0
    return {"es": es, "em": float(prediction == target)}
