#!/usr/bin/env python3
"""E10-split: fixed repo-disjoint 2-fold assignment for the 8004 CADRE/E9 tasks (DEV_SPLIT_PLAN.md, Track A).

Unit = repository (tasks of one repo share code; a task-level split leaks). Written once to dev_folds.json and
never regenerated: its sha256 goes into HANDOFF.
    RepoEval (line + API share 8 repos): repos sorted by sha256(SEED + repo), alternately assigned -> 4 / 4.
    CCEval py / java: repos sorted by sha256(SEED + repo), each put in the fold with fewer tasks so far
        (balanced on task count; ~470 / ~240 repos).
Repo of a CCEval task = metadata.repository of the official test.jsonl (AlignCoder dataset copy), joined on
task_id; RepoEval repo = task_id prefix.
usage: python dev_split.py [--limit N]   (limit only for a smoke run; writes dev_folds_smoke.json)
"""
import argparse
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = "e10-2026-10-28"
CCEVAL = os.path.join(HERE, "..", "..", "docs", "research", "audits", "aligncoder_dataset", "data", "cceval")
TASKS = os.path.join(HERE, "data_in", "e9data", "tasks.jsonl.gz")


def h(repo):
    return hashlib.sha256((SEED + repo).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cc_repo = {}
    for lang in ("python", "java"):
        for l in open(os.path.join(CCEVAL, lang, "test.jsonl"), encoding="utf-8"):
            m = json.loads(l)["metadata"]
            cc_repo[m["task_id"]] = m["repository"]
    tasks = [(r["benchmark"], r["task_id"]) for r in map(json.loads, gzip.open(TASKS, "rt", encoding="utf-8"))]
    if a.limit:
        tasks = tasks[:a.limit]
    repo = {}
    for bm, t in tasks:
        repo[(bm, t)] = cc_repo[t] if bm.startswith("cceval") else t.split("/")[0]
    fold_of = {}
    re_repos = sorted({repo[k] for k in tasks if k[0].startswith("repoeval")}, key=h)
    for i, r in enumerate(re_repos):
        fold_of[("repoeval", r)] = i % 2
    for bm in ("cceval_python", "cceval_java"):
        size = Counter(repo[k] for k in tasks if k[0] == bm)
        load = [0, 0]
        for r in sorted(size, key=h):
            f = 0 if load[0] <= load[1] else 1
            fold_of[(bm, r)] = f
            load[f] += size[r]
    out = {"seed": SEED, "tasks": {}}
    stats = defaultdict(lambda: [[0, 0], [set(), set()]])
    for bm, t in tasks:
        grp = "repoeval" if bm.startswith("repoeval") else bm
        f = fold_of[(grp, repo[(bm, t)])]
        out["tasks"][f"{bm}|{t}"] = {"repo": repo[(bm, t)], "fold": f}
        stats[bm][0][f] += 1
        stats[bm][1][f].add(repo[(bm, t)])
    out["stats"] = {bm: {"tasks": s[0], "repos": [len(s[1][0]), len(s[1][1])]} for bm, s in stats.items()}
    path = os.path.join(HERE, "dev_folds_smoke.json" if a.limit else "dev_folds.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0, sort_keys=True)
    print(json.dumps(out["stats"], indent=1))
    print("wrote", path, hashlib.sha256(open(path, "rb").read()).hexdigest()[:8])


if __name__ == "__main__":
    main()
