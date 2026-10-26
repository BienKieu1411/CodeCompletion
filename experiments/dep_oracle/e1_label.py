#!/usr/bin/env python3
"""E1 step 1: dLL labels on TRAINING repos (Data4RLCoder, the RLCoder training corpus), same pool + prompt as eval.

Per example (one Python file of a Data4RLCoder repo, cut at a random point):
    cross pool  = related files -> AlignCoder-style mini_blocks -> BM25 top-50 (query = last 20 non-empty lines)
                  -> AlignRetriever dense top-10 (mimics the saved AlignCoder candidates used at eval)
    in-file     = hidden in-file chunks at budget B (joint_retrieval.hidden_infile_blocks), the 20 nearest the cursor
    dLL(c)      = log p(GT | c + path + in-file W(B)) - log p(GT | path + in-file W(B))   (chunk_pilot.label_prompts)
B = 1024 only (middle of the pilot budgets). Cuts mimic both benchmarks: 50% at a line start (RepoEval-like, GT = the
whole next line), 50% inside a line (CCEval-like, GT = rest of the line).

Leakage guard: repos sharing >= 3 file paths (depth >= 2) with the eval candidates are dropped (count is logged).
Train/dev split follows RLCoder: repo groups [:2000] -> train, [2000:2200] -> dev.

usage:
    python e1_label.py --data_root Data4RLCoder --split train --n 4000 --shard 0 --nshards 4 --out e1_labels
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk_pilot as C  # noqa: E402
import dep_oracle as D  # noqa: E402
import joint_retrieval as J  # noqa: E402
import likelihood_probe as P  # noqa: E402

BUDGET = 1024
N_CROSS, N_BM25, N_INFILE = 10, 50, 20
QUERY_LINES = 20
MIN_LINES = 100
MIN_LINES_HARD = 250  # E1b: RepoEval pools hold ~20 hidden in-file chunks per sample


def load_groups(data_root):
    import pandas as pd
    df = pd.read_parquet(os.path.join(data_root, "github_repos/python/train.parquet"))
    groups, cur = [], []
    for path, content, first in df[["path", "content", "first"]].itertuples(index=False, name=None):
        if first and cur:
            if len(cur) > 1:
                groups.append(cur)
            cur = []
        cur.append((str(path), str(content)))
    if len(cur) > 1:
        groups.append(cur)
    return groups


def eval_paths(cand_files):
    paths = set()
    for fn in cand_files:
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l)
            for c in r.get("candidates", []):
                p = c.get("file_path", "")
                if p.count("/") >= 1:
                    paths.add(p)
    return paths


DEF_RE = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_]\w{3,})|^([A-Za-z_]\w{3,})\s*=", re.M)


def cross_names(related, own):
    """Names defined (def / class / top-level assignment) in other files and not in the current file."""
    names = {a or b for _, c in related for a, b in DEF_RE.findall(c)}
    return names - {a or b for a, b in DEF_RE.findall(own)}


def make_example(group, rng, uid, hard=False):
    """None if this repo/file gives no usable cut.

    hard=True (E1b): the GT must use a name defined in another file of the repo (CrossCodeEval's construction
    rule; RepoEval api_level targets are repo API calls too) and be long enough, otherwise the frozen generator
    already predicts it without context and every dLL is ~0 (E1: top dLL 1.7 on train vs 7-10 on eval).
    """
    min_lines = MIN_LINES_HARD if hard else MIN_LINES
    long_py = [j for j, (p, c) in enumerate(group) if p.endswith(".py") and c.count("\n") >= min_lines]
    if not long_py:  # eval files are long (RepoEval left context ~3.9k tokens): keep in-file chunks in the pool
        return None
    idx = rng.choice(long_py)
    path, content = group[idx]
    lines = content.split("\n")
    related = [(p, c) for j, (p, c) in enumerate(group) if j != idx and p.endswith(".py")]
    if not related:
        return None
    ok = [i for i in range(int(0.2 * len(lines)), int(0.8 * len(lines)))
          if len(lines[i].strip()) >= (20 if hard else 8) and not lines[i].strip().startswith("#")]
    hits = {}
    if hard:
        names = cross_names(related, content)
        for i in ok:
            pos = [m.start() for m in re.finditer(r"[A-Za-z_]\w*", lines[i]) if m.group() in names]
            if pos:
                hits[i] = pos
        ok = list(hits)
    if not ok:
        return None
    i = rng.choice(ok)
    line = lines[i]
    if rng.random() < 0.5:  # RepoEval-like: cut at the line start
        left, gt = "\n".join(lines[:i]) + "\n", line
    else:  # CCEval-like: cut inside the line, after its first token
        start = len(line) - len(line.lstrip())
        m = re.search(r"\S+", line[start:])
        lo = start + (m.end() if m else 1)
        hi = min(max(hits[i]), len(line) - 10) if hard else len(line) - 3  # hard: GT keeps a cross-file name
        if lo >= hi:
            return None
        if hard:  # cut at a token start, as CrossCodeEval does
            cuts = [m.start() for m in re.finditer(r"\w+|[^\w\s]", line) if lo <= m.start() <= hi]
            if not cuts:
                return None
            k = rng.choice(cuts)
        else:
            k = rng.randrange(lo, hi)
        left, gt = "\n".join(lines[:i]) + "\n" + line[:k], line[k:]
    return {"task_id": uid, "path": path, "left": left, "gt": gt, "related": related}


def cross_blocks(related):
    out = []
    for p, content in related:
        start = 0
        for chunk in J.mini_blocks(content):
            n = chunk.count("\n") + 1
            if len([l for l in chunk.split("\n") if l.strip()]) >= 2:
                out.append({"file_path": p, "description": f"file path: {p}\nlines: {start}-{start + n - 1}",
                            "code_content": chunk, "_src": "cross"})
            start += n
    return out


def bm25_top(query, blocks, k):
    tokz = lambda s: re.findall(r"[A-Za-z_][A-Za-z_0-9]*", s.lower())
    docs = [tokz(b["code_content"]) for b in blocks]
    N = len(docs)
    if not N:
        return []
    avg = sum(map(len, docs)) / N
    df = Counter(t for d in docs for t in set(d))
    q = tokz(query)
    scores = []
    for d in docs:
        tf, s = Counter(d), 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.5 / (tf[t] + 1.5 * (0.25 + 0.75 * len(d) / max(avg, 1)))
        scores.append(s)
    order = sorted(range(N), key=lambda i: -scores[i])[:k]
    return [blocks[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="train", choices=["train", "dev"])
    ap.add_argument("--n", type=int, default=4000, help="examples over all shards")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--eval_cands", default="", help="comma-separated eval candidate files (leakage guard)")
    ap.add_argument("--out", default="e1_labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--hard", action="store_true", help="E1b: GT must use a cross-file name (see make_example)")
    a = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    import joint_dense as JD

    groups = load_groups(a.data_root)
    groups = groups[:2000] if a.split == "train" else groups[2000:2200]
    bad = eval_paths([f for f in a.eval_cands.split(",") if f]) if a.eval_cands else set()
    keep = [g for g in groups if sum(p in bad for p, _ in g if p.count("/") >= 1) < 3]
    print(f"{a.split}: {len(groups)} repo groups, {len(groups) - len(keep)} dropped by the leakage guard; "
          f"sample train paths {[p for p, _ in groups[0][:3]]}, sample eval paths {sorted(bad)[:3]}", flush=True)
    groups = keep

    # deterministic example list over all shards; this shard takes every nshards-th
    rng = random.Random(f"{a.seed}-{a.split}" + ("-hard" if a.hard else ""))
    exs, tries = [], 0
    while len(exs) < a.n and tries < a.n * 50:
        g = groups[tries % len(groups)] if tries < len(groups) else rng.choice(groups)
        tries += 1
        ex = make_example(g, rng, f"{a.split}-{len(exs)}", a.hard)
        if ex:
            exs.append(ex)
    exs = exs[a.shard::a.nshards]

    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    scorer = JD.DenseScorer(snapshot_download(JD.RETRIEVER_REPO, revision=JD.RETRIEVER_REV))
    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.split}_{a.shard:02d}.jsonl")
    done = {json.loads(l)["task_id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    t0, n_seq = time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        for k, ex in enumerate(exs):
            if ex["task_id"] in done:
                continue
            query = "\n".join([l for l in ex["left"].split("\n") if l.strip()][-QUERY_LINES:])
            cand = bm25_top(query, cross_blocks(ex["related"]), N_BM25)
            if len(cand) > N_CROSS:
                s = scorer.scores(query, cand)
                cand = [cand[i] for i in sorted(range(len(cand)), key=lambda i: -s[i])[:N_CROSS]]
            pool = cand + J.hidden_infile_blocks(tok, ex, BUDGET)[-N_INFILE:]
            if not pool:
                continue
            gt = tok.encode(ex["gt"], add_special_tokens=False)[:P.MAX_TARGET]
            base, per = C.label_prompts(tok, ex, pool, BUDGET)
            lps = C.lp_sums(model, tok, [(base, gt)] + [(x, gt) for x in per], model.device, a.batch_size)
            dense = scorer.scores(query, pool)
            n_seq += len(per) + 1
            f.write(json.dumps({
                "task_id": ex["task_id"], "path": ex["path"], "gt": ex["gt"], "budget": BUDGET,
                "query": query, "left_tail": "\n".join(ex["left"].split("\n")[-60:]),
                "base_lp": round(lps[0], 4), "n_gt_tokens": len(gt),
                "cands": [{"src": b["_src"], "text": D.block_str(b),
                           "n_tok": len(tok.encode(D.block_str(b), add_special_tokens=False)),
                           "dll": round(lp - lps[0], 4), "dense": round(s, 5)}
                          for b, lp, s in zip(pool, lps[1:], dense)]}) + "\n")
            f.flush()
            if k % 25 == 0:
                el = time.time() - t0
                print(f"shard {a.shard}: {k}/{len(exs)} {el:.0f}s {n_seq / max(el, 1):.1f} seq/s", flush=True)
    print(f"shard {a.shard} done", flush=True)


if __name__ == "__main__":
    main()
