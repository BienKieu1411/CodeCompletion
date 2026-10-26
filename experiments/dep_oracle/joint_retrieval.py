#!/usr/bin/env python3
"""Joint in-file + cross-file retrieval under one token budget (frozen generator, no training).

Idea: the part of the current file that falls outside the in-file window (everything before the last
~450 tokens) is treated as a second candidate source. Its chunks compete with AlignCoder's cross-file
chunks for the same cross-file budget, ranked by a simple lexical similarity to the query.

Conditions (same AlignCoder prompt format, same generator, greedy, 2048-token window):
    align1536          AlignCoder saved top-10, budget 1536            (default, reference)
    align512           AlignCoder saved top-10, budget 512             (best fixed on RepoEval)
    cross_lex1536      AlignCoder top-10 re-sorted by lexical similarity (control for re-ranking alone)
    joint_lex1536      hidden in-file chunks + AlignCoder top-10, ranked jointly, budget 1536
    joint_lex1024      same, budget 1024
    infile_then_cross  most similar hidden in-file chunks (<= 512 tokens) first, then AlignCoder top-10

usage (Kaggle, GPU):
    python joint_retrieval.py run     --dataset repoeval --cands prediction_with_candidates.jsonl --n 1000 --out joint
    python joint_retrieval.py analyze --out joint
Same 1000-sample subset (seed 0) as budget_sweep.py.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D  # noqa: E402

CONDITIONS = ["align1536", "align512", "cross_lex1536", "joint_lex1536", "joint_lex1024", "infile_then_cross"]
BUDGET = {"align1536": 1536, "align512": 512, "cross_lex1536": 1536, "joint_lex1536": 1536,
          "joint_lex1024": 1024, "infile_then_cross": 1536}
INFILE_FIRST_TOKENS = 512
QUERY_LINES = 10


def ids_of(text):
    return set(D.identifiers(text))


def jacc(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def mini_blocks(text, max_lines=15):
    """AlignCoder-style chunking: blank-line groups, split > max_lines, merge neighbours up to max_lines."""
    groups, cur = [], []
    for line in text.split("\n"):
        if line.strip() == "":
            if cur:
                groups.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        groups.append(cur)
    split = []
    for g in groups:
        for i in range(0, len(g), max_lines):
            split.append(g[i:i + max_lines])
    out, buf = [], []
    for g in split:
        if len(buf) + len(g) <= max_lines:
            buf.extend(g)
        else:
            if buf:
                out.append(buf)
            buf = list(g)
    if buf:
        out.append(buf)
    return ["\n".join(b) for b in out]


def hidden_infile_blocks(tok, ex, budget):
    """Chunks of the left context that do NOT fit in the in-file window when the cross-file part is full."""
    path_ids = tok.encode(f"\n\n# file path: {ex['path']}\n\n", add_special_tokens=False)
    window = D.MAX_CTX - (budget + len(path_ids) + 10)
    left_ids = tok.encode(ex["left"], add_special_tokens=False)
    if len(left_ids) <= window:
        return []
    hidden = tok.decode(left_ids[:-window])
    blocks = []
    for k, chunk in enumerate(mini_blocks(hidden)):
        if len([l for l in chunk.split("\n") if l.strip()]) < 2:
            continue
        blocks.append({"file_path": ex["path"], "description": f"file path: {ex['path']}\nin-file block {k}",
                       "code_content": chunk, "_src": "infile"})
    return blocks


def saved_blocks(saved):
    out = []
    for b in saved:
        if b["file_path"] == "":
            break
        bb = dict(b)
        bb["_src"] = "cross"
        out.append(bb)
    return out


def blocks_for(cond, tok, ex, saved):
    q = ids_of("\n".join([l for l in ex["left"].split("\n") if l.strip()][-QUERY_LINES:]))
    cross = saved_blocks(saved)
    score = lambda b: jacc(q, ids_of(b["code_content"]))
    if cond in ("align1536", "align512"):
        return saved  # untouched, incl. STOP sentinel semantics
    if cond == "cross_lex1536":
        return sorted(cross, key=score, reverse=True)
    infile = hidden_infile_blocks(tok, ex, BUDGET[cond])
    if cond in ("joint_lex1536", "joint_lex1024"):
        return sorted(infile + cross, key=score, reverse=True)
    if cond == "infile_then_cross":
        picked, used = [], 0
        for b in sorted(infile, key=score, reverse=True):
            n = len(tok.encode(D.block_str(b), add_special_tokens=False)) + 2
            if used + n > INFILE_FIRST_TOKENS:
                break
            picked.append(b)
            used += n
        return picked + cross
    raise ValueError(cond)


def fitted_sources(tok, blocks, budget):
    """How many in-file / cross-file blocks actually fit in the budget (prefix of the ordered list)."""
    used, cnt = 0, Counter()
    for b in blocks:
        if b["file_path"] == "":
            break
        n = len(tok.encode(D.block_str(b), add_special_tokens=False)) + 2
        if used + n > budget:
            break
        used += n
        cnt[b.get("_src", "cross")] += 1
    return dict(cnt)


def cmd_run(a):
    import torch
    from transformers import AutoModelForCausalLM
    cands = D.load_cands(a.cands)
    ids = D.pick_subset(list(cands), a.n, a.seed)
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs]
    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "gt.jsonl"), "w", encoding="utf-8") as f:
        for t in ids:
            f.write(json.dumps({"task_id": t, "gt": exs[t]["gt"]}) + "\n")
    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    for cond in a.conditions.split(","):
        cdir = os.path.join(a.out, cond)
        os.makedirs(cdir, exist_ok=True)
        pred_path = os.path.join(cdir, "prediction.jsonl")
        done = set()
        if os.path.exists(pred_path):
            done = {json.loads(l)["task_id"] for l in open(pred_path, encoding="utf-8")}
        todo = [t for t in ids if t not in done]
        print(f"[{cond}] {len(done)} done, {len(todo)} to go", flush=True)
        prompts = []
        with open(os.path.join(cdir, "prompt_meta.jsonl"), "a", encoding="utf-8") as meta:
            for t in todo:
                bl = blocks_for(cond, tok, exs[t], cands[t]["candidates"])
                prompt, info = D.build_prompt(tok, exs[t], bl, max_cross=BUDGET[cond])
                prompts.append((t, tok.encode(prompt)[-D.MAX_CTX:]))
                meta.write(json.dumps({"task_id": t, "cross_tokens": info["cross_tokens"],
                                       "infile_tokens": info["infile_tokens"],
                                       "fitted": fitted_sources(tok, bl, BUDGET[cond])}) + "\n")
        prompts.sort(key=lambda x: len(x[1]))
        t0 = time.time()
        with open(pred_path, "a", encoding="utf-8") as fout:
            for i in range(0, len(prompts), a.batch_size):
                batch = prompts[i:i + a.batch_size]
                L = max(len(x[1]) for x in batch)
                inp = torch.tensor([[tok.pad_token_id] * (L - len(x[1])) + x[1] for x in batch]).cuda()
                att = torch.tensor([[0] * (L - len(x[1])) + [1] * len(x[1]) for x in batch]).cuda()
                with torch.no_grad():
                    gen = model.generate(inp, attention_mask=att, max_new_tokens=D.MAX_NEW, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                for (t, _), g in zip(batch, gen[:, L:]):
                    fout.write(json.dumps({"task_id": t, "pred": tok.decode(g, skip_special_tokens=True)}) + "\n")
                fout.flush()
                if (i // a.batch_size) % 20 == 0:
                    print(f"[{cond}] {i + len(batch)}/{len(prompts)} {time.time() - t0:.0f}s", flush=True)


def cmd_analyze(a):
    gt = {}
    for l in open(os.path.join(a.out, "gt.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        gt[r["task_id"]] = r["gt"]
    em, es, fitted = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    conds = [c for c in CONDITIONS if os.path.exists(os.path.join(a.out, c, "prediction.jsonl"))]
    for c in conds:
        for l in open(os.path.join(a.out, c, "prediction.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            em[c][r["task_id"]], es[c][r["task_id"]] = D.simple_score(r["pred"], gt[r["task_id"]])
        mp = os.path.join(a.out, c, "prompt_meta.jsonl")
        if os.path.exists(mp):
            for l in open(mp, encoding="utf-8"):
                r = json.loads(l)
                fitted[c][r["task_id"]] = r.get("fitted", {})
    ids = sorted(set.intersection(*[set(em[c]) for c in conds]))
    n = len(ids)
    res = {"n": n, "conditions": {}}
    for c in conds:
        row = {"em": round(100 * sum(em[c][t] for t in ids) / n, 2),
               "es": round(sum(es[c][t] for t in ids) / n, 2)}
        for ref in ("align1536", "align512"):
            if ref in em and c != ref:
                w = sum(em[c][t] > em[ref][t] for t in ids)
                l_ = sum(em[c][t] < em[ref][t] for t in ids)
                row[f"vs_{ref}"] = {"gained": w, "lost": l_, "mcnemar_p": D.mcnemar(w, l_)}
        if fitted[c]:
            row["mean_infile_blocks_in_prompt"] = round(sum(fitted[c][t].get("infile", 0) for t in ids) / n, 2)
            row["mean_cross_blocks_in_prompt"] = round(sum(fitted[c][t].get("cross", 0) for t in ids) / n, 2)
            row["pct_samples_with_infile_block"] = round(
                100 * sum(fitted[c][t].get("infile", 0) > 0 for t in ids) / n, 1)
        res["conditions"][c] = row
    res["notes"] = ("simple EM/ES (first k lines). Paired exact McNemar per sample. Same 1000-sample subset "
                    "(seed 0) as budget_sweep.py; align1536/align512 should reproduce its b1536/b512.")
    with open(os.path.join(a.out, "joint_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="joint_out")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
