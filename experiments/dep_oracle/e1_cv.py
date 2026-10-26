#!/usr/bin/env python3
"""E1 diagnostic: is dLL learnable from (query, chunk) text IN-DOMAIN?

E1 trained on Data4RLCoder random cuts and failed G1, but those labels carry almost no signal (top dLL 1.7 vs 7-10
on eval). This removes the distribution shift: 5-fold cross-validation on the Phase-2 pilot labels themselves
(RepoEval + CCEval, 200 samples each, budgets 512/1024/1536), folds grouped by repository, both datasets pooled for
training. Same scorer, init (AlignRetriever) and loss as e1_train.py. Diagnostic only, never a reported result:
the folds are eval data.

Reading (fixed before running): held-out Spearman at B=1024 >= 0.30 on both datasets -> dLL is learnable and E1's
failure is the training data; < 0.20 -> a text scorer cannot recover dLL, drop the scorer line and move to a
budget-only policy.

usage:
    python e1_cv.py --fold 0 --variant bi_soft --out e1cv --data_dir aligncoder_data \
        --pilot repoeval=prediction_with_candidates.jsonl:pilot_repoeval_chunks.jsonl \
        --pilot cceval=cceval_python_prediction_with_candidates.jsonl:pilot_cceval_chunks.jsonl
"""
import argparse
import json
import os
import random
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk_pilot as C  # noqa: E402
import dep_oracle as D  # noqa: E402
from e1_train import NULL_TEXT, Scorer, evaluate  # noqa: E402

K = 5
BUDGETS = [512, 1024, 1536]


def rows_for(dataset, cands_file, labels_file, data_dir, tok):
    D.DATASET = dataset
    subset = "python" if dataset == "cceval" else "api_level"
    cands = D.load_cands(cands_file)
    ids = C.pilot_ids(cands, 1000, 0, 200)
    labels = {(r["task_id"], r["budget"]): r for r in map(json.loads, open(labels_file, encoding="utf-8"))}
    exs = D.load_examples(data_dir, subset, set(ids))
    rows = []
    for t in ids:
        if t not in exs:
            continue
        q = "\n".join([l for l in exs[t]["left"].split("\n") if l.strip()][-20:])
        for b in BUDGETS:
            pool, lab = C.pool_at(tok, exs[t], cands[t]["candidates"], b), labels[(t, b)]["cands"]
            assert len(pool) == len(lab), (t, b)
            if len(pool) < 2:
                continue
            rows.append({"dataset": dataset, "task_id": t, "budget": b, "query": q,
                         "fold": zlib.crc32(t.split("/")[0].encode()) % K,
                         "cands": [{"text": D.block_str(x), "dll": l["dll"], "dense": l["dense"]}
                                   for x, l in zip(pool, lab)]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--variant", default="bi_soft", choices=["bi_soft", "cross_soft"])
    ap.add_argument("--pilot", action="append", required=True, help="dataset=cands_file:labels_file")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="e1cv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--samples_per_step", type=int, default=4)
    ap.add_argument("--max_list", type=int, default=31)
    a = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    import joint_dense as JD

    random.seed(a.fold)
    torch.manual_seed(a.fold)
    tok = D.get_tokenizer(D.GEN_DEFAULT)
    rows = []
    for spec in a.pilot:
        ds, files = spec.split("=")
        cf, lf = files.split(":")
        rows += rows_for(ds, cf, lf, a.data_dir, tok)
    train = [r for r in rows if r["fold"] != a.fold]
    test = [r for r in rows if r["fold"] == a.fold]
    print(f"fold {a.fold}: train {len(train)} lists, test {len(test)} lists", flush=True)

    sc = Scorer(snapshot_download(JD.RETRIEVER_REPO, revision=JD.RETRIEVER_REV),
                "cross" if a.variant.startswith("cross") else "bi")

    def report(ep):
        res = {"fold": a.fold, "variant": a.variant, "epoch": ep}
        for ds in sorted({r["dataset"] for r in test}):
            for b in BUDGETS:
                sub = [r for r in test if r["dataset"] == ds and r["budget"] == b]
                if sub:
                    m = evaluate(sc, sub)
                    m["dense_spearman"] = round(sum(x for x in (C.spearman([c["dense"] for c in r["cands"]],
                                                                           [c["dll"] for c in r["cands"]])
                                                                 for r in sub) if x is not None) / len(sub), 4)
                    res[f"{ds}_B{b}"] = m
        print(json.dumps(res), flush=True)
        with open(os.path.join(a.out, f"{a.variant}_fold{a.fold}.jsonl"), "a") as f:
            f.write(json.dumps(res) + "\n")

    os.makedirs(a.out, exist_ok=True)
    sc.enc.eval()
    report(0)
    opt = torch.optim.AdamW(sc.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(train) // a.samples_per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, int(0.05 * steps))) * max(0.0, 1 - s / max(steps, 1)))
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        sc.enc.train()
        random.shuffle(train)
        for i in range(0, len(train) - a.samples_per_step + 1, a.samples_per_step):
            for r in train[i:i + a.samples_per_step]:  # as e1_train.py (soft listwise KL, NULL with dLL 0)
                cs = r["cands"] if len(r["cands"]) <= a.max_list else random.sample(r["cands"], a.max_list)
                d = torch.tensor([c["dll"] for c in cs] + [0.0], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s = sc.forward(r["query"], [c["text"] for c in cs] + [NULL_TEXT]).float()
                tgt = torch.softmax(d, -1)
                loss = (tgt * (torch.log(tgt + 1e-12) - torch.log_softmax(s, -1))).sum()
                (loss / a.samples_per_step).backward()
            torch.nn.utils.clip_grad_norm_(sc.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        print(f"epoch {ep} {time.time() - t0:.0f}s", flush=True)
        sc.enc.eval()
        report(ep)


if __name__ == "__main__":
    main()
