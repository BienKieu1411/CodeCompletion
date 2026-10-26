#!/usr/bin/env python3
"""E1 step 2: train a utility scorer on the e1_label.py dLL labels (RLCoder-style training, AlignRetriever init).

Variants (same data, same init, same steps):
    bi_soft     bi-encoder (AlignRetriever), listwise KL( softmax(dLL / T) || softmax(score) )
    bi_onehot   bi-encoder, cross-entropy on argmax dLL -- RLCoder's objective (one best candidate per sample)
    cross_soft  cross-encoder ([query ; chunk] -> scalar), listwise KL as bi_soft
Every list holds a NULL candidate with dLL = 0 (RLCoder's "Don't need cross file context" stop signal), so a chunk
scored above NULL is predicted to help: that is the stop rule used at eval.

usage:
    python e1_train.py --labels e1_labels --variant bi_soft --out e1_models
"""
import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NULL_TEXT = "Don't need cross file context for completion"  # RLCoder NULL_CONTEXT
Q_LEN, C_LEN, X_LEN, X_Q = 256, 320, 512, 192  # bi query / bi cand / cross total / cross query part
VARIANTS = ["bi_soft", "bi_onehot", "cross_soft"]


class Scorer:
    """score(query, texts) -> list of floats. Used by training (with grad) and by e1_eval.py (no grad)."""

    def __init__(self, path, kind, device="cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.kind, self.device = torch, kind, device
        self.tok = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.enc = AutoModel.from_pretrained(path).to(device)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1).to(device) if kind == "cross" else None
        self.scale = 20.0  # cosine -> logits (temperature 0.05, RLCoder-like)
        hp = os.path.join(path, "head.pt")
        if self.head is not None and os.path.exists(hp):
            self.head.load_state_dict(torch.load(hp, map_location=device))

    def parameters(self):
        return list(self.enc.parameters()) + (list(self.head.parameters()) if self.head is not None else [])

    def _wrap(self, tokens):
        t = self.tok
        return t.convert_tokens_to_ids([t.cls_token, "<encoder-only>", t.sep_token] + tokens + [t.sep_token])

    def _batch(self, seqs):
        torch = self.torch
        L = max(map(len, seqs))
        x = torch.tensor([s + [self.tok.pad_token_id] * (L - len(s)) for s in seqs], device=self.device)
        return x, x.ne(self.tok.pad_token_id)

    def _pool(self, x, m):
        h = self.enc(x, attention_mask=m)[0]
        return (h * m.unsqueeze(-1)).sum(1) / m.sum(-1, keepdim=True)

    def forward(self, query, texts):
        torch = self.torch
        tq = self.tok.tokenize(query)
        if self.kind == "bi":
            q = self._pool(*self._batch([self._wrap(tq[-(Q_LEN - 4):])]))
            c = self._pool(*self._batch([self._wrap(self.tok.tokenize(s)[:C_LEN - 4]) for s in texts]))
            q, c = torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(c, dim=-1)
            return self.scale * (c @ q.t()).squeeze(-1)
        qq = tq[-X_Q:]
        seqs = [self._wrap(qq + [self.tok.sep_token] + self.tok.tokenize(s)[:X_LEN - 5 - len(qq)]) for s in texts]
        return self.head(self._pool(*self._batch(seqs))).squeeze(-1)

    def score(self, query, texts, bs=64):
        torch = self.torch
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(texts), bs):
                out += self.forward(query, texts[i:i + bs]).float().cpu().tolist()
        return out

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.enc.save_pretrained(path)
        self.tok.save_pretrained(path)
        if self.head is not None:
            self.torch.save(self.head.state_dict(), os.path.join(path, "head.pt"))


def spearman(x, y):
    import chunk_pilot as C
    return C.spearman(x, y)


def load_rows(dirs, split, min_top=None):
    """dirs: comma-separated label dirs. min_top: drop samples whose best dLL < min_top (no signal to learn)."""
    rows = []
    for d in dirs.split(","):
        for fn in sorted(glob.glob(os.path.join(d, f"{split}_*.jsonl"))):
            rows += [json.loads(l) for l in open(fn, encoding="utf-8")]
    rows = [r for r in rows if len(r["cands"]) >= 2]
    return [r for r in rows if max(c["dll"] for c in r["cands"]) >= min_top] if min_top is not None else rows


def evaluate(sc, rows):
    """Within-sample Spearman(score, dLL) over real chunks, NULL-rule accuracy, dLL of the top-1 pick."""
    rho, acc, top, best = [], [], [], []
    for r in rows:
        s = sc.score(r["query"], [c["text"] for c in r["cands"]] + [NULL_TEXT])
        d = [c["dll"] for c in r["cands"]]
        x = spearman(s[:-1], d)
        if x is not None:
            rho.append(x)
        acc += [(si > s[-1]) == (di > 0) for si, di in zip(s[:-1], d)]
        top.append(d[max(range(len(d)), key=lambda i: s[i])])
        best.append(max(d))
    n = len(rows)
    return {"spearman": round(sum(rho) / len(rho), 4), "null_rule_acc": round(sum(acc) / len(acc), 4),
            "top1_dll": round(sum(top) / n, 4), "oracle_top1_dll": round(sum(best) / n, 4), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--out", default="e1_models")
    ap.add_argument("--init", default=None, help="default: AlignRetriever at the eval revision")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=1.0, help="T in softmax(dLL / T)")
    ap.add_argument("--samples_per_step", type=int, default=4)
    ap.add_argument("--max_list", type=int, default=31, help="chunks per sample (+ NULL); larger lists subsampled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev_labels", default=None, help="label dir(s) for dev; default = --labels")
    ap.add_argument("--min_top", type=float, default=None, help="train only on samples with best dLL >= this")
    a = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    import joint_dense as JD

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    init = a.init or snapshot_download(JD.RETRIEVER_REPO, revision=JD.RETRIEVER_REV)
    sc = Scorer(init, "cross" if a.variant.startswith("cross") else "bi")
    train = load_rows(a.labels, "train", a.min_top)
    dev = load_rows(a.dev_labels or a.labels, "dev")
    print(f"train {len(train)} dev {len(dev)}", flush=True)
    out = os.path.join(a.out, a.variant)
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "metrics.jsonl"), "a")

    sc.enc.eval()
    m0 = evaluate(sc, dev)
    m0["dense_spearman"] = round(sum(x for x in (spearman([c["dense"] for c in r["cands"]],
                                                          [c["dll"] for c in r["cands"]]) for r in dev)
                                    if x is not None) / len(dev), 4)
    print("epoch 0", m0, flush=True)
    log.write(json.dumps({"epoch": 0, **m0}) + "\n")
    log.flush()

    opt = torch.optim.AdamW(sc.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(train) // a.samples_per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, int(0.05 * steps))) * max(0.0, 1 - s / max(steps, 1)))
    step, t0 = 0, time.time()
    for ep in range(1, a.epochs + 1):
        sc.enc.train()
        random.shuffle(train)
        for i in range(0, len(train) - a.samples_per_step + 1, a.samples_per_step):
            tot = 0.0
            for r in train[i:i + a.samples_per_step]:  # backward per sample: keeps the cross-encoder in memory
                cs = r["cands"] if len(r["cands"]) <= a.max_list else random.sample(r["cands"], a.max_list)
                d = torch.tensor([c["dll"] for c in cs] + [0.0], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s = sc.forward(r["query"], [c["text"] for c in cs] + [NULL_TEXT]).float()
                logp = torch.log_softmax(s, -1)
                if a.variant.endswith("onehot"):
                    loss = -logp[int(d.argmax())]
                else:
                    tgt = torch.softmax(d / a.temp, -1)
                    loss = (tgt * (torch.log(tgt + 1e-12) - logp)).sum()
                (loss / a.samples_per_step).backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(sc.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{steps} loss {tot / a.samples_per_step:.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        sc.enc.eval()
        m = evaluate(sc, dev)
        print(f"epoch {ep}", m, flush=True)
        log.write(json.dumps({"epoch": ep, **m}) + "\n")
        log.flush()
        sc.save(os.path.join(out, f"epoch{ep}"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
