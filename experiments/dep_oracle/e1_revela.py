#!/usr/bin/env python3
"""Zero-shot Revela-code-500M as the chunk scorer, on the Phase-2 pilot pools (same check as E1 / AMI).

Revela (ICLR 2026) trains a retriever with a language-modeling objective (in-batch attention: chunks that help an
LM predict the next tokens get high similarity), so its similarity may track utility better than AlignRetriever.
The released checkpoint is only the retriever (Qwen2.5-0.5B + LoRA, EOS pooling, "Query: " / "Passage: " prefixes;
loading as experiments/revela_kaggle/benchmark.py). Query = last 20 non-empty lines; passage = block_str (as
AlignCoder's str(CodeBlock)).

Generation as e1_eval.py (whole chunks that fit B, best chunk nearest the cursor): revela_fill_B, B in 512/1024/1536.

Decision (fixed before running), per dataset:
    G1: Spearman(Revela score, dLL) at B=1024 >= 0.30   (AlignRetriever 0.07-0.14; best E1b scorer / AMI 0.14-0.20)
    G2: best revela_fill EM >= dense_fill at the same B + 3 (RepoEval) / + 2 (CCEval)   (dense_fill from E1b eval)

usage:
    python e1_revela.py run     --dataset repoeval --cands ... --labels pilot/chunks.jsonl --out e1rev
    python e1_revela.py analyze --dataset repoeval --out modal_out/e1rev_repoeval --prev_dir probe_inputs/repoeval \
                                --pilot_dir modal_out/pilot_repoeval --e1_dir modal_out/e1heval_repoeval
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk_pilot as C  # noqa: E402
import dep_oracle as D  # noqa: E402
import likelihood_probe as P  # noqa: E402
from e1_ami import pack  # noqa: E402

REVELA = "trumancai/Revela-code-500M"
BUDGETS = [512, 1024, 1536]
SLACK = 32


class Revela:
    def __init__(self, max_tokens=512, batch_size=32):
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModel, AutoTokenizer
        cfg = PeftConfig.from_pretrained(REVELA)
        self.tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        base = AutoModel.from_pretrained(cfg.base_model_name_or_path, torch_dtype=torch.float16,
                                         attn_implementation="sdpa")
        self.model = PeftModel.from_pretrained(base, REVELA).merge_and_unload().cuda().eval()
        self.torch, self.max_tokens, self.bs = torch, max_tokens, batch_size

    def encode(self, texts, query):
        torch = self.torch
        prefix = "Query: " if query else "Passage: "
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.bs):
                ids = self.tok([prefix + t for t in texts[i:i + self.bs]], add_special_tokens=True, truncation=True,
                               max_length=self.max_tokens - 1, return_attention_mask=False)["input_ids"]
                ids = [r + [self.tok.eos_token_id] for r in ids]
                batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
                h = self.model(**batch, use_cache=False).last_hidden_state
                end = batch.attention_mask.sum(1) - 1
                out.append(torch.nn.functional.normalize(h[torch.arange(len(ids), device="cuda"), end].float(), dim=-1))
        return torch.cat(out)

    def scores(self, query, texts):
        q = self.encode([query], True)
        return (self.encode(texts, False) @ q.t()).squeeze(-1).cpu().tolist()


def cmd_run(a):
    import torch
    from transformers import AutoModelForCausalLM

    cands = D.load_cands(a.cands)
    ids = C.pilot_ids(cands, a.n, a.seed, a.n_pilot)
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs]
    if a.limit:
        ids = ids[:a.limit]
    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    os.makedirs(a.out, exist_ok=True)
    pools = {(t, b): C.pool_at(tok, exs[t], cands[t]["candidates"], b) for t in ids for b in BUDGETS}

    sc_path = os.path.join(a.out, "scores.jsonl")
    scores = {}
    if os.path.exists(sc_path):
        for r in map(json.loads, open(sc_path)):
            scores[(r["task_id"], r["budget"])] = r["s"]
    if not all(k in scores for k in pools):
        rv = Revela()
        with open(sc_path, "a") as f:
            for t in ids:
                q = "\n".join([l for l in exs[t]["left"].split("\n") if l.strip()][-20:])
                for b in BUDGETS:
                    if (t, b) in scores:
                        continue
                    s = rv.scores(q, [D.block_str(x) for x in pools[(t, b)]]) if pools[(t, b)] else []
                    scores[(t, b)] = s
                    f.write(json.dumps({"task_id": t, "budget": b, "s": s}) + "\n")
        del rv
        torch.cuda.empty_cache()
        print("scored", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    for b in BUDGETS:
        cond = f"revela_fill_{b}"
        cdir = os.path.join(a.out, cond)
        os.makedirs(cdir, exist_ok=True)
        pred_path = os.path.join(cdir, "prediction.jsonl")
        done = {json.loads(l)["task_id"] for l in open(pred_path)} if os.path.exists(pred_path) else set()
        prompts = []
        for t in ids:
            if t in done:
                continue
            pool, s = pools[(t, b)], scores[(t, b)]
            bl = pack(tok, [pool[i] for i in sorted(range(len(pool)), key=lambda i: -s[i])], b)
            prompt, info = D.build_prompt(tok, exs[t], bl, max_cross=b + SLACK)
            prompts.append((t, tok.encode(prompt)[-D.MAX_CTX:], info["cross_tokens"], [x["_src"] for x in bl]))
        prompts.sort(key=lambda x: len(x[1]))
        with open(pred_path, "a") as fout:
            for i in range(0, len(prompts), a.gen_batch_size):
                batch = prompts[i:i + a.gen_batch_size]
                L = max(len(x[1]) for x in batch)
                inp = torch.tensor([[tok.pad_token_id] * (L - len(x[1])) + x[1] for x in batch]).cuda()
                att = torch.tensor([[0] * (L - len(x[1])) + [1] * len(x[1]) for x in batch]).cuda()
                with torch.no_grad():
                    gen = model.generate(inp, attention_mask=att, max_new_tokens=D.MAX_NEW, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                for (t, _, ct, srcs), g in zip(batch, gen[:, L:]):
                    fout.write(json.dumps({"task_id": t, "pred": tok.decode(g, skip_special_tokens=True),
                                           "cross_tokens": ct, "srcs": srcs}) + "\n")
                fout.flush()
        print(f"[{cond}] done", flush=True)


def cmd_analyze(a):
    gt = {t: r["gt"] for t, r in P.load_jsonl(os.path.join(a.prev_dir, "gt.jsonl")).items()}
    labels = {(r["task_id"], r["budget"]): r for r in map(json.loads, open(os.path.join(a.pilot_dir, "chunks.jsonl")))}
    scores = {(r["task_id"], r["budget"]): r["s"] for r in map(json.loads, open(os.path.join(a.out, "scores.jsonl")))}
    res = {"dataset": a.dataset, "spearman": {}, "em": {}}
    for b in BUDGETS:
        rv, dn = [], []
        for (t, bb), s in scores.items():
            if bb != b:
                continue
            cs = labels[(t, b)]["cands"]
            assert len(cs) == len(s), (t, b)
            x = C.spearman(s, [c["dll"] for c in cs])
            y = C.spearman([c["dense"] for c in cs], [c["dll"] for c in cs])
            if x is not None:
                rv.append(x)
            if y is not None:
                dn.append(y)
        res["spearman"][f"B{b}"] = {"revela": round(sum(rv) / len(rv), 4), "dense": round(sum(dn) / len(dn), 4)}

    def em_of(p):
        return {t: D.simple_score(r["pred"], gt[t])[0] for t, r in P.load_jsonl(p).items()}
    preds = {f"revela_fill_{b}": em_of(os.path.join(a.out, f"revela_fill_{b}", "prediction.jsonl")) for b in BUDGETS}
    for b in BUDGETS:
        for m in ("dense", "cross_soft"):
            p = os.path.join(a.e1_dir, f"{m}_fill_{b}", "prediction.jsonl")
            if os.path.exists(p):
                preds[f"ref_{m}_fill_{b}"] = em_of(p)
    p0 = json.load(open(os.path.join(a.pilot_dir, "pilot_summary.json")))["P0"]
    preds["ref_regen_P0"] = em_of(os.path.join(a.pilot_dir, "regen_" + p0, "prediction.jsonl"))
    ids = sorted(set.intersection(*[set(v) for v in preds.values()]))
    for c, em in preds.items():
        b = c.rsplit("_", 1)[1]
        ref = preds.get(f"ref_dense_fill_{b}", preds["ref_regen_P0"])
        w = sum(em[t] > ref[t] for t in ids)
        l_ = sum(em[t] < ref[t] for t in ids)
        res["em"][c] = {"em": round(100 * sum(em[t] for t in ids) / len(ids), 2),
                        "vs_dense_same_B": [w, l_, round(D.mcnemar(w, l_), 4)]}
    res["n"] = len(ids)
    need = 3.0 if a.dataset == "repoeval" else 2.0
    gains = {b: res["em"][f"revela_fill_{b}"]["em"] - res["em"][f"ref_dense_fill_{b}"]["em"] for b in BUDGETS
             if f"ref_dense_fill_{b}" in res["em"]}
    res["decision"] = {"G1_spearman_1024": res["spearman"]["B1024"]["revela"],
                       "G1": "PASS" if res["spearman"]["B1024"]["revela"] >= 0.30 else "FAIL",
                       "G2_gain_vs_dense": gains, "G2": "PASS" if max(gains.values()) >= need else "FAIL"}
    with open(os.path.join(a.out, "revela_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--labels")
    ap.add_argument("--prev_dir")
    ap.add_argument("--pilot_dir")
    ap.add_argument("--e1_dir")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="e1rev")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_pilot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
