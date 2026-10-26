#!/usr/bin/env python3
"""Training-free decode-time improvement pilot for CADRE (small subset).

One vLLM generation pass per task on the CADRE dual-channel prompt (8k, prompt_token_ids):
  greedy + N sampled candidates (with token logprobs).
Then, offline in the same script, compare selectors over the SAME candidate pool:
  greedy | maxlogp | vote | grounded | mbr | grounded_mbr
Hyper-parameters are tuned on half of the tasks and reported on the held-out half (2-fold).
Run on Colab: python run_rerank_pilot.py  (needs /content/mb_spt with the bundle extracted).
"""
from __future__ import annotations

import gzip, itertools, json, os, random, re, sys, tarfile, time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path("/content/mb_spt")
BUNDLE = Path("/content/full_benchmark_8004_bundle.tar.gz")
N_PER_BENCH = int(os.environ.get("N_PER_BENCH", "100"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
TEMP = float(os.environ.get("TEMP", "0.5"))
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "/content")

import subprocess
def ensure():
    miss = []
    for mod, pkg in (("fuzzywuzzy", "fuzzywuzzy"), ("Levenshtein", "python-Levenshtein"), ("rank_bm25", "rank_bm25"), ("nltk", "nltk")):
        try: __import__(mod)
        except ImportError: miss.append(pkg)
    try:
        from importlib.metadata import version
        if version("tree-sitter") != "0.20.4": miss.append("tree-sitter==0.20.4")
    except Exception: miss.append("tree-sitter==0.20.4")
    try:
        import vllm  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
        subprocess.check_call(["uv", "pip", "install", "--system", "vllm", "--torch-backend=auto"])
    if miss: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *miss])
ensure()

import run_full_benchmark as rfb  # noqa: E402  (reuses prompt builder + postprocess + metrics)
from transformers import AutoTokenizer  # noqa: E402
from tree_sitter import Language, Parser  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

GEN = "deepseek-ai/deepseek-coder-1.3b-base"
REV = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
BENCHES = ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]

if not (ROOT / "tasks.jsonl.gz").exists():
    with tarfile.open(BUNDLE, "r:gz") as t:
        t.extractall(ROOT)

tok = AutoTokenizer.from_pretrained(GEN, revision=REV, use_fast=True)
tasks_all = [json.loads(l) for l in gzip.open(ROOT / "tasks.jsonl.gz", "rt")]
traces = {(r["benchmark"], r["task_id"]): r for r in map(json.loads, gzip.open(ROOT / "retrieval_trace.jsonl.gz", "rt"))}
files = {r["file_id"]: {"path": r["path"], "text": r["text"]} for r in map(json.loads, gzip.open(ROOT / "files.jsonl.gz", "rt"))}
pools = {}
if (ROOT / "pools.jsonl.gz").exists():
    pools = {r["pool_id"]: r["file_ids"] for r in map(json.loads, gzip.open(ROOT / "pools.jsonl.gz", "rt"))}
parsers = {}
for lang in ("python", "java"):
    p = Parser(); p.set_language(Language(str(ROOT / "parsers" / f"{lang}-lang-parser.so"), lang)); parsers[lang] = p

rng = random.Random(20261026)
tasks = []
for b in BENCHES:
    bt = [t for t in tasks_all if t["benchmark"] == b]
    tasks += rng.sample(bt, min(N_PER_BENCH, len(bt)))
print(f"pilot tasks: {len(tasks)}", flush=True)

llm = LLM(model=GEN, tokenizer=GEN, revision=REV, tokenizer_revision=REV, dtype="float16", max_model_len=8192,
          max_num_seqs=128, max_num_batched_tokens=16384, gpu_memory_utilization=0.85, seed=1, enforce_eager=True,
          disable_log_stats=True)

# ---------- prompts ----------
prompts, meta = [], []
for t in tasks:
    tr = traces.get((t["benchmark"], t["task_id"])) or traces.get(t["task_id"])
    if not tr: continue
    if "file_ids_by_path" in t:
        pf = [files[f] for f in t["file_ids_by_path"].values() if f in files]
    else:
        pf = [files[f] for f in pools.get(t.get("pool_id"), []) if f in files]
    ids, _, _ = rfb.build_full_prompt("cadre_full_dual_channel", t, tr, pf, tok)
    prompts.append({"prompt_token_ids": ids}); meta.append((t, ids))

# ---------- generation: greedy + sampled, with logprobs ----------
sp_g = SamplingParams(max_tokens=48, temperature=0.0, logprobs=0)
sp_s = SamplingParams(max_tokens=48, temperature=TEMP, top_p=0.95, n=N_SAMPLES, logprobs=0, seed=7)
t0 = time.time()
og = llm.generate(prompts, sp_g)
os_ = llm.generate(prompts, sp_s)
print(f"generation done in {time.time()-t0:.0f}s", flush=True)

ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

def end_char(raw: str, bm: str, stmt: str) -> int:
    if bm.startswith("repoeval"):
        return len(raw)  # whole raw (truncated to n lines upstream); logprob over all tokens is ok proxy
    return len(stmt) if raw.startswith(stmt) and stmt else len(raw)

def make_cand(out, t, is_greedy):
    ids = list(out.token_ids)
    raw = tok.decode(ids, skip_special_tokens=True)
    bm, lang = t["benchmark"], t["language"]
    if bm.startswith("cceval"):
        stmt = rfb.postprocess_code_lines(t["left_context"], raw, parsers[lang], lang)
    else:
        stmt = rfb.postprocess_repoeval(raw, t["groundtruth"])
    lps = [list(d.values())[0].logprob if len(d) == 1 else d[i].logprob for i, d in enumerate([{k: v} for k, v in (out.logprobs[j].items() if False else [])] )] if False else None
    tl = []
    for j, d in enumerate(out.logprobs or []):
        tid = ids[j]
        tl.append(d[tid].logprob if tid in d else list(d.values())[0].logprob)
    ec = end_char(raw, bm, stmt)
    k = len(ids)
    for kk in range(1, len(ids) + 1):
        if len(tok.decode(ids[:kk], skip_special_tokens=True)) >= ec:
            k = kk; break
    k = max(1, k)
    mlp = sum(tl[:k]) / k if tl else 0.0
    clean = rfb.remove_comments(stmt)
    key = "".join(clean.split())
    return {"stmt": stmt, "clean": clean, "key": key, "mlp": mlp, "greedy": is_greedy}

pool = []
for (t, ids), g, s in zip(meta, og, os_):
    cands = [make_cand(g.outputs[0], t, True)] + [make_cand(o, t, False) for o in s.outputs]
    prompt_text = tok.decode(ids, skip_special_tokens=True)
    universe = set(rfb.extract_identifiers(prompt_text, t["language"]))
    for c in cands:
        cid = rfb.extract_identifiers(c["clean"], t["language"])
        c["n_ung"] = sum(1 for x in set(cid) if x not in universe)
        c["n_id"] = len(set(cid))
    pool.append((t, cands))

# ---------- scoring helpers ----------
def metric(t, c):
    clean_gt = rfb.remove_comments(t["groundtruth"])
    pl = [l.strip() for l in c["clean"].splitlines() if l.strip()]
    gl = [l.strip() for l in clean_gt.splitlines() if l.strip()]
    if t["benchmark"].startswith("repoeval"):
        pe = pl[: len(gl)]
        em = int(len(gl) == len(pe) and gl == pe)
        a, b = "\n".join(gl), "\n".join(pe)
        m = max(len(a), len(b))
        es = (1 - rfb.levenshtein_distance(a, b) / m) * 100 if m else 100.0
    else:
        em = int(pl == gl)
        es = rfb.cal_edit_sim_single(c["clean"], clean_gt)
    return em, es

from fuzzywuzzy import fuzz

def select(cands, cfg):
    groups = defaultdict(list)
    for c in cands: groups[c["key"]].append(c)
    n = len(cands)
    reps = []
    for k, g in groups.items():
        reps.append({"c": g[0], "votes": len(g), "mlp": max(x["mlp"] for x in g), "ung": g[0]["n_ung"], "nid": g[0]["n_id"]})
    mode = cfg["mode"]
    if mode == "greedy":
        return cands[0]
    if mode == "maxlogp":
        return max(reps, key=lambda r: r["mlp"])["c"]
    if mode == "vote":
        return max(reps, key=lambda r: (r["votes"], r["mlp"]))["c"]
    a, b, gam = cfg.get("a", 1.0), cfg.get("b", 0.0), cfg.get("g", 0.0)
    import math
    best, bs = None, -1e18
    for r in reps:
        s = a * math.log(r["votes"] / n) + b * r["mlp"] - gam * r["ung"]
        if mode == "mbr":
            s = a * sum(o["votes"] * fuzz.ratio(r["c"]["clean"].strip(), o["c"]["clean"].strip()) / 100.0 for o in reps) / n + b * r["mlp"] - gam * r["ung"]
        if r["c"]["greedy"]: s += cfg.get("greedy_bonus", 0.0)
        if s > bs: bs, best = s, r["c"]
    return best

def evaluate(idxs, cfg):
    by = defaultdict(list)
    for i in idxs:
        t, cands = pool[i]
        em, es = metric(t, select(cands, cfg))
        by[t["benchmark"]].append((em, es))
    per = {b: (100 * sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)) for b, v in by.items()}
    macro_em = sum(v[0] for v in per.values()) / len(per)
    macro_es = sum(v[1] for v in per.values()) / len(per)
    return macro_em, macro_es, per

grid = [{"mode": "greedy"}, {"mode": "maxlogp"}, {"mode": "vote"}]
for a, b, g, gb in itertools.product([0.0, 1.0], [0.0, 1.0, 3.0], [0.0, 0.5, 1.0, 2.0], [0.0, 0.5]):
    if a == 0 and b == 0 and g == 0: continue
    grid.append({"mode": "score", "a": a, "b": b, "g": g, "greedy_bonus": gb})
    grid.append({"mode": "mbr", "a": max(a, 1.0), "b": b, "g": g, "greedy_bonus": gb})

idx_all = list(range(len(pool)))
rng2 = random.Random(3); rng2.shuffle(idx_all)
fold = [idx_all[0::2], idx_all[1::2]]
report = {"n_tasks": len(pool), "n_samples": N_SAMPLES, "temp": TEMP}
rep_cv = {}
base_all = evaluate(idx_all, {"mode": "greedy"})
report["greedy_all"] = {"em": base_all[0], "es": base_all[1], "per": base_all[2]}
orc = 0
for t, cands in pool:
    orc += max(metric(t, c)[0] for c in cands)
report["oracle_em_any_candidate"] = 100 * orc / len(pool)  # upper bound (per-task, not macro)

cv = []
for k in (0, 1):
    tr_i, te_i = fold[k], fold[1 - k]
    scored = [(evaluate(tr_i, c)[0] + 0.05 * evaluate(tr_i, c)[1], c) for c in grid]
    best_cfg = max(scored, key=lambda x: x[0])[1]
    te = evaluate(te_i, best_cfg); bs = evaluate(te_i, {"mode": "greedy"})
    cv.append({"cfg": best_cfg, "heldout_em": te[0], "heldout_es": te[1], "greedy_em": bs[0], "greedy_es": bs[1]})
report["cv"] = cv
report["cv_mean_gain_em"] = sum(c["heldout_em"] - c["greedy_em"] for c in cv) / 2
report["cv_mean_gain_es"] = sum(c["heldout_es"] - c["greedy_es"] for c in cv) / 2
report["fixed_selectors_all"] = {}
for name, cfg in (("maxlogp", {"mode": "maxlogp"}), ("vote", {"mode": "vote"}),
                  ("mbr_b1", {"mode": "mbr", "a": 1.0, "b": 1.0, "g": 0.0}),
                  ("grounded_mbr", {"mode": "mbr", "a": 1.0, "b": 1.0, "g": 1.0}),
                  ("grounded_vote", {"mode": "score", "a": 1.0, "b": 1.0, "g": 1.0})):
    r = evaluate(idx_all, cfg)
    report["fixed_selectors_all"][name] = {"em": r[0], "es": r[1], "per": r[2]}
(OUT / "rerank_pilot_metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
with gzip.open(OUT / "rerank_pilot_pool.jsonl.gz", "wt") as f:
    for t, cands in pool:
        f.write(json.dumps({"task_id": t["task_id"], "benchmark": t["benchmark"], "gt": t["groundtruth"], "cands": cands}) + "\n")
print(json.dumps(report, indent=1), flush=True)
