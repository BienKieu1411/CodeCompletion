#!/usr/bin/env python3
"""E9: E2's per-sample budget choice inside CADRE v2.8 (RLCoder retrieval, training-free).

Pipeline = partner's CADRE v2.8 (`cadre_v28_unified.py`, verbatim copy of BienKieu ba5983c
run_cadre_v28_unified.py, sha1 3f6b6907): window 3072 (the partner's predictions match 3072, 397/400
exact on a 400-task probe, not the CLI default 2560; 2560 kept as `--window 2560`), RLCoder top-10 chunks from the stored trace, tree stub
(CSSIContractBuilder), target spec from the pass-1 tree draft, copy gate 0.95 -> fallback to pass1_raw.
Only `max_exemplar_tokens` (cross-file chunk cap) changes: 256 / 1024 / 1536. 1536 is CADRE's default (the cap is
then bound by min_left 1024, ~1.3-1.4k in practice), so b1536 == CADRE v2.8. The tree block and the target spec
are built once (spec from the default-budget tree draft) and shared across budgets, so only the budget varies.
Per budget: raw_b (gate fallback) + tree_b (tasks without spec) or tree_spec_b (tasks with spec), then the gate.
Conf = first-line mean log-prob of the final output of that budget (generator's own tokens, no GT).

Pre-registered (2026-10-27, before any run). Full test; PRIMARY sets RepoEval-API 1600 + CCEval-Python 2665
(same as E2), CCEval-Java / RepoEval-line reported as secondary. Two scorers: CADRE official
(`evaluate_records_official`, RepoEval = per-repo macro) and ours (`simple_score`, micro).
    Anchor: b1536 final vs partner's cadre_v28_final: official EM within 0.5 per benchmark, per-sample prediction
        agreement >= 98%.
    E9a (axis exists): oracle over the 3 budgets - best fixed budget >= 3 EM (simple) on both PRIMARY sets.
    E9b (PRIMARY): conf3 > max(CADRE default b1536, best fixed budget) on both PRIMARY sets under BOTH scorers,
        AND pooled McNemar (per-sample official EM, conf3 vs b1536, both sets) p < 0.05.
    No learned parameters: conf3 and the budget set were fixed on AlignCoder dev (E2), so this test is out of sample.
E9c (pre-registered 2026-10-27 after E9 FAIL, before any E9c run): E9 kept {256,1024,1536} from the 2048 window,
    so the top budget was 50% of 3072 instead of E2's 75%. Rescale by E2's window fractions 12.5/50/75% of 3072 ->
    --budgets 384,1536,2304. CADRE keeps >= 1024 in-file tokens, which caps cross-file at ~1.5k, so budgets above
    1536 use --min-left-big 256 (b1536 and below keep 1024, so b1536 is still CADRE v2.8). Same anchor, same
    E9a / E9b thresholds, b1536 is still the reference. E9 itself stays FAIL whatever E9c gives.
E9d (pre-registered 2026-10-28, before E9c results were read): conf over 6 candidates per task = {raw, main} x the
    E9c budgets (both already generated per budget; the copy gate is replaced by conf). Run also stores raw_conf /
    main_conf, so it reproduces E9c (conf3 on the gated `final`) in the same run.
    E9d PASS: conf6 > conf3 on both PRIMARY sets under BOTH scorers AND pooled McNemar (official EM, conf6 vs
    conf3) p < 0.05. Oracle over the 6 candidates reported as the ceiling.

usage (Colab):
    python e9_cadre_budget.py run --data-root /content/e9data --cadre /content/cadre_official --out $D/runs/e9 [--limit 8]
    python e9_cadre_budget.py analyze --out $D/runs/e9 --ref <cadre_v28_8004_predictions.jsonl.gz>
"""
import argparse
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BUDGETS = [256, 1024, 1536]  # overridden by --budgets
PRIMARY = ["repoeval_api", "cceval_python"]
BMS = ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]


def cmd_run(a):
    sys.path.insert(0, a.cadre)
    import cadre_v28_unified as C
    from cadre.cssi import CSSIContractBuilder
    from cadre.intent_parser import IntentParser, clean_bpe_text, strip_method_affixes
    C.CSSIContractBuilder, C.IntentParser = CSSIContractBuilder, IntentParser  # in case the module's import failed
    C.clean_bpe_text, C.strip_method_affixes = clean_bpe_text, strip_method_affixes
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import budget_sweep as BS

    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.rev, use_fast=True)
    files = {r["file_id"]: {"path": r["path"], "text": r["text"]}
             for r in map(json.loads, gzip.open(root / "files.jsonl.gz", "rt"))}
    traces = {(r["benchmark"], r["task_id"]): r
              for r in map(json.loads, gzip.open(root / "rlcoder_full_8004_retrieval_trace.jsonl.gz", "rt"))}
    tasks = []
    for t in map(json.loads, gzip.open(root / "tasks.jsonl.gz", "rt")):
        if t["benchmark"] in BMS and (not a.limit or sum(x["benchmark"] == t["benchmark"] for x in tasks) < a.limit):
            tasks.append(t)
    trs = [traces.get((t["benchmark"], t["task_id"]), {}) for t in tasks]
    pools = [[files[f] for f in t.get("file_ids_by_path", {}).values() if f in files] for t in tasks]
    trees = [CSSIContractBuilder.synthesize_contract(language=t["language"], left_context=t["left_context"],
                                                     pool_files=pf, mode="decorated_stub", max_methods_per_class=6)
             for t, pf in zip(tasks, pools)]
    print(f"{len(tasks)} tasks, {sum(map(bool, trees))} trees", flush=True)

    llm = LLM(model=a.model, tokenizer=a.model, revision=a.rev, tokenizer_revision=a.rev, dtype="float16",
              max_model_len=a.window, max_num_seqs=128, max_num_batched_tokens=32768, gpu_memory_utilization=a.gpu_mem,
              seed=20261026, enforce_eager=True, disable_log_stats=True)
    sp = SamplingParams(max_tokens=64, temperature=0.0, top_p=1.0, logprobs=1)

    def gen(prompts):  # partner's decode rule + chosen-token log-probs
        t0, res = time.time(), []
        for o in llm.generate([{"prompt_token_ids": p} for p in prompts], sampling_params=sp):
            c = o.outputs[0]
            ids = list(c.token_ids)
            dec = tok.decode(ids, skip_special_tokens=True)
            text = dec if sum(ch.isspace() for ch in dec) >= sum(ch.isspace() for ch in c.text) else c.text
            keep = ids.index(tok.eos_token_id) if tok.eos_token_id in ids else len(ids)
            lp = [round(c.logprobs[j][ids[j]].logprob, 4) for j in range(keep)]
            res.append({"text": text, "lp": lp, "pieces": [tok.decode([i]) for i in ids[:keep]]})
        print(f"  gen {len(prompts)} in {time.time() - t0:.0f}s", flush=True)
        return res

    P = lambda b, t, tr, tb="", sb="": C.build_cadre_prompt_v28(tok, t, tr, tree_cssi_block=tb, target_spec_block=sb,
                                                               max_exemplar_tokens=b, max_total_tokens=a.window,
                                                               min_left_tokens=1024 if b <= 1536 else a.min_left_big)
    # spec from the default-budget tree draft, shared by every budget
    tree_def = gen([P(1536, t, tr, tb) for t, tr, tb in zip(tasks, trs, trees)])
    specs = [C.resolve_target_spec_with_hierarchy(t, tr, pf, d["text"]) for t, tr, pf, d in zip(tasks, trs, pools, tree_def)]
    print(f"spec triggered {sum(map(bool, specs))}/{len(tasks)}", flush=True)
    for b in BUDGETS:
        f_out = out / f"b{b}.jsonl.gz"
        if f_out.exists():
            continue
        raw = gen([P(b, t, tr) for t, tr in zip(tasks, trs)])
        idx = [i for i in range(len(tasks)) if specs[i] or b != 1536]
        main = dict(zip(idx, gen([P(b, tasks[i], trs[i], trees[i], specs[i]) for i in idx])))
        with gzip.open(f_out, "wt", encoding="utf-8") as f:
            for i, t in enumerate(tasks):
                c = main.get(i, tree_def[i])
                refs = [x.get("code") or x.get("text") or x.get("code_content") or "" for x in
                        (trs[i].get("rlcoder_top10") or trs[i].get("aligncoder_top10") or trs[i].get("spc_chunks")
                         or trs[i].get("retrieved_chunks") or [])[:5]]
                lines = c["text"].splitlines()
                sim = C.max_line_similarity(lines[0] if lines else c["text"], [r for r in refs if r.strip()])
                fin = raw[i] if sim >= a.copy_threshold else c
                f.write(json.dumps({"task_id": t["task_id"], "benchmark": t["benchmark"], "language": t["language"],
                                    "groundtruth": t["groundtruth"], "raw": raw[i]["text"], "main": c["text"],
                                    "final": fin["text"], "gate": "copy" if fin is raw[i] else "main",
                                    "conf": BS.first_line_conf(fin["pieces"], fin["lp"]),
                                    "raw_conf": BS.first_line_conf(raw[i]["pieces"], raw[i]["lp"]),
                                    "main_conf": BS.first_line_conf(c["pieces"], c["lp"])}, ensure_ascii=False) + "\n")
        print(f">>> b{b} written", flush=True)


def cmd_analyze(a):
    import cadre_v28_unified as C
    import dep_oracle as D
    out = Path(a.out)
    R = {b: {(r["benchmark"], r["task_id"]): r for r in map(json.loads, gzip.open(out / f"b{b}.jsonl.gz", "rt",
                                                                                   encoding="utf-8"))}
         for b in BUDGETS}
    keys = sorted(set.intersection(*[set(R[b]) for b in BUDGETS]))
    pick = {k: max(BUDGETS, key=lambda b: R[b][k]["conf"]) for k in keys}

    def official(pred_of):  # CADRE scorer on a chosen prediction per task (per-sample em kept for McNemar)
        recs = [dict(R[1536][k], pred=pred_of(k)) for k in keys]
        per = {}
        for r in recs:
            per[(r["benchmark"], r["task_id"])] = C.evaluate_records_official([r], "pred")[r["benchmark"]]["em"] > 0
        return C.evaluate_records_official(recs, "pred"), per

    sys_pred = {f"b{b}": (lambda b: lambda k: R[b][k]["final"])(b) for b in BUDGETS}
    sys_pred["conf3"] = lambda k: R[pick[k]][k]["final"]
    has6 = all("raw_conf" in R[b][k] for b in BUDGETS for k in keys)
    if has6:  # E9d
        cand = [(b, v) for b in BUDGETS for v in ("raw", "main")]
        pick6 = {k: max(cand, key=lambda bv: R[bv[0]][k][bv[1] + "_conf"]) for k in keys}
        sys_pred["conf6"] = lambda k: R[pick6[k][0]][k][pick6[k][1]]
    off, off_per, simp = {}, {}, {}
    for s, f in sys_pred.items():
        off[s], off_per[s] = official(f)
        simp[s] = {k: D.simple_score(f(k), R[1536][k]["groundtruth"])[0] for k in keys}
    res, pooled = {}, Counter()
    for bm in BMS:
        ks = [k for k in keys if k[0] == bm]
        if not ks:
            continue
        n = len(ks)
        sm = {s: round(100.0 * sum(simp[s][k] for k in ks) / n, 2) for s in sys_pred}
        om = {s: off[s][bm]["em"] for s in sys_pred}
        best_s = max(BUDGETS, key=lambda b: sm[f"b{b}"])
        best_o = max(BUDGETS, key=lambda b: om[f"b{b}"])
        orc = round(100.0 * sum(max(simp[f"b{b}"][k] for b in BUDGETS) for k in ks) / n, 2)
        w = sum(off_per["conf3"][k] and not off_per["b1536"][k] for k in ks)
        l_ = sum(off_per["b1536"][k] and not off_per["conf3"][k] for k in ks)
        if bm in PRIMARY:
            pooled["w"] += w
            pooled["l"] += l_
        if has6:
            o6 = round(100.0 * sum(max(D.simple_score(R[b][k][v], R[1536][k]["groundtruth"])[0] for b, v in cand)
                                   for k in ks) / n, 2)
            w6 = sum(off_per["conf6"][k] and not off_per["conf3"][k] for k in ks)
            l6 = sum(off_per["conf3"][k] and not off_per["conf6"][k] for k in ks)
            if bm in PRIMARY:
                pooled["w6"] += w6
                pooled["l6"] += l6
        res[bm] = {"n": n, "simple": sm, "official": om, "es_official": {s: off[s][bm]["es"] for s in sys_pred},
                   "best_fixed_simple": best_s, "best_fixed_official": best_o, "oracle3_simple": orc,
                   "oracle_gap_simple": round(orc - sm[f"b{best_s}"], 2),
                   "conf3_vs_b1536_official": [w, l_, round(D.mcnemar(w, l_), 4)],
                   "conf3_mix": dict(Counter(pick[k] for k in ks)),
                   "gate_copy_pct_b1536": round(100.0 * sum(R[1536][k]["gate"] == "copy" for k in ks) / n, 2),
                   "pass_simple": sm["conf3"] > max(sm["b1536"], sm[f"b{best_s}"]),
                   "pass_official": om["conf3"] > max(om["b1536"], om[f"b{best_o}"])}
        if has6:
            res[bm].update({"oracle6_simple": o6, "conf6_vs_conf3_official": [w6, l6, round(D.mcnemar(w6, l6), 4)],
                            "conf6_mix": dict(Counter(f"{b}{v[0]}" for b, v in (pick6[k] for k in ks))),
                            "pass6": sm["conf6"] > sm["conf3"] and om["conf6"] > om["conf3"]})
    if a.ref:
        ref = {(r["benchmark"], r["task_id"]): r for r in map(json.loads, gzip.open(a.ref, "rt", encoding="utf-8"))}
        for bm in res:
            ks = [k for k in keys if k[0] == bm and k in ref]
            rs = C.evaluate_records_official([ref[k] for k in ks], "cadre_v28_final")
            res[bm]["anchor_ref_official_em"] = rs[bm]["em"]
            res[bm]["anchor_pred_agree_pct"] = round(100.0 * sum(R[1536][k]["final"] == ref[k]["cadre_v28_final"]
                                                                 for k in ks) / max(1, len(ks)), 2)
            res[bm]["anchor_ok"] = abs(res[bm]["official"]["b1536"] - rs[bm]["em"]) <= 0.5 and \
                res[bm]["anchor_pred_agree_pct"] >= 98.0
    p = D.mcnemar(pooled["w"], pooled["l"])
    res["pooled_primary_conf3_vs_b1536"] = [pooled["w"], pooled["l"], round(p, 4)]
    res["E9a"] = "PASS" if all(res[b]["oracle_gap_simple"] >= 3 for b in PRIMARY if b in res) else "FAIL"
    res["E9b_primary"] = "PASS" if all(res[b]["pass_simple"] and res[b]["pass_official"] for b in PRIMARY
                                       if b in res) and p < 0.05 else "FAIL"
    if has6:
        p6 = D.mcnemar(pooled["w6"], pooled["l6"])
        res["pooled_primary_conf6_vs_conf3"] = [pooled["w6"], pooled["l6"], round(p6, 4)]
        res["E9d"] = "PASS" if all(res[b]["pass6"] for b in PRIMARY if b in res) and p6 < 0.05 else "FAIL"
    json.dump(res, open(out / "e9_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="/content/e9data")
    ap.add_argument("--cadre", default="/content/cadre_official")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--limit", type=int, default=0, help="per benchmark")
    ap.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-base")
    ap.add_argument("--rev", default="c919139c3a9b4070729c8b2cca4847ab29ca8d94")
    ap.add_argument("--window", type=int, default=3072)
    ap.add_argument("--budgets", default="256,1024,1536")
    ap.add_argument("--min-left-big", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--copy-threshold", type=float, default=0.95)
    a = ap.parse_args()
    global BUDGETS
    BUDGETS = [int(x) for x in a.budgets.split(",")]
    cmd_run(a) if a.cmd == "run" else cmd_analyze(a)


if __name__ == "__main__":
    main()
