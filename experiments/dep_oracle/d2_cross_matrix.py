#!/usr/bin/env python3
r"""D2 (Prefill-only GPU ~15-20 min L4, then CPU analyze): Counterfactual Cross-Context Likelihood Matrix (C2).

Hypothesis:
  Self-confidence log P(y_i | x, C_i) (the diagonal M_{i,i} used by E2 and Ciniselli et al.) suffers from
  "Echo-Chamber / Copycat bias": if context C_i contains a distractor line, the generator copies it into y_i
  with high self-confidence under C_i, whereas other context views C_j disagree.
  By computing the full Cross-Context Likelihood Matrix M_{i,j} = mean_logprob(y_i_line1 | x, C_j) across
  i in {256, 512, 1024, 1536} and j in {0, 256, 512, 1024, 1536} (requiring ONLY prefill forward passes,
  ZERO generation), we can extract:
    - Cross-Context Consensus : mean_j M_{i,j}
    - Causal Context Lift     : M_{i,i} - M_{i,0} (evidence support vs no-cross-file baseline B=0)
    - Echo-Chamber Penalty    : M_{i,i} - mean_{j != i} M_{i,j} (flags copycat hallucinations)

Pre-registered Gate (from directions_beyond_budget_2026-10-25.md):
  - PASS if cross-fitted D2 verifier beats E2 (`PRIMARY learned_256_1024_1536`) on BOTH datasets and
    pooled McNemar p < 0.05.

Usage:
  # 1. GPU prefill scoring on Colab L4 (resumable JSONL):
  python d2_cross_matrix.py score --dataset repoeval --cands prediction_with_candidates.jsonl --data_dir $DATA --out $D/runs/d2_repoeval
  python d2_cross_matrix.py score --dataset cceval --cands cceval_python_prediction_with_candidates.jsonl --data_dir $DATA --out $D/runs/d2_cceval
  # 2. CPU evaluation (cross-fit dev <-> hold-out, exact same protocol as e2_confirm.py):
  python d2_cross_matrix.py analyze --repoeval_dir modal_out/d2_repoeval --cceval_dir modal_out/d2_cceval
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep as BS  # noqa: E402
import dep_oracle as D  # noqa: E402
import e2_confirm as E2  # noqa: E402
import e7_errors as E7  # noqa: E402
import likelihood_probe as LP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CAND_B = [256, 512, 1024, 1536]
CTX_B = [0, 256, 512, 1024, 1536]


def first_line_with_nl(text):
    lines = text.split("\n")
    return lines[0] + ("\n" if len(lines) > 1 else "")


def cmd_score(a):
    import torch
    from transformers import AutoModelForCausalLM

    subset, cpath = E7.CFG[a.dataset]
    D.DATASET = a.dataset
    cands_path = a.cands if os.path.isabs(a.cands) else os.path.join(HERE, a.cands)
    cands = D.load_cands(cands_path)
    preds_by_b = E7.load_preds(a.dataset)
    all_ids = sorted(set.intersection(*[set(preds_by_b[b]) for b in CAND_B]))
    if a.limit:
        all_ids = all_ids[:a.limit]
    exs = D.load_examples(a.data_dir, subset, set(all_ids))

    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    os.makedirs(a.out, exist_ok=True)
    out_file = os.path.join(a.out, "matrix.jsonl")
    done = set()
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["task_id"])
    todo = [t for t in all_ids if t not in done]
    print(f"[D2 {a.dataset}] Total={len(all_ids)}, Done={len(done)}, Todo={len(todo)}")
    if not todo:
        return

    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    t0 = time.time()
    with open(out_file, "a", encoding="utf-8") as fout:
        for idx, t in enumerate(todo, 1):
            ex = exs[t]
            saved = cands[t]["candidates"]
            # Build context prompt token IDs for each context budget j in CTX_B
            ctx_pids = {}
            for bj in CTX_B:
                pr, _ = D.build_prompt(tok, ex, saved if bj > 0 else [], max_cross=bj)
                ctx_pids[bj] = tok.encode(pr, add_special_tokens=False)

            # De-duplicate unique first-line targets across candidate budgets i in CAND_B
            cand_lines = {bi: first_line_with_nl(preds_by_b[bi][t]) for bi in CAND_B}
            unique_lines = sorted(set(cand_lines.values()))
            line_to_tids = {l: tok.encode(l, add_special_tokens=False)[:64] for l in unique_lines}

            # Build (prompt_ids, target_ids) pairs only for unique lines
            seqs, keys = [], []
            for l in unique_lines:
                tids = line_to_tids[l]
                if not tids:
                    continue
                for bj in CTX_B:
                    seqs.append((ctx_pids[bj], tids))
                    keys.append((l, bj))

            # Score in mini-batches of a.batch_size
            lp_map = {}
            for k in range(0, len(seqs), a.batch_size):
                sub_s = seqs[k: k + a.batch_size]
                sub_k = keys[k: k + a.batch_size]
                res_lps = LP.target_logprobs(model, tok, sub_s, model.device, with_argmax=False)
                for (l, bj), lps in zip(sub_k, res_lps):
                    lp_map[(l, bj)] = round(sum(lps) / len(lps), 5) if lps else -10.0

            # Assemble 4x5 matrix M[str(bi)][str(bj)]
            mat = {}
            for bi in CAND_B:
                l = cand_lines[bi]
                mat[str(bi)] = {str(bj): lp_map.get((l, bj), -10.0) for bj in CTX_B}

            row = {"task_id": t, "matrix": mat}
            fout.write(json.dumps(row) + "\n")
            if idx % 20 == 0 or idx == len(todo):
                fout.flush()
                el = time.time() - t0
                print(f"  [{idx}/{len(todo)}] elapsed={el:.1f}s ({el/idx:.2f}s/item)")


def matrix_features_for_budget(mat_row, b, S):
    """Extract causal & consensus features for candidate budget `b` from the cross-context matrix."""
    m_b = mat_row[str(b)]
    self_lp = m_b[str(b)]
    null_lp = m_b["0"]
    other_lps = [m_b[str(bj)] for bj in S if bj != b]
    all_lps = [m_b[str(bj)] for bj in S]
    mean_all = sum(all_lps) / len(all_lps)
    mean_other = sum(other_lps) / max(len(other_lps), 1)
    min_all = min(all_lps)
    lift = self_lp - null_lp            # Causal Context Lift over B=0
    echo = self_lp - mean_other         # Echo-Chamber / Copycat divergence
    return [self_lp, mean_all, min_all, lift, echo]


def cmd_analyze(a):
    import numpy as np

    dev = E2.pooled({ds: os.path.join(HERE, f"kaggle_sweep_{ds}_out", "sweep") for ds in E2.DEV_BEST})
    hold = E2.pooled({ds: os.path.join(HERE, "modal_out", f"e2hold_{ds}") for ds in E2.DEV_BEST})
    did, dem, dconf, dst, dds = dev
    hid, hem, hconf, hst, hds = hold

    # Load matrices
    mats = {}
    for ds, mdir in [("repoeval", a.repoeval_dir), ("cceval", a.cceval_dir)]:
        mfile = os.path.join(mdir, "matrix.jsonl")
        if os.path.exists(mfile):
            for r in map(json.loads, open(mfile, encoding="utf-8")):
                mats[r["task_id"]] = r["matrix"]

    did = [t for t in did if t in mats]
    hid = [t for t in hid if t in mats]
    all_ids = did + hid
    em_all = {b: {**dem[b], **hem[b]} for b in CAND_B}
    conf_all = {b: {**dconf[b], **hconf[b]} for b in CAND_B}
    st_all = {**dst, **hst}
    ds_all = {**dds, **hds}

    S = [256, 1024, 1536]

    def crossfit(feat_fn):
        ch = {}
        for tr_ids, te_ids in [(did, hid), (hid, did)]:
            Xtr = [feat_fn(t) for t in tr_ids]
            ytr = {b: [em_all[b][t] for t in tr_ids] for b in S}
            Xte = [feat_fn(t) for t in te_ids]
            preds = E2.fit_predict(Xtr, ytr, Xte, S)
            ch.update(zip(te_ids, preds))
        return ch

    pols = {}
    pols["per_dataset_best_fixed"] = {t: E2.DEV_BEST[ds_all[t]] for t in all_ids}
    pols["E2_conf_256_1024_1536"] = {t: max(S, key=lambda b: conf_all[b][t]) for t in all_ids}
    pols["E2_learned_256_1024_1536"] = crossfit(lambda t: st_all[t] + [conf_all[b][t] for b in S])

    # Training-free Cross-Context Consensus & Echo-Penalized Consensus
    pols["D2_consensus_mean"] = {
        t: max(S, key=lambda b: sum(mats[t][str(b)][str(bj)] for bj in S) / len(S)) for t in all_ids
    }
    pols["D2_consensus_minus_echo"] = {
        t: max(S, key=lambda b: (
            sum(mats[t][str(b)][str(bj)] for bj in S) / len(S)
            - 0.25 * max(0.0, mats[t][str(b)][str(b)] - sum(mats[t][str(b)][str(bj)] for bj in S if bj != b) / (len(S) - 1))
        )) for t in all_ids
    }
    # Learned D2 Verifier (static + cross-context matrix features)
    pols["D2_learned_matrix"] = crossfit(
        lambda t: st_all[t] + [x for b in S for x in matrix_features_for_budget(mats[t], b, S)]
    )
    pols["oracle_3budgets"] = {t: max(S, key=lambda b: em_all[b][t]) for t in all_ids}

    ref = pols["E2_learned_256_1024_1536"]
    for name, ch in pols.items():
        parts = []
        for ds in ("repoeval", "cceval"):
            sub = [t for t in all_ids if ds_all[t] == ds]
            em_v = 100 * sum(em_all[ch[t]][t] for t in sub) / max(len(sub), 1)
            w = sum(em_all[ch[t]][t] > em_all[ref[t]][t] for t in sub)
            l_ = sum(em_all[ch[t]][t] < em_all[ref[t]][t] for t in sub)
            parts.append(f"{ds} {em_v:5.2f} [+{w}/-{l_} p={D.mcnemar(w, l_):.3f}]")
        print(f"{name:28s}: " + " | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("score")
    ps.add_argument("--dataset", choices=["repoeval", "cceval"], required=True)
    ps.add_argument("--cands", required=True)
    ps.add_argument("--data_dir", default="aligncoder_data")
    ps.add_argument("--out", required=True)
    ps.add_argument("--generator", default=D.GEN_DEFAULT)
    ps.add_argument("--batch_size", type=int, default=8)
    ps.add_argument("--limit", type=int, default=0)

    pa = sub.add_parser("analyze")
    pa.add_argument("--repoeval_dir", default="modal_out/d2_repoeval")
    pa.add_argument("--cceval_dir", default="modal_out/d2_cceval")

    a = ap.parse_args()
    if a.cmd == "score":
        cmd_score(a)
    elif a.cmd == "analyze":
        cmd_analyze(a)


if __name__ == "__main__":
    main()
