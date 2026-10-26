#!/usr/bin/env python3
r"""D1-CPU (Gate 0, 0 CU): Repo Call-Site Extractor & Empirical Verification vs API Definitions.

Hypotheses tested on CPU before spending any Colab GPU Compute Units:
  H1a (Call-Site > Definition coverage of missing args/kwargs):
      For samples where the baseline generator predicts the right callee `fn(` (or in general fails),
      extracting top-3 repo Call-Sites (`oracle-U` and zero-leakage `self-U` triggered by the draft's `fn(`)
      recovers the exact missing GT argument/keyword/member tokens at a significantly higher rate than
      `oracle-D` (definitions/signatures).
  H1b (Zero-leakage `self-U` trigger accuracy):
      Triggering on the first syntactic boundary (`fn(` or `obj.`) of the baseline greedy draft
      identifies the true GT callee/receiver and retrieves valid repo call-sites without using GT.
  H1c (Offline Call-Site + Scope-Partitioned Reranking on E2/E5 pools):
      Scoring saved candidates by (i) repo call-site argument/keyword pattern match and
      (ii) anti-copycat scope check (penalizing foreign local variables from cross-file chunks that
      never appear in in-file left context) improves prefix-only selection on E5/E2.

Usage:
  .\.venv-revela\Scripts\python.exe experiments/dep_oracle/d1_callsite_cpu.py --limit 10
  .\.venv-revela\Scripts\python.exe experiments/dep_oracle/d1_callsite_cpu.py
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D  # noqa: E402
import e2_confirm as E2  # noqa: E402
import e7_errors as E7  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
DOT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)")
KW_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)")


def extract_triggers(line):
    """Extract syntactic triggers from a single line of code:
    Returns (callees, dot_pairs) where callees are function/method names before '('
    and dot_pairs are (receiver, attr) around '.'."""
    code = D._COMMENT_RE.sub(" ", D._STR_RE.sub('""', line))
    callees = []
    for m in CALL_RE.finditer(code):
        full = m.group(1)
        short = full.split(".")[-1]
        if short not in D._SKIP:
            callees.append((full, short))
    dots = []
    for m in DOT_RE.finditer(code):
        rec, attr = m.group(1), m.group(2)
        if attr not in D._SKIP:
            dots.append((rec, attr))
    return callees, dots


def extract_call_Touch(lines, idx, max_span=6):
    """Given lines and line index idx where a call starts, grab 1 preceding line + up to max_span lines
    until parentheses balance (or max_span reached)."""
    start = max(0, idx - 1)
    out = []
    paren = 0
    started = False
    for j in range(start, min(len(lines), idx + max_span)):
        l = lines[j].rstrip()
        out.append(l)
        if j >= idx:
            code_part = l.split("#")[0]
            for ch in code_part:
                if ch == "(":
                    paren += 1
                    started = True
                elif ch == ")":
                    paren -= 1
            if started and paren <= 0:
                break
    return "\n".join(out).strip()


class RepoCallSiteIndex:
    """Indexes all call-sites and member accesses across the repo (files + left context, NO right context)."""

    def __init__(self, ex):
        self.ex = ex
        self.imported_stems = set()
        for mod in re.findall(r"^\s*(?:from|import)\s+([\w\.]+)", ex["left"], re.M):
            self.imported_stems.add(mod.split(".")[-1])
        self.local_recent_ids = set(D.identifiers("\n".join(ex["left"].splitlines()[-30:])))
        self.sources = []
        # Cross-file sources
        for f in ex["files"]:
            self.sources.append((f["path"], f["text"], False))
        # In-file left context (excluding last 3 lines right at cursor to avoid trivial self-match)
        left_lines = ex["left"].splitlines()
        if len(left_lines) > 5:
            self.sources.append((ex["path"], "\n".join(left_lines[:-3]), True))

    def find_callsites(self, symbol, max_sites=3):
        """Find top-k call-sites of `symbol(` across repo files and earlier left context."""
        pat = re.compile(rf"\b{re.escape(symbol)}\s*\(")
        def_pat = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b")
        candidates = []
        seen_snips = set()
        for path, text, is_infile in self.sources:
            lines = text.splitlines()
            stem = os.path.splitext(os.path.basename(path))[0]
            is_imp = stem in self.imported_stems
            for i, line in enumerate(lines):
                if not pat.search(line):
                    continue
                if def_pat.search(line):
                    continue  # skip def header; we specifically want USAGE / CALL-SITES!
                snip = extract_call_Touch(lines, i, max_span=5)
                norm_snip = " ".join(snip.split())
                if not norm_snip or norm_snip in seen_snips:
                    continue
                seen_snips.add(norm_snip)
                snip_ids = set(D.identifiers(snip))
                overlap = len(snip_ids & self.local_recent_ids)
                has_kwargs = bool(KW_RE.search(snip))
                # Sort key: prefer imported/same-file, higher local identifier overlap, has explicit kwargs, shorter snippet
                score = (int(is_imp or is_infile), overlap, int(has_kwargs), -len(snip))
                candidates.append((score, path, snip))
        candidates.sort(key=lambda x: x[0], reverse=True)
        blocks = []
        for _, path, snip in candidates[:max_sites]:
            desc = f"file path: {path}\nusage call-site of {symbol}"
            blocks.append({"file_path": path, "description": desc, "code_content": snip})
        return blocks

    def find_member_usages(self, receiver_or_cls, max_sites=3):
        """Find lines accessing `receiver_or_cls.<attr>` across the repo."""
        pat = re.compile(rf"\b{re.escape(receiver_or_cls)}\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)")
        candidates = []
        seen = set()
        for path, text, is_infile in self.sources:
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if not pat.search(line):
                    continue
                snip = "\n".join(l.rstrip() for l in lines[max(0, i - 1): min(len(lines), i + 2)]).strip()
                norm_snip = " ".join(snip.split())
                if norm_snip in seen:
                    continue
                seen.add(norm_snip)
                snip_ids = set(D.identifiers(snip))
                overlap = len(snip_ids & self.local_recent_ids)
                candidates.append(((int(is_infile), overlap, -len(snip)), path, snip))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [{"file_path": p, "description": f"file path: {p}\nmember usage of {receiver_or_cls}", "code_content": s}
                for _, p, s in candidates[:max_sites]]


def analyze_dataset(ds, limit=0):
    subset, cpath = E7.CFG[ds]
    D.DATASET = ds
    e7_rows = [json.loads(l) for l in open(os.path.join(HERE, "modal_out", f"e7_rows_{ds}.jsonl"), encoding="utf-8")]
    if limit:
        e7_rows = e7_rows[:limit]
    wanted = {r["task_id"] for r in e7_rows}
    cands = D.load_cands(os.path.join(HERE, cpath))
    preds_by_b = E7.load_preds(ds)
    best_b = E2.DEV_BEST[ds]
    preds = preds_by_b[best_b]
    exs = D.load_examples(os.path.join(HERE, "aligncoder_data"), subset, wanted)

    stats = Counter()
    examples_log = []

    for r in e7_rows:
        t = r["task_id"]
        ex = exs[t]
        saved = cands[t]["candidates"]
        g1 = E7.first_line(ex["gt"]).strip()
        p1 = E7.first_line(preds[t]).strip()
        em_base = D.simple_score(preds[t], ex["gt"])[0]

        stats["n_total"] += 1
        if em_base:
            stats["n_base_correct"] += 1
            continue
        stats["n_base_wrong"] += 1

        line_pfx = "" if ex["left"].endswith("\n") else ex["left"].split("\n")[-1]
        g_full_line = line_pfx + g1
        p_full_line = line_pfx + p1

        g_callees, g_dots = extract_triggers(g1)
        p_callees, p_dots = extract_triggers(p1)
        # Include triggers from full line (especially when cursor is right after '.' or '(' in CCEval)
        gf_callees, gf_dots = extract_triggers(g_full_line)
        pf_callees, pf_dots = extract_triggers(p_full_line)
        g_all_callees = g_callees + [x for x in gf_callees if x not in g_callees]
        p_all_callees = p_callees + [x for x in pf_callees if x not in p_callees]
        g_all_dots = g_dots + [x for x in gf_dots if x not in g_dots]
        p_all_dots = p_dots + [x for x in pf_dots if x not in p_dots]

        g_ids = set(D.identifiers(g1))
        p_ids = set(D.identifiers(p1))
        missing_gt_ids = g_ids - p_ids
        g_kwargs = set(KW_RE.findall(g1))
        p_kwargs = set(KW_RE.findall(p1))
        missing_kwargs = g_kwargs - p_kwargs

        defs_idx = D.ExampleDefs(ex)
        cs_idx = RepoCallSiteIndex(ex)

        # 1. Check trigger match between self-draft (p_full_line) and groundtruth (g_full_line)
        same_callee = bool(g_all_callees and p_all_callees and g_all_callees[0][1] == p_all_callees[0][1])
        same_dot_rec = bool(g_all_dots and p_all_dots and g_all_dots[0][0] == p_all_dots[0][0])
        if same_callee:
            stats["wrong_same_callee"] += 1
        if same_dot_rec:
            stats["wrong_same_dot_receiver"] += 1
        if same_callee or same_dot_rec:
            stats["wrong_has_valid_self_trigger"] += 1

        # 2. Oracle-D (Definitions of GT symbols) vs Oracle-U (Call-sites of GT triggers) vs Self-U (Call-sites of Pred triggers)
        def_blocks = []
        for name, w, _ in D.gt_symbols(ex, defs_idx):
            if w in ("right", "repo"):
                def_blocks.extend(defs_idx.blocks_for(name, max_defs=2))
        def_text = "\n".join(b["code_content"] for b in def_blocks)
        def_ids = set(D.identifiers(def_text))
        def_kws = set(KW_RE.findall(def_text))

        # Oracle-U blocks
        ou_blocks = []
        for _, short in g_all_callees[:2]:
            ou_blocks.extend(cs_idx.find_callsites(short, max_sites=3))
        for rec, _ in g_all_dots[:2]:
            ou_blocks.extend(cs_idx.find_member_usages(rec, max_sites=2))
        ou_text = "\n".join(b["code_content"] for b in ou_blocks)
        ou_ids = set(D.identifiers(ou_text))
        ou_kws = set(KW_RE.findall(ou_text))

        # Self-U blocks (NO GT used! Only pred's first-line triggers + line_pfx triggers)
        su_blocks = []
        for _, short in p_all_callees[:2]:
            su_blocks.extend(cs_idx.find_callsites(short, max_sites=3))
        for rec, _ in p_all_dots[:2]:
            su_blocks.extend(cs_idx.find_member_usages(rec, max_sites=2))
        # Self-UD: Self-U call-sites + definitions of triggered callees/receivers (still zero GT leak!)
        sud_blocks = list(su_blocks)
        for _, short in p_all_callees[:2]:
            if defs_idx.where(short) == "repo":
                sud_blocks.extend(defs_idx.blocks_for(short, max_defs=1))
        for rec, attr in p_all_dots[:2]:
            for sym in (rec, attr):
                if defs_idx.where(sym) == "repo":
                    sud_blocks.extend(defs_idx.blocks_for(sym, max_defs=1))
        su_text = "\n".join(b["code_content"] for b in su_blocks)
        su_ids = set(D.identifiers(su_text))
        su_kws = set(KW_RE.findall(su_text))
        sud_text = "\n".join(b["code_content"] for b in sud_blocks)
        sud_ids = set(D.identifiers(sud_text))
        sud_kws = set(KW_RE.findall(sud_text))

        if def_blocks:
            stats["oracle_D_nonempty"] += 1
        if ou_blocks:
            stats["oracle_U_nonempty"] += 1
        if su_blocks:
            stats["self_U_nonempty"] += 1

        if missing_gt_ids:
            stats["has_missing_gt_ids"] += 1
            if missing_gt_ids & def_ids:
                stats["oracle_D_hits_missing_id"] += 1
            if missing_gt_ids & ou_ids:
                stats["oracle_U_hits_missing_id"] += 1
            if missing_gt_ids & su_ids:
                stats["self_U_hits_missing_id"] += 1
            if missing_gt_ids & sud_ids:
                stats["self_UD_hits_missing_id"] += 1
            if missing_gt_ids <= (ou_ids | cs_idx.local_recent_ids):
                stats["oracle_U_plus_local30_covers_ALL_missing_ids"] += 1
            if missing_gt_ids <= (su_ids | cs_idx.local_recent_ids):
                stats["self_U_plus_local30_covers_ALL_missing_ids"] += 1
            if missing_gt_ids <= (sud_ids | cs_idx.local_recent_ids):
                stats["self_UD_plus_local30_covers_ALL_missing_ids"] += 1
            if missing_gt_ids <= (def_ids | cs_idx.local_recent_ids):
                stats["oracle_D_plus_local30_covers_ALL_missing_ids"] += 1

        if missing_kwargs:
            stats["has_missing_kwargs"] += 1
            if missing_kwargs & def_kws:
                stats["oracle_D_hits_missing_kw"] += 1
            if missing_kwargs & ou_kws:
                stats["oracle_U_hits_missing_kw"] += 1
            if missing_kwargs & su_kws:
                stats["self_U_hits_missing_kw"] += 1

        if same_callee and (missing_gt_ids & su_ids) and len(examples_log) < 5:
            examples_log.append({
                "task_id": t,
                "gt": g1,
                "pred": p1,
                "missing_ids_recovered_by_self_U": sorted(missing_gt_ids & su_ids),
                "self_U_top_callsite": su_blocks[0]["code_content"] if su_blocks else "",
                "oracle_D_snippet": def_blocks[0]["code_content"] if def_blocks else "NONE"
            })

    return dict(stats), examples_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = {}
    for ds in ("repoeval", "cceval"):
        stats, ex_log = analyze_dataset(ds, a.limit)
        out[ds] = {"stats": stats, "examples": ex_log}
        nw = max(stats.get("n_base_wrong", 1), 1)
        nm = max(stats.get("has_missing_gt_ids", 1), 1)
        print(f"\n=== {ds.upper()} (Total={stats['n_total']}, BaseWrong={stats['n_base_wrong']}) ===")
        print(f"  Self-Trigger Valid (same callee or dot receiver as GT): {stats['wrong_has_valid_self_trigger']} ({100*stats['wrong_has_valid_self_trigger']/nw:.1f}% of wrong)")
        print(f"  Non-empty blocks retrieved: Oracle-D={stats['oracle_D_nonempty']} | Oracle-U={stats['oracle_U_nonempty']} | Self-U={stats['self_U_nonempty']}")
        print(f"  Samples with missing GT identifiers (n={nm}):")
        print(f"    Oracle-D (Definitions, leaks GT) hits >=1 missing ID : {stats['oracle_D_hits_missing_id']:4d} ({100*stats['oracle_D_hits_missing_id']/nm:5.1f}%)")
        print(f"    Oracle-U (Call-Sites,  leaks GT) hits >=1 missing ID : {stats['oracle_U_hits_missing_id']:4d} ({100*stats['oracle_U_hits_missing_id']/nm:5.1f}%)")
        print(f"    Self-U   (Call-Sites, NO GT!)    hits >=1 missing ID : {stats['self_U_hits_missing_id']:4d} ({100*stats['self_U_hits_missing_id']/nm:5.1f}%)")
        print(f"    Self-UD  (Call+Def,   NO GT!)    hits >=1 missing ID : {stats['self_UD_hits_missing_id']:4d} ({100*stats['self_UD_hits_missing_id']/nm:5.1f}%)")
        print(f"  Covers 100% of missing IDs (combined with last 30 in-file lines):")
        print(f"    Oracle-D + Local30 : {stats['oracle_D_plus_local30_covers_ALL_missing_ids']:4d} ({100*stats['oracle_D_plus_local30_covers_ALL_missing_ids']/nm:5.1f}%)")
        print(f"    Oracle-U + Local30 : {stats['oracle_U_plus_local30_covers_ALL_missing_ids']:4d} ({100*stats['oracle_U_plus_local30_covers_ALL_missing_ids']/nm:5.1f}%)")
        print(f"    Self-U   + Local30 : {stats['self_U_plus_local30_covers_ALL_missing_ids']:4d} ({100*stats['self_U_plus_local30_covers_ALL_missing_ids']/nm:5.1f}%)")
        print(f"    Self-UD  + Local30 : {stats['self_UD_plus_local30_covers_ALL_missing_ids']:4d} ({100*stats['self_UD_plus_local30_covers_ALL_missing_ids']/nm:5.1f}%)")
        for ex in ex_log[:3]:
            print(f"  [Example {ex['task_id']}]")
            print(f"    GT  : {ex['gt']}")
            print(f"    PRED: {ex['pred']}")
            print(f"    Recovered IDs by Self-U: {ex['missing_ids_recovered_by_self_U']}")
            print(f"    Self-U Call-Site: {ex['self_U_top_callsite'].splitlines()[0] if ex['self_U_top_callsite'] else ''} ...")
    if not a.limit:
        os.makedirs(os.path.join(HERE, "modal_out"), exist_ok=True)
        with open(os.path.join(HERE, "modal_out", "d1_callsite_cpu_summary.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
