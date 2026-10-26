#!/usr/bin/env python3
"""Dependency-context diagnostic + oracle for RepoEval (AlignCoder lineage).

Question: when the frozen generator fails, is it because the definitions of the
repo symbols it has to call (functions/classes) never reach the prompt? If so,
at which stage are they lost?
    pool  -> the BM25 + api_blocks candidate pool does not contain the definition
    rank  -> it is in the repo, but the saved AlignCoder top-k does not include it
    budget-> it was retrieved but cut by the 1536-token cross-file budget

Subcommands
    coverage  CPU only. Where are the definitions of the ground-truth symbols,
              and do they survive into AlignCoder's prompt?  (no GPU)
    generate  GPU. Re-generate with a frozen generator under 4 prompt conditions,
              re-using the SAVED AlignCoder candidates (no retriever run):
                none      : in-file context only
                base      : saved AlignCoder candidates (reproduces the prompt)
                oracle    : definitions of GT symbols + base   (upper bound, leaks GT)
                prefixdep : definitions of symbols used near the cursor + base
                            (realistic, no GT; uses current-file suffix + other files)
              The cross-file budget is identical (1536 tokens) in every condition.
    score     CPU. EM/ES per condition + paired comparison against `base`.

Typical Kaggle use (see README.md):
    python dep_oracle.py coverage --cands prediction_with_candidates.jsonl --out out
    python dep_oracle.py generate --cands prediction_with_candidates.jsonl --out out --n 400
    python dep_oracle.py score --out out [--aligncoder_src /kaggle/input/aligncoder-source]
"""
import argparse
import builtins
import difflib
import glob
import json
import keyword
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

DATA_REPO = "AlignCoder/Data4AlignCoder"
DATA_REV = "ce72fe3ef4987e15a9219fea4174bb80700e742c"  # same revision as run_eval.py
GEN_DEFAULT = "deepseek-ai/deepseek-coder-1.3b-base"
MAX_CROSS, MAX_CTX, MAX_NEW = 1536, 2048, 64  # AlignCoder official config
CONDITIONS = ["none", "base", "oracle", "prefixdep", "base_cross512", "oracle_cross512"]
# condition -> cross-file token budget (in-file context gets the rest of MAX_CTX)
CROSS_BUDGET = {"base_cross512": 512, "oracle_cross512": 512}

# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------

DATASET = "repoeval"  # set from --dataset in main(); "cceval" uses subset "python"


def data_folder(data_dir, subset):
    return os.path.join(data_dir, "data", DATASET, subset)


def ensure_data(data_dir, subset):
    pattern = os.path.join(data_folder(data_dir, subset), "test*.parquet")
    if not glob.glob(pattern):
        from huggingface_hub import snapshot_download
        snapshot_download(DATA_REPO, repo_type="dataset", revision=DATA_REV, local_dir=data_dir,
                          allow_patterns=[f"data/{DATASET}/{subset}/*"])
    return sorted(glob.glob(pattern))


def load_examples(data_dir, subset, wanted=None):
    import pandas as pd
    files = ensure_data(data_dir, subset)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = {}
    for row in df[["task_id", "path", "left_context", "right_context", "crossfile_context",
                   "groundtruth"]].itertuples(index=False):
        if wanted is not None and row.task_id not in wanted:
            continue
        cf = row.crossfile_context
        if isinstance(cf, str):
            cf = json.loads(cf)
        files_ = [{"path": x["path"], "text": x["text"]} for x in cf]
        out[row.task_id] = {"task_id": row.task_id, "path": row.path, "left": row.left_context,
                            "right": row.right_context or "", "files": files_,
                            "gt": row.groundtruth}
    return out


def load_cands(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["task_id"]] = r
    return rows


def pick_subset(task_ids, n, seed=0):
    """Stratified by repository, deterministic."""
    if not n or n >= len(task_ids):
        return sorted(task_ids)
    by_repo = defaultdict(list)
    for t in sorted(task_ids):
        by_repo[t.split("/")[0]].append(t)
    rng = random.Random(seed)
    for v in by_repo.values():
        rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < n:
        for repo in sorted(by_repo):
            if i < len(by_repo[repo]) and len(picked) < n:
                picked.append(by_repo[repo][i])
        i += 1
    return sorted(picked)

# ----------------------------------------------------------------------------------
# Symbols and definitions
# ----------------------------------------------------------------------------------

_STR_RE = re.compile(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\')')
_COMMENT_RE = re.compile(r"#[^\n]*")
_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SKIP = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls", "__init__", "__call__"}
DEF_RE = re.compile(r"^([ \t]*)(?:async[ \t]+)?(def|class)[ \t]+([A-Za-z_]\w*)", re.M)


def identifiers(code):
    code = _COMMENT_RE.sub(" ", _STR_RE.sub(" ", code))
    seen, out = set(), []
    for m in _ID_RE.finditer(code):
        name = m.group(0)
        if name not in _SKIP and name not in seen and not name[0].isdigit():
            seen.add(name)
            out.append(name)
    return out


def _signature(lines, i, max_lines=8):
    sig = []
    for j in range(i, min(i + max_lines, len(lines))):
        sig.append(lines[j].rstrip())
        code = lines[j].split("#")[0].rstrip()
        if code.endswith(":"):
            break
    return "\n".join(sig)


def _snippet(text, m, max_methods=25):
    """def -> full signature; class -> header + direct method signatures (AlignCoder style)."""
    lines = text[m.start():].split("\n")
    indent = len(m.group(1).expandtabs())
    if m.group(2) == "def":
        return _signature(lines, 0)
    out = [_signature(lines, 0)]
    body_indent = None
    i, n_methods = 1, 0
    while i < len(lines) and n_methods < max_methods:
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        ind = len(line) - len(line.lstrip())
        if ind <= indent:
            break
        if body_indent is None:
            body_indent = ind
        if ind == body_indent and re.match(r"\s*(async\s+)?def\s", line):
            out.append(_signature(lines, i))
            n_methods += 1
        i += 1
    return "\n".join(out)


def index_defs(text):
    """name -> list of (kind, snippet, enclosing_class)"""
    idx = defaultdict(list)
    classes = []  # stack of (indent, name)
    for m in DEF_RE.finditer(text):
        ind = len(m.group(1).expandtabs())
        while classes and classes[-1][0] >= ind:
            classes.pop()
        encl = classes[-1][1] if classes else None
        idx[m.group(3)].append((m.group(2), m, encl))
        if m.group(2) == "class":
            classes.append((ind, m.group(3)))
    return idx


class ExampleDefs:
    def __init__(self, ex):
        self.ex = ex
        self.left = index_defs(ex["left"])
        self.right = index_defs(ex["right"])
        self.repo = defaultdict(list)  # name -> [(path, kind, m, encl, text)]
        for f in ex["files"]:
            for name, lst in index_defs(f["text"]).items():
                for kind, m, encl in lst:
                    self.repo[name].append((f["path"], kind, m, encl, f["text"]))
        self.imported_stems = set()
        for mod in re.findall(r"^\s*(?:from|import)\s+([\w\.]+)", ex["left"], re.M):
            self.imported_stems.add(mod.split(".")[-1])

    def where(self, name):
        if name in self.left:
            return "left"
        if name in self.right:
            return "right"
        if name in self.repo:
            return "repo"
        return None

    def blocks_for(self, name, max_defs=2):
        """CodeBlock-like dicts holding the definition of `name` outside the left context."""
        blocks = []
        if name in self.right:
            for kind, m, encl in self.right[name][:max_defs]:
                snip = _snippet(self.ex["right"], m)
                desc = f"file path: {self.ex['path']}\ndefinition of {name}" + (f" (in class {encl})" if encl else "")
                blocks.append({"file_path": self.ex["path"], "description": desc, "code_content": snip})
            return blocks
        defs = self.repo.get(name, [])
        defs = sorted(defs, key=lambda d: (os.path.splitext(os.path.basename(d[0]))[0] not in self.imported_stems,
                                           d[1] != "class", len(d[0])))
        for path, kind, m, encl, text in defs[:max_defs]:
            desc = f"file path: {path}\ndefinition of {name}" + (f" (in class {encl})" if encl else "")
            blocks.append({"file_path": path, "description": desc, "code_content": _snippet(text, m)})
        return blocks


def gt_symbols(ex, d):
    """GT identifiers that have a def/class somewhere (current file or repo)."""
    out = []
    for name in identifiers(ex["gt"]):
        w = d.where(name)
        if w:
            out.append((name, w, name in ex["left"]))
    return out


def prefix_symbols(ex, d, last_lines=40):
    """Realistic: identifiers in the last N non-empty lines before the cursor, most recent first,
    whose definition is NOT in the left context (so the model cannot see it otherwise)."""
    lines = [l for l in ex["left"].split("\n") if l.strip()][-last_lines:]
    names = identifiers("\n".join(lines))[::-1]  # rough recency: later first-occurrences first
    return [n for n in names if d.where(n) in ("right", "repo")]

# ----------------------------------------------------------------------------------
# Prompt (mirrors AlignCoder generator.CustomDataset.construct_prompts)
# ----------------------------------------------------------------------------------

def block_str(b, comment="#"):
    s = "\n".join(f"{comment} {cl}" for cl in b["description"].strip().split("\n") if cl) + "\n"
    s += "\n".join(f"{comment} {cl}" for cl in b["code_content"].split("\n") if cl)
    return s.strip()


def build_prompt(tok, ex, blocks, max_cross=MAX_CROSS, max_ctx=MAX_CTX):
    filt = []
    for b in blocks:
        if b["file_path"] == "":
            break
        filt.append(b)
    cross = "\n\n".join(block_str(b) for b in filt)
    cross_ids = tok.encode(cross[: max_cross * 10], add_special_tokens=False)[:max_cross]
    path_ids = tok.encode(f"\n\n# file path: {ex['path']}\n\n", add_special_tokens=False)
    allowed = max_ctx - (len(cross_ids) + len(path_ids) + 10)
    infile_ids = tok.encode(ex["left"], add_special_tokens=False)[-allowed:]
    prompt = tok.decode(cross_ids + path_ids + infile_ids)
    return prompt, {"cross_tokens": len(cross_ids), "infile_tokens": len(infile_ids),
                    "cross_text": tok.decode(cross_ids), "infile_text": tok.decode(infile_ids)}


def has_def(text, name):
    return re.search(rf"\b(?:def|class)\s+{re.escape(name)}\b", text) is not None


def condition_blocks(cond, ex, d, saved, budget_names=12):
    if cond == "none":
        return []
    if cond in ("base", "base_cross512"):
        return saved
    if cond == "oracle_cross512":  # oracle definitions only, small budget, rest goes to in-file context
        inj = []
        for n in [n for n, w, _ in gt_symbols(ex, d) if w in ("right", "repo")][:budget_names]:
            inj.extend(d.blocks_for(n))
        return inj
    if cond == "oracle":
        names = [n for n, w, _ in gt_symbols(ex, d) if w in ("right", "repo")]
    elif cond == "prefixdep":
        names = prefix_symbols(ex, d)
    else:
        raise ValueError(cond)
    inj = []
    for n in names[:budget_names]:
        inj.extend(d.blocks_for(n))
    return inj + [b for b in saved]  # injected first so they survive the 1536-token cut

# ----------------------------------------------------------------------------------
# coverage
# ----------------------------------------------------------------------------------

def get_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.model_max_length = 10 ** 10
    return tok


def cmd_coverage(a):
    cands = load_cands(a.cands)
    ids = pick_subset(list(cands), a.n, a.seed)
    exs = load_examples(a.data_dir, a.subset, set(ids))
    tok = get_tokenizer(a.generator)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for t in ids:
        ex, c = exs[t], cands[t]
        d = ExampleDefs(ex)
        saved = c["candidates"]
        pool_text = "\n".join(b["code_content"] for b in saved)  # saved top-k incl. api blocks
        _, info = build_prompt(tok, ex, saved)
        stop_pos = next((i for i, b in enumerate(saved) if b["file_path"] == ""), len(saved))
        syms = []
        for name, where, used_in_prefix in gt_symbols(ex, d):
            in_topk = has_def(pool_text, name)
            in_prompt_cross = has_def(info["cross_text"], name)
            in_prompt_infile = has_def(info["infile_text"], name)
            syms.append({"name": name, "where": where, "used_in_prefix": used_in_prefix,
                         "n_repo_defs": len(d.repo.get(name, [])),
                         "in_saved_topk": in_topk, "in_prompt": in_prompt_cross or in_prompt_infile})
        rows.append({"task_id": t, "symbols": syms, "stop_pos": stop_pos,
                     "n_saved": len(saved), "cross_tokens": info["cross_tokens"],
                     "infile_tokens": info["infile_tokens"],
                     "left_tokens": len(tok.encode(ex["left"], add_special_tokens=False)),
                     "orig_pred": c.get("pred", ""),
                     "orig_simple_em": simple_score(c.get("pred", ""), ex["gt"])[0]})
    with open(os.path.join(a.out, "coverage_rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize_coverage(rows)
    with open(os.path.join(a.out, "coverage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def summarize_coverage(rows):
    n = len(rows)
    per_where = Counter()
    lost = Counter()  # for symbols defined outside the left context
    ex_needs_ext = 0
    ex_ext_all_in_prompt = 0
    ex_left_truncated = 0
    for r in rows:
        ext = [s for s in r["symbols"] if s["where"] in ("right", "repo")]
        for s in r["symbols"]:
            per_where[s["where"]] += 1
            if s["where"] == "left" and not s["in_prompt"]:
                ex_left_truncated += 1
        if ext:
            ex_needs_ext += 1
            if all(s["in_prompt"] for s in ext):
                ex_ext_all_in_prompt += 1
        for s in ext:
            if s["in_prompt"]:
                lost["reached_prompt"] += 1
            elif s["in_saved_topk"]:
                lost["retrieved_but_cut_by_budget"] += 1
            else:
                lost["not_retrieved"] += 1
            lost[f"where={s['where']}"] += 1
    tot_ext = lost["reached_prompt"] + lost["retrieved_but_cut_by_budget"] + lost["not_retrieved"]
    pct = lambda x, y: round(100.0 * x / y, 2) if y else None
    # same breakdown restricted to unambiguous names (<=3 definitions in the repo)
    unamb = Counter()
    for r in rows:
        for s in r["symbols"]:
            if s["where"] in ("right", "repo") and s["n_repo_defs"] <= 3:
                key = "reached_prompt" if s["in_prompt"] else ("cut" if s["in_saved_topk"] else "not_retrieved")
                unamb[key] += 1
    tot_u = sum(unamb.values())
    # does the ORIGINAL AlignCoder prediction succeed more often when the defs reached the prompt?
    grp = defaultdict(list)
    for r in rows:
        ext = [s for s in r["symbols"] if s["where"] in ("right", "repo")]
        g = "no_external_needed" if not ext else ("all_ext_defs_in_prompt" if all(s["in_prompt"] for s in ext)
                                                 else "some_ext_def_missing")
        grp[g].append(r.get("orig_simple_em", 0))
    em_by_group = {g: {"n": len(v), "orig_simple_em_pct": pct(sum(v), len(v))} for g, v in grp.items()}
    return {
        "orig_alignCoder_simple_em_by_group": em_by_group,
        "unambiguous_external_symbols_total": tot_u,
        "unambiguous_reached_prompt_pct": pct(unamb["reached_prompt"], tot_u),
        "unambiguous_retrieved_but_cut_pct": pct(unamb["cut"], tot_u),
        "unambiguous_not_retrieved_pct": pct(unamb["not_retrieved"], tot_u),
        "examples": n,
        "examples_whose_GT_uses_a_symbol_defined_outside_left_context": ex_needs_ext,
        "pct_examples_needing_external_def": pct(ex_needs_ext, n),
        "pct_of_those_with_all_external_defs_in_prompt": pct(ex_ext_all_in_prompt, ex_needs_ext),
        "gt_symbol_occurrences_by_definition_site": dict(per_where),
        "external_symbols_total": tot_ext,
        "external_symbols_reached_prompt_pct": pct(lost["reached_prompt"], tot_ext),
        "external_symbols_retrieved_but_cut_pct": pct(lost["retrieved_but_cut_by_budget"], tot_ext),
        "external_symbols_not_retrieved_pct": pct(lost["not_retrieved"], tot_ext),
        "external_symbols_defined_in_current_file_suffix": lost["where=right"],
        "external_symbols_defined_in_other_files": lost["where=repo"],
        "left_defined_symbols_truncated_out_of_prompt": ex_left_truncated,
        "mean_cross_tokens": round(sum(r["cross_tokens"] for r in rows) / n, 1),
        "mean_infile_tokens_in_prompt": round(sum(r["infile_tokens"] for r in rows) / n, 1),
        "mean_left_context_tokens": round(sum(r["left_tokens"] for r in rows) / n, 1),
        "pct_examples_left_context_truncated": pct(sum(r["left_tokens"] > r["infile_tokens"] for r in rows), n),
        "pct_examples_retriever_chose_STOP_first": pct(sum(r["stop_pos"] == 0 for r in rows), n),
    }

# ----------------------------------------------------------------------------------
# funnel (needs the full rerank pool: cceval_extract.py output)
# ----------------------------------------------------------------------------------

def cmd_funnel(a):
    import gzip
    comp = {}
    with gzip.open(a.compact, "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            comp[r["task_id"]] = r
    exs = load_examples(a.data_dir, a.subset, set(comp))
    os.makedirs(a.out, exist_ok=True)
    stage = Counter()
    stage_unamb = Counter()
    grp = defaultdict(list)
    ranks_not_selected = []
    missing_ids = []
    rows_out = []
    for t, r in comp.items():
        if t not in exs:
            missing_ids.append(t)
            continue
        ex = exs[t]
        d = ExampleDefs(ex)
        pool_defs, sel_defs, best_rank = set(), set(), {}
        for c in r["pool"]:
            for n in c["defs"]:
                pool_defs.add(n)
                if c["rank"] is not None:
                    best_rank[n] = min(best_rank.get(n, 10 ** 9), c["rank"])
                if c["selected"]:
                    sel_defs.add(n)
        in_prompt = set(r["prompt_cross_defs"]) | set(r["prompt_infile_defs"])
        sts = []
        for name, where, used in gt_symbols(ex, d):
            if where == "left":
                continue
            if name in in_prompt:
                s = "reached_prompt"
            elif name in sel_defs:
                s = "selected_but_cut_by_budget"
            elif name in pool_defs:
                s = "in_pool_not_selected_by_reranker"
                ranks_not_selected.append(best_rank.get(name))
            else:
                s = "not_in_bm25_pool"
            stage[s] += 1
            if len(d.repo.get(name, [])) <= 3:
                stage_unamb[s] += 1
            sts.append({"name": name, "where": where, "stage": s})
        g = ("no_external_needed" if not sts else
             "all_ext_defs_in_prompt" if all(x["stage"] == "reached_prompt" for x in sts) else "some_ext_def_missing")
        grp[g].append(r["official"]["em"] if r.get("official") else 0)
        rows_out.append({"task_id": t, "symbols": sts, "official": r.get("official")})
    pct = lambda x, y: round(100.0 * x / y, 2) if y else None
    tot, totu = sum(stage.values()), sum(stage_unamb.values())
    rk = sorted(x for x in ranks_not_selected if x is not None)
    summary = {
        "examples": len(rows_out), "missing_in_parquet": len(missing_ids),
        "external_symbols_total": tot,
        "stage_pct": {k: pct(v, tot) for k, v in stage.items()},
        "unambiguous_total": totu,
        "unambiguous_stage_pct": {k: pct(v, totu) for k, v in stage_unamb.items()},
        "official_em_by_group": {g: {"n": len(v), "em_pct": pct(sum(v), len(v))} for g, v in grp.items()},
        "reranker_rank_of_missed_defs": {"median": rk[len(rk) // 2] if rk else None,
                                         "pct_rank_le_20": pct(sum(x <= 20 for x in rk), len(rk)),
                                         "pct_rank_le_30": pct(sum(x <= 30 for x in rk), len(rk))},
    }
    with open(os.path.join(a.out, "funnel_rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, "funnel_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

# ----------------------------------------------------------------------------------
# generate
# ----------------------------------------------------------------------------------

def cmd_generate(a):
    import torch
    from transformers import AutoModelForCausalLM
    cands = load_cands(a.cands)
    ids = pick_subset(list(cands), a.n, a.seed)
    exs = load_examples(a.data_dir, a.subset, set(ids))
    tok = get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    os.makedirs(a.out, exist_ok=True)
    conds = a.conditions.split(",")
    for cond in conds:
        cdir = os.path.join(a.out, cond)
        os.makedirs(cdir, exist_ok=True)
        pred_path = os.path.join(cdir, "prediction.jsonl")
        done = set()
        if os.path.exists(pred_path):
            done = {json.loads(l)["task_id"] for l in open(pred_path, encoding="utf-8")}
        todo = [t for t in ids if t not in done]
        print(f"[{cond}] {len(done)} done, {len(todo)} to go", flush=True)
        prompts = []
        meta = open(os.path.join(cdir, "prompt_meta.jsonl"), "a", encoding="utf-8")
        for t in todo:
            ex = exs[t]
            d = ExampleDefs(ex)
            blocks = condition_blocks(cond, ex, d, cands[t]["candidates"])
            prompt, info = build_prompt(tok, ex, blocks, max_cross=CROSS_BUDGET.get(cond, MAX_CROSS))
            ids_ = tok.encode(prompt)[-MAX_CTX:]  # as in CustomDataset (adds BOS)
            prompts.append((t, ids_))
            meta.write(json.dumps({"task_id": t, "cross_tokens": info["cross_tokens"],
                                   "infile_tokens": info["infile_tokens"],
                                   "n_blocks": len(blocks)}) + "\n")
        meta.close()
        prompts.sort(key=lambda x: len(x[1]))
        t0 = time.time()
        with open(pred_path, "a", encoding="utf-8") as fout:
            for i in range(0, len(prompts), a.batch_size):
                batch = prompts[i:i + a.batch_size]
                L = max(len(x[1]) for x in batch)
                inp = torch.tensor([[tok.pad_token_id] * (L - len(x[1])) + x[1] for x in batch]).cuda()
                att = torch.tensor([[0] * (L - len(x[1])) + [1] * len(x[1]) for x in batch]).cuda()
                with torch.no_grad():
                    gen = model.generate(inp, attention_mask=att, max_new_tokens=MAX_NEW, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                for (t, _), g in zip(batch, gen[:, L:]):
                    fout.write(json.dumps({"task_id": t, "pred": tok.decode(g, skip_special_tokens=True)}) + "\n")
                fout.flush()
                if (i // a.batch_size) % 10 == 0:
                    print(f"[{cond}] {i + len(batch)}/{len(prompts)}  {time.time() - t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------------
# score
# ----------------------------------------------------------------------------------

def _ratio(a, b):
    try:
        from fuzzywuzzy import fuzz
        return fuzz.ratio(a, b)
    except Exception:
        return round(100 * difflib.SequenceMatcher(None, a, b).ratio())


def simple_score(pred, gt):
    """Fallback (no tree-sitter): first k non-empty lines, k = #non-empty GT lines."""
    strip = lambda s: [l.strip() for l in _COMMENT_RE.sub("", s).split("\n") if l.strip()]
    g = strip(gt)
    p = strip(pred)[: len(g)]
    return int(p == g), _ratio("\n".join(p), "\n".join(g))


def official_score(src, cdir, jsonl):
    """Runs AlignCoder's own compute_metric_stmt (needs its utils/ + tree-sitter .so)."""
    cwd = os.getcwd()
    sys.path.insert(0, src)
    os.chdir(src)
    try:
        from utils.eval_metric import compute_metric_stmt
        res = compute_metric_stmt(os.path.abspath(os.path.join(cwd, cdir)), jsonl, language="python",
                                  ts_lib=os.path.join(src, "utils/build/python-lang-parser.so"))
        per = {}
        with open(os.path.join(cwd, cdir, "detailed_results.json")) as f:
            for l in f:
                r = json.loads(l)
                per[r["task_id"]] = (r["em"], r["es"])
        return res, per
    finally:
        os.chdir(cwd)


def cmd_score(a):
    exs_gt = None
    jsonl = os.path.abspath(os.path.join(data_folder(a.data_dir, a.subset), "test.jsonl"))
    if os.path.exists(jsonl):
        exs_gt = {}
        for l in open(jsonl, encoding="utf-8"):
            r = json.loads(l)
            exs_gt[r["metadata"]["task_id"]] = r["groundtruth"]
    else:
        ensure_data(a.data_dir, a.subset)
        exs_gt = {}
        for l in open(jsonl, encoding="utf-8"):
            r = json.loads(l)
            exs_gt[r["metadata"]["task_id"]] = r["groundtruth"]
    per_cond, table = {}, {}
    for cond in CONDITIONS:
        cdir = os.path.join(a.out, cond)
        p = os.path.join(cdir, "prediction.jsonl")
        if not os.path.exists(p):
            continue
        preds = {}
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            preds[r["task_id"]] = r["pred"]
        per = {t: simple_score(pr, exs_gt[t]) for t, pr in preds.items()}
        res = {"n": len(per), "simple_em": round(100 * sum(v[0] for v in per.values()) / len(per), 2),
               "simple_es": round(sum(v[1] for v in per.values()) / len(per), 2)}
        if a.aligncoder_src:
            try:
                # evaluator expects exactly these ids in prediction.jsonl; dedupe first
                with open(p, "w", encoding="utf-8") as f:
                    for t, pr in preds.items():
                        f.write(json.dumps({"task_id": t, "pred": pr}) + "\n")
                ores, oper = official_score(a.aligncoder_src, cdir, jsonl)
                res["official"] = ores
                per = {t: (oper[t][0], oper[t][1]) for t in oper}
            except Exception as e:  # keep going with the simple metric
                res["official_error"] = repr(e)
        per_cond[cond], table[cond] = per, res
    if "base" in per_cond:
        for cond, per in per_cond.items():
            if cond == "base":
                continue
            common = sorted(set(per) & set(per_cond["base"]))
            win = sum(per[t][0] == 1 and per_cond["base"][t][0] == 0 for t in common)
            loss = sum(per[t][0] == 0 and per_cond["base"][t][0] == 1 for t in common)
            table[cond]["paired_vs_base"] = {"n": len(common), "em_gained": win, "em_lost": loss,
                                             "mcnemar_p": mcnemar(win, loss)}
    with open(os.path.join(a.out, "scores.json"), "w") as f:
        json.dump(table, f, indent=2)
    print(json.dumps(table, indent=2))


def mcnemar(b, c):
    """Exact two-sided McNemar (binomial) p-value."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 5)

# ----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["coverage", "generate", "score", "funnel"])
    ap.add_argument("--cands", help="AlignCoder results/<task>/prediction_with_candidates.jsonl")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None, help="repoeval: api_level|line_level (default api_level); cceval: python")
    ap.add_argument("--compact", default=None, help="cceval_extract.py output, for the `funnel` command")
    ap.add_argument("--out", default="dep_oracle_out")
    ap.add_argument("--n", type=int, default=0, help="stratified subset size (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generator", default=GEN_DEFAULT)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--aligncoder_src", default=None, help="folder with AlignCoder utils/ (optional)")
    a = ap.parse_args()
    global DATASET
    DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    {"coverage": cmd_coverage, "generate": cmd_generate, "score": cmd_score, "funnel": cmd_funnel}[a.cmd](a)


if __name__ == "__main__":
    main()
