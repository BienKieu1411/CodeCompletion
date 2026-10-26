import json
import gzip
import math
import re
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path("results/aligncoder-rq1-deepseek13b-20261013-225803")
TARGET_RUN_ID = "aligncoder-rq1-deepseek13b-20261013-225803"

def iter_gz(p):
    proc = subprocess.Popen(["gzip", "-dc", str(p)], stdout=subprocess.PIPE, text=True, encoding="utf-8")
    for line in proc.stdout:
        if line.strip(): yield json.loads(line)
    proc.wait()

ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PY_STOP = {"def", "class", "return", "if", "else", "elif", "for", "while", "in", "not", "and", "or", "True", "False", "None", "self", "import", "from", "as", "with", "try", "except", "raise", "pass", "break", "continue", "lambda", "yield", "is", "assert", "del", "global", "nonlocal", "int", "str", "float", "bool", "list", "dict", "set", "tuple", "len", "range", "print", "super"}
JAVA_STOP = {"public", "private", "protected", "static", "final", "void", "int", "long", "double", "float", "boolean", "char", "byte", "short", "new", "return", "if", "else", "for", "while", "do", "switch", "case", "default", "break", "continue", "try", "catch", "finally", "throw", "throws", "class", "interface", "extends", "implements", "this", "super", "null", "true", "false", "import", "package", "String", "Object", "List", "Map", "Set", "Override"}

def extract_ids(code: str, lang: str) -> set[str]:
    stops = PY_STOP if lang == "python" else JAVA_STOP
    return {m for m in ID_RE.findall(code) if m not in stops and len(m) > 2}

LINE_RANGE_RE = re.compile(r"lines:\s*(\d+)-(\d+)")
def parse_rng(desc: str):
    m = LINE_RANGE_RE.search(desc or "")
    return (int(m.group(1)), int(m.group(2))) if m else None

def format_scit_block(desc: str, code_content: str, lang: str, kept_lines=None, ri=None) -> str:
    comment_label = "#" if lang == "python" else "//"
    desc_lines = [f"{comment_label} {cl}" for cl in desc.strip().split("\n") if cl]
    header = "\n".join(desc_lines) + "\n"
    
    raw_lines = code_content.split("\n")
    if kept_lines is not None and ri is not None and len(raw_lines) == (ri[1] - ri[0] + 1):
        body_lines = [f"{comment_label} {raw_lines[idx]}" for idx, ln in enumerate(range(ri[0], ri[1] + 1)) if ln in kept_lines]
    else:
        body_lines = [f"{comment_label} {cl}" for cl in raw_lines]
    body = "\n".join(body_lines)
    return header + body

def assemble_benchmark_slice(limit_per_lang=100):
    all_tasks = []
    
    for ds_name, lang in [("cceval_python", "python"), ("cceval_java", "java")]:
        obs_dir = BASE / "observability" / ds_name
        raw_enh = {str(r["task_id"]): r for r in iter_gz(obs_dir / "generation_raw_events.jsonl.gz") if r["run_id"] == TARGET_RUN_ID and r["event"] == "query_enhancement_raw"}
        raw_final = {str(r["task_id"]): r for r in iter_gz(obs_dir / "generation_raw_events.jsonl.gz") if r["run_id"] == TARGET_RUN_ID and r["event"] == "final_completion_raw"}
        
        task_inputs, rerank_pool = {}, {}
        for r in iter_gz(obs_dir / "retrieval_events.jsonl.gz"):
            if r["run_id"] != TARGET_RUN_ID: continue
            tid = str(r["task_id"])
            if r["event"] == "task_input": task_inputs[tid] = r
            elif r["event"] == "unixcoder_rerank_pool": rerank_pool[tid] = r

        lang_candidates = []
        for tid, t_in in task_inputs.items():
            target = t_in["target"]
            left_ctx = t_in["left_context"]
            target_ids = extract_ids(target, lang)
            left_ids = extract_ids(left_ctx, lang)
            novel_ids = target_ids - left_ids
            if not novel_ids: continue

            rr = rerank_pool.get(tid)
            if not rr: continue
            pool = [c for c in rr["candidate_pool_after_bm25_and_api"] if c["file_path"] != ""]
            selected = sorted([c for c in pool if c["selected"]], key=lambda x: x["selected_rank"])
            if not pool or not selected: continue

            rf_rec = raw_final.get(tid, {})
            final_prompt = rf_rec.get("prompt", "") or ""
            sep = f"# file path: {t_in['file_path']}"
            if sep not in final_prompt: continue

            parts = final_prompt.split(sep, 1)
            crossfile_part = parts[0]
            in_file_suffix = sep + parts[1]
            exact_char_budget = len(crossfile_part)
            if exact_char_budget < 500: continue

            # Precompute cand IDs and line ranges
            cand_ids = {c["candidate_index"]: extract_ids(c["code_content"], lang) for c in pool}
            cand_rng = {c["candidate_index"]: parse_rng(c["description"]) for c in pool}

            scores = [c["cosine_score"] for c in pool]
            s_max, s_min = max(scores), min(scores)
            s_span = max(s_max - s_min, 1e-6)
            s0 = {c["candidate_index"]: (c["cosine_score"] - s_min) / s_span for c in pool}

            # Self-info
            df = Counter()
            for c in pool:
                for u in cand_ids[c["candidate_index"]]: df[u] += 1
            N_p = len(pool)
            self_info = {u: math.log(1.0 + N_p / (1.0 + df_u)) for u, df_u in df.items()}

            # Posterior weights from K=8 drafts
            raw_8 = raw_enh.get(tid, {}).get("raw_completions", [])
            K = max(len(raw_8), 1)
            mc_counts = Counter()
            draft_had_em1 = False
            for comp in raw_8:
                c_text = comp if isinstance(comp, str) else comp.get("completion_text", "")
                c_target = comp.get("target_code", "") if isinstance(comp, dict) else ""
                if c_text.strip().splitlines() and c_target and c_target.strip().splitlines():
                    if c_text.strip().splitlines()[0] == c_target.strip().splitlines()[0]:
                        draft_had_em1 = True
                for u in extract_ids(c_text, lang):
                    mc_counts[u] += 1

            post_wt = {}
            for u in self_info:
                if u in left_ids:
                    w = 0.0
                else:
                    k_u = mc_counts[u]
                    p_prior = 1.0 / (1.0 + df[u])
                    p_post = (k_u + 1.0) / (K + 2.0)
                    w = self_info[u] * (0.80 * p_prior + 0.20 * p_post)
                post_wt[u] = w

            # Greedy SCIT candidate selection over full pool (Budget = 10 blocks)
            chosen_S = []
            covered_syms = set()
            covered_lines = defaultdict(set)
            anchor_files = {}

            for step in range(10):
                best_gain = -1.0
                best_c = None
                for c in pool:
                    if c in chosen_S: continue
                    ci = c["candidate_index"]
                    fp = c["file_path"]
                    ri = cand_rng[ci]
                    c_len = max(len(c["code_content"]), 50)
                    new_syms = cand_ids[ci] - covered_syms
                    sym_gain = sum(post_wt.get(u, 0.0) for u in new_syms)

                    val = 0.40 * s0[ci] + 0.60 * sym_gain
                    if not c["api_block"] and ri and fp in covered_lines:
                        lines_c = set(range(ri[0], ri[1] + 1))
                        ov_lines = len(lines_c & covered_lines[fp])
                        new_line_ratio = 1.0 - (ov_lines / max(len(lines_c), 1))
                        c_len = max(int(c_len * (0.20 + 0.80 * new_line_ratio)), 80)
                        if new_line_ratio < 0.20:
                            val *= new_line_ratio
                        elif fp in anchor_files and anchor_files[fp] >= 0.85:
                            is_cont = any(0 <= abs(ri[0] - l) <= 5 or 0 <= abs(ri[1] - l) <= 5 for l in covered_lines[fp])
                            if is_cont and sym_gain > 0:
                                val += 0.06 * anchor_files[fp] * new_line_ratio

                    density = val / ((c_len / 400.0) ** 0.15)
                    if density > best_gain:
                        best_gain, best_c = density, c

                if best_c is None: break
                chosen_S.append(best_c)
                ci = best_c["candidate_index"]
                covered_syms |= cand_ids[ci]
                if not best_c["api_block"]:
                    fp = best_c["file_path"]
                    anchor_files[fp] = max(anchor_files.get(fp, 0.0), s0[ci])
                    if cand_rng[ci]:
                        covered_lines[fp] |= set(range(cand_rng[ci][0], cand_rng[ci][1] + 1))

            # Build SCIT prompt using quotient interval union and correct comment formatting
            seen_file_lines = defaultdict(set)
            scit_parts = []
            for c in chosen_S:
                ri = cand_rng[c["candidate_index"]]
                fp = c["file_path"]
                lines = c["code_content"].split("\n")
                if not c["api_block"] and ri and len(lines) == (ri[1] - ri[0] + 1):
                    kept_set = set(ln for ln in range(ri[0], ri[1] + 1) if ln not in seen_file_lines[fp])
                    seen_file_lines[fp].update(range(ri[0], ri[1] + 1))
                    if kept_set:
                        block_str = format_scit_block(c["description"], c["code_content"], lang, kept_lines=kept_set, ri=ri)
                        scit_parts.append(block_str)
                else:
                    block_str = format_scit_block(c["description"], c["code_content"], lang)
                    scit_parts.append(block_str)

            scit_crossfile = "\n\n".join(scit_parts) + "\n\n"
            if len(scit_crossfile) > exact_char_budget:
                truncated = scit_crossfile[:exact_char_budget]
                last_nl = truncated.rfind("\n\n")
                if last_nl > 500:
                    scit_crossfile = truncated[:last_nl + 2]
                else:
                    last_single = truncated.rfind("\n")
                    scit_crossfile = truncated[:last_single + 1] if last_single > 500 else truncated
            
            prompt_scit = scit_crossfile + in_file_suffix
            prompt_baseline = final_prompt
            prompt_no_retrieval = in_file_suffix.lstrip()

            # Measure if outside_top10 was rescued
            t10_cands = set(c["candidate_index"] for c in selected)
            scit_cands = set(c["candidate_index"] for c in chosen_S)
            has_outside_rescue = bool(scit_cands - t10_cands)

            lang_candidates.append({
                "task_id": tid,
                "dataset": ds_name,
                "language": lang,
                "file_path": t_in["file_path"],
                "target": target,
                "prompt_baseline": prompt_baseline,
                "prompt_scit": prompt_scit,
                "prompt_no_retrieval": prompt_no_retrieval,
                "has_outside_rescue": has_outside_rescue,
                "draft_had_em1": draft_had_em1,
            })

        # Stratified selection to ensure high signal: 50 rescue tasks, 25 draft em1 tasks, 25 regular
        rescue_tasks = [x for x in lang_candidates if x["has_outside_rescue"]]
        draft_tasks = [x for x in lang_candidates if x["draft_had_em1"] and not x["has_outside_rescue"]]
        other_tasks = [x for x in lang_candidates if not x["has_outside_rescue"] and not x["draft_had_em1"]]
        
        selected_tasks = rescue_tasks[:50] + draft_tasks[:25] + other_tasks[:25]
        if len(selected_tasks) < limit_per_lang:
            selected_tasks += [x for x in lang_candidates if x not in selected_tasks][:limit_per_lang - len(selected_tasks)]
        
        print(f"Selected {len(selected_tasks)} tasks for {ds_name} (Rescued: {sum(x['has_outside_rescue'] for x in selected_tasks)}, Draft EM=1: {sum(x['draft_had_em1'] for x in selected_tasks)})")
        all_tasks.extend(selected_tasks)

    # Save to json.gz
    out_path = Path("kaggle_eval_scit/benchmark_tasks_200.json.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(all_tasks, f)
    print(f"Saved {len(all_tasks)} benchmark tasks to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    assemble_benchmark_slice(100)
