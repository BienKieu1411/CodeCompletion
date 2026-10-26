import gzip
import json
import re
import difflib
import ast
from collections import defaultdict
from pathlib import Path

def extract_identifiers(code: str, language: str) -> list[str]:
    kws = {"def", "class", "return", "if", "else", "for", "while", "in", "import", "from", "public", "private", "protected", "static", "void", "int", "final"}
    words = re.findall(r"\b[_A-Za-z][_A-Za-z0-9]*\b", code)
    return [w for w in words if w not in kws]

def get_py_statement_ast(prompt: str, raw_pred: str) -> str:
    lines = raw_pred.splitlines()
    if not lines: return raw_pred.strip()
    cur = ""
    for l in lines:
        if cur: cur += "\n" + l
        else: cur = l
        code = prompt + cur
        try:
            ast.parse(code)
            return cur.strip()
        except SyntaxError:
            continue
    return lines[0].strip()

def get_java_statement(raw_pred: str) -> str:
    end_idx = None
    for i in range(len(raw_pred)):
        if raw_pred[i] in [";", "}", "{"]:
            end_idx = i
            break
    return raw_pred[:end_idx + 1].strip() if end_idx is not None else raw_pred.strip()

def evaluate_predictions_file(pred_file: Path) -> dict:
    records = []
    with gzip.open(pred_file, "rt") as f:
        for line in f:
            records.append(json.loads(line))

    by_arm_bm = defaultdict(lambda: defaultdict(list))
    by_arm_prefix = defaultdict(lambda: defaultdict(list))

    for r in records:
        arm = r["arm"]
        bm = r["benchmark"]
        raw_pred = r["raw_prediction"]
        gt = r["groundtruth"]
        prompt = r.get("left_context", "")
        lang = r.get("language", "python")

        # 1. AlignCoder Standard Metric (AST Statement Match)
        if "repoeval" in bm:
            gt_lines = [l.strip() for l in gt.splitlines() if l.strip()]
            n = max(1, len(gt_lines))
            pred_lines = [l.strip() for l in raw_pred.splitlines() if l.strip()][:n]
            pred_align = "\n".join(pred_lines)
        elif lang == "python":
            pred_align = get_py_statement_ast(prompt, raw_pred)
        else:
            pred_align = get_java_statement(raw_pred)

        em_align = int(pred_align.strip() == gt.strip())
        by_arm_bm[arm][bm].append(em_align)

        # 2. AIRCoder Prefix Match Metric
        gt_clean = "".join([l.strip() for l in gt.splitlines() if l.strip()])
        pred_clean = "".join([l.strip() for l in raw_pred.splitlines() if l.strip()])
        pred_trunc = pred_clean[:len(gt_clean)]
        em_prefix = int(pred_trunc == gt_clean) if gt_clean else 0
        by_arm_prefix[arm][bm].append(em_prefix)

    print("="*85)
    print("CHUẨN 1: ALIGNCODER STANDARD METRIC (Tree-sitter AST Statement Match - STRICT)")
    print(f"{'Arm':<24} | {'Micro EM':<9} | {'Python':<8} | {'Java':<8} | {'API':<8} | {'Line':<8}")
    print("-" * 85)
    for arm in sorted(by_arm_bm.keys()):
        bms = by_arm_bm[arm]
        py = sum(bms['cceval_python'])/len(bms['cceval_python'])*100 if bms['cceval_python'] else 0
        jv = sum(bms['cceval_java'])/len(bms['cceval_java'])*100 if bms['cceval_java'] else 0
        api = sum(bms['repoeval_api'])/len(bms['repoeval_api'])*100 if bms['repoeval_api'] else 0
        ln = sum(bms['repoeval_line'])/len(bms['repoeval_line'])*100 if bms['repoeval_line'] else 0
        tot = sum(sum(v) for v in bms.values()) / sum(len(v) for v in bms.values()) * 100
        print(f"{arm:<24} | {tot:>7.2f}% | {py:>6.2f}% | {jv:>6.2f}% | {api:>6.2f}% | {ln:>6.2f}%")
    print("="*85)

    print("\n" + "="*85)
    print("CHUẨN 2: AIRCODER PAPER OFFICIAL METRIC (Prefix Match Truncation [:len(target)])")
    print(f"{'Arm':<24} | {'Micro EM':<9} | {'Python':<8} | {'Java':<8} | {'API':<8} | {'Line':<8}")
    print("-" * 85)
    for arm in sorted(by_arm_prefix.keys()):
        bms = by_arm_prefix[arm]
        py = sum(bms['cceval_python'])/len(bms['cceval_python'])*100 if bms['cceval_python'] else 0
        jv = sum(bms['cceval_java'])/len(bms['cceval_java'])*100 if bms['cceval_java'] else 0
        api = sum(bms['repoeval_api'])/len(bms['repoeval_api'])*100 if bms['repoeval_api'] else 0
        ln = sum(bms['repoeval_line'])/len(bms['repoeval_line'])*100 if bms['repoeval_line'] else 0
        tot = sum(sum(v) for v in bms.values()) / sum(len(v) for v in bms.values()) * 100
        print(f"{arm:<24} | {tot:>7.2f}% | {py:>6.2f}% | {jv:>6.2f}% | {api:>6.2f}% | {ln:>6.2f}%")
    print("="*85 + "\n")

if __name__ == "__main__":
    import sys
    fpath = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/mb_spt_iterative/outputs/cadre_plus_pilot_256_predictions.jsonl.gz")
    evaluate_predictions_file(fpath)
