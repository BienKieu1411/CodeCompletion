#!/usr/bin/env python3
"""Test RLCoder retrieval and CADRE+ prompt pipeline on 5 tasks locally."""

from __future__ import annotations

import gzip
import json
import tarfile
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

BUNDLE_PATH = Path("experiments/mb_spt_iterative/full_benchmark_8004_bundle.tar.gz")

def split_into_smaller_blocks(file_path, code_content, language):
    mini_blocks = []
    current_block = []
    for line in code_content.splitlines():
        if line.strip() == "":
            if current_block:
                mini_blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        mini_blocks.append(current_block)

    max_len = 15
    temp_mini_blocks = []
    for mb in mini_blocks:
        if len(mb) > max_len:
            for idx in range(0, len(mb), max_len):
                temp_mini_blocks.append(mb[idx:idx+max_len])
        else:
            temp_mini_blocks.append(mb)
    mini_blocks = temp_mini_blocks

    smaller_blocks = []
    current_content = []
    total_lines = 0
    for block in mini_blocks:
        if total_lines >= 5000:
            break
        if len(current_content) + len(block) <= 15:
            current_content.extend(block)
            total_lines += len(block)
        else:
            if current_content:
                smaller_blocks.append({
                    "file_path": file_path,
                    "line_start": total_lines - len(current_content) + 1,
                    "line_end": total_lines,
                    "code_content": "\n".join(current_content),
                    "language": language,
                })
            current_content = block
            total_lines += len(block)
    if current_content:
        smaller_blocks.append({
            "file_path": file_path,
            "line_start": total_lines - len(current_content) + 1,
            "line_end": total_lines,
            "code_content": "\n".join(current_content),
            "language": language,
        })
    return smaller_blocks


def main():
    print("Loading bundle sample...")
    with tarfile.open(BUNDLE_PATH, "r:gz") as tar:
        f_tasks = tar.extractfile("tasks.jsonl.gz")
        with gzip.open(f_tasks, "rt") as gz:
            sample_tasks = [json.loads(gz.readline()) for _ in range(5)]
        
        f_files = tar.extractfile("files.jsonl.gz")
        files_by_id = {}
        with gzip.open(f_files, "rt") as gz:
            for line in gz:
                d = json.loads(line)
                files_by_id[d["file_id"]] = d

    print(f"Loaded {len(sample_tasks)} tasks, {len(files_by_id)} files indexed.")

    # Load RLCoder tokenizer & model
    print("Loading nov3630/RLCoder...")
    tok = AutoTokenizer.from_pretrained("nov3630/RLCoder")
    model = AutoModel.from_pretrained("nov3630/RLCoder").eval()

    for idx, t in enumerate(sample_tasks):
        lang = t["language"]
        t_id = t["task_id"]
        path = t["path"]
        print(f"\nTask {idx}: {t_id} ({t['benchmark']}, lang={lang}, path={path})")

        # 1. Gather all blocks
        blocks = []
        for rel_path, fid in t.get("file_ids_by_path", {}).items():
            f_entry = files_by_id.get(fid)
            if not f_entry: continue
            file_blocks = split_into_smaller_blocks(rel_path, f_entry["text"], lang)
            blocks.extend(file_blocks)

        print(f"  Total candidate blocks from repo: {len(blocks)}")
        if not blocks:
            print("  No blocks found!")
            continue

        # 2. BM25
        corpus = [b["code_content"].lower().split() for b in blocks]
        bm25 = BM25Okapi(corpus)
        query_lines = [l for l in t["left_context"].splitlines() if l.strip()][-20:]
        query_text = "\n".join(query_lines)
        q_tokens = query_text.lower().split()
        scores = bm25.get_scores(q_tokens)
        top100_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:100]
        top100_blocks = [blocks[i] for i in top100_idx]

        # Add stop block
        top100_blocks.append({
            "file_path": "",
            "line_start": 0,
            "line_end": 0,
            "code_content": "Don't need cross file context for completion",
            "language": lang,
            "is_stop": True,
        })

        # 3. Neural Reranking via nov3630/RLCoder
        def encode_batch(texts, is_query, max_length):
            input_ids = []
            for text in texts:
                tokens = tok.tokenize(text)
                if is_query:
                    tokens = tokens[-(max_length - 4):]
                else:
                    tokens = tokens[:(max_length - 4)]
                tokens = [tok.cls_token, "<encoder-only>", tok.sep_token] + tokens + [tok.sep_token]
                ids = tok.convert_tokens_to_ids(tokens)
                padding = max_length - len(ids)
                ids += [tok.pad_token_id] * padding
                input_ids.append(ids)
            tensor = torch.tensor(input_ids, dtype=torch.long)
            mask = tensor.ne(tok.pad_token_id)
            with torch.no_grad():
                emb = model(tensor, attention_mask=mask)[0]
                sent_emb = (emb * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
                sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            return sent_emb

        q_emb = encode_batch([query_text], is_query=True, max_length=256)
        
        cand_texts = []
        for b in top100_blocks:
            if b.get("is_stop"):
                cand_texts.append("Don't need cross file context for completion")
            else:
                cs = "#" if lang == "python" else "//"
                cand_texts.append(f"{cs} file path: {b['file_path']}\n{cs} lines: {b['line_start']}-{b['line_end']}\n{cs} {b['code_content']}")
        
        c_emb = encode_batch(cand_texts, is_query=False, max_length=512)
        sims = torch.mm(q_emb, c_emb.t())[0].numpy()
        top10_idx = sims.argsort()[-10:][::-1]

        top10_cands = [top100_blocks[i] for i in top10_idx]
        print("  Top 5 RLCoder retrieved blocks:")
        for rk, i in enumerate(top10_idx[:5]):
            b = top100_blocks[i]
            fp = b.get("file_path") or "[STOP BLOCK]"
            print(f"    #{rk+1}: sim={sims[i]:.4f}, file={fp}, lines={b.get('line_start')}-{b.get('line_end')}")

    print("\nLocal test passed successfully!")

if __name__ == "__main__":
    main()
