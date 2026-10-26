# %% [markdown]
# # One-pass RLRetriever on RepoBench-R
#
# This notebook evaluates the trained RLRetriever checkpoint on the one-pass
# RepoBench-R ranking task.  It uses the retriever input
# template, token lengths,
# mean-pooling, and cosine ranking.  RepoBench-R contains exactly one labelled
# gold cross-file snippet per example, so Accuracy@k equals Recall@k.
#
# This is a controlled retriever baseline: no generator, query enhancement,
# or BM25 shortlist is used.  It ranks each RepoBench-R candidate pool directly.

# %%
# In Kaggle, install once in a separate cell if these packages are absent:
# !pip -q install "transformers>=4.44,<5" "huggingface_hub>=0.24" "accelerate>=0.33" sentencepiece

# %%
import gc
import gzip
import math
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Enable a Kaggle GPU accelerator before running the benchmark."
print("device:", torch.cuda.get_device_name(0))

# Set to an integer for a smoke test.  Keep None for the full official splits.
LIMIT = None
QUERY_BATCH_SIZE = 256
CODE_BATCH_SIZE = 256
QUERY_MAX_LENGTH = 256
CODE_MAX_LENGTH = 512
TOP_KS = (1, 3, 5)

MODELS = {
    "RLRetriever": "nov3630/RLRetriever",
}
CONFIGS = ("python_cff", "python_cfr", "java_cff", "java_cfr")
SPLITS = ("test_easy", "test_hard")

# %%
def make_query(row):
    # RLCoder/AlignCoder use the final 20 nonempty lines of left context.
    # RepoBench-R's `code` is precisely that in-file prefix.
    nonempty_lines = [line for line in row["code"].splitlines() if line.strip()]
    return "\n".join(nonempty_lines[-20:])


def make_candidate(snippet, language):
    # Mirror CodeBlock.__str__ in both upstream repositories.  The original
    # corpus has paths; RepoBench-R only releases snippets, so use a constant
    # synthetic path.  A constant path lets identical snippets be encoded once
    # and reused across all queries in a split.
    comment = "#" if language == "python" else "//"
    nonempty = [line for line in snippet.splitlines() if line.strip()]
    header = [
        f"{comment} file path: repobench_candidate",
        f"{comment} lines: 0-{len(snippet.splitlines())}",
    ]
    return "\n".join(header + [f"{comment} {line}" for line in nonempty]).strip()


def load_repobench_r(config, split):
    # RepoBench-R is distributed as a gzipped Python pickle behind an old
    # `datasets` loading script.  New datasets versions reject such scripts,
    # so download the official file and load its documented split directly.
    archive = hf_hub_download(
        repo_id="tianyang/repobench-r",
        repo_type="dataset",
        filename=f"data/{config}.gz",
    )
    with gzip.open(archive, "rb") as handle:
        raw = pickle.load(handle)
    subset, difficulty = split.split("_", maxsplit=1)
    return raw[subset][difficulty]


def load_dataset(_legacy_path, config, split, **_ignored):
    """Compatibility shim for stale notebook cells from the first version.

    It deliberately shadows the old Hugging Face datasets helper, whose
    script-backed RepoBench-R invocation now fails on current Kaggle images.
    """
    return load_repobench_r(config, split)


def upstream_tokenize(tokenizer, text, max_length, is_query):
    # Exact token construction in RLCoder/AlignCoder retriever.py.
    tokens = tokenizer.tokenize(text)
    if is_query:
        tokens = tokens[-max_length + 4 :]
    else:
        tokens = tokens[: max_length - 4]
    tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
    return input_ids


def print_progress(label, current, total, width=32):
    """Notebook-friendly progress bar implemented only with print."""
    ratio = current / total if total else 1.0
    filled = int(width * ratio)
    bar = "=" * width if current >= total else "=" * filled + ">" + "." * max(0, width - filled - 1)
    print(f"\r    {label:<18} [{bar}] {ratio:6.2%} ({current}/{total})", end="", flush=True)
    if current == total:
        print()


@torch.inference_mode()
def encode(model, tokenizer, texts, max_length, is_query, batch_size):
    vectors = []
    label = "encoding queries" if is_query else "encoding snippets"
    total_batches = math.ceil(len(texts) / batch_size)
    print_progress(label, 0, total_batches)
    for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch_texts = texts[start : start + batch_size]
        ids = torch.tensor(
            [upstream_tokenize(tokenizer, text, max_length, is_query) for text in batch_texts],
            dtype=torch.long,
            device=DEVICE,
        )
        mask = ids.ne(tokenizer.pad_token_id)
        token_vectors = model(ids, attention_mask=mask).last_hidden_state
        pooled = (token_vectors * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
        vectors.append(F.normalize(pooled, p=2, dim=1).cpu())
        print_progress(label, batch_number, total_batches)
    return torch.cat(vectors, dim=0)


def retrieval_metrics(rankings, gold_indices):
    out = {}
    ranks = []
    for ranked, gold in zip(rankings, gold_indices):
        ranks.append(int(np.where(ranked == gold)[0][0]) + 1)
    ranks = np.asarray(ranks)
    for k in TOP_KS:
        out[f"Recall@{k}"] = float((ranks <= k).mean())
    out["MRR"] = float((1.0 / ranks).mean())
    out["mean_rank"] = float(ranks.mean())
    return out


def run_split(model_name, model_id, config, split):
    language = "python" if config.startswith("python") else "java"
    # RepoBench-R is an old script-backed dataset; modern dataset loaders
    # intentionally reject it, so always use the direct archive loader here.
    ds = load_repobench_r(config, split)
    if LIMIT is not None:
        ds = ds[: min(LIMIT, len(ds))]

    total_examples = len(ds)
    # The released pickle contains a small number of malformed examples whose
    # gold index falls outside their released candidate list.  Such examples
    # have no possible correct rank, so exclude them and report the count.
    rows = [
        row for row in ds
        if 0 <= row["golden_snippet_index"] < len(row["context"])
    ]
    invalid_gold_indices = total_examples - len(rows)
    if invalid_gold_indices:
        print(f"    skipped {invalid_gold_indices}/{total_examples} examples with an invalid gold index")
    if not rows:
        raise ValueError(f"No valid examples remain in {config}/{split}")
    queries = [make_query(row) for row in rows]
    unique_candidates = []
    candidate_to_id = {}
    candidate_ids_per_row = []
    total_candidate_occurrences = 0
    for row in rows:
        local_ids = []
        for snippet in row["context"]:
            candidate = make_candidate(snippet, language)
            candidate_id = candidate_to_id.get(candidate)
            if candidate_id is None:
                candidate_id = len(unique_candidates)
                candidate_to_id[candidate] = candidate_id
                unique_candidates.append(candidate)
            local_ids.append(candidate_id)
        candidate_ids_per_row.append(local_ids)
        total_candidate_occurrences += len(local_ids)
    print(
        f"    valid examples: {len(rows):,}; candidate occurrences: {total_candidate_occurrences:,}; "
        f"unique snippets: {len(unique_candidates):,}"
    )
    gold_indices = [row["golden_snippet_index"] for row in rows]

    query_vectors = encode(model, tokenizer, queries, QUERY_MAX_LENGTH, True, QUERY_BATCH_SIZE)
    candidate_vectors = encode(model, tokenizer, unique_candidates, CODE_MAX_LENGTH, False, CODE_BATCH_SIZE)

    rankings = []
    for i, candidate_ids in enumerate(candidate_ids_per_row):
        scores = torch.mv(candidate_vectors[candidate_ids], query_vectors[i])
        rankings.append(torch.argsort(scores, descending=True).numpy())

    metrics = retrieval_metrics(rankings, gold_indices)
    metrics.update({
        "model": model_name,
        "model_id": model_id,
        "config": config,
        "split": split,
        "examples": len(rows),
        "source_examples": total_examples,
        "invalid_gold_indices": invalid_gold_indices,
        "avg_candidates": float(total_candidate_occurrences / len(rows)),
        "unique_candidate_snippets": len(unique_candidates),
    })
    return metrics


# %%
all_results = []

for model_name, model_id in MODELS.items():
    print(f"\nLoading {model_name}: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()

    for config in CONFIGS:
        for split in SPLITS:
            print(f"  {config}/{split}")
            all_results.append(run_split(model_name, model_id, config, split))
            pd.DataFrame(all_results).to_csv("rlretriever_repobench_r_results.csv", index=False)
            gc.collect()
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

results = pd.DataFrame(all_results)
display(results.sort_values(["config", "split", "model"]))

# Macro-average over the four RepoBench-R configurations for each difficulty.
summary = (
    results.groupby(["model", "split"], as_index=False)[["Recall@1", "Recall@3", "Recall@5", "MRR", "mean_rank"]]
    .mean()
    .sort_values(["split", "Recall@1"], ascending=[True, False])
)
display(summary)
results.to_csv("rlretriever_repobench_r_results.csv", index=False)
summary.to_csv("rlretriever_repobench_r_summary.csv", index=False)
print("Saved: rlretriever_repobench_r_results.csv and rlretriever_repobench_r_summary.csv")
