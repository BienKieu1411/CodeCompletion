# %% [markdown]
# # Adapted AlignCoder enhanced-query retrieval on RepoBench-R
#
# RepoBench-R provides a *closed candidate pool* rather than a whole repository.
# This script therefore uses each supplied pool as a mini corpus:
#
# `prefix -> BM25(pool) -> sampled completion hypotheses -> enhanced query
#         -> adapted AlignRetriever rerank -> greedy completion -> append prefix`
#
# This follows AlignCoder's *forward generation* control flow.  RepoBench-R
# still supplies a closed candidate pool (rather than a whole repository), and
# has a retrieval label only for the original cursor; therefore Recall/MRR are
# explicitly computed from forward step 1 only.  Candidate embeddings are cached
# once per unique formatted candidate within each split.

# %%
import gc
import gzip
import json
import math
import os
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator first."
DEVICE = torch.device("cuda:0")
print("retriever/generator device:", torch.cuda.get_device_name(0))

# Kaggle Dataset input.  It may either be the contents of ``result_3`` itself
# (the usual attached-Dataset layout), or a notebook-output folder containing
# ``.../retriever_ckpt/result_3``.  The resolver supports both layouts.
CHECKPOINT_INPUT = Path(os.environ.get(
    "ALIGNCODER_CHECKPOINT_INPUT",
    "/kaggle/input/datasets/thanhnguyenvinh/aligncoder-retriever-adapted",
))
TARGET_CHECKPOINT_NAME = "result_3"
GENERATOR_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"  # paper's family/default

# Full RepoBench-R: 4 language/settings configurations × easy/hard.
# This is very expensive with generator-enhanced queries.  Set LIMIT=1000 for
# a development smoke test; leave None for the requested complete evaluation.
LIMIT = None
CONFIGS = ("python_cff", "python_cfr", "java_cff", "java_cfr")
SPLITS = ("test_easy", "test_hard")
SUPPORTED_CONFIGS = {"python_cff", "python_cfr", "java_cff", "java_cfr"}

# Parentheses alone do not create a one-item tuple in Python: ``("java_cff")``
# is a string and would make the loop request bogus files such as ``data/j.gz``.
# Accept that common notebook edit safely, while rejecting actual misspellings.
if isinstance(CONFIGS, str):
    CONFIGS = (CONFIGS,)
unknown_configs = set(CONFIGS) - SUPPORTED_CONFIGS
if unknown_configs:
    raise ValueError(
        f"Unsupported RepoBench-R config(s): {sorted(unknown_configs)}; "
        f"choose from {sorted(SUPPORTED_CONFIGS)}"
    )

# The adapted retriever was trained with ``inference_type=unixcoder_with_rl``
# and ``sample_number=8``.  In AlignCoder this selects BM25 top-(8 * 10) before
# UniXcoder reranking, then keeps the best 8 blocks for generation.
SAMPLE_NUMBER = 8
BM25_CONTEXT_TOPK = SAMPLE_NUMBER * 10
DENSE_TOPK = SAMPLE_NUMBER
# Number of sampled completion hypotheses used to build one enhanced query.
# This is separate from SAMPLE_NUMBER, which controls retrieved code blocks.
NUM_GENERATIONS = 3
# A retrieval call enhances its query once.  This is distinct from
# ``forward_generation_times``: AlignCoder does not repeatedly enhance one
# fixed cursor.  It greedily generates code, appends it to the cursor, then
# makes a new retrieval call at the next forward step.
QUERY_ENHANCEMENT_ROUNDS = 1
FORWARD_GENERATION_STEPS = 3
assert QUERY_ENHANCEMENT_ROUNDS == 1, "AlignCoder performs one query enhancement per forward step"
GEN_BATCH_SIZE = 2
GEN_MAX_CONTEXT = 1024
GEN_MAX_NEW_TOKENS = 64
QUERY_BATCH_SIZE = 256
CODE_BATCH_SIZE = 256
QUERY_MAX_LENGTH = 256
CODE_MAX_LENGTH = 512
TOP_KS = (1, 3, 5)

# Resume support.  In the same Kaggle session this working-path CSV is reused
# automatically.  For a fresh notebook, upload/attach the partial CSV as a
# Kaggle Dataset and set this to its /kaggle/input/... path.
ARTIFACT_DIR = Path("/kaggle/working/aligncoder_repobench_r_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = ARTIFACT_DIR / "aligncoder_forward_generation_repobench_r_results.csv"
# Do not resume the old query-only evaluator: it measured a different protocol.
LEGACY_RESULTS_CSV = Path("/kaggle/working/aligncoder_forward_generation_repobench_r_results.csv")
PREVIOUS_RESULTS_CSV = None

(ARTIFACT_DIR / "run_manifest.json").write_text(json.dumps({
    "checkpoint_input": str(CHECKPOINT_INPUT),
    "generator_model": GENERATOR_MODEL,
    "limit": LIMIT,
    "configs": CONFIGS,
    "splits": SPLITS,
    "bm25_context_topk": BM25_CONTEXT_TOPK,
    "dense_topk": DENSE_TOPK,
    "target_checkpoint_name": TARGET_CHECKPOINT_NAME,
    "num_generations": NUM_GENERATIONS,
    "query_enhancement_rounds": QUERY_ENHANCEMENT_ROUNDS,
    "forward_generation_steps": FORWARD_GENERATION_STEPS,
    "retrieval_metric_forward_step": 1,
    "generator_max_context": GEN_MAX_CONTEXT,
    "generator_max_new_tokens": GEN_MAX_NEW_TOKENS,
}, indent=2), encoding="utf-8")


def print_progress(label, current, total, width=32):
    ratio = current / total if total else 1.0
    filled = int(width * ratio)
    bar = "=" * width if current >= total else "=" * filled + ">" + "." * max(0, width - filled - 1)
    print(f"\r    {label:<24} [{bar}] {ratio:6.2%} ({current}/{total})", end="", flush=True)
    if current >= total:
        print()


def resolve_checkpoint(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint input does not exist: {root}")
    direct_weights = [root / "model.safetensors", root / "pytorch_model.bin"]
    if (root / "config.json").is_file() and any(path.is_file() for path in direct_weights):
        print("Using attached checkpoint root:", root)
        return root
    matches = list(root.rglob("model.safetensors")) + list(root.rglob("pytorch_model.bin"))
    target = [p.parent for p in matches if p.parent.name == TARGET_CHECKPOINT_NAME]
    candidates = target or [p.parent for p in matches]
    if not candidates:
        raise FileNotFoundError(f"No Hugging Face model weights found under {root}")
    if not target:
        available = sorted({p.parent.name for p in matches})
        raise FileNotFoundError(
            f"Requested checkpoint {TARGET_CHECKPOINT_NAME!r} was not found under {root}; "
            f"available checkpoint folders: {available}"
        )
    checkpoint = candidates[0]
    print("Using adapted retriever checkpoint:", checkpoint)
    return checkpoint


def load_repobench_r(config, split):
    archive = hf_hub_download(
        repo_id="tianyang/repobench-r",
        repo_type="dataset",
        filename=f"data/{config}.gz",
    )
    with gzip.open(archive, "rb") as handle:
        raw = pickle.load(handle)
    subset, difficulty = split.split("_", maxsplit=1)
    return raw[subset][difficulty]


def retrieval_query(left_context):
    """AlignCoder retrieval query: the last 20 non-empty source lines."""
    lines = [line for line in left_context.splitlines() if line.strip()]
    return "\n".join(lines[-20:])


def make_candidate(snippet, language):
    # Same CodeBlock-like text format used in the retriever evaluator.
    comment = "#" if language == "python" else "//"
    lines = [line for line in snippet.splitlines() if line.strip()]
    return "\n".join(
        [f"{comment} file path: repobench_candidate", f"{comment} lines: 0-{len(snippet.splitlines())}"]
        + [f"{comment} {line}" for line in lines]
    ).strip()


def upstream_tokenize(tokenizer, text, max_length, is_query):
    tokens = tokenizer.tokenize(text)
    tokens = tokens[-max_length + 4:] if is_query else tokens[:max_length - 4]
    tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + tokens + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids + [tokenizer.pad_token_id] * (max_length - len(ids))


@torch.inference_mode()
def encode_retriever(model, tokenizer, texts, max_length, is_query, batch_size):
    vectors = []
    label = "encoding enhanced queries" if is_query else "caching unique candidates"
    total = math.ceil(len(texts) / batch_size)
    print_progress(label, 0, total)
    for number, start in enumerate(range(0, len(texts), batch_size), 1):
        ids = torch.tensor(
            [upstream_tokenize(tokenizer, x, max_length, is_query) for x in texts[start:start + batch_size]],
            dtype=torch.long,
            device=DEVICE,
        )
        mask = ids.ne(tokenizer.pad_token_id)
        hidden = model(ids, attention_mask=mask).last_hidden_state
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
        vectors.append(F.normalize(pooled, p=2, dim=1).cpu())
        print_progress(label, number, total)
    return torch.cat(vectors, dim=0)


def bm25_candidate_indices(row, base_query, topk):
    """Offline BM25 first-stage retrieval over one RepoBench-R candidate pool.

    Kaggle notebooks with Internet disabled cannot install ``rank-bm25``.  This
    compact implementation has the same Okapi BM25 scoring role and avoids any
    runtime package download.
    """
    corpus = [snippet.lower().split() for snippet in row["context"]]
    if not corpus:
        return np.empty(0, dtype=np.int64)
    query_terms = base_query.lower().split()
    if not query_terms:
        return np.empty(0, dtype=np.int64)
    document_count = len(corpus)
    document_lengths = np.asarray([len(document) for document in corpus], dtype=np.float32)
    average_length = max(1.0, float(document_lengths.mean()))
    document_frequency = Counter(term for document in corpus for term in set(document))
    term_frequencies = [Counter(document) for document in corpus]
    scores = np.zeros(document_count, dtype=np.float32)
    k1, b = 1.5, 0.75
    for term in set(query_terms):
        frequency = document_frequency.get(term, 0)
        if frequency == 0:
            continue
        inverse_frequency = math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
        for index, counts in enumerate(term_frequencies):
            term_count = counts.get(term, 0)
            if term_count:
                normalizer = term_count + k1 * (1.0 - b + b * document_lengths[index] / average_length)
                scores[index] += inverse_frequency * term_count * (k1 + 1.0) / normalizer
    return np.argsort(scores)[::-1][:min(topk, len(corpus))]


def bm25_contexts(row, base_query, topk):
    indices = bm25_candidate_indices(row, base_query, topk)
    return [row["context"][int(i)] for i in indices]


def generator_prompt(tokenizer, left_context, contexts, language):
    # Mirrors AlignCoder's prompt layout; RepoBench-R has no source file path.
    blocks = "\n\n".join(make_candidate(code, language) for code in contexts)
    block_ids = tokenizer.encode(blocks, add_special_tokens=False)[:512]
    path_ids = tokenizer.encode("\n\n# file path: repobench_target\n\n", add_special_tokens=False)
    allowed = max(1, GEN_MAX_CONTEXT - len(block_ids) - len(path_ids) - 10)
    context_ids = tokenizer.encode(left_context, add_special_tokens=False)[-allowed:]
    return tokenizer.decode(block_ids + path_ids + context_ids)


def compose_enhanced_query(base_query, hypotheses):
    """AlignCoder query layout: source query first, sampled answers afterward."""
    unique = list(dict.fromkeys(x.strip() for x in hypotheses if x.strip()))
    if not unique:
        return base_query
    hypotheses_text = "\n\n".join(
        f"# candidate answer {index}:\n{completion}"
        for index, completion in enumerate(unique, 1)
    )
    return base_query + "\n\n" + hypotheses_text


@torch.inference_mode()
def enhance_queries_once(
    generator,
    tokenizer,
    rows,
    retrieval_queries,
    full_left_contexts,
    language,
    forward_step,
):
    """Sample predictions once, using BM25 from the short retrieval query.

    As in AlignCoder, retrieval sees the last 20 non-empty lines while the
    generator prompt retains the token-budgeted tail of the full left context.
    """
    prompts = [
        generator_prompt(
            tokenizer,
            full_left_context,
            bm25_contexts(row, query, BM25_CONTEXT_TOPK),
            language,
        )
        for row, query, full_left_context in zip(rows, retrieval_queries, full_left_contexts)
    ]
    tokenizer.padding_side = "left"
    next_hypotheses = []
    total = math.ceil(len(prompts) / GEN_BATCH_SIZE)
    progress_label = f"enhancement step {forward_step}"
    print_progress(progress_label, 0, total)
    for number, start in enumerate(range(0, len(prompts), GEN_BATCH_SIZE), 1):
        batch_prompts = prompts[start:start + GEN_BATCH_SIZE]
        batch = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=GEN_MAX_CONTEXT).to(DEVICE)
        inputs = {key: value.repeat_interleave(NUM_GENERATIONS, dim=0) for key, value in batch.items()}
        torch.manual_seed(SEED + number)
        generated = generator.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for offset in range(0, len(completions), NUM_GENERATIONS):
            next_hypotheses.append(completions[offset:offset + NUM_GENERATIONS])
        print_progress(progress_label, number, total)
    return next_hypotheses


@torch.inference_mode()
def greedy_generate_once(generator, tokenizer, prompts, forward_step):
    """AlignCoder forward generation: greedy, one completion per prompt."""
    tokenizer.padding_side = "left"
    completions = []
    total = math.ceil(len(prompts) / GEN_BATCH_SIZE)
    print_progress(f"forward generation {forward_step}", 0, total)
    for number, start in enumerate(range(0, len(prompts), GEN_BATCH_SIZE), 1):
        batch = tokenizer(
            prompts[start:start + GEN_BATCH_SIZE],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=GEN_MAX_CONTEXT,
        ).to(DEVICE)
        generated = generator.generate(
            **batch,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = generated[:, batch["input_ids"].shape[1]:]
        completions.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        print_progress(f"forward generation {forward_step}", number, total)
    return completions


def metrics(rankings, gold_indices):
    # Dense reranking is deliberately applied only to the BM25 pool, as in
    # AlignCoder.  A missing gold block is ranked immediately after that pool.
    ranks = []
    for ranking, gold in zip(rankings, gold_indices):
        positions = np.flatnonzero(ranking == gold)
        ranks.append(int(positions[0]) + 1 if len(positions) else len(ranking) + 1)
    ranks = np.asarray(ranks)
    result = {f"Recall@{k}": float((ranks <= k).mean()) for k in TOP_KS}
    result.update(MRR=float((1 / ranks).mean()), mean_rank=float(ranks.mean()))
    return result


def run_split(retriever, retriever_tokenizer, generator, generator_tokenizer, config, split):
    language = "python" if config.startswith("python") else "java"
    source = load_repobench_r(config, split)
    if LIMIT is not None:
        source = source[:min(LIMIT, len(source))]
    rows = [r for r in source if 0 <= r["golden_snippet_index"] < len(r["context"])]
    print(f"    valid examples: {len(rows):,}/{len(source):,}")
    original_left_contexts = [row["code"] for row in rows]

    # Deduplicate only static candidate representations.  Enhanced queries are
    # necessarily encoded per example after generation.
    unique, lookup, ids_per_row = [], {}, []
    for row in rows:
        local = []
        for snippet in row["context"]:
            candidate = make_candidate(snippet, language)
            if candidate not in lookup:
                lookup[candidate] = len(unique)
                unique.append(candidate)
            local.append(lookup[candidate])
        ids_per_row.append(local)
    print(f"    candidate occurrences: {sum(map(len, ids_per_row)):,}; unique cached: {len(unique):,}")

    candidate_vectors = encode_retriever(retriever, retriever_tokenizer, unique, CODE_MAX_LENGTH, False, CODE_BATCH_SIZE)
    # This is equivalent to AlignCoder/main.py's forward-generation loop:
    # retrieve with a one-pass enhanced query, greedy-decode, then replace the
    # working left context by original-prefix + all generated text so far.
    working_left_contexts = list(original_left_contexts)
    accumulated_generations = ["" for _ in rows]
    rankings = None
    first_step_pool_sizes = None
    for forward_step in range(1, FORWARD_GENERATION_STEPS + 1):
        # AlignCoder derives a fresh short retrieval query at every moving
        # cursor, but gives the generator the full token-budgeted left context.
        retrieval_queries = [retrieval_query(context) for context in working_left_contexts]
        hypotheses = enhance_queries_once(
            generator,
            generator_tokenizer,
            rows,
            retrieval_queries,
            working_left_contexts,
            language,
            forward_step,
        )
        enhanced = [
            compose_enhanced_query(query, sample_hypotheses)
            for query, sample_hypotheses in zip(retrieval_queries, hypotheses)
        ]
        query_vectors = encode_retriever(
            retriever, retriever_tokenizer, enhanced,
            QUERY_MAX_LENGTH, True, QUERY_BATCH_SIZE,
        )
        step_rankings, step_contexts = [], []
        for index, (row, local_ids) in enumerate(zip(rows, ids_per_row)):
            bm25_indices = bm25_candidate_indices(row, enhanced[index], BM25_CONTEXT_TOPK)
            # Convert local candidate indices to cached-vector indices before
            # scoring, then convert the final rank order back to local indices.
            vector_indices = [local_ids[int(candidate)] for candidate in bm25_indices]
            scores = torch.mv(candidate_vectors[vector_indices], query_vectors[index])
            order = torch.argsort(scores, descending=True).cpu().numpy()
            dense_ranking = bm25_indices[order]
            step_rankings.append(dense_ranking)
            top_indices = dense_ranking[:DENSE_TOPK]
            step_contexts.append([row["context"][int(candidate)] for candidate in top_indices])
        if forward_step == 1:
            rankings = step_rankings
            first_step_pool_sizes = [len(rank) for rank in rankings]
        prompts = [
            generator_prompt(generator_tokenizer, left_context, contexts, language)
            for left_context, contexts in zip(working_left_contexts, step_contexts)
        ]
        generated = greedy_generate_once(generator, generator_tokenizer, prompts, forward_step)
        accumulated_generations = [old + new for old, new in zip(accumulated_generations, generated)]
        working_left_contexts = [
            original + generated
            for original, generated in zip(original_left_contexts, accumulated_generations)
        ]

    result = metrics(rankings, [r["golden_snippet_index"] for r in rows])
    result.update(model="AlignCoder_enhanced_adapted", config=config, split=split,
                  examples=len(rows), source_examples=len(source),
                  avg_candidates=float(sum(map(len, ids_per_row)) / len(rows)),
                  avg_bm25_pool=float(np.mean(first_step_pool_sizes)),
                  unique_candidate_snippets=len(unique), generator=GENERATOR_MODEL,
                  retriever_checkpoint=str(resolve_checkpoint(CHECKPOINT_INPUT)),
                  forward_generation_steps=FORWARD_GENERATION_STEPS,
                  query_enhancement_rounds=QUERY_ENHANCEMENT_ROUNDS,
                  retrieval_metric_forward_step=1)
    prediction_path = ARTIFACT_DIR / f"{config}_{split}_forward_generations.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row, completion in zip(rows, accumulated_generations):
            handle.write(json.dumps({
                "code_before_cursor": row["code"],
                "next_line": row.get("next_line"),
                "forward_generated_code": completion,
            }, ensure_ascii=False) + "\n")
    print(f"    wrote forward-generation outputs: {prediction_path}")
    return result


# %%
checkpoint = resolve_checkpoint(CHECKPOINT_INPUT)
print("Loading adapted AlignRetriever...")
retriever_tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
retriever = AutoModel.from_pretrained(checkpoint).to(DEVICE).eval()

print("Loading frozen generator:", GENERATOR_MODEL)
generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
if generator_tokenizer.pad_token_id is None:
    generator_tokenizer.pad_token_id = generator_tokenizer.eos_token_id
generator_tokenizer.padding_side = "left"
generator_tokenizer.truncation_side = "left"
generator = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL, torch_dtype=torch.float16).to(DEVICE).eval()

if PREVIOUS_RESULTS_CSV is not None:
    previous_path = Path(PREVIOUS_RESULTS_CSV)
elif RESULTS_CSV.exists():
    previous_path = RESULTS_CSV
elif LEGACY_RESULTS_CSV.exists():
    previous_path = LEGACY_RESULTS_CSV
else:
    previous_path = None

if previous_path is not None and previous_path.exists():
    results = pd.read_csv(previous_path).to_dict("records")
    print(f"Resuming from {previous_path}: {len(results)} completed split(s)")
else:
    results = []

completed = {(row["config"], row["split"]) for row in results}
for config in CONFIGS:
    for split in SPLITS:
        if (config, split) in completed:
            print(f"\n{config}/{split}: already present in results CSV; skipping")
            continue
        print(f"\n{config}/{split}")
        results.append(run_split(retriever, retriever_tokenizer, generator, generator_tokenizer, config, split))
        pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
        print(f"    checkpointed partial results: {RESULTS_CSV}")
        torch.cuda.empty_cache()
        gc.collect()

results = pd.DataFrame(results)
results = results.sort_values(["config", "split"]).reset_index(drop=True)
display(results)
results.to_csv(RESULTS_CSV, index=False)

# Equal-weight macro average over the four RepoBench-R configurations.  Keep
# easy and hard separate because they have materially different pool sizes.
metric_columns = ["Recall@1", "Recall@3", "Recall@5", "MRR", "mean_rank"]
summary = (
    results.groupby("split", as_index=False)[metric_columns]
    .mean()
    .assign(model="AlignCoder_enhanced_adapted", aggregation="macro_avg_over_4_configs")
    [["model", "aggregation", "split", *metric_columns]]
    .sort_values("split")
    .reset_index(drop=True)
)
display(summary)
SUMMARY_CSV = ARTIFACT_DIR / "aligncoder_forward_generation_repobench_r_summary.csv"
summary.to_csv(SUMMARY_CSV, index=False)
print(f"Saved artifact folder: {ARTIFACT_DIR}")
