"""Sample repository tasks and annotate measured context interventions via vLLM.

Run separately from supervised training. One committed SQLite bundle per
query makes annotation resumable without repeating completed requests.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sqlite3
import time

import requests

from src.data.ast_repo_pool import sample_repo_task
from src.data.ast_training_data import DataConfig
from src.data.audit_completion_data import TOKENIZER_REVISION
from src.data.build_ast_repo_pool import RET_REVISION
from src.data.repo_index import digest_file
from src.train.context_contract import ContextConfig, ContextRenderer, fingerprint
from src.train.epoch_data_loader import EpochDataLoader, pack, unpack
from src.train.utility_metrics import VERSION as METRIC_VERSION, completion_score

MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
BUNDLE_VERSION = "cur_bundles_stratified20_v3"
MAX_CONTEXTS = 20


def intervention_graph(renderer, seed, *, audit=None, rank_candidates=None):
    """Three cardinality strata, <=20 scored sets with shared endpoints.

    Empty + 3 * (S, S+a, S+b, S+a+b, S-i, S-i+a) + singleton a.
    Each feasible four-node diamond measures a local discrete interaction.
    Sparse, label-free coverage is not exhaustive attribution or Shapley value.
    """
    rng = random.Random(seed)
    nodes, lookup, edges = [], {}, {}
    priority = {"member": 0, "add": 1, "joint": 2, "swap": 3}
    n = len(renderer.candidates)

    def node(indices):
        key = renderer.canonical(indices)
        if not renderer.feasible(key):
            return None
        if key not in lookup:
            if len(nodes) >= MAX_CONTEXTS:
                raise AssertionError("Annotation exceeded twenty contexts")
            lookup[key] = len(nodes)
            nodes.append({"indices": list(key), "cost": renderer.render(key)[1]})
        return lookup[key]

    def edge(a, b, group):
        s, t = node(a), node(b)
        if s is None or t is None or s == t:
            return
        s, t = sorted((s, t))
        old = edges.get((s, t))
        if old is None or priority[group] < priority[old["group"]]:
            edges[s, t] = {"source": s, "target": t, "group": group}

    node(())
    modes = ["lexical", "random", "model" if rank_candidates is not None else "path_cluster"]
    rng.shuffle(modes)  # Do not confound cardinality with ordering strategy.
    records, singleton = [], None
    for (low, high), mode in zip(((1, 2), (4, 5), (7, 8)), modes):
        requested = rng.randint(low, high)
        size = min(requested, max(0, n - 2), max(0, renderer.config.max_snippets - 2))
        order = list(range(n))
        if mode == "random":
            rng.shuffle(order)
        elif mode == "path_cluster":
            order.sort(key=lambda i: (renderer.candidates[i]["path"], i))
        state = []
        while len(state) < size:
            possible = [i for i in order if i not in state]
            feasible = renderer.feasible_many([state + [i] for i in possible])
            available = [i for i, ok in zip(possible, feasible) if ok]
            if not available:
                break
            if mode == "model":
                available = rank_candidates(state, available)
            state.append(available[0])
        state = renderer.canonical(state)
        node(state)
        outside = [i for i in range(n) if i not in state]
        feasible = renderer.feasible_many([(*state, i) for i in outside])
        feasible_adds = [i for i, ok in zip(outside, feasible) if ok]
        # Prefer measurable adds; if full by tokens, still try a replacement.
        probes = feasible_adds or outside
        if rank_candidates is not None and feasible_adds:
            probes = rank_candidates(state, probes)
        chosen = probes[:1]
        if len(probes) > 1:
            chosen.append(rng.choice(probes[1:]))
        for i in chosen:
            edge(state, (*state, i), "add")
        diamond = None
        if len(chosen) == 2:
            a, b = chosen
            four = [state, (*state, a), (*state, b), (*state, a, b)]
            if all(renderer.feasible_many(four)):
                edge(state, four[3], "joint")
                edge(four[1], four[3], "joint")
                edge(four[2], four[3], "joint")
                diamond = [lookup[renderer.canonical(s)] for s in four]
        removed_id, swap_id, old = None, None, None
        if state:
            old = rng.choice(state)
            removed = tuple(i for i in state if i != old)
            edge(removed, state, "member")
            removed_id = lookup.get(renderer.canonical(removed))
            if chosen:
                replacement = (*removed, chosen[0])
                edge(state, replacement, "swap")
                swap_id = lookup.get(renderer.canonical(replacement))
        if chosen and renderer.feasible_many([(chosen[0],)])[0]:
            singleton = chosen[0]
        records.append({"requested_size": requested, "actual_size": len(state),
                        "mode": mode, "base": lookup[state], "probes": chosen,
                        "removed_candidate": old, "removed_node": removed_id,
                        "swap_node": swap_id, "diamond": diamond})
    if singleton is not None:
        edge((), (singleton,), "add")
    if audit is not None:
        audit.update(max_contexts=MAX_CONTEXTS, strata=records,
                     unique_contexts=len(nodes), unique_edges=len(edges),
                     complete_diamonds=sum(r["diamond"] is not None for r in records))
    return nodes, list(edges.values())


class BundleStore:
    def __init__(self, path, manifest):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, value TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS bundles (task_key TEXT PRIMARY KEY, payload BLOB)")
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, completion TEXT)")
        previous = self.db.execute("SELECT value FROM metadata WHERE id=1").fetchone()
        if previous and json.loads(previous[0]) != manifest:
            self.db.close()
            raise ValueError("Annotation contract changed; use a new bundle file")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES (1, ?)", (json.dumps(manifest),))
        self.db.commit()

    def close(self):
        self.db.close()


class CompletionOracle:
    def __init__(self, base_url, store, revision, config, batch_size=108, *, request_timeout=900,
                 check_continue=None):
        self.base_url, self.store = base_url.rstrip("/"), store
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        if batch_size < 1:
            raise ValueError("Positive generator batch size required")
        self.batch_size = batch_size
        if request_timeout <= 0:
            raise ValueError("Positive request timeout required")
        self.request_timeout, self.check_continue = request_timeout, check_continue
        self.session = requests.Session()
        # Never put credentials in notebooks, manifests, cache keys or logs.
        if os.environ.get("MODAL_KEY") and os.environ.get("MODAL_SECRET"):
            self.session.headers.update({"Modal-Key": os.environ["MODAL_KEY"],
                                         "Modal-Secret": os.environ["MODAL_SECRET"]})
        elif os.environ.get("VLLM_API_KEY"):
            self.session.headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
        self.contract = {"model": MODEL, "revision": revision,
                         "temperature": 0.0, "max_tokens": config.output_tokens,
                         "seed": 123, "add_special_tokens": False, "n": 1}

    def verify_served_model(self):
        response = self.session.get(self.base_url + "/v1/models", timeout=30)
        if not response.ok:
            raise RuntimeError(f"Generator model probe HTTP {response.status_code}")
        names = {item["id"] for item in response.json().get("data", [])}
        if MODEL not in names:
            raise ValueError(f"Endpoint must serve {MODEL}; got {sorted(names)}")
        # OpenAI-compatible /models does not prove the HF weights revision.
        # Deployment must pin the revision recorded in the annotation manifest.

    def generate(self, prompts):
        keys = [fingerprint({"generation": self.contract, "prompt": p}) for p in prompts]
        found, missing = {}, {}
        for key, prompt in zip(keys, prompts):
            row = (self.store.db.execute("SELECT completion FROM responses WHERE key=?", (key,)).fetchone()
                   if self.store is not None else None)
            if row:
                found[key] = row[0]
            else:
                missing[key] = prompt
        pending = list(missing.items())
        print(json.dumps({"phase": "oracle", "prompts": len(prompts),
                          "uncached_unique": len(pending)}), flush=True)
        for start in range(0, len(pending), self.batch_size):
            if self.check_continue is not None:
                self.check_continue()
            batch = pending[start:start+self.batch_size]
            request = {k: v for k, v in self.contract.items() if k != "revision"}
            request["prompt"] = [p for _, p in batch]
            response = self.session.post(self.base_url + "/v1/completions", json=request,
                                         timeout=(10, self.request_timeout))
            if not response.ok:
                raise RuntimeError(f"Generator HTTP {response.status_code}: {response.text[:600]}")
            choices = response.json().get("choices", [])
            by_index = {choice["index"]: choice for choice in choices}
            if len(choices) != len(batch) or set(by_index) != set(range(len(batch))):
                raise ValueError("Generator returned missing/duplicate completion indices")
            for i, (key, _) in enumerate(batch):
                completion = by_index[i]["text"]
                if not isinstance(completion, str):
                    raise ValueError("Generator completion must be a string")
                found[key] = completion
                if self.store is not None:
                    self.store.db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, completion))
            if self.store is not None:
                self.store.db.commit()
            print(f"Oracle generated {min(start+self.batch_size, len(pending))}/{len(pending)}", flush=True)
        return [found[key] for key in keys]


def prepare_annotation(task, generator, config, seed):
    renderer = ContextRenderer(task, generator, config)
    design = {}
    nodes, edges = intervention_graph(renderer, seed, audit=design)
    if not edges:
        raise ValueError(f"Task has no feasible intervention: {task['task_id']}")
    prompts = [renderer.render(node["indices"])[0] for node in nodes]
    return {"task": task, "renderer": renderer, "nodes": nodes, "edges": edges,
            "design": design, "prompts": prompts, "seed": seed}


def finish_annotation(prepared, predictions):
    task, renderer = prepared["task"], prepared["renderer"]
    nodes, edges, design = prepared["nodes"], prepared["edges"], prepared["design"]
    if len(predictions) != len(nodes):
        raise ValueError("Annotation predictions do not match scored contexts")
    for node, completion in zip(nodes, predictions):
        node.update(completion_score(completion, task["target_code"], task["target_kind"]))
        node["completion"] = completion
    # Explicit input allowlist. Dependency evidence and other gold-derived
    # diagnostics cannot enter the learned model's features.
    model_task = {k: task[k] for k in ("task_id", "repo_uid", "language", "file_path", "candidates")}
    model_task["left_context"] = renderer.left_context
    return {"version": BUNDLE_VERSION, "task": model_task, "nodes": nodes, "edges": edges,
            "design": design,
            "target_kind": task["target_kind"], "context_contract": renderer.contract(),
            "candidate_costs": [renderer.count(piece) for piece in renderer.pieces],
            "prefix_metadata": renderer.prefix_metadata,
            "metric": METRIC_VERSION, "seed": prepared["seed"]}


def annotate_many(prepared_tasks, oracle):
    """Feed multiple independent tasks into one request stream, not more labels/task."""
    predictions = oracle.generate([p for item in prepared_tasks for p in item["prompts"]])
    if len(predictions) != sum(len(item["nodes"]) for item in prepared_tasks):
        raise ValueError("Batched annotation predictions do not match scored contexts")
    bundles, start = [], 0
    for item in prepared_tasks:
        end = start + len(item["nodes"])
        bundles.append(finish_annotation(item, predictions[start:end]))
        start = end
    return bundles


def annotate(task, generator, oracle, config, seed):
    return annotate_many([prepare_annotation(task, generator, config, seed)], oracle)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="Prepared train.parquet repository pool")
    parser.add_argument("--output", type=Path, required=True, help="Resumable bundle .sqlite file")
    parser.add_argument("--base-url", required=True, help="Existing vLLM endpoint")
    parser.add_argument("--generator-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--split", choices=["train"], default="train",
                        help="Compatibility flag; repository validation is disabled")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=108)
    parser.add_argument("--tasks-per-batch", type=int, default=8,
                        help="Prepare several tasks to fill generator batches; labels/task still <=20")
    parser.add_argument("--limit", type=int, help="Explicit development subset, recorded in manifest")
    args = parser.parse_args()
    if min(args.epoch, args.batch_size, args.tasks_per_batch) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Positive epoch/batch sizes/limit required")
    from transformers import AutoTokenizer
    generator = AutoTokenizer.from_pretrained(MODEL, revision=args.generator_revision)
    retriever = AutoTokenizer.from_pretrained("microsoft/unixcoder-base", revision=RET_REVISION)
    generator.model_max_length = retriever.model_max_length = 10**9
    import pyarrow.parquet as pq
    pool_metadata = pq.ParquetFile(args.pool).schema_arrow.metadata or {}
    if b"preparation_contract" not in pool_metadata:
        raise ValueError("Expected the prepared repository-pool train.parquet schema")
    pool_contract = json.loads(pool_metadata[b"preparation_contract"])
    if (pool_contract["generator_tokenizer_revision"] != args.generator_revision
            or pool_contract["retriever_tokenizer_revision"] != RET_REVISION):
        raise ValueError("Annotation tokenizer revisions must match the prepared token budgets")
    data_config, context_config = DataConfig(**pool_contract["config"]), ContextConfig()
    loader = EpochDataLoader(args.pool, generator, retriever, data_config)
    indices = loader.select_epoch_indices(args.epoch, args.seed, split="train")
    if args.limit:
        indices = indices[:args.limit]
    manifest = {"version": BUNDLE_VERSION, "max_contexts": MAX_CONTEXTS, "pool_sha256": digest_file(args.pool),
                "generator": MODEL, "generator_revision": args.generator_revision,
                "retriever_revision": RET_REVISION, "metric": METRIC_VERSION,
                "context": asdict(context_config), "data": asdict(data_config),
                "split": "train", "validation": "disabled",
                "epoch": args.epoch, "seed": args.seed, "limit": args.limit,
                "training_sampling": {
                    "epoch_size": loader.epoch_size,
                    "python_quota": loader.python_quota,
                    "selection": loader.last_selection,
                },
                "selected_repos": [loader.repo_rows[i]["repo_uid"] for i in indices],
                "decoding": {"temperature": 0., "seed": 123, "n": 1, "add_special_tokens": False},
                "batching": {"http_batch_size": args.batch_size, "tasks_per_batch": args.tasks_per_batch},
                "fuzzywuzzy_version": importlib.metadata.version("fuzzywuzzy"),
                "fuzzy_backend": __import__("fuzzywuzzy.fuzz", fromlist=["SequenceMatcher"]).SequenceMatcher.__module__,
                "code_hashes": {name: digest_file(Path(__file__).with_name(name)) for name in (
                    "build_utility_bundles.py", "context_contract.py", "utility_metrics.py", "epoch_data_loader.py")},
                "sampler_sha256": digest_file(Path(__file__).parent.parent / "data/ast_repo_pool.py")}
    if not indices:
        raise ValueError("No repositories in training pool")
    store = BundleStore(args.output, manifest)
    oracle = CompletionOracle(args.base_url, store, args.generator_revision, context_config, args.batch_size)
    started = time.monotonic()
    pending = []
    window_started = started

    def flush():
        nonlocal window_started
        if not pending:
            return
        scoring_started = time.monotonic()
        bundles = annotate_many([p[2] for p in pending], oracle)
        for (position, task_key, _), bundle in zip(pending, bundles):
            store.db.execute("INSERT INTO bundles VALUES (?, ?)", (task_key, pack(bundle)))
            store.db.commit()
            print(json.dumps({"phase": "bundle_saved", "position": position + 1,
                              "total": len(indices), "nodes": len(bundle["nodes"]),
                              "complete_diamonds": bundle["design"]["complete_diamonds"],
                              "edges": len(bundle["edges"]), "seconds": time.monotonic()-started}), flush=True)
        seconds = time.monotonic() - window_started
        print(json.dumps({"phase": "annotation_window", "tasks": len(pending),
                          "contexts": sum(len(b["nodes"]) for b in bundles),
                          "window_seconds": seconds,
                          "scoring_and_save_seconds": time.monotonic() - scoring_started,
                          "tasks_per_second": len(pending) / max(seconds, 1e-9)}), flush=True)
        pending.clear()
        window_started = time.monotonic()

    try:
        if store.db.execute("SELECT COUNT(*) FROM bundles").fetchone()[0] < len(indices):
            oracle.verify_served_model()
        occurrences = {}
        for position, index in enumerate(indices):
            # One repository decompressed at a time, including resume.
            uid = loader.repo_rows[index]["repo_uid"]
            occurrence = occurrences.get(uid, 0)
            occurrences[uid] = occurrence + 1
            seed_key = [args.seed, args.epoch, uid]
            if occurrence:
                seed_key.append(occurrence)
            seed = int(fingerprint(seed_key)[:16], 16)
            task_key = fingerprint([uid, seed])
            if store.db.execute("SELECT 1 FROM bundles WHERE task_key=?", (task_key,)).fetchone():
                continue
            payload = loader._load_payloads([index])[0]
            task = sample_repo_task(payload, generator, retriever, random.Random(seed), data_config)
            if task is None:
                raise ValueError(f"No eligible task for prepared repository {uid}")
            pending.append((position, task_key, prepare_annotation(task, generator, context_config, seed)))
            if len(pending) >= args.tasks_per_batch:
                flush()
        flush()
    finally:
        store.close()


if __name__ == "__main__":
    main()
