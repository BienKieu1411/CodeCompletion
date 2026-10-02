"""Online CUR: current-model proposals -> frozen generator -> one encoder update.

Requires an already running vLLM server. Both models remain resident; generation
and backward alternate on the same GPU. No offline annotation artifact required.
"""
import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import torch

from src.data.ast_repo_pool import sample_repo_task
from src.data.ast_training_data import DataConfig
from src.data.audit_completion_data import TOKENIZER_REVISION
from src.data.build_ast_repo_pool import RET_REVISION
from src.data.repo_index import digest_file
from src.train.build_utility_bundles import MODEL, CompletionOracle, annotate_many, intervention_graph
from src.train.benchmark_validation import evaluate as evaluate_benchmark, validate_file as validate_validation_file
from src.train.conditional_utility_model import ConditionalUtilityModel, backward_bundle, encode_inputs
from src.train.context_contract import ContextConfig, ContextRenderer, fingerprint
from src.train.epoch_data_loader import EpochDataLoader
from src.train.train_conditional_utility import atomic_save, restore_rng, rng_state
from src.train.utility_metrics import VERSION as METRIC_VERSION


class OnlineStop(Exception):
    """Cooperative cancellation before the next optimizer mutation."""


@torch.no_grad()
def prepare_online_annotation(model, task, generator, retriever, config, seed, microbatch):
    """Gold-free, weight-dependent conditional ranking plus lexical/random exploration."""
    model.eval()
    renderer = ContextRenderer(task, generator, config)
    ids, mask, _ = encode_inputs({**task, "left_context": renderer.left_context}, retriever)
    device = next(model.parameters()).device
    features = torch.cat([model.encode(ids[i:i+microbatch].to(device), mask[i:i+microbatch].to(device))
                          for i in range(0, len(ids), microbatch)])
    n = len(task["candidates"])
    item_costs = torch.tensor([cost / config.cross_file_tokens
                               for cost in renderer.count_many(renderer.pieces)], device=device)

    def rank(state, choices):
        sets = [renderer.canonical((*state, i)) for i in choices]
        membership = torch.zeros((len(sets), n), device=device)
        for row, indices in enumerate(sets):
            membership[row, list(indices)] = 1
        costs = torch.tensor([cost / config.cross_file_tokens
                              for cost in renderer.cost_many(sets)], device=device)
        # F(S+i)-F(S) has the same ordering as F(S+i); S is common.
        scores = model.potential(features, membership, costs, item_costs).float()
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite online proposal score")
        values = scores.cpu().tolist()
        return [choices[j] for j in sorted(range(len(choices)), key=lambda j: (-values[j], choices[j]))]

    design = {}
    nodes, edges = intervention_graph(renderer, seed, audit=design, rank_candidates=rank)
    if not edges:
        raise ValueError(f"Task has no feasible intervention: {task['task_id']}")
    # Do not retain encoder features/closures/graphs while waiting for generator.
    return {"task": task, "renderer": renderer, "nodes": nodes, "edges": edges,
            "design": design, "prompts": [renderer.render(n["indices"])[0] for n in nodes], "seed": seed}


def online_update(model, tasks, generator, retriever, oracle, optimizer, *, config=ContextConfig(),
                  microbatch=64, replay=False, max_grad_norm=2., precision_context=nullcontext,
                  check_continue=lambda: None):
    """All proposals use theta_t. No optimizer step until every batch label is ready."""
    if not tasks:
        raise ValueError("Online update needs at least one task")
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    prepared = []
    for task, seed in tasks:
        check_continue()
        with precision_context():
            prepared.append(prepare_online_annotation(model, task, generator, retriever,
                                                      config, seed, microbatch))
    proposal_seconds = time.monotonic() - start
    scoring_start = time.monotonic()
    check_continue()
    bundles = annotate_many(prepared, oracle)
    scoring_seconds = time.monotonic() - scoring_start
    print(json.dumps({"phase": "online_scored", "tasks": len(tasks),
                      "contexts": sum(len(b["nodes"]) for b in bundles),
                      "generator_seconds": scoring_seconds}), flush=True)
    update_start = time.monotonic()
    model.train()
    totals = {"loss": 0., "gain_mse": 0., "set_mse": 0.}
    for bundle in bundles:
        check_continue()
        ids, mask, _ = encode_inputs(bundle["task"], retriever)
        with precision_context():
            metrics = backward_bundle(model, ids, mask, bundle, microbatch=microbatch,
                                      replay=replay, divisor=len(bundles))
        for key, value in metrics.items():
            totals[key] += value / len(bundles)
    check_continue()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
        raise FloatingPointError("Nonfinite parameter after online optimizer update")
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {**totals, "grad_norm": float(norm), "tasks": len(tasks),
            "contexts": sum(len(b["nodes"]) for b in bundles),
            "proposal_seconds": proposal_seconds, "generator_seconds": scoring_seconds,
            "update_seconds": time.monotonic() - update_start}


def should_validate_epoch(completed_epoch, interval):
    """Always keep the pre-training baseline; zero disables epoch evaluations."""
    return completed_epoch == 0 or (interval > 0 and completed_epoch % interval == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--valid", type=Path,
                        help="Prepared benchmark valid.parquet; evaluation only, never used for gradients")
    parser.add_argument("--valid-every-epochs", type=int, default=1,
                        help="0 disables post-epoch validation; initial baseline is retained")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--algorithm", choices=["potential", "gain24"], default="potential")
    parser.add_argument("--prepare-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5, help="Fresh 2000-task epochs, not offline reuse passes")
    parser.add_argument("--batch-size", type=int, default=16, help="Tasks per optimizer step")
    parser.add_argument("--generator-batch-size", type=int, default=128, help="HTTP prompts, not active vLLM sequences")
    parser.add_argument("--encoder-microbatch", type=int, default=64)
    parser.add_argument("--candidate-pool-size", type=int, default=100,
                        help="Top BM25 cross-file chunks passed to UniXcoder; separate from K")
    parser.add_argument("--encoder-backward", choices=["direct", "replay"], default="direct")
    parser.add_argument("--encoder-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--schedule", choices=["linear", "warmup_constant"], default="linear")
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=2.)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-minutes", type=float, default=645)
    parser.add_argument("--deadline-unix", type=float, help="Shared job deadline, including server startup")
    parser.add_argument("--stop-file", type=Path, help="Supervisor creates this to request a safe stop")
    parser.add_argument("--checkpoint-volume", help="Commit this mounted Modal Volume after each atomic save")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()
    if args.valid_every_epochs < 0:
        parser.error("valid-every-epochs must be nonnegative")
    if min(args.epochs, args.batch_size, args.generator_batch_size, args.encoder_microbatch,
           args.candidate_pool_size, args.prepare_workers, args.warmup_updates,
           args.encoder_lr, args.head_lr, args.max_grad_norm, args.save_every, args.max_minutes) <= 0:
        parser.error("Positive training limits required")
    device = torch.device(args.device)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("bf16 requires a supporting CUDA GPU; use --precision fp32 on CPU")
    precision_context = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.precision == "bf16" else nullcontext
    # Start the local clock before model/data loading, not after annotation.
    deadline = time.monotonic() + args.max_minutes * 60
    if args.deadline_unix is not None:
        deadline = min(deadline, time.monotonic() + args.deadline_unix - time.time())

    def check_continue():
        if time.monotonic() >= deadline or (args.stop_file is not None and args.stop_file.exists()):
            raise OnlineStop()

    check_continue()
    from transformers import AutoModel, AutoTokenizer
    import pyarrow.parquet as pq
    if args.algorithm == "gain24":
        import pyarrow as pa
        # Eight task workers share bounded native thread pools; no 8x8 nesting.
        pa.set_cpu_count(2)
        pa.set_io_thread_count(2)
        torch.set_num_threads(2)
    metadata = pq.ParquetFile(args.pool).schema_arrow.metadata or {}
    if b"preparation_contract" not in metadata:
        raise ValueError("Expected prepared repository-pool train.parquet, not old task-row data")
    prepared_contract = json.loads(metadata[b"preparation_contract"])
    if (prepared_contract["generator_tokenizer_revision"] != TOKENIZER_REVISION
            or prepared_contract["retriever_tokenizer_revision"] != RET_REVISION):
        raise ValueError("Prepared data tokenizer revision mismatch")
    # Repository artifacts retain all chunks. Override online mining only; do
    # not rewrite preparation metadata or regenerate targets/chunks on disk.
    data_config = replace(DataConfig(**prepared_contract["config"]), pool_size=args.candidate_pool_size)
    config = ContextConfig()
    if args.valid is not None:
        validate_validation_file(args.valid)
    generator = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION)
    retriever = AutoTokenizer.from_pretrained("microsoft/unixcoder-base", revision=RET_REVISION)
    loader = EpochDataLoader(args.pool, generator, retriever, data_config)
    contract = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in {"output", "resume", "max_minutes", "device", "base_url", "pool",
                             "valid", "deadline_unix", "stop_file", "checkpoint_volume"}}
    contract.update(version="cur_online_v1", pool_sha256=digest_file(args.pool),
                    model="microsoft/unixcoder-base", revision=RET_REVISION,
                    generator=MODEL, generator_revision=TOKENIZER_REVISION,
                    context=asdict(config), data=asdict(data_config), metric=METRIC_VERSION,
                    epoch_size=loader.epoch_size, python_quota=loader.python_quota,
                    scale=1., anchor_weight=.25, encoder_dropout=False,
                    weight_decay=.01, warmup_fraction=.05, persistent_label_cache=False,
                    fuzzywuzzy_version=importlib.metadata.version("fuzzywuzzy"),
                    fuzzy_backend=__import__("fuzzywuzzy.fuzz", fromlist=["SequenceMatcher"]).SequenceMatcher.__module__,
                    code_hashes={str(p.relative_to(Path(__file__).parents[1])): digest_file(p)
                                 for folder in (Path(__file__).parent, Path(__file__).parents[1] / "data")
                                 for p in sorted(folder.glob("*.py"))})
    if args.algorithm == "gain24":
        contract.update(version="conditional_gain24_softsign_twohot_v2", scale=None, anchor_weight=0.,
                        utility="mean_target_logprob_separate_continuation_ids_no_eos",
                        head_width=256, histogram_bins=257, head_precision="fp32",
                        target_encoding="softsign_twohot", gain_transform="g/(1+abs(g))",
                        gain_metric_domain="softsign_score", support_calibration="fixed_unit_interval",
                        max_contexts=24, state_goals=[0, 3, 6, 9], probes_per_state=5,
                        stop_threshold=0., label_cache=False)
    contract["valid_sha256"] = digest_file(args.valid) if args.valid is not None else None
    contract_id = fingerprint(contract)
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "latest.pt"
    if latest.exists() and not args.resume:
        raise ValueError("Output already contains latest.pt; specify --resume or a fresh directory")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.algorithm == "gain24":
        from src.train.distributional_gain_model import ConditionalGainModel
        from src.train.online_gain import online_gain_update
        from src.train.likelihood_oracle import LikelihoodOracle
        from src.train.select_gain_context import select_gain_context
        model = ConditionalGainModel(AutoModel.from_pretrained(contract["model"], revision=RET_REVISION),
                                     bins=contract["histogram_bins"],
                                     target_encoding=contract["target_encoding"]).to(device)
    else:
        model = ConditionalUtilityModel(AutoModel.from_pretrained(contract["model"], revision=RET_REVISION)).to(device)
    model.encoder.gradient_checkpointing_disable()
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")],
         "lr": args.head_lr}], weight_decay=.01)
    total_steps = args.epochs * math.ceil(loader.epoch_size / args.batch_size)
    warmup = args.warmup_updates if args.schedule == "warmup_constant" else max(1, int(total_steps * .05))
    def learning_rate_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        if args.schedule == "warmup_constant":
            return 1.
        # Preserve the existing linear decay for the legacy potential control.
        return max(0., (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    state = {"epoch": 1, "cursor": 0, "updates": 0}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        stored_contract = payload.get("contract")
        if not isinstance(stored_contract, dict):
            raise ValueError("Online checkpoint is missing its training contract")
        # A stopped run may be continued with more epochs, but every other
        # data/model/optimization setting must remain byte-for-byte equivalent.
        mismatches = {}
        for key in set(stored_contract) | set(contract):
            if key == "epochs":
                continue
            if key == "code_hashes":
                old_hashes = stored_contract.get(key) or {}
                new_hashes = contract.get(key) or {}
                changed_files = {path for path in set(old_hashes) | set(new_hashes)
                                 if old_hashes.get(path) != new_hashes.get(path)}
                # The resume compatibility guard itself lives in this file,
                # so its hash necessarily changes when migrating a checkpoint.
                # Do not make that migration disable all other integrity checks.
                if changed_files - {"train/train_online_utility.py", "train/modal_online_utility.py",
                                    "train/benchmark_validation.py"}:
                    mismatches[key] = sorted(changed_files)
                continue
            if stored_contract.get(key) != contract.get(key):
                mismatches[key] = (stored_contract.get(key), contract.get(key))
        stored_epochs = stored_contract.get("epochs")
        if stored_epochs is None:
            raise ValueError("Online checkpoint contract has no saved epoch count")
        if args.epochs < int(stored_epochs):
            mismatches["epochs"] = (stored_epochs, args.epochs)
        if mismatches:
            raise ValueError(f"Online resume contract mismatch; only increasing epochs is allowed: {mismatches}")
        if payload.get("contract_id") != fingerprint(stored_contract):
            raise ValueError("Online checkpoint contract id is invalid")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        state = payload["state"]
        restore_rng(payload["rng"])
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    checkpoint_volume = None
    if args.checkpoint_volume:
        import modal
        checkpoint_volume = modal.Volume.from_name(args.checkpoint_volume)

    def save(path=latest):
        atomic_save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "rng": rng_state(), "state": dict(state),
                            "contract": contract, "contract_id": contract_id})
        if checkpoint_volume is not None:
            checkpoint_volume.commit()

    def status(reason):
        (args.output / "train_status.json").write_text(json.dumps({"status": reason, **state}) + "\n")

    def validate_benchmark(completed_epoch):
        if args.valid is None:
            print(json.dumps({"phase": "validation_disabled", "n": 0}), flush=True)
            return
        print(json.dumps({"phase": "validation_start", "path": str(args.valid)}), flush=True)
        with precision_context():
            report = evaluate_benchmark(
                model, args.valid, generator, retriever, oracle, config=config,
                encoder_microbatch=args.encoder_microbatch, check_continue=check_continue,
                selector=select_gain_context if args.algorithm == "gain24" else None)
        serializable = {key: value for key, value in report.items() if key != "details"}
        print(json.dumps({"phase": "validation", **serializable}), flush=True)
        with (args.output / "validation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": completed_epoch, **report}, ensure_ascii=False) + "\n")
        mean_es = sum(row["es"] for row in report["benchmarks"].values()) / len(report["benchmarks"])
        if mean_es > state.get("best_valid_es", -1.):
            state["best_valid_es"] = mean_es
            state["best_valid_epoch"] = completed_epoch
            save(args.output / "best.pt")
        state["validated_epoch"] = completed_epoch
        save()

    # None means no persistent response/bundle cache. Only exact duplicates
    # within the current HTTP stream share one generation.
    oracle_type = LikelihoodOracle if args.algorithm == "gain24" else CompletionOracle
    oracle = oracle_type(args.base_url, None, TOKENIZER_REVISION, config, args.generator_batch_size,
                         request_timeout=180, check_continue=check_continue)
    save()
    prepare_workers = ThreadPoolExecutor(max_workers=args.prepare_workers, thread_name_prefix="prepare")
    ema_rate = None  # Runtime telemetry is not part of deterministic resume state.

    def prepare_task(item):
        repo, seed, position = item
        check_continue()
        task = sample_repo_task(repo, generator, retriever, random.Random(seed), data_config)
        if task is None:
            raise ValueError(f"No eligible task at epoch {state['epoch']} position {position}")
        return task, seed

    try:
        oracle.verify_served_model()
        if args.algorithm == "gain24":
            probe = oracle.score(["def add(a, b):\n    "], ["return a + b"], generator)
            print(json.dumps({"phase": "likelihood_probe", "mean_logprob": probe[0]}), flush=True)
            # Include epoch zero and repeat an interrupted end-of-epoch validation on resume.
            previous_epoch = state["epoch"] - 1
            if (should_validate_epoch(previous_epoch, args.valid_every_epochs)
                    and state.get("validated_epoch", -1) < previous_epoch):
                validate_benchmark(previous_epoch)
        while state["epoch"] <= args.epochs:
            indices = loader.select_epoch_indices(state["epoch"], args.seed)
            print(json.dumps({"phase": "online_epoch", **state, "rows": len(indices),
                              "encoder_lr_peak": args.encoder_lr, "head_lr_peak": args.head_lr,
                              "batch_size": args.batch_size, "candidate_pool_size": data_config.pool_size,
                              "persistent_label_cache": False}), flush=True)
            while state["cursor"] < len(indices):
                check_continue()
                start = time.monotonic()
                group = indices[state["cursor"]:state["cursor"] + args.batch_size]
                # Read/decompress each required row group once per batch, not once per sample.
                repositories = loader._load_payloads(group)
                preparation = []
                for offset, repo in enumerate(repositories):
                    position = state["cursor"] + offset
                    seed = int(fingerprint([args.seed, state["epoch"], position])[:16], 16)
                    preparation.append((repo, seed, position))
                tasks = list(prepare_workers.map(prepare_task, preparation))
                del repositories, preparation
                preparation_seconds = time.monotonic() - start
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                print(json.dumps({"phase": "online_batch", **state, "tasks": len(tasks),
                                  "candidate_counts": [len(task["candidates"]) for task, _ in tasks]}), flush=True)
                update_function = online_gain_update if args.algorithm == "gain24" else online_update
                metrics = update_function(model, tasks, generator, retriever, oracle, optimizer,
                                        config=config, microbatch=args.encoder_microbatch,
                                        replay=args.encoder_backward == "replay", max_grad_norm=args.max_grad_norm,
                                        precision_context=precision_context, check_continue=check_continue)
                scheduler.step()
                state["updates"] += 1
                state["cursor"] += len(group)
                if state["updates"] % args.save_every == 0 or state["updates"] == 1:
                    save()
                seconds = time.monotonic() - start
                rate = len(group) / max(seconds, 1e-9)
                ema_rate = rate if ema_rate is None else .2 * rate + .8 * ema_rate
                remaining_tasks = (args.epochs - state["epoch"]) * loader.epoch_size + len(indices) - state["cursor"]
                print(json.dumps({**state, **metrics, "batch_seconds": seconds,
                                  "preparation_seconds": preparation_seconds,
                                  "ema_tasks_per_second": ema_rate,
                                  "train_eta_hours_excluding_validation": remaining_tasks / ema_rate / 3600,
                                  "epoch_eta_minutes": (len(indices)-state["cursor"]) / ema_rate / 60,
                                  "estimated_tasks_before_deadline": int(max(0., deadline-time.monotonic()) * ema_rate),
                                  "tasks_per_second": len(group) / max(seconds, 1e-9),
                                  "next_encoder_lr": optimizer.param_groups[0]["lr"],
                                  "peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                                         if device.type == "cuda" else None),
                                  "remaining_minutes": max(0., (deadline-time.monotonic())/60)}), flush=True)
            completed_epoch = state["epoch"]
            state["epoch"] += 1
            state["cursor"] = 0
            save()
            if should_validate_epoch(completed_epoch, args.valid_every_epochs):
                validate_benchmark(completed_epoch)
        status("complete")
    except OnlineStop:
        optimizer.zero_grad(set_to_none=True)
        save()  # Current batch, if unfinished, has not mutated weights/cursor.
        status("stopped")
        print("Safe stop: saved last complete optimizer boundary", flush=True)
    except BaseException as error:
        # An optimizer step may have failed partway: retain the previous durable
        # boundary, never overwrite it with possibly corrupted weights.
        (args.output / "failure.json").write_text(json.dumps({"error_type": type(error).__name__,
                                                              "resume_from": str(latest)}, indent=2) + "\n")
        status("failed")
        raise
    finally:
        prepare_workers.shutdown(wait=True, cancel_futures=True)
        oracle.session.close()
        if checkpoint_volume is not None:
            checkpoint_volume.commit()


if __name__ == "__main__":
    main()
