"""Full UniXcoder fine-tuning on cached CUR bundles, with boundary-safe resume."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time

import torch

from src.data.build_ast_repo_pool import RET_REVISION
from src.data.repo_index import digest_file
from src.train.conditional_utility_model import (
    ConditionalUtilityModel, backward_bundle, bundle_tensors, encode_inputs, utility_loss)
from src.train.context_contract import fingerprint
from src.train.epoch_data_loader import unpack


class Bundles:
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        self.manifest = json.loads(self.db.execute("SELECT value FROM metadata WHERE id=1").fetchone()[0])
        self.keys = [r[0] for r in self.db.execute("SELECT task_key FROM bundles ORDER BY task_key")]
        if not self.keys or len(self.keys) != len(self.manifest["selected_repos"]):
            raise ValueError("Bundle annotations are empty/incomplete; finish annotation before training")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return unpack(self.db.execute("SELECT payload FROM bundles WHERE task_key=?",
                                      (self.keys[index],)).fetchone()[0])


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(value):
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in value["cuda"]])


@torch.no_grad()
def validate(model, data, tokenizer, microbatch, precision_context, deadline=math.inf):
    model.eval()
    totals = {"loss": 0., "gain_mse": 0., "set_mse": 0.}
    device = next(model.parameters()).device
    for index in range(len(data)):
        if time.monotonic() >= deadline:
            return None
        bundle = data[index]
        ids, mask, _ = encode_inputs(bundle["task"], tokenizer)
        with precision_context():
            features = torch.cat([model.encode(ids[i:i+microbatch].to(device), mask[i:i+microbatch].to(device))
                                  for i in range(0, len(ids), microbatch)])
            prediction = model.potential(features, *bundle_tensors(bundle, device))
            loss, metrics = utility_loss(prediction, bundle)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        totals["loss"] += float(loss)
        for k, v in metrics.items():
            totals[k] += float(v)
    return {k: v / len(data) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bundles", nargs="+", type=Path, required=True,
                        help="One bundle artifact per sampled data epoch, in training order")
    parser.add_argument("--valid-bundles", type=Path,
                        help="Optional held-out bundle; omitted because repository validation is disabled")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--passes", type=int, default=3, help="Supervised reuse passes per data epoch")
    parser.add_argument("--encoder-microbatch", type=int, default=8)
    parser.add_argument("--encoder-backward", choices=["replay", "direct"], default="replay",
                        help="direct saves forward work but retains one whole query's encoder graphs")
    parser.add_argument("--queries-per-update", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=25, help="Durable optimizer-boundary checkpoints")
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=2.)
    parser.add_argument("--max-minutes", type=float, default=690)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()
    if min(args.passes, args.encoder_microbatch, args.queries_per_update, args.max_minutes, args.save_every,
           args.encoder_lr, args.head_lr, args.max_grad_norm) <= 0:
        parser.error("Positive training limits required")
    device = torch.device(args.device)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("bf16 requires a supporting CUDA GPU; use fp32 on CPU/T4")
    precision_context = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.precision == "bf16" else nullcontext
    train = [Bundles(path) for path in args.train_bundles]
    valid = Bundles(args.valid_bundles) if args.valid_bundles else None
    if any(d.manifest["split"] != "train" for d in train):
        raise ValueError("Training bundle split mismatch")
    shared_fields = ("pool_sha256", "version", "metric", "generator", "generator_revision", "context", "retriever_revision",
                     "decoding", "fuzzywuzzy_version", "fuzzy_backend", "code_hashes", "sampler_sha256", "data")
    # Repository validation may be disabled, but annotation contracts must
    # still agree across data epochs (especially old 8 versus new 20 recipes).
    for dataset in train[1:]:
        for field in shared_fields:
            if dataset.manifest[field] != train[0].manifest[field]:
                raise ValueError(f"Training epoch annotation contract mismatch: {field}")
    if valid is not None:
        if valid.manifest["split"] != "valid":
            raise ValueError("Validation bundle split mismatch")
        for dataset in train:
            for field in shared_fields:
                if dataset.manifest[field] != valid.manifest[field]:
                    raise ValueError(f"Train/valid annotation contract mismatch: {field}")
            if set(dataset.manifest["selected_repos"]) & set(valid.manifest["selected_repos"]):
                raise ValueError("Train/validation repository overlap")
    contract = {k: v for k, v in vars(args).items()
                if k not in {"output", "resume", "train_bundles", "valid_bundles", "max_minutes", "device"}}
    contract.update(train_hashes=[digest_file(p) for p in args.train_bundles],
                    valid_hash=digest_file(args.valid_bundles) if valid is not None else None,
                    model="microsoft/unixcoder-base", revision=RET_REVISION,
                    architecture="cur_global_v1", scale=1.0, anchor_weight=0.25,
                    encoder_dropout=False, width=64, hidden=128,
                    code_hashes={name: digest_file(Path(__file__).with_name(name)) for name in (
                        "train_conditional_utility.py", "conditional_utility_model.py")})
    contract_id = fingerprint(contract)
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "latest.pt"
    if latest.exists() and not args.resume:
        raise ValueError("Output already has latest.pt; specify --resume or use a new directory")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(contract["model"], revision=RET_REVISION)
    model = ConditionalUtilityModel(AutoModel.from_pretrained(contract["model"], revision=RET_REVISION)).to(device)
    encoder_params = list(model.encoder.parameters())
    heads = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": args.encoder_lr},
                                  {"params": heads, "lr": args.head_lr}], weight_decay=.01)
    total_steps = args.passes * sum(math.ceil(len(d) / args.queries_per_update) for d in train)
    warmup = max(1, int(total_steps * .05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
        (step + 1) / warmup if step < warmup else
        max(0., (total_steps - step) / max(1, total_steps - warmup))))
    state = {"round": 0, "pass": 0, "cursor": 0, "updates": 0, "best_valid_loss": math.inf}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["contract_id"] != contract_id:
            raise ValueError("Resume contract differs: data/model/optimizer/precision must stay fixed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        state = checkpoint["state"]
        restore_rng(checkpoint["rng"])
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    def save(path=latest):
        atomic_save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "state": dict(state), "rng": rng_state(),
                           "contract": contract, "contract_id": contract_id,
                           "annotation_contract": valid.manifest if valid is not None
                           else {"split": "disabled"}})

    save()  # A valid, durable recovery point before the first update.
    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    try:
        while state["round"] < len(train):
            dataset = train[state["round"]]
            order = list(range(len(dataset)))
            random.Random(args.seed + state["round"] * 1_000_003 + state["pass"]).shuffle(order)
            while state["cursor"] < len(order):
                if time.monotonic() >= deadline:
                    save()
                    print("Time budget reached; latest.pt contains the last complete update", flush=True)
                    return
                group = order[state["cursor"]:state["cursor"] + args.queries_per_update]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                totals = {"loss": 0., "gain_mse": 0., "set_mse": 0.}
                truncated = 0
                batch_start = time.monotonic()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                for index in group:
                    bundle = dataset[index]
                    ids, mask, count = encode_inputs(bundle["task"], tokenizer)
                    truncated += count
                    with precision_context():
                        metrics = backward_bundle(model, ids, mask, bundle,
                                                  microbatch=args.encoder_microbatch, divisor=len(group),
                                                  replay=args.encoder_backward == "replay")
                    for k, v in metrics.items():
                        totals[k] += v / len(group)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
                    raise FloatingPointError("Nonfinite parameter after optimizer update")
                scheduler.step()
                state["cursor"] += len(group)
                state["updates"] += 1
                if state["updates"] % args.save_every == 0:
                    save()  # Boundary snapshot only, never a partial accumulated update.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic() - batch_start
                print(json.dumps({**state, **totals, "grad_norm": float(norm),
                                  "truncated_sequences": truncated,
                                  "encoder_backward": args.encoder_backward,
                                  "update_seconds": elapsed,
                                  "queries_per_second": len(group) / max(elapsed, 1e-9),
                                  "peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                                         if device.type == "cuda" else None),
                                  "remaining_minutes": max(0., (deadline-time.monotonic())/60)}), flush=True)
            if time.monotonic() >= deadline:
                save()
                return  # Validation and pass transition run on resume.
            # Save trained weights before validation: a validation failure does
            # not force repeating the supervised pass.
            save()
            improved = False
            if valid is not None:
                metrics = validate(model, valid, tokenizer, args.encoder_microbatch, precision_context, deadline)
                if metrics is None:
                    return
                improved = metrics["loss"] < state["best_valid_loss"]
                if improved:
                    state["best_valid_loss"] = metrics["loss"]
                print(json.dumps({"phase": "validation", "n": len(valid), **metrics}), flush=True)
            else:
                print(json.dumps({"phase": "validation_disabled", "n": 0}), flush=True)
            state["cursor"] = 0
            state["pass"] += 1
            if state["pass"] == args.passes:
                state["pass"] = 0
                state["round"] += 1
            save()
            if improved:
                save(args.output / "best.pt")
    except BaseException as error:
        # Do not overwrite latest with half a gradient update or corrupted
        # optimizer state. Resume replays updates since the last durable save
        # (at most save_every updates, or one with --save-every 1).
        (args.output / "failure.json").write_text(json.dumps({"error_type": type(error).__name__,
                                                              "resume_from": str(latest)}, indent=2) + "\n")
        raise
    finally:
        for data in [*train, *([valid] if valid is not None else [])]:
            data.db.close()


if __name__ == "__main__":
    main()
