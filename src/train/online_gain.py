"""24-context online conditional gain training, one update per task batch."""
from contextlib import nullcontext
import json
import time

import torch

from src.train.conditional_probes import matched_probe_graph
from src.train.conditional_utility_model import encode_inputs
from src.train.context_contract import ContextConfig, ContextRenderer
from src.train.distributional_gain_model import histogram_loss
from src.train.select_gain_context import score_candidates


@torch.no_grad()
def prepare_gain_annotation(model, task, generator, retriever, config, seed, microbatch):
    model.eval()
    renderer = ContextRenderer(task, generator, config)
    model_task = {**task, "left_context": renderer.left_context}
    ids, mask, truncated = encode_inputs(model_task, retriever)
    device = next(model.parameters()).device
    features = torch.cat([model.encode(ids[i:i+microbatch].to(device), mask[i:i+microbatch].to(device))
                          for i in range(0, len(ids), microbatch)])
    item_costs = torch.tensor(renderer.count_many(renderer.pieces), device=device).float() / config.cross_file_tokens

    def rank(state, choices):
        values = score_candidates(model, features, renderer, state, choices, item_costs).cpu().tolist()
        return [choices[i] for i in sorted(range(len(choices)), key=lambda i: (-values[i], choices[i]))]

    nodes, edges, audit = matched_probe_graph(renderer, seed, rank_candidates=rank)
    if not edges:
        raise ValueError("No feasible conditional gain labels for task " + task["task_id"])
    return {"task": model_task, "renderer": renderer, "nodes": nodes, "edges": edges,
            "design": audit, "truncated_sequences": truncated,
            "prompts": [renderer.render(node["indices"])[0] for node in nodes]}


def compact_bundle(bundle, retriever):
    """Re-encode only the union of observed state/probe sequences with autograd."""
    used = sorted({i for node in bundle["nodes"] for i in node["indices"]})
    remap = {old: new for new, old in enumerate(used)}
    task = {**bundle["task"], "candidates": [bundle["task"]["candidates"][i] for i in used]}
    ids, mask, _ = encode_inputs(task, retriever)
    states = [tuple(remap[i] for i in bundle["nodes"][edge["source"]]["indices"])
              for edge in bundle["edges"]]
    candidates = [remap[edge["candidate"]] for edge in bundle["edges"]]
    budget = bundle["renderer"].config.cross_file_tokens
    state_costs = [bundle["nodes"][e["source"]]["cost"] / budget for e in bundle["edges"]]
    item_costs = [bundle["renderer"].count(bundle["renderer"].pieces[i]) / budget for i in used]
    return ids, mask, states, candidates, state_costs, item_costs


def gradient_norm(parameters):
    terms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    if not terms:
        return 0.
    return float(torch.stack(terms).sum().sqrt())


def online_gain_update(model, tasks, generator, retriever, oracle, optimizer, *,
                       config=ContextConfig(), microbatch=64, replay=False,
                       max_grad_norm=2., precision_context=nullcontext,
                       check_continue=lambda: None):
    if not tasks:
        raise ValueError("Empty training batch")
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    bundles = []
    for task, seed in tasks:
        check_continue()
        with precision_context():
            bundles.append(prepare_gain_annotation(model, task, generator, retriever, config, seed, microbatch))
    proposal_seconds = time.monotonic() - started
    scoring_start = time.monotonic()
    prompts = [p for bundle in bundles for p in bundle["prompts"]]
    targets = [bundle["task"]["target_code"] for bundle in bundles for _ in bundle["prompts"]]
    scores = oracle.score(prompts, targets, generator)
    if len(scores) != len(prompts):
        raise ValueError("Missing teacher scores")
    offset, all_gains = 0, []
    for bundle in bundles:
        values = scores[offset:offset+len(bundle["nodes"])]
        offset += len(values)
        bundle["gains"] = [values[e["target"]] - values[e["source"]] for e in bundle["edges"]]
        all_gains.extend(bundle["gains"])
    gains_cpu = torch.tensor(all_gains, dtype=torch.float32)
    if not torch.isfinite(gains_cpu).all():
        raise FloatingPointError("Nonfinite measured gain")
    if not bool(model.support_calibrated):
        if float(gains_cpu.abs().max()) == 0.:
            raise ValueError("All measured gains are zero; audit teacher supervision before training")
        model.configure_support(gains_cpu, radius=max(2., 3. * float(gains_cpu.abs().max())))
    scoring_seconds = time.monotonic() - scoring_start
    print(json.dumps({"phase": "gain_scored", "tasks": len(tasks), "contexts": len(prompts),
                      "edges": len(all_gains), "gain_min": float(gains_cpu.min()),
                      "gain_max": float(gains_cpu.max()), "gain_std": float(gains_cpu.std(unbiased=False)),
                      "gain_positive": int((gains_cpu > .001).sum()),
                      "gain_negative": int((gains_cpu < -.001).sum()),
                      "support_radius": float(model.support_radius), "generator_seconds": scoring_seconds}), flush=True)
    device = next(model.parameters()).device
    model.train()
    update_start, totals = time.monotonic(), {}
    for bundle in bundles:
        check_continue()
        ids, mask, states, candidates, costs, item_costs = compact_bundle(bundle, retriever)
        if replay:
            with torch.no_grad(), precision_context():
                features = torch.cat([model.encode(ids[i:i+microbatch].to(device), mask[i:i+microbatch].to(device))
                                      for i in range(0, len(ids), microbatch)])
            features.requires_grad_(True)
        else:
            with precision_context():
                features = torch.cat([model.encode(ids[i:i+microbatch].to(device), mask[i:i+microbatch].to(device))
                                      for i in range(0, len(ids), microbatch)])
        logits = model.gain_logits(features, states, candidates,
                                   torch.tensor(costs, device=device), torch.tensor(item_costs, device=device))
        loss, metrics = histogram_loss(logits, torch.tensor(bundle["gains"], device=device), model)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite histogram loss")
        (loss / len(bundles)).backward()
        if replay:
            feature_grad = features.grad.detach()
            for i in range(0, len(ids), microbatch):
                check_continue()
                with precision_context():
                    value = model.encode(ids[i:i+microbatch].to(device), mask[i:i+microbatch].to(device))
                value.backward(feature_grad[i:i+microbatch])
        for key, value in {"loss": loss.detach(), **metrics}.items():
            totals[key] = totals.get(key, 0.) + float(value) / len(bundles)
    check_continue()
    encoder_norm = gradient_norm(model.encoder.parameters())
    head_norm = gradient_norm(p for name, p in model.named_parameters() if not name.startswith("encoder."))
    if not (0 < encoder_norm < float("inf") and 0 < head_norm < float("inf")):
        raise FloatingPointError("Disconnected/zero/nonfinite encoder or head gradient; refusing optimizer step")
    missing = [name for name, p in model.encoder.named_parameters()
               if p.requires_grad and p.grad is None and not name.startswith("pooler.")]
    if missing:
        raise RuntimeError("Encoder parameters disconnected from loss: " + str(missing[:8]))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    # A small real encoder matrix, not just a classifier bias, proves updates survive FP32 storage.
    probes = [(name, p) for name, p in model.encoder.named_parameters()
              if p.grad is not None and p.ndim == 2 and "embedding" not in name]
    if not probes:
        probes = [(name, p) for name, p in model.encoder.named_parameters() if p.grad is not None]
    probe_name, probe = probes[0]
    previous = probe.detach().clone()
    optimizer.step()
    delta = float((probe.detach().float()-previous.float()).norm())
    relative_delta = delta / max(float(previous.float().norm()), 1e-12)
    if delta == 0 or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Encoder update vanished or nonfinite weights after optimizer step")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {**totals, "grad_norm": float(norm), "encoder_grad_norm": encoder_norm,
            "head_grad_norm": head_norm, "encoder_probe": probe_name,
            "encoder_update_relative": relative_delta, "low_gradient_warning": encoder_norm < 1e-8,
            "clip_scale": min(1., max_grad_norm / max(float(norm), 1e-12)),
            "tasks": len(tasks), "contexts": len(prompts), "gain_labels": len(all_gains),
            "gain_std": float(gains_cpu.std(unbiased=False)),
            "proposal_seconds": proposal_seconds, "generator_seconds": scoring_seconds,
            "update_seconds": time.monotonic()-update_start}
