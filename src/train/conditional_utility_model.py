"""Stage-1 CUR: full UniXcoder + nonlinear, signed set potential."""
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalUtilityModel(nn.Module):
    def __init__(self, encoder, width=64, hidden=128):
        super().__init__()
        self.encoder = encoder
        self.project = nn.Linear(encoder.config.hidden_size, width)
        self.item = nn.Sequential(nn.Linear(width * 4 + 2, hidden), nn.GELU(),
                                  nn.Linear(hidden, width + 1))
        self.interaction = nn.Sequential(nn.Linear(width * 2 + 2, hidden), nn.GELU(),
                                         nn.Linear(hidden, 1))
        self.width, self.hidden = width, hidden

    def encode(self, ids, mask):
        # Fixed encoder dropout for replay equivalence. eval() does NOT freeze
        # weights: all parameters retain autograd and are optimizer parameters.
        self.encoder.eval()
        states = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = (states * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(pooled.float(), dim=-1)

    def potential(self, features, membership, set_cost, candidate_cost):
        q, c = features[0], features[1:]
        pq, pc = self.project(q), self.project(c)
        pq = pq.expand_as(pc)
        item_input = torch.cat([pq, pc, pq * pc, (pq - pc).abs(),
                                (c * q).sum(-1, keepdim=True), candidate_cost[:, None]], -1)
        item = self.item(item_input)
        counts = membership.sum(-1, keepdim=True)
        summary = membership @ item[:, 1:]
        q_sets = self.project(q).expand(len(membership), -1)
        base = self.interaction(torch.cat([q_sets, torch.zeros_like(summary),
                                          torch.zeros_like(counts), torch.zeros_like(counts)], -1))
        joint = self.interaction(torch.cat([q_sets, summary, counts / 10,
                                           set_cost[:, None]], -1))
        return membership @ item[:, 0] + (joint - base).squeeze(-1)


def encode_inputs(task, tokenizer, max_length=512):
    if not 8 <= max_length <= 512:
        raise ValueError("UniXcoder sequence cap must be in [8, 512]")
    mode = tokenizer.convert_tokens_to_ids("<encoder-only>")
    if mode is None or mode == tokenizer.unk_token_id:
        raise ValueError("Tokenizer lacks UniXcoder <encoder-only> token")
    texts = [task["left_context"]] + [c.get("retrieval_text", c["text"]) for c in task["candidates"]]
    if callable(tokenizer):
        try:
            encoded = tokenizer(texts, add_special_tokens=False, padding=False,
                                truncation=False, return_attention_mask=False)
            cores = encoded["input_ids"]
        except (AttributeError, KeyError, TypeError, ValueError):
            cores = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    else:
        cores = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    rows, truncated = [], 0
    for i, core in enumerate(cores):
        truncated += len(core) > max_length - 4
        core = core[-(max_length - 4):] if i == 0 else core[:max_length - 4]
        rows.append([tokenizer.cls_token_id, mode, tokenizer.sep_token_id, *core,
                     tokenizer.sep_token_id])
    length = max(map(len, rows))
    ids = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row)
        mask[i, :len(row)] = 1
    return ids, mask, truncated


def bundle_tensors(bundle, device):
    nodes, n = bundle["nodes"], len(bundle["task"]["candidates"])
    members = torch.zeros((len(nodes), n), device=device)
    for i, node in enumerate(nodes):
        members[i, node["indices"]] = 1
    budget = bundle["context_contract"]["cross_file_tokens"]
    costs = torch.tensor([node["cost"] / budget for node in nodes], device=device)
    candidate_costs = torch.tensor(bundle["candidate_costs"], device=device) / budget
    return members, costs, candidate_costs


def utility_loss(predicted, bundle, scale=1.0, anchor_weight=0.25):
    if scale <= 0 or anchor_weight < 0:
        raise ValueError("Invalid utility scale/anchor weight")
    observed = predicted.new_tensor([node["es"] for node in bundle["nodes"]]) / scale
    groups = {}
    for edge in bundle["edges"]:
        s, t = edge["source"], edge["target"]
        error = predicted[t] - predicted[s] - (observed[t] - observed[s])
        groups.setdefault(edge["group"], []).append(error.square())
    if not groups:
        raise ValueError("Utility bundle has no measured intervention edges")
    gain = torch.stack([torch.stack(values).mean() for values in groups.values()]).mean()
    empty = next(i for i, node in enumerate(bundle["nodes"]) if not node["indices"])
    nonempty = [i for i, node in enumerate(bundle["nodes"]) if node["indices"]]
    anchor = (predicted[nonempty] - (observed[nonempty] - observed[empty])).square().mean()
    return gain + anchor_weight * anchor, {"gain_mse": gain.detach(), "set_mse": anchor.detach()}


def backward_bundle(model, ids, mask, bundle, *, microbatch=8, replay=True,
                    scale=1.0, anchor_weight=0.25, divisor=1):
    """Direct backward or exact representation-gradient replay.

    The graph through all set heads is built once. In replay mode encoder graphs
    live for one sequence microbatch at a time; direct mode retains one whole
    task's encoder graphs. Weights/dropout stay fixed throughout each update.
    """
    if microbatch < 1 or divisor < 1:
        raise ValueError("Positive batch sizes required")
    device = next(model.parameters()).device
    members, costs, item_costs = bundle_tensors(bundle, device)
    if replay:
        with torch.no_grad():
            features = torch.cat([model.encode(ids[i:i+microbatch].to(device),
                                               mask[i:i+microbatch].to(device))
                                  for i in range(0, len(ids), microbatch)])
        features.requires_grad_(True)
    else:
        # Direct mode retains all these encoder graphs until one loss backward.
        # Chunking controls kernel batch shape, NOT total saved-activation memory.
        features = torch.cat([model.encode(ids[i:i+microbatch].to(device),
                                           mask[i:i+microbatch].to(device))
                              for i in range(0, len(ids), microbatch)])
    predicted = model.potential(features, members, costs, item_costs)
    loss, metrics = utility_loss(predicted, bundle, scale, anchor_weight)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite CUR loss")
    (loss / divisor).backward()
    if replay:
        feature_grad = features.grad.detach()
        for i in range(0, len(ids), microbatch):
            value = model.encode(ids[i:i+microbatch].to(device), mask[i:i+microbatch].to(device))
            value.backward(feature_grad[i:i+microbatch])
    return {"loss": float(loss.detach()), **{k: float(v) for k, v in metrics.items()}}
