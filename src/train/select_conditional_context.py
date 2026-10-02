"""Select add/remove/joint/swap moves using learned utility, without oracle labels."""
import itertools

import torch

from src.train.conditional_utility_model import encode_inputs
from src.train.context_contract import ContextConfig, ContextRenderer


@torch.no_grad()
def select_context(model, task, generator_tokenizer, retriever_tokenizer,
                   config=ContextConfig(), token_penalty=.01, threshold=0.,
                   max_moves=30, pair_shortlist=8, encoder_microbatch=8):
    if min(max_moves, pair_shortlist, encoder_microbatch) < 1 or token_penalty < 0 or threshold < 0:
        raise ValueError("Invalid search limits/cost/threshold")
    model.eval()
    renderer = ContextRenderer(task, generator_tokenizer, config)
    model_task = {**task, "left_context": renderer.left_context}
    ids, mask, truncated = encode_inputs(model_task, retriever_tokenizer)
    device = next(model.parameters()).device
    features = torch.cat([model.encode(ids[i:i+encoder_microbatch].to(device), mask[i:i+encoder_microbatch].to(device))
                          for i in range(0, len(ids), encoder_microbatch)])
    n = len(task["candidates"])
    item_costs = torch.tensor([renderer.count(p) / config.cross_file_tokens for p in renderer.pieces], device=device)
    values = {}

    def score(sets):
        new = list(dict.fromkeys(s for s in sets if s not in values))
        if new:
            membership = torch.zeros((len(new), n), device=device)
            for i, s in enumerate(new):
                membership[i, list(s)] = 1
            cost = torch.tensor([renderer.render(s)[1] / config.cross_file_tokens for s in new], device=device)
            predicted = model.potential(features, membership, cost, item_costs)
            objectives = predicted - token_penalty * cost  # Training scale = 1.0.
            values.update(zip(new, objectives.cpu().tolist()))
        return [values[s] for s in sets]

    current, visited, trace = (), {()}, []
    score([current])
    for _ in range(max_moves):
        outside = [i for i in range(n) if i not in current]
        additions = [renderer.canonical((*current, i)) for i in outside]
        additions = [s for s in additions if renderer.feasible(s)]
        score(additions)
        ranked = sorted(additions, key=lambda s: -values[s])
        shortlist = [next(i for i in s if i not in current) for s in ranked[:pair_shortlist // 2]]
        # Deterministic lexical/path diversity also works at full cardinality,
        # where no single addition is feasible but swaps may still help.
        paths = {task["candidates"][i]["path"] for i in shortlist}
        for i in outside:
            if len(shortlist) >= pair_shortlist:
                break
            if i not in shortlist and task["candidates"][i]["path"] not in paths:
                shortlist.append(i)
                paths.add(task["candidates"][i]["path"])
        for i in outside:
            if len(shortlist) >= pair_shortlist:
                break
            if i not in shortlist:
                shortlist.append(i)
        proposals = list(additions)
        proposals += [tuple(i for i in current if i != old) for old in current]
        proposals += [renderer.canonical((*current, a, b)) for a, b in itertools.combinations(shortlist, 2)]
        proposals += [renderer.canonical((*[i for i in current if i != old], new))
                      for old in current for new in shortlist]
        proposals = list(dict.fromkeys(s for s in proposals if s not in visited and renderer.feasible(s)))
        score(proposals)
        best = max(proposals, key=lambda s: values[s], default=current)
        gain = values[best] - values[current]
        if gain <= threshold:
            return {"indices": list(current), "trace": trace, "stop_reason": "no_predicted_improvement",
                    "search_censored": False, "truncated_sequences": truncated,
                    "cost": renderer.render(current)[1]}
        trace.append({"before": list(current), "after": list(best), "net_gain": gain})
        visited.add(best)
        current = best
    return {"indices": list(current), "trace": trace, "stop_reason": "move_cap",
            "search_censored": True, "truncated_sequences": truncated,
            "cost": renderer.render(current)[1]}
