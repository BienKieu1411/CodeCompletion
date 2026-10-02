"""Greedy conditional marginal-gain selection; STOP is the zero reference."""
import torch

from src.train.conditional_utility_model import encode_inputs
from src.train.context_contract import ContextConfig, ContextRenderer


def score_candidates(model, features, renderer, state, candidates, item_costs):
    device = features.device
    costs = torch.full((len(candidates),), renderer.cross_file(state)[1] /
                       renderer.config.cross_file_tokens, device=device)
    logits = model.gain_logits(features, [tuple(state)] * len(candidates), candidates,
                               costs, item_costs)
    values = model.expected_gain(logits)
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite predicted marginal gain")
    return values


@torch.no_grad()
def select_gain_context(model, task, generator_tokenizer, retriever_tokenizer,
                        config=ContextConfig(), *, threshold=0., encoder_microbatch=64):
    if threshold < 0 or encoder_microbatch < 1:
        raise ValueError("Invalid selection threshold/microbatch")
    model.eval()
    renderer = ContextRenderer(task, generator_tokenizer, config)
    ids, mask, truncated = encode_inputs({**task, "left_context": renderer.left_context},
                                         retriever_tokenizer)
    device = next(model.parameters()).device
    features = torch.cat([model.encode(ids[i:i+encoder_microbatch].to(device),
                                       mask[i:i+encoder_microbatch].to(device))
                          for i in range(0, len(ids), encoder_microbatch)])
    item_costs = torch.tensor(renderer.count_many(renderer.pieces), device=device).float() / config.cross_file_tokens
    selected, trace = (), []
    reason = "snippet_cap"
    while len(selected) < config.max_snippets:
        outside = [i for i in range(len(task["candidates"])) if i not in selected]
        feasible = renderer.feasible_many([(*selected, i) for i in outside])
        choices = [i for i, ok in zip(outside, feasible) if ok]
        if not choices:
            reason = "no_feasible_addition"
            break
        values = score_candidates(model, features, renderer, selected, choices, item_costs)
        best = int(values.argmax())
        gain = float(values[best])
        if gain <= threshold:
            reason = "no_predicted_improvement"
            break
        trace.append({"candidate": choices[best], "predicted_gain": gain})
        selected = renderer.canonical((*selected, choices[best]))
    return {"indices": list(selected), "trace": trace, "stop_reason": reason,
            "cost": renderer.render(selected)[1], "truncated_sequences": truncated}
