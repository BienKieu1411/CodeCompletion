from __future__ import annotations

from importlib import import_module

__all__ = ["GraphTraversalPolicy", "SemanticStateQuantizer", "RetrievalRewardModel"]

_LAZY_IMPORTS = {
    "GraphTraversalPolicy": ("graphcoder_rl.rl.graph_traversal_policy", "GraphTraversalPolicy"),
    "SemanticStateQuantizer": ("graphcoder_rl.rl.semantic_state_quantizer", "SemanticStateQuantizer"),
    "RetrievalRewardModel": ("graphcoder_rl.rl.retrieval_reward_model", "RetrievalRewardModel"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    mod_name, attr = _LAZY_IMPORTS[name]
    return getattr(import_module(mod_name), attr)
