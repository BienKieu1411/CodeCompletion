from __future__ import annotations

from importlib import import_module

__all__ = [
    "DatasetLoader",
    "LeftContextAnchorExtractor",
    "MultiHopGraphRetriever",
    "GraphTraversalPolicy",
    "SemanticStateQuantizer",
    "RetrievalRewardModel",
]

_LAZY_IMPORTS = {
    "DatasetLoader": ("graphcoder_rl.data.repository_dataset_loader", "DatasetLoader"),
    "LeftContextAnchorExtractor": ("graphcoder_rl.data.left_context_anchor_extractor", "LeftContextAnchorExtractor"),
    "MultiHopGraphRetriever": ("graphcoder_rl.retrieval.multi_hop_graph_retriever", "MultiHopGraphRetriever"),
    "GraphTraversalPolicy": ("graphcoder_rl.rl.graph_traversal_policy", "GraphTraversalPolicy"),
    "SemanticStateQuantizer": ("graphcoder_rl.rl.semantic_state_quantizer", "SemanticStateQuantizer"),
    "RetrievalRewardModel": ("graphcoder_rl.rl.retrieval_reward_model", "RetrievalRewardModel"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    mod_name, attr = _LAZY_IMPORTS[name]
    module = import_module(mod_name)
    return getattr(module, attr)
