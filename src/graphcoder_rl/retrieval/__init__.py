from __future__ import annotations

from importlib import import_module

__all__ = ["CoarseDenseRetriever", "MultiHopGraphRetriever"]

_LAZY_IMPORTS = {
    "CoarseDenseRetriever": ("graphcoder_rl.retrieval.coarse_dense_retriever", "CoarseDenseRetriever"),
    "MultiHopGraphRetriever": ("graphcoder_rl.retrieval.multi_hop_graph_retriever", "MultiHopGraphRetriever"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    mod_name, attr = _LAZY_IMPORTS[name]
    return getattr(import_module(mod_name), attr)
