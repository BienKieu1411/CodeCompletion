from __future__ import annotations

from importlib import import_module

__all__ = ["DatasetLoader", "LeftContextAnchorExtractor", "LocalGraph"]

_LAZY_IMPORTS = {
    "DatasetLoader": ("co_retrieval.data.repository_dataset_loader", "DatasetLoader"),
    "LeftContextAnchorExtractor": ("co_retrieval.data.left_context_anchor_extractor", "LeftContextAnchorExtractor"),
    "LocalGraph": ("co_retrieval.data.left_context_anchor_extractor", "LocalGraph"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    mod_name, attr = _LAZY_IMPORTS[name]
    return getattr(import_module(mod_name), attr)
