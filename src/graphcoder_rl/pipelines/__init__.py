from __future__ import annotations

from importlib import import_module

__all__ = ["infer", "build_graph_cache", "build_ppl_cache"]

_LAZY_IMPORTS = {
    "infer": ("graphcoder_rl.pipelines.graphcoder_rl_infer", "infer"),
    "build_graph_cache": ("graphcoder_rl.pipelines.cache_builders", "build_graph_cache"),
    "build_ppl_cache": ("graphcoder_rl.pipelines.cache_builders", "build_ppl_cache"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    mod_name, attr = _LAZY_IMPORTS[name]
    return getattr(import_module(mod_name), attr)
