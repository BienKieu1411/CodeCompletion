from __future__ import annotations

from importlib import import_module

__all__ = ["train", "run_contrastive_pretrain"]

_LAZY_IMPORTS = {
    "train": ("graphcoder_rl.training.graphcoder_rl_train", "train"),
    "run_contrastive_pretrain": ("graphcoder_rl.training.contrastive_pretrain", "run_contrastive_pretrain"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    mod_name, attr = _LAZY_IMPORTS[name]
    return getattr(import_module(mod_name), attr)
