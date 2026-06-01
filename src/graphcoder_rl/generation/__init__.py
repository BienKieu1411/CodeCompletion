from __future__ import annotations

from importlib import import_module

__all__ = ["GraphContextComposer", "GraphCoderPromptBuilder", "CodeLLMGenerator"]

_LAZY_IMPORTS = {
    "GraphContextComposer": ("graphcoder_rl.generation.graph_context_composer", "GraphContextComposer"),
    "GraphCoderPromptBuilder": ("graphcoder_rl.generation.graphcoder_prompt_builder", "GraphCoderPromptBuilder"),
    "CodeLLMGenerator": ("graphcoder_rl.generation.code_llm_generator", "CodeLLMGenerator"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    mod_name, attr = _LAZY_IMPORTS[name]
    return getattr(import_module(mod_name), attr)
