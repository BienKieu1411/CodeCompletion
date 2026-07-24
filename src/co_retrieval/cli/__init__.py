"""CLI package exports without eager-loading the runnable module."""

from __future__ import annotations

from typing import Any

__all__ = ["build_parser", "main"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from co_retrieval.cli import co_retrieval_cli

        return getattr(co_retrieval_cli, name)
    raise AttributeError(name)
