"""
Causal Prompt Generator
Dùng cho chuẩn Repository-Level Code Completion trong môi trường IDE thực tế.
"""

from __future__ import annotations

from typing import Optional

from graphcoder_rl.generation.graph_context_composer import GraphContextComposer


class GraphCoderPromptBuilder:
    """Tạo Prompt theo chuẩn Causal LM (Next-Token Prediction)."""

    def __init__(self, model_name: str = "deepseek-coder", max_length: int = 2048):
        self.max_length = max_length
        self.model_name = model_name
        self.composer = GraphContextComposer(max_length=max_length)

    def construct_prompt(
        self,
        retrieved_context: str,
        left_context: str,
        file_path: str = "current_file.py",
        local_graph: Optional[object] = None,
        immediate_left_lines: int = 40,
        max_imports: int = 16,
    ) -> str:
        """Context Composer wrapper theo GraphCoder-RL."""
        return self.composer.compose(
            retrieved_context=retrieved_context,
            left_context=left_context,
            file_path=file_path,
            local_graph=local_graph,
            immediate_left_lines=immediate_left_lines,
            max_imports=max_imports,
        )
