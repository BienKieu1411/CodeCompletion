"""Contract tests for consuming precomputed test-set AST chunks."""

import ast
import os
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import tree_sitter as ts
import tree_sitter_java as ts_java
import tree_sitter_python as ts_python
from rank_bm25 import BM25Okapi


SOURCE = Path(__file__).with_name("ast_ppo_unixcoder_eval_kaggle.py")
TERMINAL_PPO_EVAL_SOURCE = Path(__file__).with_name(
    "ast_ppo_unixcoder_ppo_eval_kaggle.py"
)


class CharacterTokenizer:
    cls_token = "<s>"
    sep_token = "</s>"
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text)

    def tokenize(self, text):
        return list(text)

    def convert_tokens_to_ids(self, tokens):
        return list(range(1, len(tokens) + 1))


class EvalPrecomputedChunksTest(unittest.TestCase):
    def load_eval_functions(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        constant_names = {
            "AST_CHUNK_TOKENS", "PARSERS", "RETRIEVAL_LINE_NODE_TYPES",
            "RETRIEVAL_BLOCK_NODE_TYPES", "RETRIEVAL_NODE_TYPES",
            "SCOPE_NODE_TYPES", "IDENTIFIER_RE",
        }
        constants = [node for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name)
                             and target.id in constant_names
                             for target in node.targets)]
        function_names = {"ast_chunks", "build_row", "benchmark_data_path",
                          "lexical_tokens", "render_chunk"}
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name in function_names]
        self.assertEqual({node.name for node in functions}, function_names)

        tokenizer = CharacterTokenizer()
        env = {
            "Path": Path, "np": np, "os": os, "re": re, "ts": ts,
            "ts_java": ts_java, "ts_python": ts_python,
            "BM25Okapi": BM25Okapi,
            "GEN_TOKENIZER": tokenizer, "RET_TOKENIZER": tokenizer,
            "AST_CHUNK_TOKENS": 128,
            "MAX_RELATED_FILES": 24,
            "CANDIDATE_POOL_SIZE": 64,
            "CROSSFILE_TOKEN_BUDGET": 4096,
            "RETRIEVER_QUERY_LENGTH": 256,
            "RETRIEVER_CANDIDATE_LENGTH": 2048,
            "token_count": lambda text: len(text),
        }
        exec(compile(ast.Module(body=constants + functions, type_ignores=[]),
                     str(SOURCE), "exec"), env)
        # The test isolates chunk-order behavior from prompt construction and
        # UniXcoder tokenization; these have independent contract tests.
        env["left_anchors"] = lambda _left, _language: []
        env["retrieval_query"] = lambda _example, _anchors: (
            "calculate_priority needle helper")
        env["unixcoder_ids"] = lambda *_args: [1]
        return env

    def test_cceval_path_resolves_prepared_flat_files_and_raw_fallback(self):
        env = self.load_eval_functions()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            flat = root / "cceval_python.parquet"
            flat.touch()
            nested = root / "cceval" / "python" / "test.parquet"
            nested.parent.mkdir(parents=True)
            nested.touch()

            self.assertEqual(
                env["benchmark_data_path"](root, "cceval_python.parquet", "python"),
                flat,
            )
            flat.unlink()
            self.assertEqual(
                env["benchmark_data_path"](root, "cceval_python.parquet", "python"),
                nested,
            )

    def test_terminal_ppo_eval_targets_only_cceval_and_resolves_flat_files(self):
        tree = ast.parse(TERMINAL_PPO_EVAL_SOURCE.read_text(encoding="utf-8"))
        assignment = next(node for node in tree.body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name)
                                  and target.id == "BENCHMARKS"
                                  for target in node.targets))
        benchmark_names = [item[0] for item in ast.literal_eval(assignment.value)]
        self.assertEqual(benchmark_names, ["cceval_python", "cceval_java"])

        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "benchmark_data_path")
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(TERMINAL_PPO_EVAL_SOURCE), "exec"), namespace)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            flat_java = root / "cceval_java.parquet"
            flat_java.touch()
            self.assertEqual(
                namespace["benchmark_data_path"](
                    root, "cceval_java.parquet", "java"),
                flat_java,
            )

    def test_precomputed_chunks_match_online_chunking_after_bm25_file_reorder(self):
        env = self.load_eval_functions()
        related_files = [
            (f"ordinary_{index}.py",
             f"def ordinary_{index}():\n    return {index}\n")
            for index in range(30)
        ]
        related_files.append((
            "late_needle.py",
            "def calculate_priority(value):\n    return helper(value)\n\n"
            "def helper(value):\n    return value + 1\n",
        ))
        example = {
            "task_id": "sample", "language": "python", "file_path": "main.py",
            "left_context": "def caller():\n    calculate_priority(",
            "target_code": "return calculate_priority(value)",
            "related_files": related_files,
        }

        online = env["build_row"](example)
        precomputed = []
        expected_errors = []
        for path, code in related_files:
            chunks, had_error = env["ast_chunks"](path, code, "python")
            precomputed.extend(chunks)
            if had_error:
                expected_errors.append(path)
        cached_example = {
            **example,
            "crossfile_ast_chunks": precomputed,
            "crossfile_ast_parse_error_paths": expected_errors,
        }

        def no_online_chunking(*_args, **_kwargs):
            raise AssertionError("Prepared test chunks should bypass AST parsing")

        env["ast_chunks"] = no_online_chunking
        cached = env["build_row"](cached_example)

        self.assertIn("late_needle.py", online["related_file_paths_used"])
        self.assertNotEqual(online["related_file_paths_used"],
                            [path for path, _ in related_files[:24]])
        for key in ("candidate_paths", "candidate_types", "candidate_raw_texts",
                    "candidate_texts", "candidate_costs", "parse_errors",
                    "all_ast_chunks", "related_file_paths_used"):
            self.assertEqual(cached[key], online[key], key)


if __name__ == "__main__":
    unittest.main()
