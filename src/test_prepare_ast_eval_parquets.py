import sqlite3
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.prepare_ast_eval_parquets import merge_prepared_shards, prepare_parquet


class PrepareAstEvalParquetTest(unittest.TestCase):
    def test_adds_chunks_without_changing_any_source_field_or_label(self):
        context_type = pa.list_(pa.struct([
            pa.field("path", pa.string()),
            pa.field("text", pa.string()),
        ]))
        source_schema = pa.schema([
            ("task_id", pa.string()),
            ("path", pa.string()),
            ("left_context", pa.string()),
            ("groundtruth", pa.string()),
            ("crossfile_context", context_type),
        ], metadata={b"source": b"original"})
        rows = [
            {"task_id": "repo/a.py", "path": "a.py",
             "left_context": "def a():\n    ", "groundtruth": "return helper()",
             "crossfile_context": [{"path": "helper.py",
                                    "text": "def helper():\n    return 1\n"}]},
            {"task_id": "repo/b.py", "path": "b.py",
             "left_context": "def b():\n    ", "groundtruth": "return helper()",
             "crossfile_context": [{"path": "helper.py",
                                    "text": "def helper():\n    return 1\n"}]},
        ]

        def fake_ast_chunks(path, code, language):
            self.assertEqual(language, "python")
            return ([{"path": path, "text": code, "type": "function_definition",
                      "start": 0, "end": len(code.encode("utf-8"))}], False)

        chunker = {"AST_CHUNK_TOKENS": 384, "ast_chunks": fake_ast_chunks}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_path, output_path = root / "source.parquet", root / "out.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=source_schema), source_path)
            with sqlite3.connect(root / "cache.sqlite") as connection:
                result = prepare_parquet(source_path, output_path, "python", chunker,
                                         connection, batch_size=1, progress_every=100)

            source = pq.read_table(source_path)
            output = pq.read_table(output_path)
            self.assertTrue(source.equals(output.select(source.column_names),
                                          check_metadata=False))
            self.assertEqual(output.column("groundtruth").to_pylist(),
                             ["return helper()", "return helper()"])
            chunks = output.column("crossfile_ast_chunks").to_pylist()
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0][0]["path"], "helper.py")
            self.assertEqual(chunks[1][0]["text"], "def helper():\n    return 1\n")
            self.assertEqual(result["original_values_verified"], 2)
            self.assertEqual(result["unique_source_chunks"], 1)
            self.assertEqual(result["chunk_references"], 2)

    def test_merges_repoeval_shards_in_order_without_changing_fields(self):
        schema = pa.schema([
            ("task_id", pa.string()),
            ("groundtruth", pa.string()),
            ("crossfile_ast_chunks", pa.list_(pa.struct([
                ("path", pa.string()), ("text", pa.string()),
                ("type", pa.string()), ("start", pa.int64()), ("end", pa.int64()),
            ]))),
        ], metadata={b"prepared": b"true"})
        shard_rows = [
            [{"task_id": "repo/a", "groundtruth": "first",
              "crossfile_ast_chunks": [{"path": "a.py", "text": "x=1",
                                         "type": "assignment", "start": 0, "end": 3}] }],
            [{"task_id": "repo/b", "groundtruth": "second",
              "crossfile_ast_chunks": [{"path": "b.py", "text": "x=2",
                                         "type": "assignment", "start": 0, "end": 3}] }],
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / "test_0.parquet", root / "test_1.parquet"]
            for path, rows in zip(paths, shard_rows):
                pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
            output_path = root / "test.parquet"
            result = merge_prepared_shards(paths, output_path, batch_size=1)
            merged = pq.read_table(output_path)
            self.assertEqual(merged.column("task_id").to_pylist(),
                             ["repo/a", "repo/b"])
            self.assertEqual(merged.column("groundtruth").to_pylist(),
                             ["first", "second"])
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["original_values_verified"], 2)


if __name__ == "__main__":
    unittest.main()
