"""Small CPU contract checks for the offline train-data artifact."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

import pyarrow as pa
import pyarrow.parquet as pq

from src.prepare_ast_rrpo_data import prepare_language


class OfflinePreparationTest(unittest.TestCase):
    def test_filters_singleton_repositories_and_caches_all_eligible_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.parquet"
            filtered = root / "filtered.parquet"
            pq.write_table(pa.table({
                "path": ["a.py", "b.py", "alone.py", "c.py", "d.py", "e.py"],
                "content": ["a = 1\r\nb = 2", "b = 3", "alone", "c = 4",
                            "d = 5", "a = 1\r\nb = 2"],
                "first": [True, False, True, True, False, False],
            }), raw, row_group_size=2)
            connection = sqlite3.connect(root / "ast.sqlite")
            connection.execute("CREATE TABLE ast_entries (language TEXT, "
                               "source_sha256 TEXT, spans BLOB, chunks BLOB, "
                               "had_error INTEGER, PRIMARY KEY(language, source_sha256))")
            visited = []

            def chunks(path, code, language):
                self.assertEqual(language, "python")
                visited.append(("chunk", code))
                return ([{"path": path, "text": code, "start": 0,
                          "end": len(code.encode()), "type": "assignment"}], False)

            def spans(code, language):
                self.assertNotIn("\r", code)
                visited.append(("span", code))
                return {"line": [(0, len(code.encode()), "assignment")], "block": []}

            stats = prepare_language("python", raw, filtered, connection,
                                     {"ast_chunks": chunks, "target_spans": spans})
            table = pq.read_table(filtered).to_pydict()
            self.assertEqual(stats["eligible_repos"], 2)
            self.assertEqual(stats["eligible_files"], 5)
            self.assertEqual(stats["singletons_dropped"], 1)
            self.assertEqual(table["path"], ["a.py", "b.py", "c.py", "d.py", "e.py"])
            self.assertEqual(table["first"], [True, False, True, False, False])
            self.assertNotIn(("chunk", "alone"), visited)
            self.assertEqual(sum(kind == "chunk" for kind, _ in visited), 4)
            normalized = "a = 1\nb = 2"
            self.assertIn(("span", normalized), visited)
            rows = connection.execute("SELECT spans FROM ast_entries WHERE spans IS NOT NULL")
            self.assertTrue(all(json.loads(zlib.decompress(row[0]))["line"]
                                for row in rows))
            connection.close()


if __name__ == "__main__":
    unittest.main()
