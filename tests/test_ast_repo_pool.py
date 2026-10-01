import random
import unittest

from src.data.ast_repo_pool import build_repo_pool_record, sample_repo_epoch
from src.data.ast_training_data import DataConfig


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class RepoPoolTests(unittest.TestCase):
    def setUp(self):
        self.config = DataConfig(
            min_substantive_files=2, min_file_code_lines=2,
            min_repo_code_lines=1, max_repo_code_lines=10_000,
            min_prefix_lines=1, min_prefix_tokens=1, min_target_tokens=1,
            chunk_tokens=80, min_chunk_tokens=1, retriever_tokens=120,
            line_target_tokens=100, api_target_tokens=200,
            max_target_spans_per_file=32, max_target_spans_per_repo=64,
            pool_size=8, repos_per_epoch=2,
        )
        self.repo = {
            "uid": "repo-1", "language": "python", "repo_id": 7, "split": "train",
            "files": {
                "root.py": "import helper\n" + "x = 1\n" * 12,
                "target.py": "import helper\n" + "x = 1\n" * 8 + "answer = helper.compute(1)\n",
                "helper.py": "def compute(x):\n    return x + 1\n" * 4,
                "other.py": "def unrelated(x):\n    return x\n" * 4,
            },
        }

    def test_pool_keeps_quality_and_excludes_anchor_targets(self):
        record = build_repo_pool_record(
            self.repo, CharTokenizer(), CharTokenizer(), self.config)
        self.assertEqual(record["status"], "kept")
        payload = record["payload"]
        self.assertGreater(payload["stats"]["chunk_count"], 0)
        self.assertGreater(payload["stats"]["target_count"], 0)
        self.assertNotIn(payload["root_path"], {target["path"] for target in payload["target_pool"]})

    def test_epoch_has_at_most_one_task_per_repo(self):
        record = build_repo_pool_record(
            self.repo, CharTokenizer(), CharTokenizer(), self.config)["payload"]
        rows = sample_repo_epoch([record, {**record, "repo_uid": "repo-2", "repo_id": 8}],
                                 CharTokenizer(), CharTokenizer(), 1, self.config, 123)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row["repo_uid"] for row in rows}), 2)
        self.assertTrue(all(row["candidates"] for row in rows))


if __name__ == "__main__":
    unittest.main()
