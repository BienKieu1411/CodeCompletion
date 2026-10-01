import unittest

from src.data.cross_file_budget import select_cross_file_context


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class CrossFileBudgetTests(unittest.TestCase):
    def test_budget_is_exact_and_pool_is_not_mutated(self):
        candidates = [
            {"path": "a.py", "start": 0, "end": 10,
             "source_sha256": "a", "model_text": "alpha\n" * 4,
             "bm25_score": 3.0},
            {"path": "b.py", "start": 0, "end": 10,
             "source_sha256": "b", "model_text": "beta\n" * 4,
             "bm25_score": 2.0},
            {"path": "a.py", "start": 11, "end": 20,
             "source_sha256": "a", "model_text": "gamma\n" * 4,
             "bm25_score": 1.0},
        ]
        original = [dict(item) for item in candidates]
        result = select_cross_file_context(
            candidates, CharTokenizer(), "python", budget_tokens=80, max_snippets=2)
        self.assertLessEqual(result["token_count"], 80)
        self.assertLessEqual(result["selected_count"], 2)
        self.assertEqual(candidates, original)
        self.assertEqual({item["path"] for item in result["items"]}, {"a.py", "b.py"})

    def test_oversized_chunk_is_not_truncated(self):
        candidate = {"path": "a.py", "start": 0, "end": 100,
                     "source_sha256": "a", "model_text": "x" * 100}
        result = select_cross_file_context(
            [candidate], CharTokenizer(), "python", budget_tokens=10, max_snippets=10)
        self.assertEqual(result["items"], [])
        self.assertEqual(candidate["model_text"], "x" * 100)


if __name__ == "__main__":
    unittest.main()
