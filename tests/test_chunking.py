from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from co_retrieval.chunking import RepositoryChunker


class ChunkingTests(unittest.TestCase):
    def test_chunker_extracts_methods_and_metadata(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "service.py"
            source.write_text(
                '''import os

class UserService(BaseService):
    """docs"""
    def fetch_user(self, user_id):
        return self.db.fetch_user(user_id)

def helper():
    return os.getcwd()
''',
                encoding="utf-8",
            )
            chunks = RepositoryChunker().chunk_file(source, repo_root=root)

        self.assertTrue(any(c.chunk_type == "class_header" for c in chunks))
        method = next(c for c in chunks if c.chunk_type == "method")
        self.assertEqual("UserService", method.parent_class)
        self.assertIn("fetch_user", method.defined_symbols)
        self.assertIn("fetch_user", method.call_names)

    def test_chunker_falls_back_on_invalid_python(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "broken.py"
            source.write_text("def broken(:\n    pass\n", encoding="utf-8")
            chunks = RepositoryChunker(fallback_lines=1).chunk_file(source, repo_root=root)

        self.assertTrue(chunks)
        self.assertTrue(all(c.chunk_type == "fallback" for c in chunks))


if __name__ == "__main__":
    unittest.main()
