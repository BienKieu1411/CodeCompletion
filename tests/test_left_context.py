import unittest

from src.data.left_context import pack_left_context


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class LeftContextTests(unittest.TestCase):
    def test_long_prefix_keeps_cursor_tail_and_ast_hints(self):
        source = ("from helper import compute\n"
                  "# useful API comment\n"
                  "class Service:\n"
                  "    def run(self, value):\n"
                  + "        padding = 'x'\n" * 12
                  + "        result = compute(value)\n"
                  + "        AFTER_CURSOR = 'must not appear'\n")
        cursor = source.index("        AFTER_CURSOR")
        result = pack_left_context(
            source, cursor, "python", CharTokenizer(),
            max_tokens=180, tail_tokens=90, hint_tokens=70)
        self.assertLessEqual(result["tokens"], 180)
        self.assertTrue(result["truncated"])
        self.assertIn("result = compute(value)", result["text"])
        self.assertIn("from helper import compute", result["text"])
        self.assertIn("class Service:", result["text"])
        self.assertNotIn("AFTER_CURSOR", result["text"])

    def test_short_prefix_is_preserved_after_cleanup(self):
        source = "# useful comment\nx = 1\nresult = x\n"
        cursor = source.index("result")
        result = pack_left_context(source, cursor, "python", CharTokenizer(),
                                   max_tokens=200, tail_tokens=100, hint_tokens=50)
        self.assertEqual(result["text"], source[:cursor])
        self.assertIn("useful comment", result["text"])

    def test_ast_hints_do_not_depend_on_right_context(self):
        prefix = ("from helper import compute\n"
                  "class Service:\n"
                  "    def run(self, value):\n"
                  + "        padding = 'x'\n" * 14
                  + "        result = compute(value)\n")
        left = pack_left_context(
            prefix + "def after_cursor():\n    return 'right'\n",
            len(prefix.encode()), "python", CharTokenizer(),
            max_tokens=150, tail_tokens=70, hint_tokens=60)
        right_changed = pack_left_context(
            prefix + "class A completely different: pass\n",
            len(prefix.encode()), "python", CharTokenizer(),
            max_tokens=150, tail_tokens=70, hint_tokens=60)
        self.assertEqual(left["text"], right_changed["text"])


if __name__ == "__main__":
    unittest.main()
