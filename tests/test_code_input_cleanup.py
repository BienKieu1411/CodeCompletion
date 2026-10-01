import unittest

from src.data.code_input_cleanup import clean_chunk, clean_code_input


class CleanupTests(unittest.TestCase):
    def clean(self, code, language='python'):
        return clean_code_input(code, language)['text']

    def test_non_english_comments_and_identifiers_not_noise(self):
        code = '# Tính tổng, không làm tròn\n# 计算总和\nsố = 1\n'
        self.assertEqual(self.clean(code), code)

    def test_hash_and_slashes_in_literals_untouched(self):
        code = 'url = "http://example.com/#part"\npattern = "# copyright"\n'
        self.assertEqual(self.clean(code), code)
        java = 'class A { String s = "// copyright"; char c = \'#\'; }\n'
        self.assertEqual(self.clean(java, 'java'), java)

    def test_multiline_docstrings_and_blank_lines_inside_unchanged(self):
        code = 'def f():\n    """API\n\n\n\n# Copyright is literal here\n    """\n    return 1\n'
        self.assertEqual(self.clean(code), code)

    def test_newlines_collapsed_only_outside_literals(self):
        code = 'x = 1\n\n\n\n\ny = "a\\n\\n"\n'
        self.assertEqual(self.clean(code), 'x = 1\n\n\ny = "a\\n\\n"\n')

    def test_license_separator_repeat_and_pragmas(self):
        code = '# Copyright 2020 Author\n# ----------------\n# useful\n# useful\n# noqa\n# type: ignore\nx = 1\n'
        result = self.clean(code)
        self.assertNotIn('Copyright', result)
        self.assertNotIn('----', result)
        self.assertEqual(result.count('# useful'), 1)
        self.assertIn('# noqa', result)
        self.assertIn('# type: ignore', result)

    def test_unicode_and_mixed_punctuation_banners_are_removed(self):
        code = '# ……，。。。。！！！\n# ====== --- ***\n# useful API: call foo(...) here\nx = 1\n'
        result = self.clean(code)
        self.assertNotIn('……', result)
        self.assertNotIn('======', result)
        self.assertIn('useful API: call foo(...) here', result)

    def test_punctuation_inside_literals_is_preserved(self):
        code = 'message = "……，。。。!!!"\npattern = r"---...___"\n'
        self.assertEqual(self.clean(code), code)

    def test_inline_java_comment_removal_cannot_merge_tokens(self):
        code = 'class A { int/* Copyright 2020 */value = 1; }'
        result = self.clean(code, 'java')
        self.assertNotIn('intvalue', result)
        self.assertIn('value', result)

    def test_cursor_indentation_and_dot_preserved(self):
        code = '# Copyright 2020\nx = 1\n\n\n\nanswer = helper.'
        result = self.clean(code)
        self.assertTrue(result.endswith('answer = helper.'))
        self.assertNotIn('Copyright', result)
        self.assertEqual(self.clean('x = 1\n\n\n\n    ')[-4:], '    ')

    def test_crlf_inside_string_preserved(self):
        code = 'x = """a\r\n\r\n\r\nb"""\r\ny = 2\r\n'
        result = self.clean(code)
        self.assertIn('a\r\n\r\n\r\nb', result)
        self.assertTrue(result.endswith('y = 2\n'))

    def test_invalid_or_incomplete_string_not_guessed(self):
        code = 'x = """\n\n\n# Copyright\nunterminated'
        self.assertEqual(self.clean(code), code)

    def test_string_crossing_chunk_boundary_remains_exact(self):
        code = 'x = """\n# Copyright\n\n\n\ny\n"""\n'
        raw = code.encode()
        start, end = raw.index(b'#'), raw.index(b'y\n')
        chunk = {'start': start, 'end': end, 'text': raw[start:end].decode()}
        self.assertEqual(clean_chunk(chunk, code, 'python')['text'], chunk['text'])

    def test_idempotent_and_no_source_mutation(self):
        code = '# Copyright 2020\nx = 1   \n\n\n\n'
        result = self.clean(code)
        self.assertEqual(result, self.clean(result))
        self.assertIn('Copyright', code)


if __name__ == '__main__':
    unittest.main()
