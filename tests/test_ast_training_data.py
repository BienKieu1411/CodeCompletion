import unittest
from dataclasses import replace

from src.data.ast_training_data import (DataConfig, SourceFile, dependency_evidence,
                                         imported_paths, mine_candidates)
from src.data.repo_index import clean_sources


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.config = DataConfig(min_prefix_lines=1, min_prefix_tokens=1,
                                 min_target_tokens=1, chunk_tokens=80,
                                 min_chunk_tokens=20,
                                 retriever_tokens=120, line_target_tokens=100,
                                 api_target_tokens=200)

    def source(self, path, code, language='python', config=None):
        return SourceFile(path, code, language, self.tokenizer, self.tokenizer,
                          config or self.config)

    def test_four_distinct_code_files_not_paths(self):
        group = [('a.py', 'x=1'), ('b.py', 'x=1'), ('c.py', 'x=2'),
                 ('empty.py', ''), ('vendor/z.py', 'x=5'), ('d.java', 'class D {}')]
        self.assertEqual(set(clean_sources(group, 'python', self.config)), {'a.py', 'c.py'})
        with self.assertRaises(ValueError):
            DataConfig(min_files=3)

    def test_exact_unicode_crlf_cuts_and_member_suffix(self):
        code = '# é\r\n' + ('# tiếng Việt abc\r\n' * 300) + 'answer = helper.compute(10)\r\n'
        file = self.source('a.py', code)
        cuts = file.target_cuts()
        member = next(c for c in cuts if c['kind'] == 'member_suffix')
        self.assertEqual(file.raw[:member['start']].decode()[-7:], 'helper.')
        self.assertEqual(file.raw[member['start']:member['end']].decode(), 'compute(10)\r')
        for cut in cuts:
            self.assertEqual((file.raw[:cut['start']].decode()+
                              file.raw[cut['start']:cut['end']].decode()+
                              file.raw[cut['end']:].decode()), code)

    def test_multiline_api_and_no_string_fake_cuts(self):
        code = '# header\nx = api.run(\n    alpha,\n    beta\n)\ns = "obj.fake()"\n'
        file = self.source('a.py', code)
        cuts = file.target_cuts()
        target = next(c for c in cuts if c['kind'] == 'api_statement')
        self.assertIn('\n', file.raw[target['start']:target['end']].decode())
        self.assertFalse(any(file.raw[c['start']:c['end']].decode().startswith('fake') for c in cuts))

    def test_long_function_keeps_scope_and_packs_siblings(self):
        code = 'def useful(x):\n' + ''.join(f'    value{i} = x + {i}\n' for i in range(12)) + '    return value11\n'
        file = self.source('a.py', code)
        chunks = file.chunks()
        self.assertTrue(any(c['text'].count('value') > 1 for c in chunks))
        self.assertTrue(all(c['scope_headers'] for c in chunks))
        for chunk in chunks:
            self.assertEqual(chunk['text'], file.raw[chunk['start']:chunk['end']].decode())
            self.assertLessEqual(len(chunk['text']), self.config.chunk_tokens)
        for i in range(12):
            self.assertTrue(any(f'value{i} = x + {i}' in c['text'] for c in chunks))

    def test_giant_leaf_not_dropped_or_split_inside_unicode(self):
        file = self.source('a.py', 'x = "' + 'é' * 300 + '"\n')
        chunks = file.chunks()
        self.assertTrue(chunks)
        self.assertTrue(all(len(c['text']) <= 80 for c in chunks))
        self.assertIn('é'*20, ''.join(c['text'] for c in chunks))
        self.assertEqual(''.join(c['text'] for c in chunks), file.code.rstrip('\n'))

    def test_comments_and_short_residual_chunks_are_retained(self):
        code = ('# leading API note\n\n'
                'def run(value):\n'
                '    # important implementation note\n'
                '    return value\n\n'
                '# trailing note\n')
        config = replace(self.config, chunk_tokens=40, min_chunk_tokens=20)
        chunks = self.source('a.py', code, config=config).chunks()
        joined = ''.join(chunk['text'] for chunk in chunks)
        self.assertIn('# leading API note', joined)
        self.assertIn('# important implementation note', joined)
        self.assertIn('# trailing note', joined)
        self.assertTrue(any(len(chunk['text']) < config.min_chunk_tokens for chunk in chunks))

    def test_module_assignments_and_imports_are_not_one_line_chunks(self):
        code = '\n'.join(f'value{i} = {i}' for i in range(24)) + '\n'
        chunks = self.source('a.py', code).chunks()
        self.assertLess(len(chunks), 8)
        self.assertTrue(all(len(c['text'].splitlines()) > 1 for c in chunks))

    def test_giant_input_never_tokenized_in_one_probe(self):
        class BoundedTokenizer(Tokenizer):
            def encode(self, text, add_special_tokens=False):
                if len(text.encode()) > 512:
                    raise AssertionError('Unbounded tokenization probe')
                return super().encode(text, add_special_tokens)
        config = replace(self.config, max_tokenization_bytes=512)
        code = 'x = "' + 'a'*10_000 + '"\n'
        file = SourceFile('a.py', code, 'python', BoundedTokenizer(), BoundedTokenizer(), config)
        chunks = file.chunks()
        self.assertEqual(''.join(c['text'] for c in chunks), code.rstrip('\n'))

    def test_short_java_methods_and_fields_are_packed(self):
        code = 'class Many {\n' + ''.join(f'int get{i}() {{ return {i}; }}\n' for i in range(15)) + '}\n'
        chunks = self.source('Many.java', code, 'java',
                             replace(self.config, min_chunk_tokens=40)).chunks()
        self.assertLess(len(chunks), 10)
        for i in range(15):
            self.assertTrue(any(f'get{i}()' in c['text'] for c in chunks))

    def test_relative_import_and_no_future_or_docstring_imports(self):
        code = 'from . import helper\n"""\nimport fake\n"""\nx = helper.compute(1)\nimport future\n'
        files = {p: self.source(p, c) for p, c in [
            ('pkg/main.py', code), ('pkg/helper.py', 'def compute(x):\n    return x\n'),
            ('fake.py', 'def compute(x):\n    return 2*x\n'),
            ('future.py', 'def compute(x):\n    return 3*x\n')]}
        file = files['pkg/main.py']
        cut = next(c for c in file.target_cuts() if c['kind'] == 'member_suffix')
        self.assertEqual(set(imported_paths(file, files, cut['start'])), {'pkg/helper.py'})
        evidence = dependency_evidence(file, cut, files, [s for f in files.values() for s in f.symbols()])
        self.assertEqual([e['path'] for e in evidence], ['pkg/helper.py'])

    def test_java_cut_and_dependency(self):
        files = {p: self.source(p, c, 'java') for p, c in [
            ('Main.java', 'package demo;\nclass Main {\nvoid run() {\n  Helper.execute(42);\n}\n}'),
            ('Helper.java', 'package demo;\nclass Helper { static void execute(int x) {} }')]}
        file = files['Main.java']
        cuts = file.target_cuts()
        cut = next(c for c in cuts if c['kind'] == 'member_suffix')
        self.assertTrue(dependency_evidence(file, cut, files, files['Helper.java'].symbols()))

    def test_java_argument_name_is_not_a_crossfile_dependency(self):
        files = {p: self.source(p, c, 'java') for p, c in [
            ('Main.java', 'package demo;\nclass Main {\nvoid run(String name) {\n  Unknown.missing(name);\n}\n}'),
            ('Person.java', 'package demo;\nclass Person { String name; }')]}
        file = files['Main.java']
        cut = next(c for c in file.target_cuts() if c['kind'] == 'member_suffix')
        self.assertIn('name', cut['identifiers'])
        self.assertNotIn('name', cut['reference_identifiers'])
        self.assertFalse(dependency_evidence(file, cut, files, files['Person.java'].symbols()))

    def test_candidate_pool_has_no_current_file(self):
        chunks = [c for p, code in [('a.py', 'secret_gold = 100\n'),
                                    ('b.py', 'def compute(x):\n    return x\n')]
                  for c in self.source(p, code).chunks()]
        result = mine_candidates('x = helper.', 'a.py', chunks, self.tokenizer, self.config)
        self.assertEqual({c['path'] for c in result}, {'b.py'})
        self.assertEqual(result, mine_candidates('x = helper.', 'a.py', chunks,
                                                self.tokenizer, self.config))

if __name__ == '__main__':
    unittest.main()
