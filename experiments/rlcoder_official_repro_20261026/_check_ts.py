
try:
    import tree_sitter_python, tree_sitter_java
    from tree_sitter import Language, Parser
    py_lang = Language(tree_sitter_python.language())
    py_parser = Parser(py_lang)
    print("Tree-sitter modern packages ready!")
except Exception as e:
    print("Tree-sitter notice:", e)
