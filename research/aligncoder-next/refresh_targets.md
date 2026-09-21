# Refresh targets

Refresh this dossier before using it in a paper or after changing the implementation.

## External sources

- AlignCoder: https://arxiv.org/abs/2601.19697
- CAST: https://arxiv.org/abs/2506.15655
- Controlled chunking study: https://arxiv.org/abs/2605.04763
- REPLUG: https://arxiv.org/abs/2301.12652
- CodeRAG: https://arxiv.org/abs/2509.16112
- CODEFILTER: https://arxiv.org/abs/2508.05970
- REVELA: https://arxiv.org/abs/2506.16552
- GrepRAG: https://arxiv.org/abs/2601.23254
- DraCo: https://arxiv.org/abs/2405.19782
- SelfRACG: https://arxiv.org/abs/2507.19033
- Repoformer: https://arxiv.org/abs/2403.10059
- ReCUBE: https://arxiv.org/abs/2603.25770
- Jina model card: https://huggingface.co/jinaai/jina-code-embeddings-1.5b

## Local items to re-check

- `doc/TCD_KD_FRAMEWORK.md`: model revisions, tokenizer budgets, loss/null equations, and implementation status.
- `doc/papers/ALIGNCODER_DEEPREAD.md`: native/local code alignment and baseline numbers.
- `doc/papers/CAST_DEEPREAD.md`: AST chunking measurements and budget definition.
- `doc/papers/CODERAG_DEEPREAD.md`: offline distillation details and benchmark protocol.
- User's AST-GR and AST chunker implementation: leakage, source-span coverage, parser versions, and languages.

## Volatile experiment fields

Pin checkpoint revisions, tokenizer revisions, parser versions, repository commit snapshots, dataset split manifests, context budgets, decoding parameters, and hardware. Any change to these fields creates a new baseline manifest.
