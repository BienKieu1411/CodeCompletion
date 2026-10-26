# RLCoder reproduction: DeepSeek-Coder 1.3B

This run evaluates the released RLCoder retriever with the requested
`deepseek-ai/deepseek-coder-1.3b-base` generator. It does not fine-tune either
model.

## Pinned artifacts

- RLCoder upstream: `DeepSoftwareAnalytics/RLCoder`, commit
  `164d8d88cde324a38f5da70c4f858cc4679ef08e`.
- Released retriever: `nov3630/RLRetriever`, revision
  `ec587f5d8635462fff3cfb06f9a946148acda08b`.
- Generator: `deepseek-ai/deepseek-coder-1.3b-base`, revision
  `c919139c3a9b4070729c8b2cca4847ab29ca8d94`.
- Test data: `nov3630/Data4RLCoder`, revision
  `cb9639f20f2374d75e5b0b8e8650f1b20802bf9f`.

## Inference settings

- BM25 retrieves up to 100 blocks per task.
- RLRetriever ranks those blocks and the stop block; the generator receives the
  top 10 entries, including STOP if it ranks in the slate.
- RLCoder block splitting, query construction, prompt construction, and metric
  functions are taken from the pinned upstream code.
- Cross-file context: 1,536 tokens; total prompt context: 2,048 tokens
  (about 512 in-file tokens);
  maximum generation: 64 tokens.
- Fixed-length candidate blocks and progressive generation are disabled.
- One A100 40 GB, one generation pass, generator batch size 32 and retriever
  batch size 64.

The upstream README specifies a 512-token in-file context and a 1,536-token
cross-file context for evaluation. The previous run incorrectly used the
parser defaults (512 cross-file / 1,024 total); its partial metrics are
diagnostic only and are not the reproduction result.

The entrypoint `run_eval.py` calls the official RLCoder retrieval, prompt,
block-splitting, and scoring functions. Generator inference uses a vLLM server
with the exact upstream prompt token IDs and greedy decoding for batched
serving. It skips loading the unrelated GitHub train/validation corpus and
skips target-token loss computation; neither is used to produce generation or
EM/ES. It runs the four official reported splits: CCEval Python, CCEval Java,
RepoEval line-level, and RepoEval API-level.

`run_info.json` will record the exact runtime versions and per-split results.
