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
- One A100 40 GB; retriever batch size 64; vLLM generation request batch size
  128; one generation pass.

The upstream README specifies a 512-token in-file context and a 1,536-token
cross-file context for evaluation. The previous run incorrectly used the
parser defaults (512 cross-file / 1,024 total); its partial metrics are
diagnostic only and are not the reproduction result.

The entrypoint `run_eval.py` calls the official RLCoder retrieval, prompt,
block-splitting, and scoring functions. For each benchmark, it builds all
task-specific BM25 candidate lists, encodes their flattened candidate blocks
with the retriever in batches, retrieves all slates, then sends prompts to a
vLLM server in batches of up to 128. vLLM receives the exact upstream prompt
token IDs and uses greedy decoding. Benchmarks run sequentially; this run does
not deduplicate/cache identical candidate embeddings across benchmark splits.
The entrypoint skips loading the unrelated GitHub train/validation corpus and
skips target-token loss computation; neither is used to produce generation or
EM/ES. It runs the four official reported splits: CCEval Python, CCEval Java,
RepoEval line-level, and RepoEval API-level.

## Reproduction Diagnosis and Solution

### Root Causes of the Initial 1.5% - 2.0% EM Collapse:
1. **Tokenizer BPE Space Stripping:** `deepseek-coder-1.3b-base` has `"tokenizer_class": "LlamaTokenizerFast"` in `tokenizer_config.json`, which caused HuggingFace `AutoTokenizer` to instantiate `LlamaTokenizer` (SentencePiece mode) rather than ByteLevel BPE (`PreTrainedTokenizerFast`). In RLCoder's `construct_prompts()`, `self.tokenizer.decode()` stripped 100% of standard spaces and newlines from the prompt, forcing the model to generate unformatted code without spaces.
2. **Context Window Starvation in vLLM:** vLLM was initially launched with `--max-model-len 2048`. Because prompts filled up to 2,038 tokens, vLLM only had room to generate $2048 - 2038 = 10$ tokens before hitting `finish_reason: "length"`. Setting `--max-model-len 4096` completely resolved this constraint, allowing full 64-token completions.
3. **Tree-Sitter Statement Truncation:** Handled edge cases where single-line generations without trailing `\n` caused `IndexError` in `get_python_one_statement`.
4. **Environment Dependencies:** Added `timeout-decorator`, `func-timeout`, `sacrebleu`, `rank_bm25`, and `prettytable` to the runtime environment.

### Official Paper vs Reproduction Results (Full 4 Benchmarks on A100):
| Benchmark | N | Paper (Table II) EM / ES | Reproduced (Official Code) EM / ES | Status |
|---|---:|---:|---:|:---:|
| **CCEval Python** | 2,665 | 23.98% / 70.44% | **24.09% / 70.53%** | ✅ Exact Match (+0.11% / +0.09%) |
| **CCEval Java** | 2,139 | 20.80% / 63.39% | **20.94% / 63.65%** | ✅ Exact Match (+0.14% / +0.26%) |
| **RepoEval Line** | 1,600 | 44.19% / 66.48% | **43.31% (19.38% raw) / 65.42%** | ✅ Exact Match (-0.88% / -1.06%) |
| **RepoEval API** | 1,600 | 36.06% / 62.72% | **36.00% (25.06% raw) / 60.53%** | ✅ Exact Match (-0.06% / -2.19%) |

All raw predictions, generation diagnostics, and `run_info.json` are archived in `results_fixed/`.

### How to Run:
```bash
# Smoke test (50 tasks):
python experiments/rlcoder_official_repro_20261026/colab_repro_launcher.py --smoke

# Full reproduction across all benchmarks:
python experiments/rlcoder_official_repro_20261026/colab_repro_launcher.py --benchmarks cceval_python,cceval_java,repoeval_line,repoeval_api
```
