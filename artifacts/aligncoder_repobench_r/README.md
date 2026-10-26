# AlignCoder × RepoBench-R artifacts

This folder indexes the reproducible Kaggle scripts and their expected outputs.

| Purpose | Source file |
| --- | --- |
| Train/adapt AlignRetriever with the original AlignCoder training pipeline and frozen DeepSeek-Coder 1.3B | `../../kaggle_adapt_aligncoder_deepseek_1p3b.py` |
| Evaluate the adapted checkpoint using BM25 → generator-enhanced query → dense reranking on all RepoBench-R configs | `../../kaggle_aligncoder_enhanced_repobench_r.py` |
| One-pass RLRetriever baseline on RepoBench-R | `../../kaggle_repobench_r_retrievers.py` |
| Aggregate prior retriever results | `../../kaggle_retriever_comparison_summary.py` |

## Kaggle output layout

The enhanced-query evaluator writes resumable output to:

```text
/kaggle/working/aligncoder_repobench_r_artifacts/
  run_manifest.json
  aligncoder_enhanced_repobench_r_results.csv
  aligncoder_enhanced_repobench_r_summary.csv
```

The results CSV is updated after each completed `config × split`; attach that one file to a later Kaggle notebook and set `PREVIOUS_RESULTS_CSV` to resume only unfinished splits.

## Download the epoch-3 retriever checkpoint

Run this from the workspace after configuring Kaggle CLI access for the account
that owns (or can view) the private notebook:

```powershell
.\download_kaggle_result_3.ps1
```

It invokes `kaggle kernels output thanhnguyenvinh/notebook2d422470cf` and keeps
only files whose output path contains `result_3`.  The enhanced-query evaluator
is configured to resolve `result_3` explicitly and performs three sequential
enhancement rounds, with four sampled completions per round to match training.
