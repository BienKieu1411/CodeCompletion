#!/usr/bin/env python3
"""Run the paired 400-task evaluation inside the named Colab session."""

import runpy
import sys

sys.argv = [
    "/content/colab_eval.py",
    "--checkpoint", "/content/pg_msld_eval_checkpoint.pt",
    "--valid", "/content/valid_400.parquet",
    "--previous-details", "/content/original_details.jsonl",
    "--output-dir", "/content/pg_msld_budget_eval",
    "--retriever-batch-size", "128",
    "--generator-batch-size", "8",
    "--max-rows", "400",
]
runpy.run_path("/content/colab_eval.py", run_name="__main__")
