# %% [markdown]
# Run this cell after (or independently of) the RLRetriever evaluation cell.
# It reads the checkpoint CSV already written to Kaggle working storage and
# creates the macro summary without re-running model inference.

# %%
import os
import pandas as pd

RESULTS_FILES = (
    "rlretriever_repobench_r_results.csv",       # optimized RLRetriever-only run
    "repobench_r_retriever_results.csv",         # earlier combined-run artifact
)
SUMMARY_FILE = "rlretriever_repobench_r_summary.csv"

RESULTS_FILE = next((path for path in RESULTS_FILES if os.path.exists(path)), None)
if RESULTS_FILE is None:
    raise FileNotFoundError(
        f"None of {RESULTS_FILES} was found in the current working directory. "
        "Run the evaluation cell through at least one completed split first."
    )

results = pd.read_csv(RESULTS_FILE)
results = results[results["model"] == "RLRetriever"].copy()
if results.empty:
    raise ValueError(f"{RESULTS_FILE} contains no RLRetriever rows.")
summary = (
    results.groupby(["model", "split"], as_index=False)
    [["Recall@1", "Recall@3", "Recall@5", "MRR", "mean_rank"]]
    .mean()
    .sort_values(["split", "Recall@1"], ascending=[True, False])
)

display(results.sort_values(["config", "split", "model"]))
display(summary)
summary.to_csv(SUMMARY_FILE, index=False)
print(f"Saved: {SUMMARY_FILE}")
