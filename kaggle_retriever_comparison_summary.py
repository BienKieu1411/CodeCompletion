# %% [markdown]
# # RepoBench-R one-pass retriever comparison summary
#
# Run this as a final Kaggle cell after the RLRetriever, AlignRetriever-one-pass,
# and UniXcoder-base jobs have written their CSV files.  It does not run any
# model inference.

# %%
import os
import pandas as pd

EXPECTED_CONFIGS = {"python_cff", "python_cfr", "java_cff", "java_cfr"}
METRICS = ["Recall@1", "Recall@3", "Recall@5", "MRR", "mean_rank"]

# Each dedicated file is preferred.  The combined CSV from an older run is a
# fallback; it is filtered to the intended model so rows are never mixed.
MODEL_SPECS = {
    "RLRetriever": {
        "model_id": "nov3630/RLRetriever",
        "files": [
            "rlretriever_repobench_r_results.csv",
            "repobench_r_retriever_results.csv",
        ],
    },
    "AlignRetriever_one_pass": {
        "model_id": "AlignCoder/AlignRetriever",
        "files": [
            "alignretriever_repobench_r_results.csv",
            "repobench_r_retriever_results.csv",
        ],
    },
    "UniXcoder_base": {
        "model_id": "microsoft/unixcoder-base",
        "files": [
            "unixcoder_base_repobench_r_results.csv",
            "repobench_r_retriever_results.csv",
        ],
    },
}


def load_one_model(model_name, spec):
    source = next((path for path in spec["files"] if os.path.exists(path)), None)
    if source is None:
        print(f"Missing results for {model_name}: checked {spec['files']}")
        return None

    frame = pd.read_csv(source)
    frame = frame[frame["model"] == model_name].copy()
    if frame.empty:
        print(f"{source} exists but has no rows for {model_name}")
        return None

    # A resumed run may append/rewrite the same config/split.  Keep the latest
    # row before macro-averaging so it is never counted twice.
    frame = frame.drop_duplicates(["config", "split"], keep="last")
    frame["source_file"] = source
    return frame


frames = []
for model_name, spec in MODEL_SPECS.items():
    frame = load_one_model(model_name, spec)
    if frame is not None:
        frames.append(frame)

if not frames:
    raise FileNotFoundError("No retriever results CSV was found in /kaggle/working.")

detail = pd.concat(frames, ignore_index=True)
summary = (
    detail.groupby(["model", "model_id", "split"], as_index=False)[METRICS]
    .mean()
    .sort_values(["split", "model"])
)
coverage = (
    detail.groupby(["model", "split"])["config"]
    .agg(lambda values: ", ".join(sorted(set(values))))
    .reset_index(name="completed_configs")
)
coverage["config_count"] = coverage["completed_configs"].str.count(",") + 1
coverage["missing_configs"] = coverage["completed_configs"].apply(
    lambda text: ", ".join(sorted(EXPECTED_CONFIGS - set(text.split(", ")))) or "none"
)
summary = summary.merge(coverage, on=["model", "split"], how="left")

detail.to_csv("retriever_comparison_detail.csv", index=False)
summary.to_csv("retriever_comparison_easy_hard_summary.csv", index=False)

display(summary)
print("Saved: retriever_comparison_detail.csv")
print("Saved: retriever_comparison_easy_hard_summary.csv")
