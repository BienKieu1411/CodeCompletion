"""E1 on Modal (L4): label (e1_label.py, sharded) -> train (e1_train.py, 3 variants in parallel) -> eval (e1_eval.py).

All outputs on the Volume `dep-oracle-probe`: /vol/e1_labels, /vol/e1_models/<variant>, /vol/e1eval_<dataset>.
The Phase-2 pilot labels (needed by eval) are shipped from modal_out/pilot_<dataset>/chunks.jsonl.

usage (from this folder, with the Modal venv):
    modal run --detach modal_e1.py --stage smoke           # 8 train + 8 dev examples, 1 tiny training step per variant
    modal run --detach modal_e1.py --stage label           # 4000 train + 300 dev, 4 + 1 shards
    modal run --detach modal_e1.py --stage train
    modal run --detach modal_e1.py --stage eval
    modal volume get dep-oracle-probe e1eval_repoeval ./modal_out/
"""
import os

import modal

from modal_probe import CANDS, HERE, REMOTE, _run_committed, image, vol  # noqa: F401

img = image
for f in ["e1_label.py", "e1_train.py", "e1_eval.py"]:
    img = img.add_local_file(os.path.join(HERE, f), f"{REMOTE}/{f}")
for ds in CANDS:
    img = img.add_local_file(os.path.join(HERE, "modal_out", f"pilot_{ds}", "chunks.jsonl"),
                             f"{REMOTE}/pilot_{ds}_chunks.jsonl")
img = img.add_local_python_source("modal_probe")
app = modal.App("dep-oracle-e1", image=img)
DATA = "/vol/Data4RLCoder"


def _data():
    from huggingface_hub import snapshot_download
    if not os.path.exists(f"{DATA}/github_repos/python/train.parquet"):
        snapshot_download("nov3630/Data4RLCoder", repo_type="dataset", local_dir=DATA,
                          allow_patterns=["*github_repos/python/train.parquet"])
        vol.commit()
    import glob
    hit = glob.glob(f"{DATA}/**/github_repos/python/train.parquet", recursive=True)
    return os.path.dirname(os.path.dirname(os.path.dirname(hit[0])))


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=6 * 3600, cpu=4.0, memory=16384)
def label(split: str, n: int, shard: int, nshards: int, out: str = "/vol/e1_labels", hard: bool = False) -> int:
    root = _data()
    cmd = ["python", f"{REMOTE}/e1_label.py", "--data_root", root, "--split", split, "--n", str(n),
           "--shard", str(shard), "--nshards", str(nshards), "--out", out,
           "--eval_cands", ",".join(f"{REMOTE}/{c}" for c in CANDS.values())] + (["--hard"] if hard else [])
    return _run_committed(cmd)


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=6 * 3600, cpu=4.0, memory=16384)
def train(variant: str, labels: str = "/vol/e1_labels", out: str = "/vol/e1_models", epochs: int = 2,
          dev_labels: str = "", min_top: float = -1e9) -> int:
    cmd = ["python", f"{REMOTE}/e1_train.py", "--labels", labels, "--variant", variant, "--out", out,
           "--epochs", str(epochs), "--min_top", str(min_top)] + (["--dev_labels", dev_labels] if dev_labels else [])
    return _run_committed(cmd)


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=4 * 3600, cpu=4.0, memory=16384)
def evaluate(dataset: str, models: str = "/vol/e1_models", tag: str = "e1eval", limit: int = 0,
             gen_budgets: str = "512,1024") -> int:
    cmd = ["python", f"{REMOTE}/e1_eval.py", "run", "--dataset", dataset, "--cands", f"{REMOTE}/{CANDS[dataset]}",
           "--labels", f"{REMOTE}/pilot_{dataset}_chunks.jsonl", "--models", models,
           "--data_dir", "/vol/aligncoder_data", "--out", f"/vol/{tag}_{dataset}", "--gen_budgets", gen_budgets]
    if limit:
        cmd += ["--limit", str(limit)]
    return _run_committed(cmd)


@app.local_entrypoint()
def main(stage: str = "smoke", nshards: int = 4, n_train: int = 4000, n_dev: int = 300):
    variants = ["bi_soft", "bi_onehot", "cross_soft"]
    if stage == "smoke":
        print(label.remote("train", 8, 0, 1, "/vol/e1s_labels"), label.remote("dev", 8, 0, 1, "/vol/e1s_labels"))
        print([c.get() for c in [train.spawn(v, "/vol/e1s_labels", "/vol/e1s_models", 1) for v in variants]])
        print([c.get() for c in [evaluate.spawn(ds, "/vol/e1s_models", "e1s_eval", 4) for ds in CANDS]])
    elif stage == "label":
        calls = [label.spawn("train", n_train, i, nshards) for i in range(nshards)] + [label.spawn("dev", n_dev, 0, 1)]
        print([c.get() for c in calls])
    elif stage == "train":
        print([c.get() for c in [train.spawn(v) for v in variants]])
    elif stage == "eval":
        print([c.get() for c in [evaluate.spawn(ds) for ds in CANDS]])
    # E1b: CrossCodeEval-style hard cuts, train on hard + old labels with best dLL >= 1
    elif stage == "label_hard":
        calls = ([label.spawn("train", n_train, i, nshards, "/vol/e1h_labels", True) for i in range(nshards)]
                 + [label.spawn("dev", n_dev, 0, 1, "/vol/e1h_labels", True)])
        print([c.get() for c in calls])
    elif stage == "train_hard":
        print([c.get() for c in [train.spawn(v, "/vol/e1h_labels,/vol/e1_labels", "/vol/e1h_models", 2,
                                             "/vol/e1h_labels", 1.0) for v in variants]])
    elif stage == "eval_hard":
        print([c.get() for c in [evaluate.spawn(ds, "/vol/e1h_models", "e1heval") for ds in CANDS]])
    elif stage == "eval_hard_1536":  # CCEval's best fixed budget is 1536; scores are already cached
        print([c.get() for c in [evaluate.spawn(ds, "/vol/e1h_models", "e1heval", 0, "1536") for ds in CANDS]])
