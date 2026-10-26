import base64
import json
from pathlib import Path

DATA_PATH = Path("kaggle_eval_scit/benchmark_tasks_200.json.gz")
NB_PATH = Path("kaggle_eval_scit/eval_scit_benchmark.ipynb")
META_PATH = Path("kaggle_eval_scit/kernel-metadata.json")

# Read binary gzip data and convert to base64
b64_data = base64.b64encode(DATA_PATH.read_bytes()).decode("ascii")

cells = []

# Cell 0: Markdown title
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# SCIT vs AlignCoder Baseline EM & ES Evaluation Benchmark\n",
        "**Target Model:** `deepseek-ai/deepseek-coder-1.3b-base` (Frozen Generator)\n",
        "**Benchmark:** CrossCodeEval (100 Python + 100 Java tasks)\n",
        "**Methods compared:**\n",
        "1. `no_retrieval`: Closed-book in-file baseline\n",
        "2. `aligncoder_baseline`: AlignCoder official retrieved prompt\n",
        "3. `scit_proposed`: Submodular-Supermodular Conditional Information Transport prompt (with companion rescue and duplicate lines removed under matched budget)\n"
    ]
})

# Cell 1: Environment & Setup
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os, sys, json, gzip, base64, time\n",
        "from pathlib import Path\n",
        "import torch\n",
        "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
        "import pandas as pd\n",
        "\n",
        "print('CUDA available:', torch.cuda.is_available())\n",
        "if torch.cuda.is_available():\n",
        "    print('Device name:', torch.cuda.get_device_name(0))\n",
        "    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')\n"
    ]
})

# Cell 2: Embedded Benchmark Data Loader
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        f"EMBEDDED_B64_GZ = \"{b64_data}\"\n",
        "\n",
        "gz_bytes = base64.b64decode(EMBEDDED_B64_GZ)\n",
        "raw_json = gzip.decompress(gz_bytes).decode('utf-8')\n",
        "benchmark_tasks = json.loads(raw_json)\n",
        "print(f'Successfully loaded {len(benchmark_tasks)} benchmark tasks:')\n",
        "py_tasks = [x for x in benchmark_tasks if x['language'] == 'python']\n",
        "ja_tasks = [x for x in benchmark_tasks if x['language'] == 'java']\n",
        "print(f' - Python tasks: {len(py_tasks)}')\n",
        "print(f' - Java tasks:   {len(ja_tasks)}')\n"
    ]
})

# Cell 3: Model Loading
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "MODEL_NAME = 'deepseek-ai/deepseek-coder-1.3b-base'\n",
        "print(f'Loading model {MODEL_NAME} in float16...')\n",
        "t0 = time.time()\n",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)\n",
        "tokenizer.padding_side = 'left'\n",
        "if tokenizer.pad_token is None:\n",
        "    tokenizer.pad_token = tokenizer.eos_token\n",
        "\n",
        "model = AutoModelForCausalLM.from_pretrained(\n",
        "    MODEL_NAME,\n",
        "    torch_dtype=torch.float16,\n",
        "    device_map='cuda:0' if torch.cuda.is_available() else 'cpu',\n",
        "    trust_remote_code=True\n",
        ")\n",
        "model.eval()\n",
        "print(f'Model loaded successfully in {time.time() - t0:.1f}s!')\n"
    ]
})

# Cell 4: Generation & Evaluation Metrics (RepoEval / CrossCodeEval standard)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def levenshtein(s1: str, s2: str) -> int:\n",
        "    if s1 == s2:\n",
        "        return 0\n",
        "    if len(s1) == 0:\n",
        "        return len(s2)\n",
        "    if len(s2) == 0:\n",
        "        return len(s1)\n",
        "    v0 = list(range(len(s2) + 1))\n",
        "    v1 = [0] * (len(s2) + 1)\n",
        "    for i in range(len(s1)):\n",
        "        v1[0] = i + 1\n",
        "        for j in range(len(s2)):\n",
        "            cost = 0 if s1[i] == s2[j] else 1\n",
        "            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)\n",
        "        v0, v1 = v1, [0] * (len(s2) + 1)\n",
        "    return v0[len(s2)]\n",
        "\n",
        "def compute_metrics(prediction: str, target: str):\n",
        "    gt_lines = [l.strip() for l in target.splitlines() if l.strip()]\n",
        "    pred_lines = [l.strip() for l in prediction.splitlines() if l.strip()][:len(gt_lines)]\n",
        "    em = 1.0 if (len(gt_lines) == len(pred_lines) and gt_lines == pred_lines) else 0.0\n",
        "    gt_str = '\\n'.join(gt_lines)\n",
        "    pred_str = '\\n'.join(pred_lines)\n",
        "    max_len = max(len(gt_str), len(pred_str))\n",
        "    if max_len == 0:\n",
        "        es = 100.0\n",
        "    else:\n",
        "        dist = levenshtein(gt_str, pred_str)\n",
        "        es = max(0.0, (1.0 - dist / max_len)) * 100.0\n",
        "    return em, es, pred_str, gt_str\n",
        "\n",
        "def generate_batch(prompts, max_new_tokens=64, batch_size=8):\n",
        "    outputs = []\n",
        "    for i in range(0, len(prompts), batch_size):\n",
        "        batch = prompts[i:i+batch_size]\n",
        "        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=2048)\n",
        "        input_ids = enc['input_ids'].to(model.device)\n",
        "        attention_mask = enc['attention_mask'].to(model.device)\n",
        "        prompt_len = input_ids.shape[1]\n",
        "        \n",
        "        with torch.no_grad():\n",
        "            gen_out = model.generate(\n",
        "                input_ids=input_ids,\n",
        "                attention_mask=attention_mask,\n",
        "                max_new_tokens=max_new_tokens,\n",
        "                do_sample=False,\n",
        "                pad_token_id=tokenizer.eos_token_id\n",
        "            )\n",
        "        \n",
        "        for j in range(len(batch)):\n",
        "            gen_tokens = gen_out[j][prompt_len:]\n",
        "            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)\n",
        "            outputs.append(text)\n",
        "    return outputs\n"
    ]
})

# Cell 5: Run Evaluation Across All 3 Methods
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "methods = ['no_retrieval', 'aligncoder_baseline', 'scit_proposed']\n",
        "prompt_map = {\n",
        "    'no_retrieval': 'prompt_no_retrieval',\n",
        "    'aligncoder_baseline': 'prompt_baseline',\n",
        "    'scit_proposed': 'prompt_scit'\n",
        "}\n",
        "all_records = []\n",
        "\n",
        "print('Starting generation and evaluation across all methods...')\n",
        "t_start = time.time()\n",
        "\n",
        "for method in methods:\n",
        "    pkey = prompt_map[method]\n",
        "    prompts = [task[pkey] for task in benchmark_tasks]\n",
        "    print(f'Generating completions for {method} ({len(prompts)} prompts)...')\n",
        "    t0 = time.time()\n",
        "    preds = generate_batch(prompts, max_new_tokens=64, batch_size=8)\n",
        "    print(f'Completed {method} in {time.time() - t0:.1f}s!')\n",
        "    \n",
        "    for task, pred in zip(benchmark_tasks, preds):\n",
        "        em, es, c_pred, c_targ = compute_metrics(pred, task['target'])\n",
        "        all_records.append({\n",
        "            'task_id': task['task_id'],\n",
        "            'dataset': task['dataset'],\n",
        "            'language': task['language'],\n",
        "            'method': method,\n",
        "            'target': task['target'],\n",
        "            'prediction': pred,\n",
        "            'clean_prediction': c_pred,\n",
        "            'clean_target': c_targ,\n",
        "            'exact_match': em,\n",
        "            'edit_similarity': es,\n",
        "            'has_outside_rescue': task['has_outside_rescue'],\n",
        "            'draft_had_em1': task['draft_had_em1'],\n",
        "        })\n",
        "\n",
        "print(f'All evaluations completed in {time.time() - t_start:.1f}s!')\n"
    ]
})

# Cell 6: Summarize & Display Results Table
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "df = pd.DataFrame(all_records)\n",
        "\n",
        "# Save full predictions and CSV summary\n",
        "out_dir = Path('/kaggle/working')\n",
        "df.to_json(out_dir / 'scit_eval_predictions.jsonl', orient='records', lines=True)\n",
        "\n",
        "summary = df.groupby(['language', 'method']).agg(\n",
        "    samples=('task_id', 'count'),\n",
        "    exact_match=('exact_match', lambda x: f'{x.mean()*100:.2f}%'),\n",
        "    edit_similarity=('edit_similarity', lambda x: f'{x.mean():.2f}'),\n",
        "    rescued_exact_match=('exact_match', lambda x: f'{x[df.loc[x.index, \"has_outside_rescue\"]].mean()*100:.2f}%' if df.loc[x.index, \"has_outside_rescue\"].sum() > 0 else 'N/A')\n",
        ").reset_index()\n",
        "\n",
        "summary.to_csv(out_dir / 'scit_benchmark_summary.csv', index=False)\n",
        "\n",
        "print('\\n' + '='*85)\n",
        "print('FINAL CCEVAL BENCHMARK EVALUATION RESULTS (deepseek-coder-1.3b-base)')\n",
        "print('='*85)\n",
        "print(summary.to_string(index=False))\n",
        "print('='*85)\n",
        "\n",
        "# Compute deltas for Python and Java\n",
        "for lang in ['python', 'java']:\n",
        "    sub = df[df['language'] == lang]\n",
        "    b_em = sub[sub['method'] == 'aligncoder_baseline']['exact_match'].mean() * 100\n",
        "    b_es = sub[sub['method'] == 'aligncoder_baseline']['edit_similarity'].mean()\n",
        "    s_em = sub[sub['method'] == 'scit_proposed']['exact_match'].mean() * 100\n",
        "    s_es = sub[sub['method'] == 'scit_proposed']['edit_similarity'].mean()\n",
        "    print(f'\\n>>> [{lang.upper()}] SCIT vs AlignCoder Baseline:')\n",
        "    print(f'    - Exact Match (EM):     {s_em:.2f}% vs {b_em:.2f}%  ({s_em - b_em:+.2f}%)')\n",
        "    print(f'    - Edit Similarity (ES): {s_es:.2f} vs {b_es:.2f}  ({s_es - b_es:+.2f})')\n",
        "    \n",
        "    # Check specifically on tasks where outside_top10 companions were rescued\n",
        "    res_sub = sub[sub['has_outside_rescue']]\n",
        "    res_b_em = res_sub[res_sub['method'] == 'aligncoder_baseline']['exact_match'].mean() * 100\n",
        "    res_s_em = res_sub[res_sub['method'] == 'scit_proposed']['exact_match'].mean() * 100\n",
        "    print(f'    - Rescue Subset EM:     {res_s_em:.2f}% vs {res_b_em:.2f}%  ({res_s_em - res_b_em:+.2f}%)')\n"
    ]
})

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print(f"Created {NB_PATH} ({NB_PATH.stat().st_size / 1024:.1f} KB)")

# Create kernel-metadata.json
metadata = {
    "id": "bienkieu/eval-scit-aligncoder",
    "title": "Eval SCIT AlignCoder EM-ES Benchmark",
    "code_file": "eval_scit_benchmark.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Created {META_PATH}")
