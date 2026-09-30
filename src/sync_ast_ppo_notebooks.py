"""Keep the two self-contained Kaggle notebooks' data/prompt code identical.

Run with the project's Python environment after editing the training .py source.
This performs only mechanical synchronization; it never executes notebook cells.
"""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).parent
TRAIN = ROOT / "ast_ppo_unixcoder_kaggle.py"
EVAL = ROOT / "ast_ppo_unixcoder_eval_kaggle.py"
SHARED_FUNCTIONS = (
    "ast_chunks", "left_anchors", "suffix_with_token_budget",
    "pack_left_context", "retrieval_query", "render_chunk", "unixcoder_ids",
    "build_row", "compose_prompt", "rlcoder_postprocess_completion",
    "identifier_match_f1", "rlcoder_score", "rlcoder_repo_macro", "rlcoder_summary",
    "encode_row", "verify_served_model",
)


def function_span(source, name):
    matches = [node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        return None
    node = matches[0]
    return node.lineno - 1, node.end_lineno


def sync_shared_functions():
    training = TRAIN.read_text(encoding="utf-8")
    evaluation = EVAL.read_text(encoding="utf-8")
    train_lines = training.splitlines(keepends=True)
    for name in SHARED_FUNCTIONS:
        span = function_span(training, name)
        if span is None:
            raise ValueError(f"Missing train function: {name}")
        replacement = "".join(train_lines[span[0]:span[1]])
        eval_span = function_span(evaluation, name)
        if eval_span is None:
            marker = "def decode_crossfile_context(value):\n"
            if marker not in evaluation:
                raise ValueError("Cannot locate eval data-helper insertion point")
            evaluation = evaluation.replace(marker, replacement + "\n" + marker, 1)
        else:
            lines = evaluation.splitlines(keepends=True)
            lines[eval_span[0]:eval_span[1]] = [replacement]
            evaluation = "".join(lines)
    compile(evaluation, str(EVAL), "exec")
    EVAL.write_text(evaluation, encoding="utf-8")


def source_cells(source):
    cells, kind, lines = [], None, []

    def finish():
        if kind is None:
            return
        if kind == "markdown":
            rendered = [line[2:] if line.startswith("# ") else
                        line[1:] if line.startswith("#") else line
                        for line in lines]
            cells.append((kind, rendered))
        else:
            cells.append((kind, list(lines)))

    for line in source.splitlines(keepends=True):
        if line.startswith("# %%"):
            finish()
            kind = "markdown" if "[markdown]" in line else "code"
            lines = []
        elif kind is not None:
            lines.append(line)
    finish()
    return cells


def sync_notebook(py_path):
    notebook_path = py_path.with_suffix(".ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    old_cells = notebook["cells"]
    parsed = source_cells(py_path.read_text(encoding="utf-8"))
    if py_path == TRAIN:
        # Older exported notebooks kept the reward-variance pilot in its own
        # cell. The current source folds startup invariants into training setup
        # and deliberately has no reward-generation pilot.
        obsolete = [cell for cell in old_cells
                    if "def reward_variance_pilot(" in "".join(cell.get("source", []))]
        if obsolete:
            old_cells = [cell for cell in old_cells if cell not in obsolete]
            notebook["cells"] = old_cells
    if len(parsed) != len(old_cells):
        raise ValueError(f"Cell count changed for {notebook_path}: "
                         f"{len(old_cells)} -> {len(parsed)}")
    new_cells = []
    for (kind, lines), old in zip(parsed, old_cells):
        if old["cell_type"] != kind:
            raise ValueError(f"Cell type changed in {notebook_path}")
        cell = dict(old)
        cell["source"] = lines
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            compile("".join(lines), str(notebook_path), "exec")
        new_cells.append(cell)
    notebook["cells"] = new_cells
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")


if __name__ == "__main__":
    sync_shared_functions()
    sync_notebook(TRAIN)
    sync_notebook(EVAL)
    print("Synchronized shared data/prompt functions and both Kaggle notebooks")
