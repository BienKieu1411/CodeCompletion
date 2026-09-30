"""Lightweight tests for the terminal PPO harmful-selection notebook variant."""

from __future__ import annotations

import ast
import json
import statistics
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "ast_ppo_unixcoder_terminal_bad_pick_k10_modal_kaggle.ipynb"
ENDPOINT = ROOT / "modal_deepseek_vllm_ppo_terminal_endpoint.py"


def notebook_cells():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return payload["cells"]


def load_top_level_function(name: str, namespace: dict):
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook_cells()
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)


class TerminalBadPickRewardTests(unittest.TestCase):
    def scorer(self, utilities):
        namespace = {
            "MAX_SLATE_STEPS": 10,
            "WRONG_PICK_PENALTY_WEIGHT": 0.5,
            "WRONG_PICK_PENALTY_CAP": 1.0,
            "WRONG_PICK_MIN_GAIN": 0.02,
            "VLLM_MAX_NUM_SEQS": 108,
            "np": SimpleNamespace(mean=statistics.fmean),
            "compose_prompt": lambda row, selected: ",".join(
                row["candidate_texts"][index] for index in selected
            ),
            "generate_completions": lambda prompts, progress_label=None: [
                utilities[prompt] for prompt in prompts
            ],
            "synthetic_reward": lambda value, _row: {
                "utility": float(value), "es": float(value), "id_f1": float(value)
            },
            "print": lambda *args, **kwargs: None,
        }
        load_top_level_function("terminal_step_rewards", namespace)
        load_top_level_function("score_ppo_episodes", namespace)
        return namespace["score_ppo_episodes"]

    def test_only_harmful_leave_one_out_picks_are_penalized(self):
        episode = {
            "row": {"candidate_texts": ["A", "B"]},
            "selected": [0, 1],
            "steps": [{}, {}, {}],
        }
        scorer = self.scorer({"A,B": 0.50, "B": 0.45, "A": 0.65})
        self.assertEqual(scorer([episode]), 3)
        self.assertAlmostEqual(episode["wrong_pick_penalty"], 0.085)
        self.assertAlmostEqual(episode["reward"], 0.415)
        self.assertEqual(episode["harmful_selection_count"], 1)
        self.assertEqual(episode["subthreshold_selection_count"], 1)
        self.assertEqual(episode["step_rewards"], [0.0, 0.0, 0.415])

    def test_empty_slate_has_no_candidate_penalty(self):
        episode = {
            "row": {"candidate_texts": []},
            "selected": [],
            "steps": [{"stop": True}],
        }
        scorer = self.scorer({"": 0.2})
        self.assertEqual(scorer([episode]), 1)
        self.assertEqual(episode["wrong_pick_penalty"], 0.0)
        self.assertEqual(episode["step_rewards"], [0.2])

    def test_redundant_pick_gets_small_penalty_to_teach_stop_count(self):
        episode = {
            "row": {"candidate_texts": ["A"]},
            "selected": [0],
            "steps": [{}, {}],
        }
        scorer = self.scorer({"A": 0.5, "": 0.5})
        scorer([episode])
        self.assertEqual(episode["harmful_selection_count"], 0)
        self.assertEqual(episode["subthreshold_selection_count"], 1)
        self.assertAlmostEqual(episode["wrong_pick_penalty"], 0.01)
        self.assertAlmostEqual(episode["reward"], 0.49)

    def test_summed_harm_is_capped_before_weighting(self):
        episode = {
            "row": {"candidate_texts": ["A", "B", "C"]},
            "selected": [0, 1, 2],
            "steps": [{}, {}, {}, {}],
        }
        scorer = self.scorer({
            "A,B,C": 0.2,
            "B,C": 0.8,
            "A,C": 0.8,
            "A,B": 0.8,
        })
        scorer([episode])
        self.assertEqual(episode["wrong_pick_penalty"], 0.5)
        self.assertAlmostEqual(episode["reward"], -0.3)
        self.assertEqual(episode["harmful_selection_count"], 3)

    def test_training_config_and_failure_handler_are_explicit(self):
        cells = notebook_cells()
        text = "\n".join("".join(cell.get("source", [])) for cell in cells)
        self.assertIn("ROLLOUT_QUERIES = 64", text)
        self.assertIn('"encoder_lr", "rollout_queries"', text)
        self.assertIn("encoder.gradient_checkpointing_disable()", text)
        self.assertNotIn("encoder.gradient_checkpointing_enable(", text)
        self.assertIn("MAX_SLATE_STEPS = 10", text)
        self.assertIn("VLLM_MAX_NUM_SEQS = 108", text)
        self.assertIn("GENERATOR_INPUT_TOKENS = 3072", text)
        self.assertIn("CROSSFILE_TOKEN_BUDGET = 2344", text)
        self.assertIn("MAX_TRAIN_EPOCHS = 2", text)
        self.assertIn("WRONG_PICK_MIN_GAIN = 0.02", text)
        self.assertIn("bienkieu-ppo-terminal-badpick-deepseek-vll-add36f", text)
        self.assertNotIn("bienkieu-ppo-terminal-badpick-deepseek-vllm-add36f", text)
        self.assertNotIn("REPLACE_WITH_DEPLOYED_MODAL_ENDPOINT", text)
        self.assertIn("except BaseException:", text)
        self.assertIn("save_checkpoint(CHECKPOINT_PATH)", text)
        self.assertIn("actor→encoder gradient", text)
        self.assertIn("scaler.scale(loss / len(batch)).backward()", text)
        self.assertNotIn("cceval/python/test.parquet", text)
        self.assertNotIn("repoeval/line_level/test_0.parquet", text)

    def test_epoch_cap_waits_for_two_complete_passes_and_resumes_partial_epoch(self):
        namespace = {
            "STATE": {"epoch": 1, "cursor": 100},
            "MAX_TRAIN_EPOCHS": 2,
            "TRAIN_EXAMPLES": 100,
        }
        load_top_level_function("max_epoch_run_complete", namespace)
        complete = namespace["max_epoch_run_complete"]
        self.assertFalse(complete())  # Epoch 1 is complete; epoch 2 must run.
        namespace["STATE"] = {"epoch": 2, "cursor": 99}
        self.assertFalse(complete())  # Resume the remaining row of epoch 2.
        namespace["STATE"] = {"epoch": 2, "cursor": 100}
        self.assertTrue(complete())

    def test_dedicated_modal_endpoint_context_and_batch(self):
        tree = ast.parse(ENDPOINT.read_text(encoding="utf-8"))
        values = {
            target.id: node.value
            for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        self.assertEqual(ast.literal_eval(values["INPUT_TOKENS"]), 3072)
        self.assertEqual(ast.literal_eval(values["OUTPUT_TOKENS"]), 96)
        self.assertEqual(ast.literal_eval(values["MAX_NUM_SEQS"]), 108)
        scaledown = eval(
            compile(ast.Expression(values["SCALEDOWN_WINDOW_SECONDS"]),
                    str(ENDPOINT), "eval"),
            {"__builtins__": {}},
        )
        self.assertEqual(scaledown, 300)
        compile(ENDPOINT.read_text(encoding="utf-8"), str(ENDPOINT), "exec")


if __name__ == "__main__":
    unittest.main()
