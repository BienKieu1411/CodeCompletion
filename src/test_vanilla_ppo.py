"""CPU contract tests for the learned-critic PPO Kaggle variant."""

import ast
import json
import math
import random
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).parent
PPO_SOURCE = ROOT / "ast_ppo_unixcoder_ppo_kaggle.py"
PPO_TRAIN_NOTEBOOK = ROOT / "repo_level_ast_retrieval_ppo_clip_k10_train_kaggle.ipynb"
RRPO_SOURCE = ROOT / "ast_ppo_unixcoder_kaggle.py"
POLICY_NAMES = {
    "SlateHeads", "state_distribution", "rollout", "terminal_step_rewards",
    "score_ppo_episodes", "prepare_advantages", "ppo_episode_loss",
    "measure_rollout_kl", "ppo_update",
}


def extract(source_path, names, namespace):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    if {node.name for node in nodes} != names:
        raise AssertionError(f"Could not extract all contract functions from {source_path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"),
         namespace)


def load_policy():
    namespace = {
        "torch": torch, "nn": nn, "F": F, "np": np, "math": math,
        "random": random, "DEVICE": torch.device("cpu"),
        "CROSSFILE_TOKEN_BUDGET": 300, "MAX_SLATE_STEPS": 2,
        "PPO_CLIP": 0.2, "ENTROPY_COEF": 0.001,
        "PPO_GAMMA": 1.0, "PPO_GAE_LAMBDA": 0.95,
        "VALUE_COEF": 0.5, "VALUE_CLIP": 0.2,
        "FINAL_UTILITY_WEIGHT": 0.7, "VLLM_MAX_NUM_SEQS": 8,
        "PPO_PASSES": 2, "ACCUMULATION_STEPS": 2,
        "TARGET_KL": 10.0, "MAX_GRAD_NORM": 1.0,
    }
    extract(PPO_SOURCE, POLICY_NAMES, namespace)
    namespace["heads"] = namespace["SlateHeads"](4)
    namespace["encoder"] = nn.Embedding(8, 4)

    def encode_row(_row):
        values = F.normalize(namespace["encoder"](torch.tensor([0, 1, 2, 3])),
                             p=2, dim=-1)
        return values[0], values[1:]

    namespace["encode_row"] = encode_row
    return namespace


class VanillaPPOContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.ns = load_policy()
        self.row = {"candidate_ids": [[1], [2], [3]],
                    "candidate_costs": [100, 120, 200]}

    def test_rollout_saves_old_policy_and_critic_values(self):
        episode = self.ns["rollout"](self.row)
        self.assertGreaterEqual(len(episode["steps"]), 1)
        for step in episode["steps"]:
            self.assertTrue(math.isfinite(step["old_value"]))
            self.assertTrue(math.isfinite(step["old_logp"]))

    def test_terminal_reward_is_attached_only_to_last_action(self):
        rewards = self.ns["terminal_step_rewards"](3, 0.51)
        self.assertEqual(rewards, [0.0, 0.0, 0.51])
        self.assertEqual(self.ns["terminal_step_rewards"](1, 0.27), [0.27])
        with self.assertRaises(ValueError):
            self.ns["terminal_step_rewards"](0, 0.2)
        with self.assertRaises(ValueError):
            self.ns["terminal_step_rewards"](4, 0.2)

        old_score_ns = {"RRPO_IDENTIFIER_WEIGHT": 0.2}
        new_score_ns = {"IDENTIFIER_F1_WEIGHT": 0.2}
        extract(RRPO_SOURCE, {"synthetic_reward"}, old_score_ns)
        extract(PPO_SOURCE, {"synthetic_reward"}, new_score_ns)
        for has_ids in (False, True):
            score = {"edit_similarity": 0.61, "identifier_f1": 0.37,
                     "target_has_identifiers": has_ids}
            old_score_ns["rlcoder_score"] = lambda *args, _score=score: _score
            new_score_ns["rlcoder_score"] = lambda *args, _score=score: _score
            row = {"target_code": "", "left_context": "", "language": "python"}
            self.assertEqual(old_score_ns["synthetic_reward"]("pred", row),
                             new_score_ns["synthetic_reward"]("pred", row))

    def test_ppo_generation_scores_one_final_slate_per_episode(self):
        ns = self.ns
        ns["compose_prompt"] = lambda _row, prefix: tuple(prefix)
        seen = []

        def generate(prompts, progress_label=None):
            seen.extend(prompts)
            return prompts

        ns["generate_completions"] = generate
        ns["synthetic_reward"] = lambda prefix, _row: {
            "utility": 0.2 + 0.1 * len(prefix), "es": 0.2, "id_f1": 0.1}
        episodes = [
            {"row": self.row, "selected": [], "steps": [
                {"selected": (), "action": 3}]},
            {"row": self.row, "selected": [0], "steps": [
                {"selected": (), "action": 0}, {"selected": (0,), "action": 3}]},
            {"row": self.row, "selected": [0, 1], "steps": [
                {"selected": (), "action": 0}, {"selected": (0,), "action": 1}]},
        ]
        count = ns["score_ppo_episodes"](episodes)
        self.assertEqual(count, 3)
        self.assertEqual(seen, [(), (0,), (0, 1)])
        self.assertTrue(np.allclose(episodes[0]["step_rewards"], [0.2]))
        self.assertTrue(np.allclose(episodes[1]["step_rewards"], [0.0, 0.3]))
        self.assertTrue(np.allclose(episodes[2]["step_rewards"], [0.0, 0.4]))
        self.assertTrue(all("prefix_utility" not in episode for episode in episodes))

    def test_gae_uses_learned_critic_and_builds_value_targets(self):
        episodes = [
            {"steps": [{"old_value": 0.2}, {"old_value": 0.3}],
             "step_rewards": [0.4, 0.5]},
            {"steps": [{"old_value": 0.0}, {"old_value": 0.1}],
             "step_rewards": [0.1, 0.1]},
        ]
        self.ns["PPO_GAE_LAMBDA"] = 1.0
        self.ns["prepare_advantages"](episodes)
        self.assertTrue(np.allclose(episodes[0]["raw_advantages"], [0.7, 0.2]))
        self.assertTrue(np.allclose(episodes[0]["returns"], [0.9, 0.5]))
        self.assertTrue(all(math.isfinite(v) for ep in episodes for v in ep["advantages"]))

    def test_value_loss_updates_learned_critic(self):
        ns = self.ns
        episode = ns["rollout"](self.row)
        episode["advantages"] = [1.0] * len(episode["steps"])
        episode["returns"] = [1.0] * len(episode["steps"])
        loss, diagnostics = ns["ppo_episode_loss"](episode)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(diagnostics["value"], 0.0)
        loss.backward()
        critic_grad = ns["heads"].value[-1].weight.grad
        self.assertIsNotNone(critic_grad)
        self.assertGreater(float(critic_grad.abs().sum()), 0.0)

    def test_full_ppo_update_trains_encoder_policy_and_critic(self):
        ns = self.ns
        ns["optimizer"] = torch.optim.AdamW([
            {"params": list(ns["encoder"].parameters()), "lr": 0.002},
            {"params": list(ns["heads"].parameters()), "lr": 0.004},
        ])
        ns["scaler"] = torch.amp.GradScaler("cpu", enabled=False)
        episodes = [ns["rollout"](self.row) for _ in range(6)]
        for episode in episodes:
            final = 0.25 + 0.1 * (0 in episode["selected"])
            episode["step_rewards"] = [0.0] * (len(episode["steps"]) - 1) + [final]
        encoder_before = ns["encoder"].weight.detach().clone()
        critic_before = ns["heads"].value[-1].weight.detach().clone()
        diagnostics = ns["ppo_update"](episodes)
        self.assertEqual(diagnostics["passes"], 2)
        self.assertTrue(math.isfinite(diagnostics["value"]))
        self.assertGreater(float((ns["encoder"].weight.detach() - encoder_before).abs().sum()), 0.0)
        self.assertGreater(float((ns["heads"].value[-1].weight.detach() - critic_before).abs().sum()),
                           0.0)

    def test_notebooks_share_requested_budget_and_distinct_checkpoints(self):
        train = json.loads(PPO_TRAIN_NOTEBOOK.read_text(encoding="utf-8"))
        evaluate = json.loads((ROOT / "ast_ppo_unixcoder_ppo_eval_kaggle.ipynb")
                              .read_text(encoding="utf-8"))
        train_text = "\n".join("".join(cell["source"]) for cell in train["cells"])
        eval_text = "\n".join("".join(cell["source"]) for cell in evaluate["cells"])
        for expected in ("GENERATOR_INPUT_TOKENS = 3072",
                         "CROSSFILE_TOKEN_BUDGET = 2344", "MAX_SLATE_STEPS = 10"):
            self.assertIn(expected, train_text)
            self.assertIn(expected, eval_text)
        self.assertIn("ast_ppo_terminal_v1_k10_ctx3072_xfb2344", train_text)
        self.assertIn('"objective": "ppo_clip_terminal_reward_learned_critic_gae_v1"',
                      train_text)
        self.assertIn('"reward": "terminal_rlcoder_es_identifier_f1_v1"', train_text)
        self.assertIn('"objective": "ppo_clip_terminal_reward_learned_critic_gae_v1"',
                      eval_text)
        self.assertIn('"reward": "terminal_rlcoder_es_identifier_f1_v1"', eval_text)
        self.assertNotIn("reference_values", train_text)


if __name__ == "__main__":
    unittest.main()
