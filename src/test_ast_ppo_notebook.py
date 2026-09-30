"""CPU-only contract tests for the self-contained Kaggle PPO notebook.

The notebook itself is not imported because its top-level cells require Kaggle GPUs,
the dataset and a vLLM server. These tests execute its actual policy definitions.
"""

import ast
from collections import defaultdict
import hashlib
import json
import keyword
import math
import os
import random
import re
import sqlite3
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE = Path(__file__).with_name("ast_ppo_unixcoder_kaggle.py")
NAMES = {"SlateHeads", "state_distribution", "rollout",
         "bm25_reference_completion", "rrpo_step_rewards", "score_rrpo_episodes",
         "prepare_advantages", "ppo_episode_loss", "measure_rollout_kl",
         "ppo_update", "checkpoint_payload",
         "save_checkpoint", "load_checkpoint", "find_resume_checkpoint",
         "validation_signature_changed", "resume_signature_compatible"}


def load_policy():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NAMES]
    assert {node.name for node in nodes} == NAMES
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np, "math": math,
                 "random": random, "os": os, "Path": Path,
                 "DEVICE": torch.device("cpu"), "CROSSFILE_TOKEN_BUDGET": 300,
                 "VLLM_MAX_NUM_SEQS": 8,
                 "MAX_SLATE_STEPS": 2, "PPO_CLIP": 0.2,
                 "ENTROPY_COEF": 0.001,
                 "PPO_PASSES": 2, "ACCUMULATION_STEPS": 2,
                 "TARGET_KL": 10.0, "MAX_GRAD_NORM": 1.0,
                 "RRPO_GAMMA": 1.0, "RRPO_GAE_LAMBDA": 0.95,
                 "VALIDATION_SIGNATURE_KEYS": {
                     "valid", "valid_examples_sha256", "target_chunk_tokens",
                     "min_target_tokens", "min_prefix_tokens",
                     "min_left_context_lines", "preferred_file_lines",
                     "preferred_file_chars", "target_cut_distribution"},
                 "RESUMABLE_TUNING_KEYS": {
                     "encode_batch_size", "accumulation_steps", "encoder_lr"},
                 "ENCODER_LR": 0.001, "HEAD_LR": 0.001,
                 "RRPO_FINAL_WEIGHT": 0.7}
    namespace["RESUME_CHECKPOINT_DIR"] = Path(
        "/kaggle/input/datasets/bienkieu/resume-checkpoint"
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    namespace["heads"] = namespace["SlateHeads"](4)
    namespace["encoder"] = nn.Embedding(8, 4)
    # The policy functions checkpoint the underlying HF encoder module.
    namespace["encoder_core"] = namespace["encoder"]

    def encode_row(_row):
        values = namespace["encoder"](torch.tensor([0, 1, 2, 3]))
        values = F.normalize(values, p=2, dim=-1)
        return values[0], values[1:]

    namespace["encode_row"] = encode_row
    return namespace


class PPOContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.ns = load_policy()
        self.row = {"candidate_ids": [[1], [2], [3]],
                    "candidate_costs": [100, 120, 200]}

    def test_rollout_support_and_old_current_identity(self):
        episode = self.ns["rollout"](self.row)
        selected = episode["selected"]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertLessEqual(sum(self.row["candidate_costs"][i] for i in selected), 300)
        query, candidates = self.ns["encode_row"](self.row)
        for step in episode["steps"]:
            log_probs, _, valid = self.ns["state_distribution"](
                self.row, query, candidates, step["selected"], step["remaining"]
            )
            self.assertEqual(valid, step["valid"])
            self.assertAlmostEqual(float(log_probs[step["action"]].detach()),
                                   step["old_logp"], places=6)
            self.assertTrue(valid[-1])  # STOP always exists.

    def test_ppo_loss_updates_encoder_and_ratio(self):
        episode = self.ns["rollout"](self.row)
        episode["reward"] = 0.8
        episode["advantages"] = [1.0] * len(episode["steps"])
        optimizer = torch.optim.AdamW(
            list(self.ns["encoder"].parameters()) + list(self.ns["heads"].parameters()),
            lr=0.03,
        )
        loss, diagnostics = self.ns["ppo_episode_loss"](episode)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(self.ns["measure_rollout_kl"]([episode]), 0.0, places=5)
        loss.backward()
        grad = self.ns["encoder"].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)
        optimizer.step()
        self.assertGreater(self.ns["measure_rollout_kl"]([episode]), 1e-6)

    def test_clipped_surrogate_handles_both_advantage_signs(self):
        ratio = torch.tensor([1.5, 0.5])
        advantage = torch.tensor([1.0, -1.0])
        surrogate = torch.minimum(ratio * advantage,
                                  ratio.clamp(0.8, 1.2) * advantage)
        self.assertTrue(torch.allclose(surrogate, torch.tensor([1.2, -0.8])))

    def test_two_pass_update_reuses_rollouts_and_changes_encoder(self):
        episodes = [self.ns["rollout"](self.row) for _ in range(4)]
        for episode in episodes:
            reward = 0.2 + (0.3 if 0 in episode["selected"] else 0.0)
            episode["step_rewards"] = [0.0] * (len(episode["steps"]) - 1) + [reward]
            episode["reference_values"] = [0.1] * len(episode["steps"])
        self.ns["optimizer"] = torch.optim.AdamW([
            {"params": list(self.ns["encoder"].parameters()), "lr": 0.001},
            {"params": list(self.ns["heads"].parameters()), "lr": 0.004},
        ])
        self.ns["scaler"] = torch.amp.GradScaler("cpu", enabled=False)
        before = self.ns["encoder"].weight.detach().clone()
        diagnostics = self.ns["ppo_update"](episodes)
        self.assertEqual(diagnostics["passes"], 2)
        self.assertTrue(np.isfinite(diagnostics["kl"]))
        self.assertAlmostEqual(diagnostics["kl"],
                               self.ns["measure_rollout_kl"](episodes), places=6)
        self.assertGreater(float((self.ns["encoder"].weight.detach() - before).abs().sum()),
                           0.0)


    def test_checkpoint_restores_optimizer_cursor_and_rng(self):
        self.ns["DATA_SIGNATURE"] = {"dataset": "tiny"}
        self.ns["STATE"] = {"episodes_seen": 12, "epoch": 2, "cursor": 4,
                            "order": [3, 2, 1, 0], "best_es": 0.4,
                            "last_save_elapsed": 900.0}
        self.ns["optimizer"] = torch.optim.AdamW([
            {"params": list(self.ns["encoder"].parameters()), "lr": 0.001},
            {"params": list(self.ns["heads"].parameters()), "lr": 0.004},
        ])
        self.ns["scaler"] = torch.amp.GradScaler("cpu", enabled=False)
        self.ns["encoder"].weight.square().sum().backward()
        self.ns["optimizer"].step()
        self.ns["optimizer"].zero_grad(set_to_none=True)
        parameter = self.ns["encoder"].weight
        saved_moment = self.ns["optimizer"].state[parameter]["exp_avg"].clone()
        random.seed(31)
        np.random.seed(31)
        torch.manual_seed(31)
        before = self.ns["encoder"].weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            self.ns["save_checkpoint"](path)
            expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
            with torch.no_grad():
                self.ns["encoder"].weight.add_(1.0)
            self.ns["STATE"]["cursor"] = 999
            self.ns["optimizer"].param_groups[0]["lr"] = 0.1
            self.ns["ENCODER_LR"] = 0.002
            self.ns["HEAD_LR"] = 0.003
            self.ns["load_checkpoint"](path)
            self.assertTrue(torch.allclose(self.ns["encoder"].weight, before))
            self.assertEqual(self.ns["STATE"]["cursor"], 4)
            self.assertEqual(self.ns["STATE"]["order"], [3, 2, 1, 0])
            self.assertEqual(self.ns["STATE"]["last_save_elapsed"], 0.0)
            self.assertEqual(self.ns["optimizer"].param_groups[0]["lr"], 0.002)
            self.assertEqual(self.ns["optimizer"].param_groups[1]["lr"], 0.003)
            self.assertTrue(torch.allclose(
                self.ns["optimizer"].state[parameter]["exp_avg"], saved_moment
            ))
            observed = (random.random(), float(np.random.rand()), float(torch.rand(1)))
            self.assertEqual(observed, expected)
            self.ns["DATA_SIGNATURE"] = {"objective": "terminal_ppo_v4"}
            with self.assertRaisesRegex(RuntimeError, "RRPO objective"):
                self.ns["load_checkpoint"](path)

    def test_resume_source_explicit_and_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.ns.update({"WORK_DIR": root,
                            "RESUME_CHECKPOINT_DIR": root / "resume-checkpoint",
                            "RESUME_CHECKPOINT_PATH": None,
                            "AUTO_DISCOVER_RESUME": False})
            self.assertIsNone(self.ns["find_resume_checkpoint"]())
            local = root / "latest.pt"
            local.write_bytes(b"checkpoint")
            self.assertIsNone(self.ns["find_resume_checkpoint"]())
            self.ns["AUTO_DISCOVER_RESUME"] = True
            self.assertEqual(self.ns["find_resume_checkpoint"](), local)
            local.unlink()
            resume_dir = root / "resume-checkpoint"
            resume_dir.mkdir()
            extensionless = resume_dir / "latest"
            extensionless.write_bytes(b"checkpoint")
            self.assertEqual(self.ns["find_resume_checkpoint"](), extensionless)
            local.write_bytes(b"stale local checkpoint")
            self.assertEqual(self.ns["find_resume_checkpoint"](), extensionless)
            self.ns["RESUME_CHECKPOINT_PATH"] = resume_dir / "latest.pt"
            self.assertEqual(self.ns["find_resume_checkpoint"](), extensionless)
            explicit = root / "attached.pt"
            explicit.write_bytes(b"attached")
            self.ns["RESUME_CHECKPOINT_PATH"] = str(explicit)
            self.assertEqual(self.ns["find_resume_checkpoint"](), explicit)

    def test_resume_recovers_newer_best_after_interrupted_latest_save(self):
        self.ns["DATA_SIGNATURE"] = {"objective": "rrpo_tiny"}
        self.ns["STATE"] = {"episodes_seen": 12, "epoch": 1, "cursor": 12,
                            "order": list(range(32)), "best_es": 0.4,
                            "last_save_elapsed": 0.0}
        self.ns["optimizer"] = torch.optim.AdamW(
            list(self.ns["encoder"].parameters()) + list(self.ns["heads"].parameters())
        )
        self.ns["scaler"] = torch.amp.GradScaler("cpu", enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, best = root / "latest.pt", root / "best.pt"
            self.ns["save_checkpoint"](latest)
            with torch.no_grad():
                self.ns["encoder"].weight.add_(0.25)
            newer_weights = self.ns["encoder"].weight.detach().clone()
            self.ns["STATE"].update({"episodes_seen": 16, "cursor": 16,
                                     "best_es": 0.6})
            self.ns["save_checkpoint"](best)
            with torch.no_grad():
                self.ns["encoder"].weight.add_(1.0)
            self.ns["load_checkpoint"](latest)
            self.assertEqual(self.ns["STATE"]["episodes_seen"], 16)
            self.assertEqual(self.ns["STATE"]["best_es"], 0.6)
            self.assertTrue(torch.allclose(self.ns["encoder"].weight, newer_weights))
            self.ns["STATE"].update({"episodes_seen": 20, "cursor": 20})
            self.ns["save_checkpoint"](latest)
            self.ns["load_checkpoint"](latest)
            self.assertEqual(self.ns["STATE"]["episodes_seen"], 20)


class CompletionCacheContractTest(unittest.TestCase):
    def test_exact_prompts_are_deduplicated_and_persisted(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"completion_cache_key", "generate_completions"}
        nodes = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in nodes}, names)
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE generation_cache (cache_key TEXT PRIMARY KEY, "
            "completion TEXT NOT NULL)"
        )

        class Tokenizer:
            @staticmethod
            def encode(text, add_special_tokens=True):
                return list(text.encode("utf-8"))

        class Response:
            ok = True
            status_code = 200
            text = ""

            def __init__(self, prompts):
                self.prompts = prompts

            def json(self):
                return {"choices": [
                    {"index": index,
                     "text": "completion:" + bytes(prompt).decode("utf-8")}
                    for index, prompt in enumerate(self.prompts)
                ]}

        class Requests:
            class RequestException(Exception):
                pass

            def __init__(self):
                self.calls = []
                self.fail_next = False

            def post(self, _url, headers, json, timeout):
                self.calls.append((json, timeout))
                if self.fail_next:
                    self.fail_next = False
                    raise self.RequestException("Response ended prematurely")
                return Response(json["prompt"])

        requests = Requests()
        clock = SimpleNamespace(monotonic=time.monotonic, sleep=lambda _seconds: None)
        namespace = {
            "hashlib": hashlib, "json": json, "math": math,
            "GENERATION_CACHE_SIGNATURE": {"model": "frozen-test", "temp": 0.0},
            "GENERATION_CACHE_CONNECTION": connection,
            "VLLM_MAX_NUM_SEQS": 8, "GEN_TOKENIZER": Tokenizer(),
            "GENERATOR_INPUT_TOKENS": 100, "GENERATOR_OUTPUT_TOKENS": 16,
            "API_BASE": "http://local/v1", "SERVED_MODEL_NAME": "test",
            "requests": requests, "time": clock,
            "TrainingDeadlineReached": type("TrainingDeadlineReached",
                                             (RuntimeError,), {}),
            "GeneratorRequestError": type("GeneratorRequestError",
                                           (RuntimeError,), {}),
            "modal_last_request_deadline": lambda: time.monotonic() + 3600,
            "MODAL_AUTH_HEADERS": {},
            "GENERATOR_REQUEST_TIMEOUT_SECONDS": 60,
            "ENDPOINT_RECOVERY_TIMEOUT_SECONDS": 30,
            "GENERATOR_RETRY_BACKOFF_SECONDS": 0.001,
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"),
             namespace)
        generate = namespace["generate_completions"]

        first = generate(["alpha", "beta", "alpha"])
        self.assertEqual(first, ["completion:alpha", "completion:beta",
                                 "completion:alpha"])
        self.assertEqual(len(requests.calls), 1)
        self.assertEqual(len(requests.calls[0][0]["prompt"]), 2)

        second = generate(["beta", "gamma"])
        self.assertEqual(second, ["completion:beta", "completion:gamma"])
        self.assertEqual(len(requests.calls), 2)
        self.assertEqual(connection.execute(
            "SELECT COUNT(*) FROM generation_cache").fetchone()[0], 3)
        requests.fail_next = True
        third = generate(["delta"])
        self.assertEqual(third, ["completion:delta"])
        self.assertEqual(len(requests.calls), 4)  # One interrupted call, then retry.
        self.assertEqual(connection.execute(
            "SELECT COUNT(*) FROM generation_cache").fetchone()[0], 4)
        namespace["modal_last_request_deadline"] = lambda: time.monotonic() - 1
        with self.assertRaises(namespace["TrainingDeadlineReached"]):
            generate(["after-cutoff"])
        self.assertEqual(len(requests.calls), 4)
        connection.close()


class RRPOContractTest(unittest.TestCase):
    def setUp(self):
        self.ns = load_policy()
        self.row = {"candidate_ids": [[1], [2], [3]],
                    "candidate_costs": [100, 120, 200]}

    def test_stop_fill_and_final_dominant_objective(self):
        rewards = self.ns["rrpo_step_rewards"]
        self.assertEqual(rewards([], True, 0.4), [0.4])
        self.assertAlmostEqual(sum(rewards([0.8], True, 0.8)), 0.8)
        self.assertAlmostEqual(sum(rewards([0.8, 0.2], False, 0.2)), 0.29)
        with self.assertRaises(ValueError):
            rewards([0.8], False, 0.8)
        self.ns["MAX_SLATE_STEPS"] = 3
        self.assertAlmostEqual(sum(rewards([0.9, 0.5, 0.1], False, 0.1)), 0.22)
        self.assertAlmostEqual(sum(rewards([0.8], True, 0.8)), 0.8)

    def test_ten_slot_return_matches_final_plus_prefix_formula(self):
        self.ns["MAX_SLATE_STEPS"] = 10
        for count in range(11):
            prefix = [0.1 + 0.07 * index for index in range(count)]
            final = prefix[-1] if prefix else 0.25
            rewards = self.ns["rrpo_step_rewards"](prefix, count < 10, final)
            expected = 0.7 * final + 0.3 * (
                sum(prefix) + (10 - count) * final) / 10
            self.assertAlmostEqual(sum(rewards), expected)
            self.assertEqual(len(rewards), count + (count < 10))

    def test_reference_conditions_on_actual_selected_set_and_budget(self):
        complete = self.ns["bm25_reference_completion"]
        self.assertEqual(complete(self.row, (), 300), [0, 1])
        self.assertEqual(complete(self.row, (1,), 180), [1, 0])
        self.assertEqual(complete(self.row, (1,), 80), [1])

    def test_batched_policy_and_reference_prefix_scores(self):
        score_by_prefix = {(): 0.4, (0,): 0.8, (0, 1): 0.2}
        seen_prompts = []
        self.ns["compose_prompt"] = lambda _row, prefix: tuple(prefix)

        def generate(prompts, progress_label=None):
            seen_prompts.extend(prompts)
            return prompts

        self.ns["generate_completions"] = generate
        self.ns["synthetic_reward"] = lambda prediction, _row: {
            "utility": score_by_prefix[prediction],
            "es": score_by_prefix[prediction], "id_f1": 0.0}
        episodes = [
            {"row": self.row, "selected": [], "steps": [
                {"selected": (), "remaining": 300, "action": 3}]},
            {"row": self.row, "selected": [0], "steps": [
                {"selected": (), "remaining": 300, "action": 0},
                {"selected": (0,), "remaining": 200, "action": 3}]},
            {"row": self.row, "selected": [0, 1], "steps": [
                {"selected": (), "remaining": 300, "action": 0},
                {"selected": (0,), "remaining": 200, "action": 1}]},
        ]
        scored = self.ns["score_rrpo_episodes"](episodes)
        self.assertEqual(scored, 7)
        self.assertEqual(len(seen_prompts), 7)  # dedup per row, not across rows
        for episode, expected_return, expected_final, expected_reference in zip(
                episodes, (0.4, 0.8, 0.29), (0.4, 0.8, 0.2),
                ([0.29], [0.29, 0.17], [0.29, 0.17])):
            self.assertAlmostEqual(episode["reward"], expected_return)
            self.assertAlmostEqual(episode["final_es"], expected_final)
            self.assertEqual(len(episode["step_rewards"]), len(episode["steps"]))
            for observed, expected in zip(episode["reference_values"], expected_reference):
                self.assertAlmostEqual(observed, expected)

    def test_empty_candidate_stop_uses_no_retrieval_score(self):
        empty_row = {"candidate_ids": [], "candidate_costs": []}
        self.ns["compose_prompt"] = lambda _row, prefix: tuple(prefix)
        self.ns["generate_completions"] = lambda prompts, progress_label=None: prompts
        self.ns["synthetic_reward"] = lambda _prediction, _row: {
            "utility": 0.35, "es": 0.35, "id_f1": 0.0}
        episode = {"row": empty_row, "selected": [], "steps": [
            {"selected": (), "remaining": 300, "action": 0}]}
        self.assertEqual(self.ns["score_rrpo_episodes"]([episode]), 1)
        self.assertAlmostEqual(episode["reward"], 0.35)
        self.assertAlmostEqual(episode["reference_values"][0], 0.35)

    def test_gae_reference_advantages_and_ties(self):
        episode = {"steps": [{}, {}], "step_rewards": [0.4, 0.4],
                   "reference_values": [0.5, 0.1]}
        self.ns["RRPO_GAE_LAMBDA"] = 1.0
        self.ns["prepare_advantages"]([episode])
        self.assertTrue(np.allclose(episode["raw_advantages"], [0.3, 0.3]))
        self.assertTrue(np.allclose(episode["advantages"], [0.0, 0.0]))
        self.ns["RRPO_GAE_LAMBDA"] = 0.0
        self.ns["prepare_advantages"]([episode])
        self.assertTrue(np.allclose(episode["raw_advantages"], [0.0, 0.3]))
        self.assertTrue(all(np.isfinite(value) for value in episode["advantages"]))

    def test_rollout_prefix_generation_reference_gae_and_update_seam(self):
        self.ns["compose_prompt"] = lambda _row, prefix: tuple(prefix)
        self.ns["generate_completions"] = lambda prompts, progress_label=None: prompts
        self.ns["synthetic_reward"] = lambda prefix, _row: {
            "utility": 0.2 + 0.1 * len(prefix) - 0.05 * (2 in prefix),
            "es": 0.2 + 0.1 * len(prefix) - 0.05 * (2 in prefix),
            "id_f1": 0.0}
        torch.manual_seed(19)
        episodes = [self.ns["rollout"](self.row) for _ in range(8)]
        self.ns["score_rrpo_episodes"](episodes)
        self.assertTrue(all(len(ep["reference_values"]) == len(ep["steps"])
                            for ep in episodes))
        self.assertTrue(all(0.0 <= ep["reward"] <= 1.0 for ep in episodes))
        self.ns["optimizer"] = torch.optim.AdamW(
            list(self.ns["encoder"].parameters()) + list(self.ns["heads"].parameters()),
            lr=0.001,
        )
        self.ns["scaler"] = torch.amp.GradScaler("cpu", enabled=False)
        before = self.ns["encoder"].weight.detach().clone()
        diagnostics = self.ns["ppo_update"](episodes)
        self.assertTrue(np.isfinite(diagnostics["kl"]))
        self.assertGreater(float((self.ns["encoder"].weight.detach() - before).abs().sum()),
                           0.0)


class CandidateEncodingTest(unittest.TestCase):
    def test_64_candidates_with_shorter_query_are_microbatched_with_gradients(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "encode_row")
        calls = []

        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(100, 6, padding_idx=0)

            def forward(self, input_ids, attention_mask):
                calls.append(tuple(input_ids.shape))
                return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))

        model = FakeEncoder()
        namespace = {"torch": torch, "F": F, "encoder": model,
                     "RET_TOKENIZER": SimpleNamespace(pad_token_id=0),
                     "DEVICE": torch.device("cpu"), "ENCODE_BATCH_SIZE": 8}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"),
             namespace)
        row = {"query_ids": [1, 2, 0, 0],
               "candidate_ids": [[3 + i % 90, 4, 5, 0, 0, 0]
                                 for i in range(64)]}
        query, candidates = namespace["encode_row"](row)
        self.assertEqual(tuple(query.shape), (6,))
        self.assertEqual(tuple(candidates.shape), (64, 6))
        self.assertEqual([size for size, _ in calls], [8] * 8 + [1])
        self.assertEqual(calls[0][1], 6)  # Query padded to candidate length.
        self.assertTrue(torch.isfinite(candidates).all())
        (query.sum() + candidates.sum()).backward()
        self.assertGreater(float(model.embeddings.weight.grad.abs().sum()), 0)


class PoolCoverageTest(unittest.TestCase):
    def test_eval_probe_distinguishes_file_and_pool_misses(self):
        evaluation = SOURCE.with_name("ast_ppo_unixcoder_eval_kaggle.py")
        tree = ast.parse(evaluation.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "pool_identifier_probe")
        namespace = {"IDENTIFIER_RE": re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*"),
                     "keyword": keyword, "os": os}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(evaluation), "exec"),
             namespace)
        example = {"task_id": "task1", "language": "python",
                   "file_path": "repo/target.py", "left_context": "obj = make()\n",
                   "target_code": "newMethod(helper_name)",
                   "related_files": [
                       {"path": "repo/target.py", "text": "def newMethod(): pass"},
                       {"path": "repo/selected.py", "text": "def helper_name(): pass"},
                       {"path": "repo/omitted.py", "text": "def newMethod(): pass"},
                   ]}
        row = {"related_file_paths_used": ["repo/selected.py"],
               "candidate_raw_texts": ["def unrelated(): pass"]}
        result = namespace["pool_identifier_probe"](example, row)
        self.assertEqual(result["novel_target_identifiers"], 2)
        self.assertEqual(result["repo_identifier_hits"], 2)
        self.assertEqual(result["selected_file_identifier_hits"], 1)
        self.assertEqual(result["pool_identifier_hits"], 0)
        self.assertEqual(result["missed_at_file_selection"], ["newMethod"])
        self.assertEqual(result["missed_after_file_selection"], ["helper_name"])
        row["candidate_raw_texts"] = ["def helper_name(): pass"]
        self.assertEqual(namespace["pool_identifier_probe"](example, row)[
            "pool_identifier_hits"], 1)


class ASTContractTest(unittest.TestCase):
    def test_prepared_cache_serves_training_targets_and_retrieval_chunks(self):
        import hashlib

        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"prepared_ast_lookup", "target_spans", "ast_chunks"}
        nodes = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        source = "def f():\n    return 1\n"
        spans = {"line": [[9, 17, "return_statement"]], "block": []}
        chunks = [{"text": "return 1", "type": "return_statement",
                   "start": 9, "end": 17}]
        digest = hashlib.sha256(source.encode()).hexdigest()
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE ast_entries (language TEXT, "
                           "source_sha256 TEXT, spans BLOB, chunks BLOB, "
                           "had_error INTEGER)")
        connection.execute("INSERT INTO ast_entries VALUES (?, ?, ?, ?, ?)", (
            "python", digest, zlib.compress(json.dumps(spans).encode()),
            zlib.compress(json.dumps(chunks).encode()), 0,
        ))
        env = {"PREPARED_AST": connection, "AST_CHUNK_TOKENS": 384,
               "AST_CACHE_REQUIRED": True, "hashlib": hashlib,
               "json": json, "sqlite3": sqlite3, "zlib": zlib}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
        self.assertEqual(env["target_spans"](source, "python"), spans)
        found, had_error = env["ast_chunks"]("pkg/f.py", source, "python")
        self.assertFalse(had_error)
        self.assertEqual(found, [{**chunks[0], "path": "pkg/f.py"}])
        with self.assertRaisesRegex(RuntimeError, "absent from the attached AST cache"):
            env["target_spans"]("different", "python")
        connection.close()

    def test_target_spans_never_descends_past_latest_target_row(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "target_spans")

        class LateNode:
            start_point = SimpleNamespace(row=90)

            @property
            def start_byte(self):
                raise AssertionError("late AST subtree was visited")

        root = SimpleNamespace(has_error=False, type="module", start_byte=0,
                               start_point=SimpleNamespace(row=0),
                               named_children=[LateNode()])
        parser = SimpleNamespace(parse=lambda _source: SimpleNamespace(root_node=root))
        env = {"PARSERS": {"python": parser}, "MIN_LEFT_CONTEXT_LINES": 30,
               "TARGET_LINE_TYPES": {"python": set()},
               "TARGET_BLOCK_TYPES": {"python": set()}}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), env)
        self.assertEqual(env["target_spans"]("x = 1\n" * 100, "python"),
                         {"line": [], "block": []})

    def test_disk_backed_index_streams_real_parquet_and_skips_singleton_repos(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"RepositoryFiles", "RelatedFileView", "read_repository_groups"}
        nodes = [n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data" / "github_repos" / "java" / "train.parquet"
            path.parent.mkdir(parents=True)
            pq.write_table(pa.table({"path": ["a0", "a1", "a2", "single", "c0", "c1"],
                                     "content": ["A0", "A1", "A2", "S", "C0", "C1"],
                                     "first": [True, False, False, True, True, False]}),
                           path, row_group_size=2)
            work_dir = root / "work"
            work_dir.mkdir()
            env = {"pq": pq, "DATA_ROOT": root / "data", "WORK_DIR": work_dir,
                   "Path": Path, "sqlite3": sqlite3, "zlib": zlib,
                   "os": os, "tempfile": tempfile,
                   "process_rss_gib": lambda: 0.0}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"),
                 env)
            groups = env["read_repository_groups"]("java", 2)
            self.assertEqual([[path for path, _ in files] for files, _ in groups],
                             [["a0", "a1", "a2"], ["c0", "c1"]])
            self.assertEqual(groups[1][0][1], ("c1", "C1"))

    def test_repository_groups_follow_first_boundaries_and_retain_all_files(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"RepositoryFiles", "RelatedFileView", "read_repository_groups"}
        nodes = [n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        self.assertEqual({node.name for node in nodes}, names)
        rows = ([{"first": i == 0, "path": f"a_{i}.py", "content": f"value = {i}"}
                 for i in range(40)] +
                [{"first": i == 0, "path": f"b_{i}.py", "content": f"value = {i}"}
                 for i in range(2)])

        class FakeBatch:
            def to_pylist(self):
                return rows

        class FakeParquetFile:
            def __init__(self, _path):
                pass

            def iter_batches(self, **_kwargs):
                return iter([FakeBatch()])

        class FakePQ:
            ParquetFile = FakeParquetFile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet_path = root / "data" / "github_repos" / "python" / "train.parquet"
            parquet_path.parent.mkdir(parents=True)
            parquet_path.write_bytes(b"fake parquet")
            work_dir = root / "work"
            work_dir.mkdir()
            env = {"pq": FakePQ, "DATA_ROOT": root / "data", "WORK_DIR": work_dir,
                   "Path": Path, "sqlite3": sqlite3, "zlib": zlib,
                   "os": os, "tempfile": tempfile,
                   "process_rss_gib": lambda: 0.0,
                   "PREFERRED_FILE_LINES": 1, "PREFERRED_FILE_CHARS": 1}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"),
                 env)
            groups = env["read_repository_groups"]("python", 2)
            self.assertEqual(len(groups), 2)
            self.assertIsInstance(groups[0][0], env["RepositoryFiles"])
            self.assertEqual([path for path, _ in groups[0][0]],
                             [f"a_{i}.py" for i in range(40)])
            self.assertEqual([path for path, _ in groups[1][0]], ["b_0.py", "b_1.py"])
            self.assertEqual(groups[0][0].preferred_indices(), list(range(40)))
            self.assertEqual([path for path, _ in groups[0][0].related_excluding(4)],
                             [f"a_{i}.py" for i in range(40) if i != 4])
            self.assertEqual(env["read_repository_groups"]("python", 2)[0][0][39],
                             ("a_39.py", "value = 39"))
            make_examples = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                 and n.name == "make_examples")
            env.update({"random": random,
                        "target_spans": lambda code, _language: {
                            "line": [(0, len(code.encode()), "assignment")], "block": []},
                        "choose_target_kind": lambda _rng: "line"})
            exec(compile(ast.Module(body=[make_examples], type_ignores=[]),
                         str(SOURCE), "exec"), env)
            example = env["make_examples"](groups[:1], 1, "train", 17)[0]
            self.assertIsInstance(example["related_files"], env["RelatedFileView"])
            self.assertEqual(len(example["related_files"]), 39)
            self.assertNotIn(example["file_path"],
                             [path for path, _ in example["related_files"]])

    def test_python_java_spans_respect_ast_and_token_cap(self):
        import tree_sitter as ts
        import tree_sitter_python as ts_python
        import tree_sitter_java as ts_java

        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"TARGET_LINE_TYPES", "TARGET_BLOCK_TYPES", "RETRIEVAL_NODE_TYPES",
                 "SCOPE_NODE_TYPES"}
        constants = [n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in names
                             for t in n.targets)]
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "ast_chunks")
        env = {"PARSERS": {
                   "python": ts.Parser(ts.Language(ts_python.language())),
                   "java": ts.Parser(ts.Language(ts_java.language())),
               }, "token_count": lambda text: len(text.split()),
               "AST_CHUNK_TOKENS": 24}
        exec(compile(ast.Module(body=constants + [node], type_ignores=[]),
                     str(SOURCE), "exec"), env)
        examples = [
            ("python", "m.py", "def alpha(x):\n    return x + 1\n\ndef beta(y):\n    return y * 2\n"),
            ("java", "Demo.java", "class Demo { int sum(int a, int b) { return a+b; } }"),
        ]
        for language, path, code in examples:
            with self.subTest(language=language):
                chunks, parse_error = env["ast_chunks"](
                    path, code, language, max_tokens=7
                )
                self.assertFalse(parse_error)
                self.assertTrue(chunks)
                raw = code.encode("utf-8")
                for chunk in chunks:
                    self.assertEqual(raw[chunk["start"]:chunk["end"]].decode(),
                                     chunk["text"])
                    self.assertLessEqual(len(chunk["text"].split()), 7)

    def test_training_targets_are_reconstructible_ast_boundaries(self):
        import tree_sitter as ts
        import tree_sitter_python as ts_python
        import tree_sitter_java as ts_java

        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"TARGET_LINE_TYPES", "TARGET_BLOCK_TYPES"}
        constants = [n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in names
                             for t in n.targets)]
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "target_spans"]
        env = {"PARSERS": {
                   "python": ts.Parser(ts.Language(ts_python.language())),
                   "java": ts.Parser(ts.Language(ts_java.language())),
               }, "token_count": lambda text: len(text.split()),
               "MIN_LEFT_CONTEXT_LINES": 3, "MIN_PREFIX_TOKENS": 1,
               "MIN_TARGET_TOKENS": 1, "TARGET_CHUNK_TOKENS": 40}
        exec(compile(ast.Module(body=constants + nodes, type_ignores=[]),
                     str(SOURCE), "exec"), env)
        cases = {
            "python": "x = 0\ny = 1\nz = 2\ndef use(value):\n    result = value + z\n    return result\n",
            "java": "class Demo {\n  int x = 1;\n  int y = 2;\n  int z = 3;\n  int use(int v) {\n    return v + z;\n  }\n}\n",
        }
        for language, code in cases.items():
            with self.subTest(language=language):
                spans = env["target_spans"](code, language)
                env["print"] = lambda *args, **kwargs: None
                self.assertEqual(spans, env["target_spans"](code, language, trace=True))
                self.assertTrue(spans["line"])
                raw = code.encode()
                for kind, items in spans.items():
                    for start, end, node_type in items:
                        prefix, target, suffix = raw[:start], raw[start:end], raw[end:]
                        self.assertEqual(prefix + target + suffix, raw)
                        self.assertTrue(target.strip())
                        self.assertNotEqual(node_type, "ERROR")
                        self.assertTrue(prefix.endswith(b"\n"))
                        self.assertEqual(start, 0 if not prefix else prefix.rfind(b"\n") + 1)
                        if kind == "line":
                            self.assertNotIn(b"\n", target)
                            if node_type == "return_statement":
                                self.assertTrue(target.startswith(b"    "))

    def test_target_sampling_rejects_docstrings_and_trailing_same_line_code(self):
        import tree_sitter as ts
        import tree_sitter_python as ts_python

        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        constants = [n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in {
                         "TARGET_LINE_TYPES", "TARGET_BLOCK_TYPES"} for t in n.targets)]
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "target_spans")
        env = {"PARSERS": {"python": ts.Parser(ts.Language(ts_python.language()))},
               "token_count": lambda text: len(text.split()),
               "MIN_LEFT_CONTEXT_LINES": 3, "MIN_PREFIX_TOKENS": 1,
               "MIN_TARGET_TOKENS": 1, "TARGET_CHUNK_TOKENS": 40}
        exec(compile(ast.Module(body=constants + [function], type_ignores=[]),
                     str(SOURCE), "exec"), env)
        code = ("x = 0\ny = 1\nz = 2\ndef outer():\n"
                "    \"\"\"A docstring.\"\"\"\n"
                "    y = x + 1\n"
                "    return y\n")
        spans = env["target_spans"](code, "python")
        labels = [code.encode()[start:end].decode() for kind in spans
                  for start, end, _ in spans[kind]]
        self.assertNotIn('    """A docstring."""', labels)
        self.assertIn("    return y", labels)

    def test_phong_style_sampling_keeps_exact_target_and_other_files_only(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"TARGET_LINE_TYPES", "TARGET_BLOCK_TYPES",
                 "TARGET_CUT_DISTRIBUTION"}
        constants = [n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in names
                             for t in n.targets)]
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {"target_spans", "choose_target_kind", "make_examples"}]
        import tree_sitter as ts
        import tree_sitter_python as ts_python
        env = {"PARSERS": {"python": ts.Parser(ts.Language(ts_python.language()))},
               "token_count": lambda text: len(text.split()),
               "MIN_LEFT_CONTEXT_LINES": 3, "MIN_PREFIX_TOKENS": 1,
               "MIN_TARGET_TOKENS": 1, "TARGET_CHUNK_TOKENS": 40,
               "PREFERRED_FILE_LINES": 10, "PREFERRED_FILE_CHARS": 100,
               "random": random}
        exec(compile(ast.Module(body=constants + functions, type_ignores=[]),
                     str(SOURCE), "exec"), env)
        source = "".join(f"value_{i} = {i}\n" for i in range(50))
        groups = [([("main.py", source), ("helper.py", "def helper(): return 1\n")],
                   "python")]
        examples = env["make_examples"](groups, 5, "train", 11)
        self.assertEqual(len(examples), 5)
        for sample in examples:
            self.assertEqual(sample["file_path"], "main.py")
            self.assertEqual(sample["target_kind"], "line")
            self.assertEqual(sample["related_files"], [("helper.py", "def helper(): return 1\n")])
            start, end = sample["target_start"], sample["target_end"]
            self.assertEqual(sample["left_context"] + sample["target_code"] +
                             source.encode()[end:].decode(), source)
            self.assertEqual(source.encode()[start:end].decode(), sample["target_code"])

    def test_target_sampling_can_choose_a_long_file_after_the_old_limit(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"TARGET_LINE_TYPES", "TARGET_BLOCK_TYPES",
                 "TARGET_CUT_DISTRIBUTION"}
        constants = [n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in names
                             for t in n.targets)]
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {"target_spans", "choose_target_kind", "make_examples"}]
        import tree_sitter as ts
        import tree_sitter_python as ts_python
        env = {"PARSERS": {"python": ts.Parser(ts.Language(ts_python.language()))},
               "token_count": lambda text: len(text.split()),
               "MIN_LEFT_CONTEXT_LINES": 3, "MIN_PREFIX_TOKENS": 1,
               "MIN_TARGET_TOKENS": 1, "TARGET_CHUNK_TOKENS": 40,
               "PREFERRED_FILE_LINES": 10, "PREFERRED_FILE_CHARS": 100,
               "random": random}
        exec(compile(ast.Module(body=constants + functions, type_ignores=[]),
                     str(SOURCE), "exec"), env)
        files = [(f"short_{i}.py", "x = 1\n") for i in range(39)]
        files.append(("late_long.py", "".join(f"value_{i} = {i}\n" for i in range(50))))
        examples = env["make_examples"]([(files, "python")], 3, "train", 11)
        self.assertEqual(len(examples), 3)
        self.assertTrue(all(example["file_path"] == "late_long.py" for example in examples))
        self.assertTrue(all(len(example["related_files"]) == 39 for example in examples))

    def test_candidate_pool_ranks_late_related_file_without_gold_or_self_file(self):
        import re
        import tree_sitter as ts
        import tree_sitter_python as ts_python
        from rank_bm25 import BM25Okapi

        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        constants = [n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in {
                         "TARGET_LINE_TYPES", "TARGET_BLOCK_TYPES", "RETRIEVAL_NODE_TYPES",
                         "SCOPE_NODE_TYPES"} for t in n.targets)]
        names = {"ast_chunks", "left_anchors", "suffix_with_token_budget",
                 "pack_left_context", "retrieval_query", "lexical_tokens",
                 "render_chunk", "unixcoder_ids", "build_row"}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in names]

        class CharacterTokenizer:
            cls_token, sep_token, pad_token_id = "<s>", "</s>", 0

            def encode(self, text, add_special_tokens=False):
                return list(text)

            def tokenize(self, text):
                return list(text)

            def convert_tokens_to_ids(self, tokens):
                return list(range(1, len(tokens) + 1))

        tok = CharacterTokenizer()
        env = {"re": re, "os": os,
               "PARSERS": {"python": ts.Parser(ts.Language(ts_python.language()))},
               "GEN_TOKENIZER": tok, "RET_TOKENIZER": tok,
               "token_count": lambda text: len(text), "IDENTIFIER_RE": re.compile(
                   r"[A-Za-z_$][A-Za-z0-9_$]*"),
               "AST_CHUNK_TOKENS": 128, "BM25Okapi": BM25Okapi,
               "MAX_RELATED_FILES": 10, "CANDIDATE_POOL_SIZE": 16,
               "CROSSFILE_TOKEN_BUDGET": 384,
               "RETRIEVER_QUERY_LENGTH": 256, "RETRIEVER_CANDIDATE_LENGTH": 256}
        exec(compile(ast.Module(body=constants + functions, type_ignores=[]),
                     str(SOURCE), "exec"), env)
        related = [(f"other_{i}.py", f"def ordinary_{i}():\n    return {i}\n")
                   for i in range(39)]
        related.append(("late_needle.py", "def needle():\n    return 42\n"))
        example = {"task_id": "sample", "language": "python",
                   "file_path": "main.py", "left_context": "def use():\n    needle(",
                   "target_code": "SECRET_GOLD_SUFFIX", "related_files": related}
        row = env["build_row"](example)
        self.assertIn("late_needle.py", row["candidate_paths"])
        self.assertNotIn("main.py", row["candidate_paths"])
        self.assertNotIn("SECRET_GOLD_SUFFIX", str(row["candidate_texts"]))
        self.assertTrue(all(cost <= 384 for cost in row["candidate_costs"]))


class RLCoderScoreContractTest(unittest.TestCase):
    def setUp(self):
        import editdistance
        import tree_sitter as ts
        import tree_sitter_python as ts_python
        import tree_sitter_java as ts_java
        from fuzzywuzzy import fuzz

        parsers = {"python": ts.Parser(ts.Language(ts_python.language())),
                   "java": ts.Parser(ts.Language(ts_java.language()))}
        local_tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        local_names = {"rlcoder_postprocess_completion", "identifier_match_f1",
                       "rlcoder_score",
                       "rlcoder_repo_macro", "rlcoder_summary", "synthetic_reward"}
        local_nodes = [node for node in local_tree.body
                       if isinstance(node, ast.FunctionDef) and node.name in local_names]
        self.local = {"PARSERS": parsers, "re": re, "fuzz": fuzz,
                      "editdistance": editdistance, "keyword": keyword,
                      "RRPO_IDENTIFIER_WEIGHT": 0.2}
        exec(compile(ast.Module(body=local_nodes, type_ignores=[]),
                     str(SOURCE), "exec"), self.local)

        reference_path = SOURCE.parent.parent / "RepoClone/RLCoder/utils/eval_utils.py"
        reference_tree = ast.parse(reference_path.read_text(encoding="utf-8"))
        reference_names = {"get_ast", "is_parse_valid", "get_python_one_statement",
                           "get_bracket_lang_statement", "postprocess_code_lines",
                           "remove_comments", "cal_edit_sim"}
        reference_nodes = [node for node in reference_tree.body
                           if isinstance(node, ast.FunctionDef)
                           and node.name in reference_names]
        # The original get_ast timeout decorator is unrelated to scoring logic.
        next(node for node in reference_nodes if node.name == "get_ast").decorator_list = []
        self.reference = {"re": re, "fuzz": fuzz}
        exec(compile(ast.Module(body=reference_nodes, type_ignores=[]),
                     str(reference_path), "exec"), self.reference)

        repo_path = SOURCE.parent.parent / "RepoClone/RLCoder/utils/eval_repoeval.py"
        repo_tree = ast.parse(repo_path.read_text(encoding="utf-8"))
        repo_names = {"compute_EM", "compute_ES", "compute_score_by_repo_with_metadata"}
        repo_nodes = [node for node in repo_tree.body
                      if isinstance(node, ast.FunctionDef) and node.name in repo_names]
        self.repo_reference = {"editdistance": editdistance, "defaultdict": defaultdict}
        exec(compile(ast.Module(body=repo_nodes, type_ignores=[]),
                     str(repo_path), "exec"), self.repo_reference)
        self.parsers = parsers

    def test_primary_and_repo_scores_match_rlcoder_source(self):
        examples = [
            ("python", "def run():\n    x = 1\n    ",
             "return x # note\nprint('extra')\n", "    return x"),
            ("python", "def run():\n    x = 1\n    ",
             "if x:\n        return x\n    print('extra')\n",
             "    if x:\n        return x"),
            ("java", "class Demo { void run() { ",
             "int x = 1; int y = 2;", "int x = 1;"),
            ("java", "class Demo { void run() { ",
             "answer(); // note\nnext();", "answer();"),
            ("python", "def unfinished(\n", "answer = 1\nextra = 2",
             "answer = 1"),
        ]
        for language, left, prediction, target in examples:
            with self.subTest(language=language, prediction=prediction):
                actual = self.local["rlcoder_score"](prediction, target, left, language)
                processed = self.reference["postprocess_code_lines"](
                    left, prediction, self.parsers[language], language)
                pred = self.reference["remove_comments"](processed)
                gold = self.reference["remove_comments"](target)
                pred_lines = [line.strip() for line in pred.split("\n") if line.strip()]
                gold_lines = [line.strip() for line in gold.split("\n") if line.strip()]
                self.assertEqual(actual["prediction_postprocessed"], pred)
                self.assertEqual(actual["reference_postprocessed"], gold)
                self.assertEqual(actual["exact_match"], int(pred_lines == gold_lines))
                self.assertEqual(actual["edit_similarity"],
                                 self.reference["cal_edit_sim"]([gold], [pred]) / 100.0)
                self.assertEqual(actual["repoeval_exact_match"],
                                 int(self.repo_reference["compute_EM"](gold, [pred], 1)))
                self.assertAlmostEqual(actual["repoeval_edit_similarity"],
                                       self.repo_reference["compute_ES"](gold, [pred], 1))
                row = {"target_code": target, "left_context": left,
                       "language": language}
                reward = self.local["synthetic_reward"](prediction, row)
                expected = (actual["edit_similarity"] if not actual["target_has_identifiers"]
                            else 0.8 * actual["edit_similarity"] +
                                 0.2 * actual["identifier_f1"])
                self.assertAlmostEqual(reward["utility"], expected)
                self.assertEqual(reward["es"], actual["edit_similarity"])

    def test_identifier_f1_detects_wrong_api_and_ignores_strings(self):
        score = self.local["identifier_match_f1"]
        exact, has_ids = score('return client.fetch("ignore")',
                               'return client.fetch("different")', "python")
        wrong, _ = score('return client.flush("ignore")',
                         'return client.fetch("different")', "python")
        self.assertTrue(has_ids)
        self.assertEqual(exact, 1.0)
        self.assertLess(wrong, exact)
        self.assertEqual(score('"no name"', '"also no name"', "python"), (0.0, False))
        self.assertEqual(score('new Client().call();', 'new Client().call();',
                               "java"), (1.0, True))

    def test_summary_reports_rlcoder_primary_and_repository_macro(self):
        samples = [
            ("repoeval_line_0", "repoA/1", "x = 1", "x = 1"),
            ("repoeval_line_1", "repoA/2", "x = 2", "different"),
            ("repoeval_line_1", "repoB/1", "y = 3", "y = 3"),
        ]
        records, reference_rows = [], []
        for dataset, task_id, target, prediction in samples:
            score = self.local["rlcoder_score"](prediction, target, "def f():\n    ",
                                                  "python")
            records.append({"dataset": dataset, "method": "ppo", "task_id": task_id,
                            **score})
            reference_rows.append({"task_id": task_id,
                                   "pred": score["prediction_postprocessed"],
                                   "target": score["reference_postprocessed"]})
        summary = self.local["rlcoder_summary"](records)
        combined = next(row for row in summary if row["dataset"] == "repoeval_line")
        self.assertEqual(combined["n"], 3)
        self.assertEqual(combined["rlcoder_es_0to100"], round(
            sum(row["edit_similarity"] for row in records) / 3 * 100, 4))
        expected_macro, _ = self.repo_reference["compute_score_by_repo_with_metadata"](
            reference_rows, "ES", passk=1)
        self.assertEqual(combined["repoeval_es_pct"], round(expected_macro * 100, 4))


class PromptContractTest(unittest.TestCase):
    def test_ast_anchors_are_extracted_from_visible_prefix_only(self):
        import tree_sitter as ts
        import tree_sitter_python as ts_python
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "left_anchors")
        env = {"PARSERS": {"python": ts.Parser(ts.Language(ts_python.language()))},
               "SCOPE_NODE_TYPES": {"class_definition", "function_definition"}}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), env)
        prefix = "import useful\n\nclass Box:\n    def run(self):\n        x = useful.make()\n        "
        anchors = env["left_anchors"](prefix, "python")
        self.assertIn(("import", "import useful"), anchors)
        self.assertTrue(any(kind == "scope" and "def run" in text
                            for kind, text in anchors))
        self.assertTrue(any(kind == "binding" and "x = useful.make()" in text
                            for kind, text in anchors))
        self.assertFalse(any("SECRET_GOLD_SUFFIX" in text for _, text in anchors))

    def test_prompt_uses_visible_prefix_and_selected_context_only(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"suffix_with_token_budget", "pack_left_context", "compose_prompt"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in names]

        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(char) for char in text]

            def decode(self, ids, clean_up_tokenization_spaces=False):
                return "".join(chr(value) for value in ids)

        env = {"GEN_TOKENIZER": CharacterTokenizer(),
               "CROSSFILE_TOKEN_BUDGET": 100,
               "GENERATOR_INPUT_TOKENS": 268,
               "GENERATOR_OUTPUT_TOKENS": 32,
               "IDENTIFIER_RE": __import__("re").compile(r"[A-Za-z_$][A-Za-z0-9_$]*"),
               "token_count": lambda text: len(text)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
        row = {"file_path": "m.py", "language": "python",
               "left_context": "def use():\n    ",
               "candidate_texts": ["# file path: lib.py\ndef helper(): pass"],
               "target_code": "SECRET_GOLD_SUFFIX"}
        prompt = env["compose_prompt"](row, [0])
        self.assertIn("lib.py", prompt)
        self.assertTrue(prompt.endswith(row["left_context"]))
        self.assertNotIn(row["target_code"], prompt)
        self.assertLessEqual(len(prompt), 268)

    def test_long_context_keeps_exact_tail_and_budgeted_anchors(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"suffix_with_token_budget", "pack_left_context"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in names]

        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text)

        env = {"IDENTIFIER_RE": __import__("re").compile(r"[A-Za-z_$][A-Za-z0-9_$]*")}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
        prefix = "import useful\n" + "filler = 0\n" * 100 + "value = useful.make()\n"
        packed = env["pack_left_context"](
            prefix, "python", [("import", "import useful")], CharacterTokenizer(), 120)
        self.assertLessEqual(len(packed), 120)
        self.assertTrue(packed.endswith("value = useful.make()\n"))
        self.assertNotIn("SECRET_GOLD_SUFFIX", packed)

    def test_large_budget_spends_unused_hint_reserve_on_cursor_tail(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"suffix_with_token_budget", "pack_left_context"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in names]

        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text)

        env = {"IDENTIFIER_RE": re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
        prefix = ("import useful\nimport stale\n" + "padding = 0\n" * 300 +
                  "answer = useful.make()\n    ")
        packed = env["pack_left_context"](
            prefix, "python", [("import", "import useful"),
                               ("import", "import stale")], CharacterTokenizer(), 1800)
        self.assertIn("# earlier import: import useful", packed)
        self.assertNotIn("import stale", packed)
        self.assertTrue(packed.endswith("answer = useful.make()\n    "))
        self.assertGreaterEqual(len(packed), 1700)
        self.assertLessEqual(len(packed), 1800)

    def test_long_line_keeps_partial_suffix_if_alignment_wastes_budget(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "suffix_with_token_budget")
        env = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), env)

        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text)

        prefix = "payload = " + "x" * 1000 + "\nnext_line = "
        suffix, start = env["suffix_with_token_budget"](prefix, 150,
                                                           CharacterTokenizer())
        self.assertEqual(suffix, prefix[start:])
        self.assertTrue(suffix.endswith("\nnext_line = "))
        self.assertGreater(len(suffix), 100)
        self.assertLessEqual(len(suffix), 150)

    def test_hint_is_removed_when_reclaimed_tail_already_contains_it(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"suffix_with_token_budget", "pack_left_context"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in names]

        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text)

        env = {"IDENTIFIER_RE": re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
        prefix = ("unrelated\n" * 100 + "import useful\n" + "x" * 220 +
                  "\nanswer = useful.make()\n")
        tokenizer = CharacterTokenizer()
        full_tail, _ = env["suffix_with_token_budget"](prefix, 300, tokenizer)
        packed = env["pack_left_context"](
            prefix, "python", [("import", "import useful")], tokenizer, 300)
        self.assertEqual(packed, full_tail)
        self.assertEqual(packed.count("import useful"), 1)

    def test_full_prompt_preserves_selected_chunk_and_cursor_suffix(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"suffix_with_token_budget", "pack_left_context", "compose_prompt"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in names]

        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text) + (["<eos>"] if add_special_tokens else [])

        env = {"GEN_TOKENIZER": CharacterTokenizer(), "token_count": len,
               "IDENTIFIER_RE": __import__("re").compile(r"[A-Za-z_$][A-Za-z0-9_$]*"),
               "CROSSFILE_TOKEN_BUDGET": 100,
               "GENERATOR_INPUT_TOKENS": 268, "GENERATOR_OUTPUT_TOKENS": 32}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
        chunk = "# file path: helper.py\n# def useful(): return 1"
        row = {"language": "python", "file_path": "main.py",
               "left_context": "import useful\n" + "filler = 0\n" * 80 +
                               "answer = useful.make()\n    ",
               "left_anchors": [("import", "import useful")],
               "candidate_texts": [chunk], "target_code": "SECRET_GOLD_SUFFIX"}
        prompt = env["compose_prompt"](row, [0])
        self.assertIn(chunk, prompt)
        self.assertTrue(prompt.endswith("answer = useful.make()\n    "))
        self.assertNotIn("SECRET_GOLD_SUFFIX", prompt)
        self.assertLessEqual(len(env["GEN_TOKENIZER"].encode(
            prompt, add_special_tokens=True)), 268)

    def test_no_retrieval_prompt_starts_at_current_file_header(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"suffix_with_token_budget", "pack_left_context", "compose_prompt"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in names]

        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text) + (["<eos>"] if add_special_tokens else [])

        env = {"GEN_TOKENIZER": CharacterTokenizer(), "token_count": len,
               "IDENTIFIER_RE": re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*"),
               "CROSSFILE_TOKEN_BUDGET": 100,
               "GENERATOR_INPUT_TOKENS": 268, "GENERATOR_OUTPUT_TOKENS": 32}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
        row = {"language": "python", "file_path": "main.py",
               "left_context": "def main():\n    ", "left_anchors": [],
               "candidate_texts": []}
        prompt = env["compose_prompt"](row, [])
        self.assertTrue(prompt.startswith("# file path: main.py\n\n"))
        self.assertTrue(prompt.endswith(row["left_context"]))
        self.assertLessEqual(len(prompt), 268)


class FullEpochSamplingTest(unittest.TestCase):
    def test_epoch_slot_reads_same_prebuilt_row_and_covers_fixed_train_split(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "build_train_row")
        seen = []

        def fake_prepared_row(group_id):
            seen.append(group_id)
            return {"task_id": f"group-{group_id}"}

        env = {"TRAIN_EXAMPLES": 3, "TRAIN_ROW_GROUPS": [7, 2, 9],
               "prepared_row": fake_prepared_row}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), env)
        first_epoch = [env["build_train_row"](1, slot)["task_id"]
                       for slot in range(3)]
        second_epoch = [env["build_train_row"](2, slot)["task_id"]
                        for slot in range(3)]
        self.assertEqual(first_epoch, ["group-7", "group-2", "group-9"])
        self.assertEqual(first_epoch, second_epoch)
        self.assertEqual(seen, [7, 2, 9, 7, 2, 9])
        with self.assertRaises(IndexError):
            env["build_train_row"](1, 3)


class NotebookArtifactTest(unittest.TestCase):
    def test_full_training_is_epoch_unbounded_but_keeps_wall_clock_guard(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("MAX_TRAIN_EPOCHS", source)
        self.assertIn(
            "TRAIN_EPISODES_CAP = None  # No epoch/episode cap; stop only at the time guard.",
            source,
        )
        self.assertIn("TRAIN_STOP_HOURS_FROM_KERNEL_START = 7.0", source)
        self.assertIn("STATE[\"epoch\"] += 1", source)
        notebook = json.loads(SOURCE.with_suffix(".ipynb").read_text(encoding="utf-8"))
        notebook_text = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("there is no epoch-count cap", notebook_text)
        self.assertNotIn("MAX_TRAIN_EPOCHS", notebook_text)

    def test_training_artifact_has_deployed_url_and_only_needs_proxy_token_secret(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "https://bien14112005--bienkieu-ast-rrpo-deepseek-vllm-"
            "deepseekvl-3f59be.us-east.modal.direct",
            source,
        )
        self.assertIn('get_secret("MODAL_PROXY_TOKEN")', source)
        self.assertNotIn('get_secret("MODAL_VLLM_BASE_URL")', source)
        notebook = json.loads(SOURCE.with_suffix(".ipynb").read_text(encoding="utf-8"))
        notebook_text = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("The deployed Modal URL is configured below", notebook_text)
        self.assertIn("scales its A100 to zero", notebook_text)
        self.assertIn("7-hour hard stop", notebook_text)

    def test_training_artifact_contains_no_heldout_test_evaluation(self):
        source = SOURCE.read_text(encoding="utf-8")
        for test_only in ("cceval/python/test.parquet", "cceval/java/test.parquet",
                          "repoeval/line_level/test_0.parquet",
                          "benchmark_files(", "RUN_HELDOUT_EVAL_AFTER_TRAINING"):
            self.assertNotIn(test_only, source)
        notebook = SOURCE.with_suffix(".ipynb")
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        notebook_text = "\n".join(
            "".join(cell.get("source", [])) for cell in payload["cells"]
        )
        self.assertNotIn("cceval/python/test.parquet", notebook_text)
        self.assertIn("Training-only finalization", notebook_text)

    def test_modal_gpu_scales_down_at_the_rrpo_wall_clock_guard(self):
        def constant(node):
            return eval(compile(ast.Expression(node), str(SOURCE), "eval"),
                        {"__builtins__": {}})

        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        train_values = {
            target.id: node.value
            for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        hard_stop_hours = constant(train_values["TRAIN_STOP_HOURS_FROM_KERNEL_START"])
        self.assertEqual(hard_stop_hours, 7.0)
        self.assertEqual(constant(train_values["STOP_NEW_BATCH_RESERVE_SECONDS"]), 600)
        self.assertEqual(constant(train_values["MODAL_SCALEDOWN_WINDOW_SECONDS"]), 300)

        endpoint = SOURCE.with_name("modal_deepseek_vllm_endpoint.py")
        endpoint_tree = ast.parse(endpoint.read_text(encoding="utf-8"))
        endpoint_values = {
            target.id: node.value
            for node in endpoint_tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        self.assertEqual(constant(endpoint_values["SCALEDOWN_WINDOW_SECONDS"]), 300)
        self.assertEqual(constant(endpoint_values["MAX_NUM_SEQS"]), 108)
        server_class = next(node for node in endpoint_tree.body
                            if isinstance(node, ast.ClassDef)
                            and node.name == "DeepSeekVLLMServer")
        server_decorator = next(decorator for decorator in server_class.decorator_list
                                if isinstance(decorator, ast.Call)
                                and isinstance(decorator.func, ast.Attribute)
                                and decorator.func.attr == "server")
        scaledown = next(keyword.value for keyword in server_decorator.keywords
                         if keyword.arg == "scaledown_window")
        self.assertEqual(scaledown.id, "SCALEDOWN_WINDOW_SECONDS")

    def test_rrpo_checkpoint_and_eval_signature_match(self):
        def dict_assignment(path, name):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            node = next(node.value for node in tree.body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == name
                                for target in node.targets))
            return {ast.literal_eval(key): value
                    for key, value in zip(node.keys, node.values)}

        train = dict_assignment(SOURCE, "DATA_SIGNATURE")
        evaluation = SOURCE.with_name("ast_ppo_unixcoder_eval_kaggle.py")
        expected = dict_assignment(evaluation, "expected")
        self.assertEqual(ast.literal_eval(train["objective"]),
                         ast.literal_eval(expected["objective"]))
        self.assertEqual(ast.literal_eval(train["reward"]),
                         ast.literal_eval(expected["reward"]))
        constants = ast.parse(SOURCE.read_text(encoding="utf-8"))
        reference = next(node.value for node in constants.body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name)
                                 and target.id == "RRPO_REFERENCE"
                                 for target in node.targets))
        self.assertEqual(ast.literal_eval(reference),
                         ast.literal_eval(expected["rrpo_reference"]))
        eval_constants = ast.parse(evaluation.read_text(encoding="utf-8"))
        for name in ("MAX_RELATED_FILES", "AST_CHUNK_TOKENS",
                     "RETRIEVER_CANDIDATE_LENGTH", "CANDIDATE_POOL_SIZE",
                     "CROSSFILE_TOKEN_BUDGET", "MAX_SLATE_STEPS",
                     "GENERATOR_INPUT_TOKENS",
                     "GENERATOR_OUTPUT_TOKENS"):
            def assigned(tree, constant):
                value = next(node.value for node in tree.body
                             if isinstance(node, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == constant
                                     for target in node.targets))
                return ast.literal_eval(value)
            self.assertEqual(assigned(constants, name),
                             assigned(eval_constants, name), name)
        self.assertEqual(assigned(constants, "MAX_SLATE_STEPS"), 5)
        self.assertEqual(assigned(constants, "AST_CHUNK_TOKENS"), 384)
        self.assertEqual(assigned(constants, "RETRIEVER_CANDIDATE_LENGTH"), 512)
        self.assertEqual(assigned(constants, "CANDIDATE_POOL_SIZE"), 64)
        self.assertEqual(assigned(constants, "CROSSFILE_TOKEN_BUDGET"), 1344)
        self.assertEqual(assigned(constants, "ENCODE_BATCH_SIZE"), 128)
        self.assertEqual(assigned(constants, "ROLLOUT_QUERIES"), 64)
        self.assertEqual(assigned(constants, "ACCUMULATION_STEPS"), 8)
        self.assertEqual(assigned(constants, "GENERATOR_INPUT_TOKENS"), 2048)
        self.assertEqual(assigned(constants, "GENERATOR_OUTPUT_TOKENS"), 96)
        vllm_seqs = next(node.value for node in constants.body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name)
                                 and target.id == "VLLM_MAX_NUM_SEQS"
                                 for target in node.targets))
        self.assertEqual(ast.literal_eval(vllm_seqs), 108)
        self.assertEqual(train["pool"].id, "CANDIDATE_POOL_SIZE")
        self.assertEqual(expected["pool"].id, "CANDIDATE_POOL_SIZE")
        self.assertEqual(train["encode_batch_size"].id, "ENCODE_BATCH_SIZE")
        self.assertEqual(expected["encode_batch_size"].id, "ENCODE_BATCH_SIZE")
        self.assertEqual(ast.literal_eval(train["prompt_packer"]),
                         ast.literal_eval(expected["prompt_packer"]))
        self.assertEqual(ast.literal_eval(expected["rrpo_final_weight"]), 0.7)
        self.assertEqual(ast.literal_eval(expected["rrpo_identifier_weight"]), 0.2)
        for key in ("rrpo_gamma", "rrpo_gae_lambda", "target_kl",
                    "accumulation_steps", "reward"):
            self.assertIn(key, train)
        eval_text = evaluation.read_text(encoding="utf-8")
        self.assertIn("rrpo_best_ast", eval_text)
        self.assertIn("rrpo_latest_ast", eval_text)
        self.assertNotIn("ppo_best_ast", eval_text)

    def test_training_starts_without_reward_variance_pilot(self):
        source = SOURCE.read_text(encoding="utf-8")
        for obsolete in ("PILOT_QUERIES", "STOP_IF_LOW_VARIANCE", "pilot_rows",
                         "reward_variance_pilot"):
            self.assertNotIn(obsolete, source)
        tree = ast.parse(source)
        policy_check = next(node for node in tree.body
                            if isinstance(node, ast.FunctionDef)
                            and node.name == "policy_invariants")
        self.assertEqual(len(policy_check.args.args), 1)
        self.assertEqual(policy_check.args.args[0].arg, "row")
        self.assertIn("starting RRPO training.", source)

    def test_resume_allows_optimizer_tuning_but_rejects_new_data(self):
        namespace = load_policy()
        namespace["DATA_SIGNATURE"] = {
            "dataset_sha256": "data-a", "objective": "rrpo-v2",
            "valid": 100, "valid_examples_sha256": "valid-a",
            "encode_batch_size": 8, "accumulation_steps": 8,
            "encoder_lr": 1e-5,
        }
        previous = dict(namespace["DATA_SIGNATURE"])
        current = dict(previous, encode_batch_size=16,
                       accumulation_steps=16, encoder_lr=5e-5)
        namespace["DATA_SIGNATURE"] = current
        self.assertTrue(namespace["resume_signature_compatible"](previous))
        self.assertFalse(namespace["validation_signature_changed"](previous))
        self.assertFalse(namespace["resume_signature_compatible"](
            dict(previous, dataset_sha256="data-b")))
        self.assertTrue(namespace["validation_signature_changed"](
            dict(previous, valid_examples_sha256="valid-b")))

    def test_vllm_server_context_contract(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "verify_served_model")
        env = {"SERVED_MODEL_NAME": "test-model", "VLLM_PORT": 8000,
               "GENERATOR_INPUT_TOKENS": 3072,
               "GENERATOR_MAX_MODEL_LEN": 3168, "print": lambda *args: None}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), env)
        verify = env["verify_served_model"]
        verify([{"id": "test-model", "max_model_len": 3168}], True)
        with self.assertRaisesRegex(RuntimeError, "below 3168"):
            verify([{"id": "test-model", "max_model_len": 2048}], True)
        with self.assertRaisesRegex(RuntimeError, "does not advertise"):
            verify([{"id": "test-model"}], True)

    def test_eval_accepts_interrupted_pair_with_newer_best(self):
        evaluation = SOURCE.with_name("ast_ppo_unixcoder_eval_kaggle.py")
        source = evaluation.read_text(encoding="utf-8")
        tree = ast.parse(source)
        guard = next(node for node in tree.body if isinstance(node, ast.If)
                     and "WARNING: best.pt is newer" in
                     ast.get_source_segment(source, node))
        messages = []
        namespace = {"policies": {"best": {"state": {"episodes_seen": 16}},
                                  "latest": {"state": {"episodes_seen": 12}}},
                     "print": lambda *parts: messages.append(" ".join(map(str, parts)))}
        exec(compile(ast.Module(body=[guard], type_ignores=[]),
                     str(evaluation), "exec"), namespace)
        self.assertEqual(len(messages), 1)
        self.assertIn("Evaluating both saved snapshots", messages[0])

    def test_notebook_embeds_all_source_code_cells(self):
        notebook_path = SOURCE.with_suffix(".ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        expected = []
        kind, body = None, []
        for line in SOURCE.read_text(encoding="utf-8").splitlines(keepends=True):
            if line.startswith("# %%"):
                if kind == "code":
                    expected.append("".join(body))
                kind = "markdown" if "[markdown]" in line else "code"
                body = []
            elif kind == "code":
                body.append(line)
        if kind == "code":
            expected.append("".join(body))
        actual = ["".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code"]
        self.assertEqual(actual, expected)
        self.assertEqual(len({cell["id"] for cell in notebook["cells"]}),
                         len(notebook["cells"]))
        for index, source in enumerate(actual):
            compile(source, f"notebook-cell-{index}", "exec")

    def test_eval_notebook_mirror_and_shared_preprocessing_are_identical(self):
        evaluation = SOURCE.with_name("ast_ppo_unixcoder_eval_kaggle.py")
        train_text = SOURCE.read_text(encoding="utf-8")
        eval_text = evaluation.read_text(encoding="utf-8")
        train_tree, eval_tree = ast.parse(train_text), ast.parse(eval_text)
        for name in ("ast_chunks", "left_anchors", "suffix_with_token_budget",
                     "pack_left_context", "retrieval_query",
                     "compose_prompt", "rlcoder_postprocess_completion",
                     "identifier_match_f1", "rlcoder_score", "rlcoder_repo_macro",
                     "encode_row"):
            train_node = next(node for node in train_tree.body
                              if isinstance(node, ast.FunctionDef) and node.name == name)
            eval_node = next(node for node in eval_tree.body
                             if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.get_source_segment(train_text, train_node),
                             ast.get_source_segment(eval_text, eval_node))
        notebook = json.loads(evaluation.with_suffix(".ipynb").read_text(encoding="utf-8"))
        expected = []
        kind, body = None, []
        for line in eval_text.splitlines(keepends=True):
            if line.startswith("# %%"):
                if kind == "code":
                    expected.append("".join(body))
                kind = "markdown" if "[markdown]" in line else "code"
                body = []
            elif kind == "code":
                body.append(line)
        if kind == "code":
            expected.append("".join(body))
        actual = ["".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code"]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
