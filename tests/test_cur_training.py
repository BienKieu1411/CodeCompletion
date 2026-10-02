import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fuzzywuzzy import fuzz
import torch
from torch import nn

from src.train.build_utility_bundles import (
    BundleStore, CompletionOracle, annotate, annotate_many, intervention_graph, prepare_annotation)
from src.train.conditional_utility_model import (
    ConditionalUtilityModel, backward_bundle, bundle_tensors, encode_inputs, utility_loss)
from src.train.context_contract import ContextConfig, ContextRenderer
from src.train.epoch_data_loader import balanced_epoch_indices, pack
from src.train.select_conditional_context import select_context
from src.train.train_conditional_utility import atomic_save, main as train_main
from src.train.utility_metrics import completion_score


class ToyTokenizer:
    cls_token_id, sep_token_id, pad_token_id, unk_token_id = 0, 2, 1, 3

    def convert_tokens_to_ids(self, value):
        return 4 if value == "<encoder-only>" else self.unk_token_id

    def encode(self, text, add_special_tokens=False):
        return [5 + ord(c) % 59 for c in text]


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(64, 8)
        self.linear = nn.Linear(8, 8)
        self.dropout = nn.Dropout(.3)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.dropout(self.linear(self.embedding(input_ids))))


def task():
    return {"task_id": "q", "repo_uid": "r", "language": "python", "file_path": "main.py",
            "left_context": "import helper\nanswer = helper.", "target_code": "compute(value)",
            "target_kind": "member_suffix", "candidates": [
                {"path": f"helper{i}.py", "start": 0, "end": 20,
                 "text": f"def compute{i}(x): return x + {i}\n"} for i in range(3)]}


def toy_bundle():
    return {"task": task(), "context_contract": {"cross_file_tokens": 300},
            "candidate_costs": [40, 40, 40],
            "nodes": [{"indices": s, "cost": len(s) * 40, "es": es}
                      for s, es in [([], .3), ([0], .6), ([1], .1), ([0, 1], .4),
                                    ([1, 2], .8), ([2], .3)]],
            "edges": [{"source": 0, "target": 1, "group": "add"},
                      {"source": 0, "target": 2, "group": "add"},
                      {"source": 1, "target": 3, "group": "member"},
                      {"source": 2, "target": 4, "group": "joint"},
                      {"source": 1, "target": 4, "group": "swap"}]}


class CurTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_epoch_repository_sampling_balances_python_and_java(self):
        rows = ([{"language": "python", "split": "train"} for _ in range(3)]
                + [{"language": "java", "split": "train"} for _ in range(7)])
        selected = balanced_epoch_indices(rows, list(range(len(rows))), 10,
                                          seed=19, python_quota=5)
        counts = {language: sum(rows[index]["language"] == language
                                for index in selected)
                  for language in ("python", "java")}
        self.assertEqual(counts, {"python": 5, "java": 5})
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(set(selected)), 8)

    def test_epoch_repository_sampling_keeps_unique_repos_when_possible(self):
        rows = ([{"language": "python", "split": "train"} for _ in range(5)]
                + [{"language": "java", "split": "train"} for _ in range(5)])
        selected = balanced_epoch_indices(rows, list(range(len(rows))), 8, seed=19,
                                          python_quota=4)
        self.assertEqual(len(selected), len(set(selected)))

    def test_epoch_repository_sampling_uses_1000_python_and_2000_java(self):
        rows = ([{"language": "python", "split": "valid"} for _ in range(1_013)]
                + [{"language": "java", "split": "train"} for _ in range(2_235)])
        selected = balanced_epoch_indices(rows, list(range(len(rows))), 3_000,
                                          seed=19, python_quota=1_000)
        counts = {language: sum(rows[index]["language"] == language
                                for index in selected)
                  for language in ("python", "java")}
        self.assertEqual(counts, {"python": 1_000, "java": 2_000})
        self.assertEqual(len(selected), 3_000)
        self.assertEqual(len(set(selected)), 3_000)

    def test_fixed_prefix_budget_order_and_gold_invariance(self):
        config = ContextConfig(prompt_tokens=500, cross_file_tokens=300)
        t = task()
        t["left_context"] = "# context\n" * 100 + "answer = helper."
        r = ContextRenderer(t, ToyTokenizer(), config)
        changed = {**t, "target_code": "DO_NOT_READ", "right_context": "DO_NOT_READ_EITHER"}
        r2 = ContextRenderer(changed, ToyTokenizer(), config)
        self.assertEqual(r.render([1, 0]), r.render([0, 1]))
        self.assertEqual(r.render([0]), r2.render([0]))
        self.assertTrue(r.render([0, 1])[0].endswith(r.left_context))
        self.assertLessEqual(r.count(r.render([0, 1])[0]), 500)
        self.assertTrue(r.left_context.endswith("answer = helper."))
        with self.assertRaises(ValueError):
            r.render([0, 0])

    def test_measured_graph_unique_edges_and_no_label_dependence(self):
        r = ContextRenderer(task(), ToyTokenizer(), ContextConfig(500, 300))
        nodes, edges = intervention_graph(r, 123)
        self.assertLessEqual(len(nodes), 20)
        self.assertEqual(nodes[0]["indices"], [])
        self.assertEqual(len(edges), len({(e["source"], e["target"]) for e in edges}))
        self.assertTrue(all(r.feasible(node["indices"]) for node in nodes))
        mutated = {**task(), "target_code": "different target"}
        self.assertEqual((nodes, edges), intervention_graph(ContextRenderer(mutated, ToyTokenizer(), ContextConfig(500, 300)), 123))

    def test_sparse_graph_large_pool_budget_coverage_and_determinism(self):
        t = task()
        t["candidates"] = [{"path": f"helper{i}.py", "start": 0, "end": 20,
                             "text": f"def f{i}(): return {i}\n"} for i in range(64)]
        renderer = ContextRenderer(t, ToyTokenizer(), ContextConfig(3000, 2400))
        groups, sizes, maximum = set(), set(), 0
        mode_by_stratum = [set(), set(), set()]
        for seed in range(100):
            audit = {}
            nodes, edges = intervention_graph(renderer, seed, audit=audit)
            self.assertEqual((nodes, edges), intervention_graph(renderer, seed))
            self.assertLessEqual(len(nodes), 20)
            self.assertEqual(audit["unique_contexts"], len(nodes))
            self.assertEqual(audit["complete_diamonds"], 3)
            for stratum, record in enumerate(audit["strata"]):
                mode_by_stratum[stratum].add(record["mode"])
                self.assertIn(record["requested_size"], ((1, 2), (4, 5), (7, 8))[stratum])
                self.assertEqual(record["actual_size"], record["requested_size"])
                s, sa, sb, sab = [set(nodes[i]["indices"]) for i in record["diamond"]]
                self.assertEqual(len(sa - s), 1)
                self.assertEqual(len(sb - s), 1)
                self.assertNotEqual(sa, sb)
                self.assertEqual(sab, sa | sb)
                self.assertEqual(sa & sb, s)
            self.assertEqual(len(nodes), len({tuple(n["indices"]) for n in nodes}))
            self.assertTrue(all(renderer.feasible(n["indices"]) for n in nodes))
            groups.update(e["group"] for e in edges)
            sizes.update(len(n["indices"]) for n in nodes)
            maximum = max(maximum, len(nodes))
        self.assertEqual(maximum, 20)
        self.assertTrue(all(m == {"lexical", "random", "path_cluster"} for m in mode_by_stratum))
        self.assertEqual(groups, {"add", "member", "joint", "swap"})
        self.assertEqual(sizes, set(range(11)))

    def test_sparse_graph_small_and_token_constrained_pools(self):
        for n in (0, 1, 2, 3):
            t = task()
            t["candidates"] = t["candidates"][:n]
            renderer = ContextRenderer(t, ToyTokenizer(), ContextConfig(500, 90))
            for seed in range(10):
                nodes, edges = intervention_graph(renderer, seed)
                self.assertLessEqual(len(nodes), 20)
                self.assertTrue(all(renderer.feasible(node["indices"]) for node in nodes))
                self.assertTrue(all(e["source"] < len(nodes) and e["target"] < len(nodes) for e in edges))

    def test_diamond_recovers_signed_interaction_and_conditional_effect(self):
        # Exact two-factor 0/1 response table; no learned model or generator.
        t = task()
        renderer = ContextRenderer(t, ToyTokenizer(), ContextConfig(500, 300))
        audit = {}
        nodes, _ = intervention_graph(renderer, 15, audit=audit)
        record = next(r for r in audit["strata"] if r["diamond"] is not None)
        a, b = record["probes"]
        for interaction in (-.3, 0., .3):
            values = [.4 - .05 * (a in n["indices"]) + .1 * (b in n["indices"])
                      + interaction * (a in n["indices"] and b in n["indices"]) for n in nodes]
            s, sa, sb, sab = (values[i] for i in record["diamond"])
            self.assertAlmostEqual(sab - sa - sb + s, interaction)
            self.assertAlmostEqual(sa - s, -.05)
            self.assertAlmostEqual(sab - sb, -.05 + interaction)

    def test_annotation_submits_only_measured_nodes_and_excludes_gold_inputs(self):
        t = task()
        t["right_context"] = "FORBIDDEN_RIGHT_CONTEXT"
        t["dependency_evidence"] = "FORBIDDEN_GOLD_METADATA"
        seen = []

        class Oracle:
            def generate(self, prompts):
                seen.extend(prompts)
                return ["compute(value)"] * len(prompts)

        bundle = annotate(t, ToyTokenizer(), Oracle(), ContextConfig(500, 300), 15)
        self.assertEqual(len(seen), len(bundle["nodes"]))
        self.assertLessEqual(len(seen), 20)
        self.assertEqual(bundle["design"]["max_contexts"], 20)
        self.assertTrue(all(n["es"] == 1. for n in bundle["nodes"]))
        self.assertNotIn("target_code", bundle["task"])
        self.assertNotIn("dependency_evidence", bundle["task"])
        self.assertNotIn("right_context", bundle["task"])
        self.assertTrue(all("FORBIDDEN" not in prompt for prompt in seen))

    def test_multiple_tasks_fill_http_batches_without_mixing_labels(self):
        tasks = []
        for i in range(8):
            t = task()
            t.update(task_id=f"q{i}", file_path=f"main_{i}.py", target_code=f"answer_{i}()")
            t["candidates"] = [{"path": f"helper{j}.py", "start": 0, "end": 20,
                                 "text": f"def f{j}(): return {j}\n"} for j in range(64)]
            tasks.append(t)
        config = ContextConfig(3000, 2400)
        prepared = [prepare_annotation(t, ToyTokenizer(), config, i) for i, t in enumerate(tasks)]
        counts = [len(p["nodes"]) for p in prepared]
        self.assertTrue(all(n <= 20 for n in counts))

        def response_for(url, json, timeout):
            predictions = []
            for index, prompt in enumerate(json["prompt"]):
                i = next(i for i in range(8) if f"# File: main_{i}.py" in prompt)
                predictions.append({"index": index, "text": f"answer_{i}()"})
            return SimpleNamespace(ok=True, json=lambda: {"choices": predictions})

        with tempfile.TemporaryDirectory() as directory:
            store = BundleStore(Path(directory) / "bundles.sqlite", {"test": "pool"})
            oracle = CompletionOracle("http://localhost:8000", store, "revision", config, 128)
            with patch.object(oracle.session, "post", side_effect=response_for) as call:
                bundles = annotate_many(prepared, oracle)
                self.assertEqual(call.call_count, (sum(counts) + 127) // 128)
                self.assertEqual(len(call.call_args_list[0].kwargs["json"]["prompt"]), 128)
                for i, bundle in enumerate(bundles):
                    self.assertEqual(bundle["task"]["task_id"], f"q{i}")
                    self.assertTrue(all(n["es"] == 1. for n in bundle["nodes"]))
                # Restarting/reusing the identical prepared window hits durable cache.
                old_calls = call.call_count
                annotate_many(prepared, oracle)
                self.assertEqual(call.call_count, old_calls)
            store.close()

    def test_direct_encoding_avoids_replay_forward_work(self):
        class CountingEncoder(TinyEncoder):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, input_ids, attention_mask):
                self.calls += 1
                return super().forward(input_ids, attention_mask)

        direct = ConditionalUtilityModel(CountingEncoder(), width=4, hidden=8)
        replay = copy.deepcopy(direct)
        bundle = toy_bundle()
        ids, mask, _ = encode_inputs(bundle["task"], ToyTokenizer(), 64)
        backward_bundle(direct, ids, mask, bundle, microbatch=2, replay=False)
        backward_bundle(replay, ids, mask, bundle, microbatch=2, replay=True)
        self.assertEqual(replay.encoder.calls, direct.encoder.calls * 2)
        for p, rp in zip(direct.parameters(), replay.parameters()):
            self.assertTrue(torch.allclose(p.grad, rp.grad, atol=1e-6, rtol=1e-4))

    def test_gain_loss_matches_incidence_form_and_anchor_removes_nullspace(self):
        # One disconnected observed component: gain-only cannot set its level.
        bundle = {"nodes": [{"indices": x, "es": y} for x, y in
                           [([], .2), ([0], .5), ([1], .1), ([1, 2], .7)]],
                  "edges": [{"source": 0, "target": 1, "group": "add"},
                            {"source": 2, "target": 3, "group": "member"}]}
        gold = torch.tensor([0., .3, -.1, .5], dtype=torch.float64)
        prediction = gold + torch.tensor([0., .1, .3, -.2], dtype=torch.float64)
        incidence = torch.tensor([[-1., 1., 0., 0.], [0., 0., -1., 1.]], dtype=torch.float64)
        error = prediction - gold
        expected = (incidence @ error).square().mean() + .25 * error[1:].square().mean()
        actual = utility_loss(prediction, bundle)[0]
        self.assertTrue(torch.allclose(expected, actual))
        shifted = gold + torch.tensor([0., 0., .7, .7], dtype=torch.float64)
        self.assertLess(float(utility_loss(shifted, bundle, anchor_weight=0)[0]), 1e-15)
        self.assertGreater(float(utility_loss(shifted, bundle)[0]), .01)
        hessian = torch.autograd.functional.hessian(
            lambda z: utility_loss(torch.cat([z.new_zeros(1), z]), bundle)[0], gold[1:])
        self.assertTrue((torch.linalg.eigvalsh(hessian) > 0).all())

    def test_potential_is_invariant_to_candidate_index_permutation(self):
        torch.manual_seed(19)
        model = ConditionalUtilityModel(TinyEncoder(), width=4, hidden=8)
        features = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        members, costs, items = bundle_tensors(toy_bundle(), "cpu")
        permutation = torch.tensor([2, 0, 1])
        original = model.potential(features, members, costs, items)
        permuted = model.potential(torch.cat([features[:1], features[1:][permutation]]),
                                   members[:, permutation], costs, items[permutation])
        self.assertTrue(torch.allclose(original, permuted, atol=1e-7))

    def test_fuzz_parity_against_checked_in_rlcoder_function(self):
        source = Path("RepoClone/RLCoder/utils/eval_utils.py").read_text()
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "cal_edit_sim")
        namespace = {"fuzz": fuzz}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "reference", "exec"), namespace)
        for gold, prediction in [("foo(12)", "foo(13)"), ("名字()", "名前()"), ("x()", "")]:
            got = completion_score(prediction + "\nEXTRA()", gold, "line")["es"] if prediction else completion_score("", gold, "line")["es"]
            self.assertEqual(got, namespace["cal_edit_sim"]([gold], [prediction]) / 100)
        metric = completion_score("  a()\n b()\nEXTRA()", "a()\nb()", "api_statement")
        self.assertEqual(metric, {"es": 1., "em": 1.})

    def test_loss_sign_empty_potential_and_gradient_replay_equivalence(self):
        torch.manual_seed(7)
        model = ConditionalUtilityModel(TinyEncoder(), width=4, hidden=8)
        replay_model = copy.deepcopy(model)
        bundle = toy_bundle()
        ids, mask, _ = encode_inputs(bundle["task"], ToyTokenizer(), 64)
        backward_bundle(model, ids, mask, bundle, replay=False)
        backward_bundle(replay_model, ids, mask, bundle, microbatch=2, replay=True)
        for (name, p), (_, rp) in zip(model.named_parameters(), replay_model.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.allclose(p.grad, rp.grad, atol=2e-7, rtol=2e-5), name)
        self.assertGreater(float(model.encoder.embedding.weight.grad.abs().sum()), 0)
        features = model.encode(ids, mask)
        features.retain_grad()
        predicted = model.potential(features, *bundle_tensors(bundle, "cpu"))
        self.assertEqual(float(predicted[0].detach()), 0.)
        utility_loss(predicted, bundle)[0].backward()
        self.assertTrue((features.grad.abs().sum(-1) > 0).all())
        ideal = torch.tensor([n["es"] - .3 for n in bundle["nodes"]])
        self.assertLess(float(utility_loss(ideal, bundle)[0]), 1e-12)

    def test_tiny_bundle_overfits_and_encoder_changes(self):
        torch.manual_seed(17)
        model = ConditionalUtilityModel(TinyEncoder(), width=8, hidden=16)
        before = model.encoder.embedding.weight.detach().clone()
        bundle = toy_bundle()
        ids, mask, _ = encode_inputs(bundle["task"], ToyTokenizer(), 64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.015)
        losses = []
        for _ in range(250):
            optimizer.zero_grad()
            metrics = backward_bundle(model, ids, mask, bundle, replay=False)
            losses.append(metrics["loss"])
            optimizer.step()
        self.assertLess(losses[-1], losses[0] * .1)
        self.assertFalse(torch.equal(before, model.encoder.embedding.weight))

    def test_replay_matches_real_roberta_attention_encoder(self):
        from transformers import RobertaConfig, RobertaModel
        torch.manual_seed(5)
        encoder = RobertaModel(RobertaConfig(vocab_size=64, hidden_size=16,
                               num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=32, max_position_embeddings=130,
                               pad_token_id=1, hidden_dropout_prob=.1))
        model = ConditionalUtilityModel(encoder, width=4, hidden=8)
        replay = copy.deepcopy(model)
        bundle = toy_bundle()
        ids, mask, _ = encode_inputs(bundle["task"], ToyTokenizer(), 64)
        backward_bundle(model, ids, mask, bundle, replay=False)
        backward_bundle(replay, ids, mask, bundle, microbatch=2)
        for (name, p), (_, rp) in zip(model.named_parameters(), replay.named_parameters()):
            if p.grad is None:  # Roberta's unused pooler is not our mean pool.
                self.assertIn("pooler", name)
                continue
            self.assertTrue(torch.allclose(p.grad, rp.grad, atol=1e-6, rtol=1e-4), name)

    def test_selector_learns_stop_and_can_add_complementary_pair(self):
        class OraclePotential(ConditionalUtilityModel):
            def potential(self, features, members, costs, item_costs):
                # Neither 0 nor 1 helps alone, jointly they help; 2 hurts.
                return .5 * members[:, 0] * members[:, 1] - .4 * members[:, 2]
        model = OraclePotential(TinyEncoder(), width=4, hidden=8)
        output = select_context(model, task(), ToyTokenizer(), ToyTokenizer(), ContextConfig(500, 300))
        self.assertEqual(set(output["indices"]), {0, 1})
        self.assertFalse(output["search_censored"])

    def test_oracle_caches_prompts_and_rejects_missing_response(self):
        with tempfile.TemporaryDirectory() as directory:
            store = BundleStore(Path(directory) / "bundles.sqlite", {"test": 1})
            oracle = CompletionOracle("http://localhost:8000", store, "revision", ContextConfig(), 4)
            response = SimpleNamespace(ok=True, json=lambda: {"choices": [{"index": 0, "text": "ok"}]})
            with patch.object(oracle.session, "post", return_value=response) as call:
                self.assertEqual(oracle.generate(["same", "same"]), ["ok", "ok"])
                self.assertEqual(oracle.generate(["same"]), ["ok"])
                self.assertEqual(call.call_count, 1)
                with self.assertRaises(ValueError):
                    oracle.generate(["one", "two"])
            store.close()

    def test_checkpoint_resume_matches_uninterrupted_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for split in ("train", "valid"):
                path = root / f"{split}.sqlite"
                manifest = {"split": split, "selected_repos": [split], "pool_sha256": "same",
                            "version": "v", "metric": "m", "generator": "g", "generator_revision": "r",
                            "context": {}, "retriever_revision": "r", "decoding": {},
                            "fuzzywuzzy_version": "test", "fuzzy_backend": "test",
                            "code_hashes": {}, "sampler_sha256": "test", "data": {}}
                store = BundleStore(path, manifest)
                store.db.execute("INSERT INTO bundles VALUES (?, ?)", (split, pack(toy_bundle())))
                store.db.commit()
                store.close()
                paths.append(path)

            def run(output, resume=None, interrupt=False):
                argv = ["train", "--train-bundles", str(paths[0]), "--valid-bundles", str(paths[1]),
                        "--output", str(output), "--passes", "2", "--save-every", "1", "--device", "cpu"]
                if resume:
                    argv += ["--resume", str(resume)]

                def save_then_interrupt(path, payload):
                    atomic_save(path, payload)
                    if interrupt and payload["state"]["updates"] == 1:
                        raise KeyboardInterrupt()

                with patch("sys.argv", argv), patch("transformers.AutoModel.from_pretrained", side_effect=lambda *a, **k: TinyEncoder()), \
                     patch("transformers.AutoTokenizer.from_pretrained", return_value=ToyTokenizer()), \
                     patch("src.train.train_conditional_utility.atomic_save", side_effect=save_then_interrupt), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train_main()

            run(root / "full")
            with self.assertRaises(KeyboardInterrupt):
                run(root / "resumed", interrupt=True)
            run(root / "resumed", resume=root / "resumed/latest.pt")
            full = torch.load(root / "full/latest.pt", weights_only=False)
            resumed = torch.load(root / "resumed/latest.pt", weights_only=False)
            self.assertEqual(full["state"], resumed["state"])
            for key in full["model"]:
                self.assertTrue(torch.equal(full["model"][key], resumed["model"][key]), key)

    def test_mixed_annotation_recipes_rejected_without_validation(self):
        first = SimpleNamespace(manifest={"split": "train", "pool_sha256": "same",
                                           "version": "cur_bundles_sparse8_v2"})
        second = SimpleNamespace(manifest={"split": "train", "pool_sha256": "same",
                                            "version": "cur_bundles_stratified20_v3"})
        with tempfile.TemporaryDirectory() as directory:
            argv = ["train", "--train-bundles", "one.sqlite", "two.sqlite",
                    "--output", directory, "--device", "cpu"]
            with patch("sys.argv", argv), \
                 patch("src.train.train_conditional_utility.Bundles", side_effect=[first, second]), \
                 patch("transformers.AutoModel.from_pretrained") as load:
                with self.assertRaisesRegex(ValueError, "Training epoch annotation contract mismatch: version"):
                    train_main()
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
