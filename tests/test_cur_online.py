import contextlib
import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.data.ast_training_data import DataConfig, mine_candidates
from src.data.audit_completion_data import TOKENIZER_REVISION
from src.data.build_ast_repo_pool import RET_REVISION
from src.train.build_utility_bundles import CompletionOracle
from src.train.conditional_utility_model import ConditionalUtilityModel, encode_inputs
from src.train.context_contract import ContextConfig, ContextRenderer
from src.train.train_conditional_utility import atomic_save
from src.train.train_online_utility import OnlineStop, main, online_update, prepare_online_annotation
from tests.test_cur_training import TinyEncoder, ToyTokenizer, task


class OnlineTests(unittest.TestCase):
    def test_batched_tokenization_preserves_scalar_context_and_encoder_inputs(self):
        class BatchToyTokenizer(ToyTokenizer):
            def __call__(self, texts, **kwargs):
                return {"input_ids": [self.encode(text, add_special_tokens=False) for text in texts]}

        slow = ContextRenderer(task(), ToyTokenizer(), ContextConfig(500, 300))
        fast = ContextRenderer(task(), BatchToyTokenizer(), ContextConfig(500, 300))
        sets = [(), (0,), (1,), (0, 1), (1, 2)]
        self.assertEqual(fast.feasible_many(sets), [slow.feasible(s) for s in sets])
        self.assertEqual(fast.cost_many(sets), [slow.cross_file(s)[1] for s in sets])
        self.assertEqual([fast.render(s) for s in sets], [slow.render(s) for s in sets])
        ids_slow, mask_slow, trunc_slow = encode_inputs(task(), ToyTokenizer())
        ids_fast, mask_fast, trunc_fast = encode_inputs(task(), BatchToyTokenizer())
        self.assertTrue(torch.equal(ids_slow, ids_fast))
        self.assertTrue(torch.equal(mask_slow, mask_fast))
        self.assertEqual(trunc_slow, trunc_fast)

    def test_bm25_top_100_then_encoder_sees_all_candidates(self):
        tok = ToyTokenizer()
        chunks = [{"path": f"helper{i:03}.py", "start": 0, "end": 20,
                   "scope_headers": [], "text": f"def helper{i}(): return value\n"}
                  for i in range(130)]
        chunks.append({**chunks[0], "path": "current.py"})
        config = replace(DataConfig(), pool_size=100)
        selected = mine_candidates("helper129(value)", "current.py", chunks, tok, config)
        self.assertEqual(len(selected), 100)
        self.assertNotIn("current.py", [c["path"] for c in selected])
        self.assertEqual(selected[0]["path"], "helper129.py")
        self.assertEqual([c["bm25_score"] for c in selected],
                         sorted([c["bm25_score"] for c in selected], reverse=True))
        self.assertEqual(len(mine_candidates("value", "current.py", chunks[:7], tok, config)), 7)
        t = {**task(), "candidates": selected}
        model = ConditionalUtilityModel(TinyEncoder(), width=4, hidden=8)
        with patch.object(model, "encode", wraps=model.encode) as encode:
            prepared = prepare_online_annotation(model, t, tok, tok, ContextConfig(5000, 4000), 12, 64)
        self.assertEqual([call.args[0].shape[0] for call in encode.call_args_list], [64, 37])
        self.assertLessEqual(len(prepared["nodes"]), 20)
        self.assertTrue(all(len(node["indices"]) <= 10 for node in prepared["nodes"]))

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_no_persistent_labels_with_in_call_deduplication(self):
        oracle = CompletionOracle("http://localhost:8000", None, "test", ContextConfig(), 128)
        response = SimpleNamespace(ok=True, json=lambda: {"choices": [{"index": 0, "text": "answer"}]})
        with patch.object(oracle.session, "post", return_value=response) as post:
            self.assertEqual(oracle.generate(["same", "same"]), ["answer", "answer"])
            self.assertEqual(oracle.generate(["same"]), ["answer"])
            self.assertEqual(post.call_count, 2)  # No reuse across optimizer batches.
        oracle.session.close()

    def test_current_model_changes_proposals_without_gold_access(self):
        class RankedModel(ConditionalUtilityModel):
            def __init__(self):
                super().__init__(TinyEncoder(), width=4, hidden=8)
                self.sign = 1.

            def potential(self, features, membership, costs, item_costs):
                return membership @ (self.sign * torch.arange(membership.shape[1], device=membership.device).float())

        t = task()
        t["candidates"] = [{**t["candidates"][0], "path": f"p{i}.py"} for i in range(12)]
        model = RankedModel()
        tok, config = ToyTokenizer(), ContextConfig(5000, 4000)
        a = prepare_online_annotation(model, t, tok, tok, config, 12, 4)
        changed = {**t, "target_code": "DO_NOT_USE", "right_context": "DO_NOT_USE"}
        b = prepare_online_annotation(model, changed, tok, tok, config, 12, 4)
        self.assertEqual(a["prompts"], b["prompts"])
        model.sign = -1.
        c = prepare_online_annotation(model, t, tok, tok, config, 12, 4)
        self.assertNotEqual(a["prompts"], c["prompts"])
        self.assertIn("model", [s["mode"] for s in c["design"]["strata"]])
        self.assertLessEqual(len(c["nodes"]), 20)
        self.assertFalse(any(p.grad is not None for p in model.parameters()))

    def test_each_batch_scores_then_updates_encoder_before_next_proposals(self):
        model = ConditionalUtilityModel(TinyEncoder(), width=4, hidden=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        tok = ToyTokenizer()
        snapshots, events = [], []

        def prepare(*args, **kwargs):
            snapshots.append(model.encoder.embedding.weight.detach().clone())
            events.append("propose")
            return prepare_online_annotation(*args, **kwargs)

        def generate(prompts):
            events.append("score")
            return ["compute(value)"] * len(prompts)

        original_step = optimizer.step

        def step():
            events.append("step")
            self.assertGreater(float(model.encoder.embedding.weight.grad.abs().sum()), 0.)
            return original_step()

        with patch("src.train.train_online_utility.prepare_online_annotation", side_effect=prepare), \
             patch.object(optimizer, "step", side_effect=step):
            for _ in range(2):
                online_update(model, [(task(), 17), (task(), 18)], tok, tok,
                              SimpleNamespace(generate=generate), optimizer, config=ContextConfig(500, 300))
        self.assertEqual(events, ["propose", "propose", "score", "step"] * 2)
        self.assertTrue(torch.equal(snapshots[0], snapshots[1]))
        self.assertFalse(torch.equal(snapshots[1], snapshots[2]))
        self.assertTrue(torch.equal(snapshots[2], snapshots[3]))

    def test_failed_generation_does_not_update_weights(self):
        model = ConditionalUtilityModel(TinyEncoder(), width=4, hidden=8)
        original = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters())
        tok = ToyTokenizer()

        def fail(prompts):
            raise RuntimeError("generator unavailable")

        with patch.object(optimizer, "step") as step:
            with self.assertRaisesRegex(RuntimeError, "generator unavailable"):
                online_update(model, [(task(), 12)], tok, tok, SimpleNamespace(generate=fail), optimizer,
                              config=ContextConfig(500, 300))
            step.assert_not_called()
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]), key)

    def test_cooperative_stop_before_step_keeps_optimizer_boundary(self):
        model = ConditionalUtilityModel(TinyEncoder(), width=4, hidden=8)
        original = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters())
        tok = ToyTokenizer()

        def check():
            # Interrupt after backward, immediately before mutation.
            if any(p.grad is not None for p in model.parameters()):
                raise OnlineStop()

        with patch.object(optimizer, "step") as step:
            with self.assertRaises(OnlineStop):
                online_update(model, [(task(), 12)], tok, tok,
                              SimpleNamespace(generate=lambda prompts: ["compute(value)"] * len(prompts)),
                              optimizer, config=ContextConfig(500, 300), check_continue=check)
            step.assert_not_called()
        self.assertEqual(len(optimizer.state), 0)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]), key)

    def test_generator_checks_stop_between_http_batches(self):
        checks = []

        def check():
            checks.append(1)
            if len(checks) == 2:
                raise OnlineStop()

        oracle = CompletionOracle("http://localhost:8000", None, "test", ContextConfig(), 1,
                                  request_timeout=180, check_continue=check)
        response = SimpleNamespace(ok=True, json=lambda: {"choices": [{"index": 0, "text": "answer"}]})
        with patch.object(oracle.session, "post", return_value=response) as post:
            with self.assertRaises(OnlineStop):
                oracle.generate(["one", "two"])
            self.assertEqual(post.call_count, 1)
            self.assertEqual(post.call_args.kwargs["timeout"], (10, 180))
        oracle.session.close()

    def test_online_resume_matches_uninterrupted_training(self):
        class Encoder(TinyEncoder):
            def gradient_checkpointing_disable(self):
                pass

        class Loader:
            epoch_size, python_quota = 3, 1

            def __init__(self, *args):
                pass

            def select_epoch_indices(self, *args):
                return [0, 1, 2]

            def _load_payloads(self, indices):
                return [{}]

        class Oracle:
            def __init__(self, *args, **kwargs):
                self.session = SimpleNamespace(close=lambda: None)

            def verify_served_model(self):
                pass

            def generate(self, prompts):
                return ["compute(value)"] * len(prompts)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = root / "train.parquet"
            pool.write_bytes(b"mock parquet")
            contract = {"config": asdict(DataConfig()), "generator_tokenizer_revision": TOKENIZER_REVISION,
                        "retriever_tokenizer_revision": RET_REVISION}
            metadata = {b"preparation_contract": json.dumps(contract).encode()}

            def run(output, resume=False, interrupt=False):
                argv = ["train", "--pool", str(pool), "--output", str(output), "--epochs", "2",
                        "--base-url", "http://localhost:8000", "--batch-size", "2", "--save-every", "1",
                        "--device", "cpu", "--precision", "fp32"]
                if resume:
                    argv += ["--resume", str(output / "latest.pt")]

                def save_then_interrupt(path, payload):
                    atomic_save(path, payload)
                    if interrupt and payload["state"]["updates"] == 1:
                        raise KeyboardInterrupt()

                with patch("sys.argv", argv), \
                     patch("pyarrow.parquet.ParquetFile", return_value=SimpleNamespace(schema_arrow=SimpleNamespace(metadata=metadata))), \
                     patch("src.train.train_online_utility.EpochDataLoader", Loader), \
                     patch("src.train.train_online_utility.CompletionOracle", Oracle), \
                     patch("src.train.train_online_utility.sample_repo_task", side_effect=lambda *a: task()), \
                     patch("src.train.train_online_utility.atomic_save", side_effect=save_then_interrupt), \
                     patch("transformers.AutoTokenizer.from_pretrained", return_value=ToyTokenizer()), \
                     patch("transformers.AutoModel.from_pretrained", side_effect=lambda *a, **k: Encoder()), \
                     contextlib.redirect_stdout(io.StringIO()):
                    main()

            run(root / "full")
            with self.assertRaises(KeyboardInterrupt):
                run(root / "resumed", interrupt=True)
            run(root / "resumed", resume=True)
            full = torch.load(root / "full/latest.pt", weights_only=False)
            resumed = torch.load(root / "resumed/latest.pt", weights_only=False)
            self.assertEqual(full["state"], {"epoch": 3, "cursor": 0, "updates": 4})
            self.assertEqual(full["state"], resumed["state"])
            self.assertEqual(full["contract"]["data"]["pool_size"], 100)
            self.assertEqual(contract["config"]["pool_size"], 64)  # Original metadata unchanged.
            for key in full["model"]:
                self.assertTrue(torch.equal(full["model"][key], resumed["model"][key]), key)


if __name__ == "__main__":
    unittest.main()
