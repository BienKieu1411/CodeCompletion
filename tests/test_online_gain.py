import contextlib
import copy
from types import SimpleNamespace
import unittest

import torch

from src.train.context_contract import ContextConfig
from src.train.distributional_gain_model import ConditionalGainModel
from src.train.online_gain import online_gain_update, prepare_gain_annotation
from src.train.select_gain_context import select_gain_context
from tests.test_cur_training import TinyEncoder, ToyTokenizer, task


def fake_scores(prompts, targets, tokenizer):
    return [-2. + .25 * ("def compute0" in p) - .12 * ("def compute1" in p)
            + .06 * ("def compute2" in p) for p in prompts]


class OnlineGainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(10)
        self.model = ConditionalGainModel(TinyEncoder(), width=16, bins=65,
                                          target_encoding="softsign_twohot")
        self.tok = ToyTokenizer()
        self.config = ContextConfig(1000, 800)

    def test_first_batch_proposals_do_not_need_labels_or_calibrated_support(self):
        a = prepare_gain_annotation(self.model, task(), self.tok, self.tok, self.config, 1, 64)
        b = prepare_gain_annotation(self.model, {**task(), "target_code": "PRIVATE_LABEL"},
                                    self.tok, self.tok, self.config, 1, 64)
        self.assertEqual(a["prompts"], b["prompts"])
        self.assertEqual(a["edges"], b["edges"])
        self.assertLessEqual(len(a["nodes"]), 24)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_score_before_single_optimizer_step_and_encoder_changes(self):
        events = []
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        original_step = optimizer.step
        old = self.model.encoder.linear.weight.detach().clone()

        def score(*args):
            events.append("score")
            self.assertTrue(torch.equal(old, self.model.encoder.linear.weight))
            return fake_scores(*args)

        def step():
            events.append("step")
            return original_step()

        optimizer.step = step
        with contextlib.redirect_stdout(None):
            metrics = online_gain_update(self.model, [(task(), 1), (task(), 2)], self.tok, self.tok,
                                         SimpleNamespace(score=score), optimizer, config=self.config)
        self.assertEqual(events, ["score", "step"])
        self.assertFalse(torch.equal(old, self.model.encoder.linear.weight))
        self.assertGreater(metrics["encoder_grad_norm"], 0.)
        self.assertGreater(metrics["encoder_update_relative"], 0.)
        self.assertLessEqual(metrics["contexts"], 48)

    def test_replay_matches_direct_encoder_update(self):
        replay_model = copy.deepcopy(self.model)
        for model, replay in ((self.model, False), (replay_model, True)):
            optimizer = torch.optim.SGD(model.parameters(), lr=.001)
            with contextlib.redirect_stdout(None):
                online_gain_update(model, [(task(), 2)], self.tok, self.tok,
                                   SimpleNamespace(score=fake_scores), optimizer,
                                   config=self.config, replay=replay, microbatch=2)
        for p, q in zip(self.model.parameters(), replay_model.parameters()):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)

    def test_next_batch_gain_above_two_still_updates(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        with contextlib.redirect_stdout(None):
            for scale in (1., 100.):
                oracle = SimpleNamespace(score=lambda *args: [scale * v for v in fake_scores(*args)])
                metrics = online_gain_update(self.model, [(task(), 2)], self.tok, self.tok,
                                             oracle, optimizer, config=self.config)
                self.assertGreater(metrics['encoder_update_relative'], 0.)
                self.assertTrue(torch.isfinite(torch.tensor(metrics['loss'])))

    def test_scoring_failure_does_not_mutate_weights(self):
        old = copy.deepcopy(self.model.state_dict())
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)

        def fail(*args):
            raise RuntimeError("generator failed")

        with self.assertRaisesRegex(RuntimeError, "generator failed"):
            online_gain_update(self.model, [(task(), 1)], self.tok, self.tok,
                               SimpleNamespace(score=fail), optimizer, config=self.config)
        for key, value in old.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]))

    def test_selector_recomputes_and_stops_without_gold(self):
        class RuleModel(ConditionalGainModel):
            def gain_logits(self, features, states, candidates, state_costs, candidate_costs):
                return torch.tensor([[.4 if not s and c == 0 else -.1] for s, c in zip(states, candidates)])

            def expected_gain(self, logits):
                return logits[:, 0]

        model = RuleModel(TinyEncoder(), width=16)
        without_gold = {key: value for key, value in task().items() if key != "target_code"}
        result = select_gain_context(model, without_gold, self.tok, self.tok, self.config)
        self.assertEqual(result["indices"], [0])
        self.assertEqual(result["stop_reason"], "no_predicted_improvement")


if __name__ == "__main__":
    unittest.main()
