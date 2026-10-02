import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.train.distributional_gain_model import ConditionalGainModel, histogram_loss


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(32, 8)
        self.projection = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.4)

    def forward(self, input_ids, attention_mask):
        values = self.dropout(self.projection(self.embedding(input_ids)))
        return SimpleNamespace(last_hidden_state=values)


class DistributionalGainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_model(self, bins=17):
        return ConditionalGainModel(TinyEncoder(), width=8, bins=bins)

    def test_signed_support_zero_mean_bias_and_persistence(self):
        model = self.make_model(bins=65)
        gains = torch.tensor([-0.8, -0.31, 0.0, 0.18, 0.71])
        model.configure_support(gains)
        targets = model.gain_targets(gains)
        radius = model.support_radius.item()
        centers = torch.linspace(-radius, radius, model.bins)
        encoded_means = targets @ centers

        self.assertTrue(model.support_calibrated.item())
        self.assertEqual(centers[model.bins // 2].item(), 0.0)
        self.assertLessEqual((encoded_means - gains).abs().max().item(),
                             0.1 * (2 * radius / (model.bins - 1)))
        restored = self.make_model(bins=65)
        restored.load_state_dict(model.state_dict())
        self.assertTrue(restored.support_calibrated.item())
        self.assertAlmostEqual(restored.support_radius.item(), radius)

    def test_bounded_twohot_handles_later_outliers_without_clipping(self):
        model = ConditionalGainModel(TinyEncoder(), width=8, bins=257,
                                     target_encoding="softsign_twohot")
        gains = torch.tensor([-1e20, -100., -2.02415, -.001, 0., .001, 2.02415, 100., 1e20])
        targets = model.gain_targets(gains)
        values = gains / (1. + gains.abs())
        torch.testing.assert_close(targets.sum(-1), torch.ones(len(gains)))
        torch.testing.assert_close(targets @ torch.linspace(-1., 1., 257), values, atol=2e-7, rtol=1e-6)
        self.assertTrue(torch.all((targets > 0).sum(-1) <= 2))
        self.assertTrue(torch.all(values[1:] > values[:-1]))
        logits = torch.zeros(len(gains), 257, requires_grad=True)
        loss, metrics = histogram_loss(logits, gains, model)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.norm()), 0.)
        self.assertLess(float(metrics['target_encoding_bias_max']), 2e-7)
        self.assertEqual(float(model.support_radius), 1.)
        with self.assertRaisesRegex(ValueError, 'finite'):
            model.gain_targets(torch.tensor([float('inf')]))

    def test_support_rejects_outliers_instead_of_clamping(self):
        model = self.make_model()
        nominal = model.expected_gain(torch.zeros(1, model.bins))
        self.assertEqual(nominal.shape, (1,))
        with self.assertRaisesRegex(ValueError, "all-zero calibration"):
            model.configure_support(torch.zeros(4))
        model.configure_support(torch.tensor([-0.4, 0.2, 0.5]))
        with self.assertRaisesRegex(ValueError, "support guard"):
            model.gain_targets(torch.tensor([model.support_radius.item()]))
        with self.assertRaisesRegex(RuntimeError, "already calibrated"):
            model.configure_support(torch.tensor([-0.5, 0.5]))

    def test_empty_set_permutation_and_state_conditioning(self):
        torch.manual_seed(9)
        model = self.make_model()
        model.configure_support(torch.tensor([-0.5, 0.6]))
        features = torch.randn(4, 8)
        candidate_costs = torch.tensor([0.1, 0.2, 0.3])

        empty = model.gain_logits(features, [()], [0], torch.tensor([0.0]),
                                  candidate_costs)
        one_state = model.gain_logits(features, [(1,)], [0], torch.tensor([0.2]),
                                      candidate_costs)
        ordered = model.gain_logits(features, [(1, 2)], [0], torch.tensor([0.3]),
                                    candidate_costs)
        permuted = model.gain_logits(features, [(2, 1)], [0], torch.tensor([0.3]),
                                     candidate_costs)

        self.assertEqual(empty.shape, (1, model.bins))
        self.assertTrue(torch.isfinite(empty).all())
        self.assertFalse(torch.allclose(empty, one_state))
        torch.testing.assert_close(ordered, permuted, atol=1e-6, rtol=1e-6)

    def test_head_and_loss_stay_fp32_under_autocast(self):
        torch.manual_seed(4)
        model = self.make_model()
        gains = torch.tensor([-0.3, 0.4])
        model.configure_support(gains)
        features = torch.randn(3, 8)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model.gain_logits(torch.randn(4, 8), [(), (1,)], [0, 2],
                                       torch.tensor([0.0, 0.2]),
                                       torch.tensor([0.1, 0.2, 0.3]))
            loss, metrics = histogram_loss(logits, gains, model)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(all(value.dtype == torch.float32 for value in metrics.values()))
        loss.backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_encoder_and_each_attention_head_block_receive_gradients(self):
        torch.manual_seed(21)
        model = self.make_model()
        ids = torch.randint(0, 32, (4, 7))
        mask = torch.ones_like(ids)
        features = model.encode(ids, mask)
        gains = torch.tensor([0.6, -0.3, 0.1])
        model.configure_support(gains)
        logits = model.gain_logits(
            features, [(), (1,), (2,)], [0, 2, 1], torch.tensor([0.0, 0.1, 0.2]),
            torch.tensor([0.1, 0.2, 0.3]))
        loss, _ = histogram_loss(logits, gains, model)
        loss.backward()

        for name, module in (
            ("encoder", model.encoder), ("project", model.project),
            ("item_cost", model.item_cost), ("extra_project", model.extra_project),
            ("attention block 0", model.attention_blocks[0]),
            ("attention block 1", model.attention_blocks[1]),
            ("gain head", model.gain_head),
        ):
            grads = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(grads and all(grad is not None for grad in grads), name)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in grads), name)
            self.assertGreater(sum(grad.norm().item() for grad in grads), 0.0, name)

        model.encoder.zero_grad(set_to_none=True)
        first = model.encode(ids, mask)
        second = model.encode(ids, mask)
        torch.testing.assert_close(first, second)
        self.assertFalse(model.encoder.training)

    def test_tiny_balanced_problem_overfits_sign_and_stop(self):
        torch.manual_seed(12)
        model = ConditionalGainModel(TinyEncoder(), width=32, bins=65,
                                     target_encoding="softsign_twohot")
        gains = torch.tensor([-0.8] * 8 + [0.0] * 8 + [0.8] * 8)
        features = torch.zeros(25, 8)
        features[1:9, 0] = -1.0
        features[17:25, 0] = 1.0
        candidate_costs = torch.zeros(24)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        initial_loss = None
        # Sanity check capacity/optimization, not a prediction of real epochs.
        for _ in range(120):
            logits = model.gain_logits(features, [()] * 24, list(range(24)),
                                       torch.zeros(24), candidate_costs)
            loss, _ = histogram_loss(logits, gains, model)
            if initial_loss is None:
                initial_loss = float(loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss.detach()), initial_loss * .5)

        logits = model.gain_logits(features, [()] * 24, list(range(24)),
                                   torch.zeros(24), candidate_costs)
        predicted = model.expected_gain(logits).detach()
        delta = 0.15
        self.assertTrue(torch.all(predicted[:8] < -delta))
        self.assertTrue(torch.all(predicted[8:16].abs() <= delta))
        self.assertTrue(torch.all(predicted[16:] > delta))


if __name__ == "__main__":
    unittest.main()
