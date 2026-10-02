"""Distributional prediction of signed marginal gains for retrieval candidates."""

import torch
from torch import nn
from torch.nn import functional as F


class _CandidateSetAttentionBlock(nn.Module):
    """One permutation-invariant cross-attention block for a candidate and set."""

    def __init__(self, width, heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.feed_forward_norm = nn.LayerNorm(width)

    def forward(self, candidate, memory, padding_mask):
        attended, _ = self.attention(
            candidate, memory, memory, key_padding_mask=padding_mask,
            need_weights=False)
        candidate = self.attention_norm(candidate + attended)
        return self.feed_forward_norm(candidate + self.feed_forward(candidate))


class ConditionalGainModel(nn.Module):
    """Predict a soft histogram for each (selected set, candidate) marginal gain.

    ``states`` and ``candidates`` are edge-aligned: row ``i`` predicts adding
    ``candidates[i]`` to ``states[i]``. Candidate indices address
    ``features[1:]``. Costs are expected to be normalized by the context budget;
    the query feature is always included in attention memory, including for an
    empty selected set. There are no positional encodings, so set order is
    irrelevant.
    """

    def __init__(self, encoder, width=256, bins=65, support_radius=1.0,
                 target_encoding="gaussian"):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError("width must be positive and divisible by four heads")
        if bins < 9 or bins % 2 == 0:
            raise ValueError("bins must be an odd integer >= 9 for a signed support")
        if not torch.isfinite(torch.tensor(float(support_radius))) or support_radius <= 0:
            raise ValueError("support_radius must be finite and positive")

        self.encoder = encoder
        self.width = width
        self.bins = bins
        if target_encoding not in {"gaussian", "softsign_twohot"}:
            raise ValueError("Unknown target encoding")
        if target_encoding == "softsign_twohot" and support_radius != 1.0:
            raise ValueError("Bounded gain encoding uses fixed support [-1, 1]")
        self.target_encoding = target_encoding
        self.project = nn.Linear(encoder.config.hidden_size, width)
        self.item_cost = nn.Linear(1, width, bias=False)
        self.extra_project = nn.Linear(3, width, bias=False)
        self.attention_blocks = nn.ModuleList(
            [_CandidateSetAttentionBlock(width, heads=4) for _ in range(2)])
        self.gain_head = nn.Linear(width, bins)

        # Legacy Gaussian targets need calibration; bounded two-hot targets use
        # fixed unit support from initialization, never the first batch's range.
        self.register_buffer("support_radius", torch.tensor(float(support_radius)))
        self.register_buffer("support_calibrated", torch.tensor(target_encoding == "softsign_twohot", dtype=torch.bool))

    def encode(self, ids, mask):
        """Mean-pool normalized encoder outputs; eval mode disables dropout only."""
        self.encoder.eval()
        states = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        weights = mask.to(dtype=torch.float32).unsqueeze(-1)
        pooled = (states * weights).sum(1) / weights.sum(1).clamp_min(1)
        return F.normalize(pooled.float(), dim=-1)

    def gain_logits(self, features, states, candidates, state_costs, candidate_costs):
        """Return FP32 logits ``[edges, bins]`` for aligned state/candidate pairs.

        ``state_costs`` has one normalized scalar per edge. ``candidate_costs``
        has one normalized scalar per candidate feature. Set cardinality is
        provided as a third scalar extra, scaled by ten as in the CUR baseline.
        """
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("features must contain a query followed by candidates")
        if len(states) != len(candidates) or not candidates:
            raise ValueError("states and candidates must be nonempty and edge-aligned")
        edge_count = len(candidates)
        candidate_count = features.shape[0] - 1
        if any(not isinstance(state, tuple) for state in states):
            raise TypeError("each state must be a tuple of selected candidate indices")
        if any(not isinstance(index, int) or not 0 <= index < candidate_count
               for state in states for index in state):
            raise IndexError("state contains an invalid candidate index")
        if any(not isinstance(index, int) or not 0 <= index < candidate_count
               for index in candidates):
            raise IndexError("candidate contains an invalid index")
        if any(candidate in state for state, candidate in zip(states, candidates)):
            raise ValueError("candidate must not already be in its selected state")

        device = features.device
        state_costs = torch.as_tensor(state_costs, device=device, dtype=torch.float32).reshape(-1)
        candidate_costs = torch.as_tensor(
            candidate_costs, device=device, dtype=torch.float32).reshape(-1)
        if state_costs.numel() != edge_count or candidate_costs.numel() != candidate_count:
            raise ValueError("cost vectors must align with edges and candidate features")
        if not torch.isfinite(state_costs).all() or not torch.isfinite(candidate_costs).all():
            raise ValueError("cost inputs must be finite")

        # Disabling autocast around the complete head keeps attention, logits,
        # and downstream loss arithmetic in FP32 even in a mixed-precision loop.
        with torch.autocast(device_type=device.type, enabled=False):
            projected = self.project(features.float())
            query = projected[0]
            candidate_ids = torch.tensor(candidates, device=device, dtype=torch.long)
            cost_features = candidate_costs[:, None]
            candidate_vectors = projected[1:] + self.item_cost(cost_features)

            max_memory = max(len(state) for state in states) + 1
            memory = projected.new_zeros((edge_count, max_memory, self.width))
            padding_mask = torch.ones(
                (edge_count, max_memory), device=device, dtype=torch.bool)
            memory[:, 0] = query
            padding_mask[:, 0] = False
            counts = []
            for row, state in enumerate(states):
                counts.append(len(state) / 10.0)
                if state:
                    indices = torch.tensor(state, device=device, dtype=torch.long)
                    end = len(state) + 1
                    memory[row, 1:end] = candidate_vectors.index_select(0, indices)
                    padding_mask[row, 1:end] = False

            selected_costs = candidate_costs.index_select(0, candidate_ids)
            count_features = torch.tensor(counts, device=device, dtype=torch.float32)
            extras = torch.stack((state_costs, selected_costs, count_features), dim=-1)
            candidate_tokens = (
                candidate_vectors.index_select(0, candidate_ids)
                + query.unsqueeze(0)
                + self.extra_project(extras)
            ).unsqueeze(1)
            for block in self.attention_blocks:
                candidate_tokens = block(candidate_tokens, memory, padding_mask)
            return self.gain_head(candidate_tokens[:, 0]).float()

    def configure_support(self, gains, *, radius=None):
        """Calibrate and freeze symmetric bin support from training gains only.

        By default the support is expanded to include the calibration batch with
        room for three smoothing standard deviations (or half the radius for a
        coarse support). Passing ``radius`` is an explicit training-only choice.
        A later target outside the safe interior raises; it is never clamped.
        """
        if bool(self.support_calibrated.item()):
            raise RuntimeError("gain support is already calibrated and frozen")
        gains = torch.as_tensor(gains, dtype=torch.float32).reshape(-1)
        if gains.numel() == 0 or not torch.isfinite(gains).all():
            raise ValueError("support calibration requires nonempty finite gains")
        max_abs = float(gains.abs().max().item())
        if radius is None:
            if max_abs == 0.0:
                raise ValueError("all-zero calibration batch; supply a training-only radius")
            guard_fraction = min(4.5 / (self.bins - 1), 0.5)
            factor = max(1.5, 1.1 / (1.0 - guard_fraction))
            radius = max_abs * factor + max(1e-6, max_abs * 1e-6)
        radius = float(radius)
        if not torch.isfinite(torch.tensor(radius)) or radius <= 0:
            raise ValueError("support radius must be finite and positive")
        self._check_gain_support(gains, radius)
        self.support_radius.fill_(radius)
        self.support_calibrated.fill_(True)

    def _bin_geometry(self, device):
        radius = self.support_radius.to(device=device, dtype=torch.float32)
        centers = torch.linspace(-1.0, 1.0, self.bins, device=device) * radius
        spacing = 2.0 * radius / (self.bins - 1)
        sigma = 0.75 * spacing
        return radius, centers, spacing, sigma

    def _check_gain_support(self, gains, radius=None):
        if radius is None:
            radius = float(self.support_radius.item())
        spacing = 2.0 * radius / (self.bins - 1)
        sigma = 0.75 * spacing
        guard = min(3.0 * sigma, 0.5 * radius)
        safe_edge = radius - guard
        if torch.any(gains.abs() >= safe_edge):
            maximum = float(gains.abs().max().item())
            raise ValueError(
                f"gain magnitude {maximum:.6g} reaches support guard {safe_edge:.6g}; "
                "widen support using training data, do not clip")

    def gain_targets(self, gains):
        """Encode gains with the configured Gaussian or bounded two-hot targets."""
        if not bool(self.support_calibrated.item()):
            raise RuntimeError("call configure_support() before constructing gain targets")
        gains = torch.as_tensor(gains, device=self.support_radius.device,
                                dtype=torch.float32).reshape(-1)
        if gains.numel() == 0 or not torch.isfinite(gains).all():
            raise ValueError("gain targets must be nonempty and finite")
        if self.target_encoding == "softsign_twohot":
            # Strictly monotone, sign-preserving, no clipping or batch-dependent
            # redefinition of bin centers. Every finite gain has a valid label.
            value = self.target_values(gains)
            position = (value + 1.) * ((self.bins - 1) / 2.)
            lower = position.floor().long()
            upper = (lower + 1).clamp_max(self.bins - 1)
            fraction = position - lower.float()
            targets = gains.new_zeros((len(gains), self.bins))
            targets.scatter_add_(1, lower[:, None], (1.-fraction)[:, None])
            targets.scatter_add_(1, upper[:, None], fraction[:, None])
            return targets
        with torch.autocast(device_type=gains.device.type, enabled=False):
            radius, centers, spacing, sigma = self._bin_geometry(gains.device)
            self._check_gain_support(gains, float(radius.item()))
            boundaries = (centers[:-1] + centers[1:]) * 0.5
            boundaries = torch.cat((
                centers.new_full((1,), -torch.inf), boundaries,
                centers.new_full((1,), torch.inf)))
            standardized = (boundaries[None, :] - gains[:, None]) / sigma
            cdf = 0.5 * (1.0 + torch.erf(standardized / (2.0 ** 0.5)))
            targets = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0)
            targets = targets / targets.sum(-1, keepdim=True).clamp_min(1e-30)
            encoded_mean = targets @ centers
            max_bias = (encoded_mean - gains).abs().max()
            tolerance = 0.1 * spacing
            if float(max_bias.item()) > float(tolerance.item()):
                raise ValueError(
                    "integrated HL-Gauss target does not preserve gain mean within "
                    f"0.1 bin width (max bias={float(max_bias.item()):.6g})")
            return targets

    def target_values(self, gains):
        """Metric/selection domain; softsign predictions are NOT raw logprob gains."""
        return gains / (1. + gains.abs()) if self.target_encoding == "softsign_twohot" else gains

    def expected_gain(self, logits):
        """Expected signed score in the configured target domain (not its inverse)."""
        if logits.ndim != 2 or logits.shape[-1] != self.bins:
            raise ValueError("logits must have shape [edges, bins]")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            _, centers, _, _ = self._bin_geometry(logits.device)
            return torch.softmax(logits.float(), dim=-1) @ centers


def histogram_loss(logits, gains, model):
    """Histogram cross entropy; MAE/bias are in the configured target domain."""
    if logits.ndim != 2 or logits.shape[-1] != model.bins:
        raise ValueError("logits must have shape [edges, model.bins]")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        gains = torch.as_tensor(gains, device=logits.device, dtype=torch.float32).reshape(-1)
        if gains.numel() != logits.shape[0]:
            raise ValueError("gain targets must align with logits")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite gain logits")
        targets = model.gain_targets(gains)
        values = model.target_values(gains)
        loss = -(targets * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        expected = model.expected_gain(logits)
        predicted_probabilities = torch.softmax(logits, dim=-1)
        radius, _, _, _ = model._bin_geometry(logits.device)
        edge_mass = predicted_probabilities[:, 0] + predicted_probabilities[:, -1]
        metrics = {
            "gain_mae": (expected - values).abs().mean().detach(),
            "gain_bias": (expected - values).mean().detach(),
            "gain_sign_accuracy": (expected.sign() == gains.sign()).float().mean().detach(),
            "target_encoding_bias_max": (targets @ model._bin_geometry(logits.device)[1]
                                          - values).abs().max().detach(),
            "predicted_edge_mass": edge_mass.mean().detach(),
            "support_radius": radius.detach(),
        }
        return loss, metrics
